# main/retrieval/make_sliding_fragment_queries_grouped.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from preproc.simplemono_preproc.npy_stream import NpyAppendWriter
from .simplemono_rel import load_decoding_cfg, infer_used_len_events


def load_queries_meta(meta_jsonl: str) -> List[Dict[str, Any]]:
    out = []
    with Path(meta_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    out.sort(key=lambda x: int(x["query_id"]))
    return out


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    return out


def _iter_starts(n_notes: int, frag_len: int, step: int, *, cover_end: bool = True):
    if n_notes < frag_len:
        return
    step = max(1, int(step))
    last = n_notes - frag_len
    st = 0
    while st <= last:
        yield st
        st += step
    if cover_end and last > 0:
        # ensure the tail is covered
        if (st - step) != last:
            yield last


def _slice_fragment(events: np.ndarray, frag_start_note: int, frag_len_notes: int, *, pad_id: int, eos_id: int) -> np.ndarray:
    """
    events: [L,3] global
    fragment is defined over note rows (excluding BOS/EOS):
      note rows are [1 : used_len-1)
    output: [L,3] global, BOS + frag_notes + EOS + PAD
    """
    L = int(events.shape[0])
    out = np.full((L, 3), int(pad_id), dtype=np.int32)
    out[0] = events[0].astype(np.int32, copy=False)

    st = 1 + int(frag_start_note)
    ed = st + int(frag_len_notes)

    out[1 : 1 + frag_len_notes] = events[st:ed].astype(np.int32, copy=False)
    out[1 + frag_len_notes] = np.array([int(eos_id), int(eos_id), int(eos_id)], dtype=np.int32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries_npy", type=str, required=True)
    ap.add_argument("--queries_meta_jsonl", type=str, required=True)
    ap.add_argument("--simplemono_pkl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    # windowing
    ap.add_argument("--frag_lens", type=str, default="24,32,48,64")
    ap.add_argument("--stride_ratio", type=float, default=0.5, help="step = round(frag_len * stride_ratio)")
    ap.add_argument("--stride_notes", type=int, default=0, help="override step_notes if >0")
    ap.add_argument("--fallback_full", action="store_true", help="if too short for all frag_lens, use full seq as 1 fragment")
    ap.add_argument("--min_notes_for_fallback", type=int, default=8)

    ap.add_argument("--write_frag_meta_jsonl", action="store_true")
    ap.add_argument("--max_queries", type=int, default=0, help="0 => all")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_dec = load_decoding_cfg(args.simplemono_pkl)
    qmeta = load_queries_meta(args.queries_meta_jsonl)

    queries = np.load(args.queries_npy, mmap_mode="r")
    if queries.ndim != 3 or queries.shape[-1] != 3:
        raise ValueError(f"Bad queries shape: {queries.shape}")

    N = min(int(queries.shape[0]), len(qmeta))
    if int(args.max_queries) > 0:
        N = min(N, int(args.max_queries))

    L = int(queries.shape[1])
    frag_lens = _parse_int_list(args.frag_lens)
    if not frag_lens:
        raise ValueError("frag_lens is empty")
    frag_lens = sorted(set(int(x) for x in frag_lens if int(x) > 0))

    w = NpyAppendWriter(out_dir / "fragments.npy", dtype=np.int32, row_shape=(L, 3))

    frag_meta_f = None
    if bool(args.write_frag_meta_jsonl):
        frag_meta_f = (out_dir / "fragments_meta.jsonl").open("w", encoding="utf-8")

    offsets = np.zeros((N + 1,), dtype=np.int32)
    total_frags = 0
    empty_q = 0

    for qid in tqdm(range(N), desc="[MakeFragments] sliding windows", dynamic_ncols=True, smoothing=0.0):
        ev = np.asarray(queries[qid], dtype=np.int32)
        used_len = infer_used_len_events(ev, cfg_dec)
        n_notes = max(0, int(used_len) - 2)

        frag_defs: List[Tuple[int, int]] = []

        for fl in frag_lens:
            if n_notes < fl:
                continue
            step = int(args.stride_notes) if int(args.stride_notes) > 0 else int(round(float(fl) * float(args.stride_ratio)))
            step = max(1, step)
            for st in _iter_starts(n_notes, fl, step, cover_end=True):
                frag_defs.append((int(st), int(fl)))

        if not frag_defs and bool(args.fallback_full) and n_notes >= int(args.min_notes_for_fallback):
            frag_defs = [(0, int(n_notes))]

        if not frag_defs:
            empty_q += 1
            offsets[qid + 1] = offsets[qid]
            continue

        for st, fl in frag_defs:
            frag = _slice_fragment(ev, st, fl, pad_id=int(cfg_dec.pad_id), eos_id=int(cfg_dec.eos_id))
            w.append(frag)
            fid = int(w.count - 1)

            if frag_meta_f is not None:
                frag_meta_f.write(json.dumps({
                    "fragment_id": fid,
                    "query_id": int(qid),
                    "target_doc_id": int(qmeta[qid]["target_doc_id"]),
                    "frag_start_note": int(st),
                    "frag_len_notes": int(fl),
                    "used_len_events": int(fl + 2),
                }, ensure_ascii=False) + "\n")

        total_frags += len(frag_defs)
        offsets[qid + 1] = int(w.count)

    w.close()
    if frag_meta_f is not None:
        frag_meta_f.close()

    np.save(out_dir / "frag_offsets.npy", offsets)

    meta = {
        "version": "sliding_fragments_grouped_v1",
        "queries_npy": str(Path(args.queries_npy)),
        "queries_meta_jsonl": str(Path(args.queries_meta_jsonl)),
        "simplemono_pkl": str(Path(args.simplemono_pkl)),
        "N_queries": int(N),
        "L_events": int(L),
        "frag_lens": frag_lens,
        "stride_ratio": float(args.stride_ratio),
        "stride_notes": int(args.stride_notes),
        "fallback_full": bool(args.fallback_full),
        "min_notes_for_fallback": int(args.min_notes_for_fallback),
        "total_fragments": int(total_frags),
        "avg_frags_per_query": float(total_frags / max(N, 1)),
        "empty_queries": int(empty_q),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[Saved] fragments.npy  (N={int(offsets[-1])}) -> {out_dir / 'fragments.npy'}")
    print(f"[Saved] frag_offsets.npy (shape={offsets.shape}) -> {out_dir / 'frag_offsets.npy'}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
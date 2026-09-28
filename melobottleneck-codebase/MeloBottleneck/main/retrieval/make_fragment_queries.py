# main/retrieval/make_fragment_queries.py
from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..vocab_utils import load_vocab_info
from ..quantization import MusicQuantizationTables
from ..ornament import MusicOrnamenter, MusicOrnamentConfig
from .simplemono_rel import load_decoding_cfg, infer_used_len_events

from .corrupt import MusicCorruptionConfig, MusicCorruptor


def _filter_kwargs_for_dataclass(dc_cls, d: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in (d or {}).items() if k in allowed}


def _invert_global2local(global2local: np.ndarray, n_local: int) -> np.ndarray:
    g2l = np.asarray(global2local, dtype=np.int64)
    local2global = np.full(n_local, -1, dtype=np.int64)
    valid = (g2l >= 0) & (g2l < n_local)
    local2global[g2l[valid]] = np.nonzero(valid)[0]
    return local2global


def build_quant_tables_from_vocabinfo(vocab) -> MusicQuantizationTables:
    return MusicQuantizationTables(
        special_n=int(vocab.special_n),
        duration_code_to_pos=vocab.duration_code_to_pos,
        duration_pos_to_code=vocab.duration_pos_to_code,
        deltatime_code_offset=int(vocab.deltatime_code_offset),
        deltatime_code_to_pos=vocab.deltatime_code_to_pos,
        deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    )


def make_one_query(
    *,
    doc_events: np.ndarray,          # [L,3] global
    used_len_doc: int,
    frag_start_note: int,
    frag_len_notes: int,
    cfg_dec,
) -> np.ndarray:
    """
    从 doc_events 里截取 fragment，返回 query_events (global IDs, [L,3])。
    """
    L = int(doc_events.shape[0])
    pad = int(cfg_dec.pad_id)
    eos = int(cfg_dec.eos_id)

    # note rows in doc: [1 : used_len-1)
    n_notes = max(0, int(used_len_doc) - 2)
    if frag_len_notes <= 0 or frag_start_note < 0 or frag_start_note + frag_len_notes > n_notes:
        raise ValueError("Bad fragment range.")

    st = 1 + int(frag_start_note)
    ed = st + int(frag_len_notes)

    out = np.full((L, 3), pad, dtype=np.int32)
    out[0] = doc_events[0].astype(np.int32, copy=False)  # copy BOS row
    out[1 : 1 + frag_len_notes] = doc_events[st:ed].astype(np.int32, copy=False)
    out[1 + frag_len_notes] = np.array([eos, eos, eos], dtype=np.int32)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_npy", type=str, required=True)
    ap.add_argument("--simplemono_pkl", type=str, required=True)

    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_queries", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--frag_min_notes", type=int, default=32)
    ap.add_argument("--frag_max_notes", type=int, default=96)

    # ornament
    ap.add_argument("--ornament_json", type=str, default=None, help="e.g. ornament_ood.json")
    ap.add_argument("--ornament_apply", action="store_true", help="Enable ornamentation")
    ap.add_argument("--ornament_p_apply", type=float, default=1.0)
    ap.add_argument("--max_extra_tokens", type=int, default=128)
    ap.add_argument("--save_pi_npy", type=str, default=None)

    # corruption
    ap.add_argument("--corrupt_apply", action="store_true", help="Enable corruption (transcription/recording errors)")
    ap.add_argument("--corrupt_json", type=str, default=None, help="Optional corruption config json")
    ap.add_argument("--corrupt_p_apply", type=float, default=1.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = np.load(args.corpus_npy, mmap_mode="r")
    if corpus.ndim != 3 or corpus.shape[-1] != 3:
        raise ValueError(f"Bad corpus shape: {corpus.shape}")
    N, L, _ = corpus.shape

    cfg_dec = load_decoding_cfg(args.simplemono_pkl)

    # ornamenter (optional)
    orn = None
    local2global_pitch = local2global_dur = local2global_dt = None

    if args.ornament_apply:
        vocab = load_vocab_info(args.simplemono_pkl)
        qt = build_quant_tables_from_vocabinfo(vocab)

        # build local->global tables
        local2global_pitch = _invert_global2local(vocab.global2local_pitch, vocab.n_pitch)
        local2global_dur = _invert_global2local(vocab.global2local_duration, vocab.n_duration)
        local2global_dt = _invert_global2local(vocab.global2local_dt, vocab.n_dt)

        # local special ids (safe)
        pad_local = int(vocab.global2local_pitch[int(vocab.pad_id)])
        bos_local = int(vocab.global2local_pitch[int(vocab.bos_id)])
        eos_local = int(vocab.global2local_pitch[int(vocab.eos_id)])

        base_cfg = MusicOrnamentConfig(
            enable=True,
            p_apply=float(args.ornament_p_apply),
            pad_id=pad_local,
            bos_id=bos_local,
            eos_id=eos_local,
            max_extra_tokens=int(args.max_extra_tokens),
            quantization_tables=qt,
        )

        if args.ornament_json is not None:
            od = json.loads(Path(args.ornament_json).read_text(encoding="utf-8"))
            od = _filter_kwargs_for_dataclass(MusicOrnamentConfig, od)
            base_cfg = replace(base_cfg, **od)

        orn = MusicOrnamenter(base_cfg)

    corruptor = None
    if args.corrupt_apply:
        cor_cfg = MusicCorruptionConfig(enable=True, p_apply=float(args.corrupt_p_apply))
        if args.corrupt_json is not None:
            cd = json.loads(Path(args.corrupt_json).read_text(encoding="utf-8"))
            cd = _filter_kwargs_for_dataclass(MusicCorruptionConfig, cd)
            cor_cfg = replace(cor_cfg, **cd)
        corruptor = MusicCorruptor(cor_cfg, cfg_dec)

        if args.save_pi_npy is not None:
            print("[WARN] save_pi_npy will reflect ornament pi BEFORE corruption; "
                  "merge/delete may invalidate alignment.")

    rng = np.random.default_rng(int(args.seed))

    queries = np.full((int(args.n_queries), int(L), 3), int(cfg_dec.pad_id), dtype=np.int32)
    pi_out = None
    if args.save_pi_npy is not None:
        pi_out = np.full((int(args.n_queries), int(L)), -2, dtype=np.int32)

    meta_path = out_dir / "queries_meta.jsonl"
    q_path = out_dir / "queries.npy"
    pi_path = Path(args.save_pi_npy) if args.save_pi_npy is not None else None

    with meta_path.open("w", encoding="utf-8") as f:
        made = 0
        pbar = tqdm(total=int(args.n_queries), desc="[MakeQueries] fragment", dynamic_ncols=True)

        while made < int(args.n_queries):
            doc_id = int(rng.integers(0, N))
            doc_ev = np.asarray(corpus[doc_id], dtype=np.int32)

            used_len_doc = infer_used_len_events(doc_ev, cfg_dec)
            n_notes = max(0, int(used_len_doc) - 2)
            if n_notes < int(args.frag_min_notes):
                continue

            frag_len = int(rng.integers(int(args.frag_min_notes), int(args.frag_max_notes) + 1))
            frag_len = min(frag_len, n_notes)
            if frag_len <= 0:
                continue

            start = int(rng.integers(0, n_notes - frag_len + 1))

            q_clean = make_one_query(
                doc_events=doc_ev,
                used_len_doc=used_len_doc,
                frag_start_note=start,
                frag_len_notes=frag_len,
                cfg_dec=cfg_dec,
            )

            q_final = q_clean
            pi = None

            if orn is not None:
                # global -> local
                vocab = load_vocab_info(args.simplemono_pkl)  # small; ok for now
                q_local = q_clean.copy()
                q_local[:, 0] = vocab.global2local_pitch[q_local[:, 0]]
                q_local[:, 1] = vocab.global2local_duration[q_local[:, 1]]
                q_local[:, 2] = vocab.global2local_dt[q_local[:, 2]]

                q_aug_local, pi = orn.augment(q_local, rng=rng, max_extra_tokens=int(args.max_extra_tokens))

                # local -> global
                q_aug_local = np.asarray(q_aug_local, dtype=np.int64)
                q_global = np.empty_like(q_aug_local, dtype=np.int64)
                q_global[:, 0] = local2global_pitch[q_aug_local[:, 0]]
                q_global[:, 1] = local2global_dur[q_aug_local[:, 1]]
                q_global[:, 2] = local2global_dt[q_aug_local[:, 2]]
                q_final = q_global.astype(np.int32)

            if corruptor is not None:
                q_final = corruptor.corrupt(q_final, rng=rng)

            queries[made] = q_final
            if pi_out is not None and pi is not None:
                pi_out[made] = np.asarray(pi, dtype=np.int32)

            used_len_q = infer_used_len_events(q_final, cfg_dec)

            f.write(json.dumps({
                "query_id": made,
                "target_doc_id": doc_id,
                "frag_start_note": start,
                "frag_len_notes": frag_len,
                "used_len_events": int(used_len_q),
                "ornament": bool(orn is not None),
                "corruption": bool(corruptor is not None),
            }, ensure_ascii=False) + "\n")

            made += 1
            pbar.update(1)

        pbar.close()

    np.save(q_path, queries)
    print(f"[Saved] {q_path} | shape={queries.shape}")

    if pi_out is not None and pi_path is not None:
        np.save(pi_path, pi_out)
        print(f"[Saved] {pi_path} | shape={pi_out.shape}")

    print(f"[Saved] {meta_path}")


if __name__ == "__main__":
    main()
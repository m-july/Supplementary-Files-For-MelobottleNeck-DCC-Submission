# preproc/preproc_tavern_vt_retrieval.py
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .simplemono_preproc.npy_stream import NpyAppendWriter
from .simplemono_preproc.vocab import SimpleMonoVocab
from .simplemono_preproc.pipelines.pretrain import default_max_len_tokens

# reuse from your existing TAVERN silver script
from .preproc_tavern_silver_benchmark import (
    list_variation_theme_pairs_from_score,
    load_treble_melody_notes,
    encode_notes_to_padded_events_global,
)


def _relpath_posix(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def _load_jsonl(path: Path, key: str) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    out.sort(key=lambda x: int(x[key]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tavern_root", type=str, required=True, help="TAVERN-master root")
    ap.add_argument("--simplemono_pkl", type=str, required=True, help="SimpleMono.pkl (bookkeeping + later decoding cfg)")
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--split_name", type=str, default="test")

    ap.add_argument("--max_len_tokens", type=int, default=0, help="0 => default_max_len_tokens(); must be divisible by 3")
    ap.add_argument("--melody_mode", type=str, default="top", choices=["top", "all"])

    ap.add_argument("--min_theme_notes", type=int, default=8)
    ap.add_argument("--min_var_notes", type=int, default=8)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 => all")

    ap.add_argument("--write_meta_jsonl", action="store_true", help="write themes_meta.jsonl + variations_meta.jsonl")
    args = ap.parse_args()

    tavern_root = Path(args.tavern_root)
    out_root = Path(args.output_dir) / str(args.split_name)

    if out_root.exists():
        raise FileExistsError(f"Output dir already exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)

    max_len_tokens = int(args.max_len_tokens)
    if max_len_tokens <= 0:
        max_len_tokens = int(default_max_len_tokens())
    if max_len_tokens % 3 != 0:
        raise ValueError(f"max_len_tokens must be divisible by 3, got {max_len_tokens}")
    max_events = max_len_tokens // 3
    max_notes = max_events - 2

    print(f"[TAVERN] {tavern_root}")
    print(f"[OUT]   {out_root}")
    print(f"[LEN]   max_len_tokens={max_len_tokens} => max_events={max_events}")
    print(f"[MODE]  melody_mode={args.melody_mode}")

    pairs = list_variation_theme_pairs_from_score(tavern_root)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    print(f"[PAIR]  found {len(pairs)} variation-theme phrase pairs.")

    vocab_global = SimpleMonoVocab.build()

    # -----------------------------
    # pass 1: filter pairs by parsing + min length (avoid dangling docs)
    # -----------------------------
    note_cache: Dict[str, list] = {}

    kept_pairs: List[Dict[str, Any]] = []
    skip = {
        "parse_or_empty_theme": 0,
        "parse_or_empty_var": 0,
        "too_short_theme": 0,
        "too_short_var": 0,
        "exception": 0,
    }

    for p in tqdm(pairs, desc="[Scan] parse+filter", dynamic_ncols=True, smoothing=0.0):
        try:
            th = load_treble_melody_notes(Path(p["theme_file"]), melody_mode=args.melody_mode, cache=note_cache)
            if not th:
                skip["parse_or_empty_theme"] += 1
                continue
            if len(th) < int(args.min_theme_notes):
                skip["too_short_theme"] += 1
                continue

            va = load_treble_melody_notes(Path(p["variation_file"]), melody_mode=args.melody_mode, cache=note_cache)
            if not va:
                skip["parse_or_empty_var"] += 1
                continue
            if len(va) < int(args.min_var_notes):
                skip["too_short_var"] += 1
                continue

            kept_pairs.append(p)
        except Exception:
            skip["exception"] += 1
            continue

    print(f"[KEEP]  kept_pairs={len(kept_pairs)} / {len(pairs)}")
    print(f"[SKIP]  {skip}")

    # -----------------------------
    # build theme doc list (unique)
    # -----------------------------
    theme_entries: Dict[str, Dict[str, Any]] = {}
    for p in kept_pairs:
        theme_path = Path(p["theme_file"]).resolve()
        k = str(theme_path)
        if k not in theme_entries:
            theme_entries[k] = {
                "path": theme_path,
                "opus": p["opus"],
                "theme_phrase_base": p["theme_phrase_base"],
            }

    themes_sorted = sorted(
        theme_entries.values(),
        key=lambda d: (str(d["opus"]), str(d["theme_phrase_base"]), str(d["path"].name)),
    )

    # -----------------------------
    # writers
    # -----------------------------
    w_docs = NpyAppendWriter(out_root / "themes.npy", dtype=np.int32, row_shape=(max_events, 3))
    w_q = NpyAppendWriter(out_root / "variations.npy", dtype=np.int32, row_shape=(max_events, 3))

    f_docs_meta = None
    f_q_meta = None
    if bool(args.write_meta_jsonl):
        f_docs_meta = (out_root / "themes_meta.jsonl").open("w", encoding="utf-8")
        f_q_meta = (out_root / "variations_meta.jsonl").open("w", encoding="utf-8")

    theme_path_to_docid: Dict[str, int] = {}

    # -----------------------------
    # encode themes (docs)
    # -----------------------------
    for th in tqdm(themes_sorted, desc="[Encode] themes", dynamic_ncols=True, smoothing=0.0):
        theme_notes = load_treble_melody_notes(th["path"], melody_mode=args.melody_mode, cache=note_cache)
        theme_notes_t = theme_notes[:max_notes]
        ev, used_len = encode_notes_to_padded_events_global(theme_notes_t, vocab_global, max_events=max_events)

        w_docs.append(ev.astype(np.int32, copy=False))
        doc_id = int(w_docs.count - 1)
        theme_path_to_docid[str(Path(th["path"]).resolve())] = doc_id

        if f_docs_meta is not None:
            f_docs_meta.write(json.dumps({
                "doc_id": doc_id,
                "opus": th["opus"],
                "theme_phrase_base": th["theme_phrase_base"],
                "theme_file": _relpath_posix(Path(th["path"]), tavern_root),
                "theme_notes": int(len(theme_notes)),
                "theme_notes_trunc": int(len(theme_notes_t)),
                "used_len_events": int(used_len),
            }, ensure_ascii=False) + "\n")

    # -----------------------------
    # encode variations (queries)
    # -----------------------------
    for p in tqdm(kept_pairs, desc="[Encode] variations", dynamic_ncols=True, smoothing=0.0):
        theme_path = str(Path(p["theme_file"]).resolve())
        tgt = theme_path_to_docid.get(theme_path, None)
        if tgt is None:
            continue

        var_path = Path(p["variation_file"]).resolve()
        var_notes = load_treble_melody_notes(var_path, melody_mode=args.melody_mode, cache=note_cache)
        var_notes_t = var_notes[:max_notes]
        ev, used_len = encode_notes_to_padded_events_global(var_notes_t, vocab_global, max_events=max_events)

        w_q.append(ev.astype(np.int32, copy=False))
        qid = int(w_q.count - 1)

        if f_q_meta is not None:
            f_q_meta.write(json.dumps({
                "query_id": qid,
                "target_doc_id": int(tgt),
                "opus": p["opus"],
                "variation": p["variation"],
                "theme_phrase_base": p["theme_phrase_base"],
                "var_phrase_key": p["var_phrase_key"],
                "theme_file": _relpath_posix(Path(p["theme_file"]), tavern_root),
                "variation_file": _relpath_posix(var_path, tavern_root),
                "var_notes": int(len(var_notes)),
                "var_notes_trunc": int(len(var_notes_t)),
                "used_len_events": int(used_len),
            }, ensure_ascii=False) + "\n")

    # close
    w_docs.close()
    w_q.close()
    if f_docs_meta is not None:
        f_docs_meta.close()
    if f_q_meta is not None:
        f_q_meta.close()

    meta = {
        "version": "tavern_vt_retrieval_v1",
        "tavern_root": str(tavern_root),
        "split_name": str(args.split_name),
        "simplemono_pkl": str(args.simplemono_pkl),
        "melody_mode": str(args.melody_mode),
        "max_len_tokens": int(max_len_tokens),
        "max_events": int(max_events),
        "max_notes": int(max_notes),
        "filters": {
            "min_theme_notes": int(args.min_theme_notes),
            "min_var_notes": int(args.min_var_notes),
        },
        "counts": {
            "pairs_found": int(len(pairs)),
            "pairs_kept": int(len(kept_pairs)),
            "themes": int(len(themes_sorted)),
            "variations": int(w_q.count),
            "skip": {k: int(v) for k, v in skip.items()},
        },
    }
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[DONE] TAVERN VT retrieval pack built.")
    print(json.dumps(meta["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
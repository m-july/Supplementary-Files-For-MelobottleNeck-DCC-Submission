# decode_npy_to_midi.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .simplemono_preproc.decoder_midi import (
    SimpleMonoDecodingConfig,
    decode_events_to_midi,
    sanitize_filename,
)


def load_metadata_by_row(metadata_path: Path, split: str) -> Dict[int, dict]:
    meta: Dict[int, dict] = {}
    with Path(metadata_path).open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("split") != split:
                continue
            meta[int(obj["row_idx"])] = obj
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--simplemono_pkl", type=str, default="",
                    help="Path to SimpleMono.pkl. If empty, will try <npy_dir>/SimpleMono.pkl")
    ap.add_argument("--metadata", type=str, default="",
                    help="Path to metadata.jsonl (optional but recommended for used_len + naming)")

    ap.add_argument("--split", type=str, default="",
                    help="train|valid|test (if empty, infer from npy filename stem)")
    ap.add_argument("--metadata_split", type=str, default="",
                    help="Split name to look up in metadata.jsonl (if different from --split / npy stem). "
                         "Useful for derived files like test_skeleton.npy whose metadata is stored under 'test'.")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--program", type=int, default=0)
    ap.add_argument("--tempo", type=float, default=80.0)
    ap.add_argument("--velocity", type=int, default=80)
    ap.add_argument("--strict", action="store_true")

    args = ap.parse_args()

    npy_path = Path(args.npy_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = args.split.strip() or npy_path.stem  # usually train/valid/test

    pkl_path = Path(args.simplemono_pkl) if args.simplemono_pkl else (npy_path.parent / "SimpleMono.pkl")
    if not pkl_path.exists():
        raise FileNotFoundError(f"SimpleMono.pkl not found: {pkl_path}")
    cfg = SimpleMonoDecodingConfig.from_simplemono_pkl(pkl_path)

    arr = np.load(npy_path, mmap_mode="r")  # (N, T, 3)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Bad npy shape: {arr.shape}")

    N = int(arr.shape[0])
    st = max(0, int(args.start))
    ed = N if args.end < 0 else min(N, int(args.end))
    if args.limit > 0:
        ed = min(ed, st + int(args.limit))

    meta_by_row: Dict[int, dict] = {}
    if args.metadata:
        meta_split = args.metadata_split.strip() or split
        meta_by_row = load_metadata_by_row(Path(args.metadata), split=meta_split)

    for i in range(st, ed):
        events = arr[i]
        meta = meta_by_row.get(i)
        used_len = int(meta["used_len_events"]) if meta is not None else None

        # build output filename
        if meta is not None:
            src = Path(meta["source_file"])  # posix relpath in jsonl, Path can handle it
            stem = sanitize_filename(src.stem)
            voice = sanitize_filename(meta.get("voice_id", "voice"))
            win = int(meta.get("window_idx", 0))

            src_rel = src.parent if not src.parent.is_absolute() else Path(src.parent.name)
            subdir = out_dir / split / src_rel
            subdir.mkdir(parents=True, exist_ok=True)

            out_mid = subdir / f"{i:08d}__{stem}__{voice}__w{win}.mid"
        else:
            subdir = out_dir / split
            subdir.mkdir(parents=True, exist_ok=True)
            out_mid = subdir / f"{i:08d}.mid"

        decode_events_to_midi(
            events,
            out_mid,
            cfg,
            used_len_events=used_len,
            strict=args.strict,
            program=args.program,
            velocity=args.velocity,
            tempo_bpm=args.tempo,
        )

    print(f"[DONE] decoded {ed - st} sequences from {npy_path} to {out_dir / split}")


if __name__ == "__main__":
    main()

# python .\decode_npy_to_midi.py --npy_path .\output\anthology_v251218_lyrics_included\train.npy --metadata .\output\anthology_v251218_lyrics_included\metadata.jsonl --out_dir .\output\anthology_v251218_lyrics_included\decoded_midis_train

# python -m preproc.decode_npy_to_midi --npy_path .\preproc\output\anthology_v251218_lyrics_included\test_skeleton.npy --metadata .\preproc\output\anthology_v251218_lyrics_included\metadata.jsonl --out_dir .\preproc\output\anthology_v251218_lyrics_included\decoded_midis_test_skeleton --metadata_split test

# python .\decode_npy_to_midi.py --npy_path .\output\anthology_v251218_lyrics_included\test.npy --metadata .\output\anthology_v251218_lyrics_included\metadata.jsonl --out_dir .\output\anthology_v251218_lyrics_included\decoded_midis_test

# python -m preproc.decode_npy_to_midi --npy_path .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\train_aug_orn.npy --metadata .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\metadata.jsonl --out_dir .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\decoded_train_aug_orn --metadata_split train

# python -m preproc.decode_npy_to_midi --npy_path .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\train.npy --metadata .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\metadata.jsonl --out_dir .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\decoded_train --metadata_split train
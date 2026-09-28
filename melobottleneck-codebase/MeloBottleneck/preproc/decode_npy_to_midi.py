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


def infer_used_len_from_events(
    events: np.ndarray,
    *,
    eos_id: int,
    pad_id: int,
) -> int:
    """
    从事件内容本身推断有效长度：
    1) 若存在 EOS，则返回 EOS 位置（含 EOS）
    2) 否则若存在 PAD，则返回首个 PAD 位置
    3) 否则返回整条长度

    注意：这里看的只是 pitch 列，因为整个工程里 EOS/PAD/BOS 等 special
    token 是按三列同步写入的，而 decode 逻辑本来也主要依赖 pitch 列判断 special。
    """
    ev = np.asarray(events)
    if ev.ndim != 2 or ev.shape[1] != 3:
        raise ValueError(f"events must be (T,3), got {ev.shape}")

    pitch = ev[:, 0].astype(np.int64, copy=False)

    eos_idx = np.flatnonzero(pitch == int(eos_id))
    if eos_idx.size > 0:
        return int(eos_idx[0]) + 1

    pad_idx = np.flatnonzero(pitch == int(pad_id))
    if pad_idx.size > 0:
        return int(pad_idx[0])

    return int(ev.shape[0])


def resolve_used_len_events(
    events: np.ndarray,
    *,
    cfg: SimpleMonoDecodingConfig,
    metadata_used_len: Optional[int],
    policy: str,
    row_idx: Optional[int] = None,
) -> Optional[int]:
    """
    policy:
      - auto: 优先相信事件内容；若与 metadata 冲突，则以事件内容为准，并打印警告
      - infer: 完全忽略 metadata，用事件内容推断
      - metadata: 沿用 metadata（旧行为）
      - none: 传 None 给 decoder，让 decoder 自己扫完整条（通常也没问题）
    """
    if policy == "none":
        return None

    inferred_len = infer_used_len_from_events(
        events,
        eos_id=int(cfg.eos_id),
        pad_id=int(cfg.pad_id),
    )

    if policy == "infer":
        return inferred_len

    if policy == "metadata":
        if metadata_used_len is None:
            return inferred_len
        return max(0, min(int(metadata_used_len), int(events.shape[0])))

    if policy != "auto":
        raise ValueError(f"Unknown used_len policy: {policy}")

    if metadata_used_len is None:
        return inferred_len

    metadata_used_len = max(0, min(int(metadata_used_len), int(events.shape[0])))
    if metadata_used_len != inferred_len:
        where = f"row {row_idx}" if row_idx is not None else "row ?"
        print(
            f"[WARN] {where}: metadata used_len_events={metadata_used_len} "
            f"!= inferred_len={inferred_len}; using inferred_len."
        )
    return inferred_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--simplemono_pkl", type=str, default="",
                    help="Path to SimpleMono.pkl. If empty, will try <npy_dir>/SimpleMono.pkl")
    ap.add_argument("--metadata", type=str, default="",
                    help="Path to metadata.jsonl (optional but recommended for naming; used_len may be stale for derived npy)")

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
    ap.add_argument(
        "--used_len_policy",
        type=str,
        default="auto",
        choices=["auto", "infer", "metadata", "none"],
        help=(
            "How to determine effective sequence length before decoding. "
            "'auto' (default) compares metadata with EOS/PAD inferred length and prefers the latter on mismatch; "
            "'infer' ignores metadata length entirely; 'metadata' keeps old behavior; 'none' passes None to decoder."
        ),
    )

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
        metadata_used_len = int(meta["used_len_events"]) if meta is not None and "used_len_events" in meta else None
        used_len = resolve_used_len_events(
            events,
            cfg=cfg,
            metadata_used_len=metadata_used_len,
            policy=args.used_len_policy,
            row_idx=i,
        )

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

# usage:
# python -m preproc.decode_npy_to_midi --npy_path .\preproc\output\anthology_v251218_lyrics_included\test_skeleton.npy --metadata .\preproc\output\anthology_v251218_lyrics_included\metadata.jsonl --out_dir .\preproc\output\anthology_v251218_lyrics_included\decoded_midis_test_skeleton --metadata_split test 
# python -m preproc.decode_npy_to_midi --npy_path .\preproc\output\anthology_v251218_lyrics_included\test_recon.npy --metadata .\preproc\output\anthology_v251218_lyrics_included\metadata.jsonl --out_dir .\preproc\output\anthology_v251218_lyrics_included\decoded_midis_test_recon --metadata_split test 
# simplemono_preproc/pipelines/manifest_task.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..constants import DEFAULT_SPLIT, WINDOW_MAX_NOTES
from ..encoder import encode_triples_to_events, iter_piece_triple_windows, pad_or_truncate_events
from ..npy_stream import NpyAppendWriter
from ..splitting import split_by_groups
from ..utils import ensure_dir_empty
from ..vocab import SimpleMonoVocab

from tqdm import tqdm


def default_max_len_tokens() -> int:
    """
    Your old pipeline (window=2048 triples) => max tokens:
      BOS(3) + 2048*3 + EOS(3) = 6150
    """
    return 3 * (WINDOW_MAX_NOTES + 2)


@dataclass
class ManifestItem:
    path: Path
    label: int
    group: str


def read_manifest_jsonl(manifest_path: Path, base_dir: Optional[Path] = None) -> List[ManifestItem]:
    """
    Each line:
      {"path": "...", "label": 0, "group": "song_xxx"}  # group optional
    """
    base_dir = Path(base_dir) if base_dir is not None else None
    items: List[ManifestItem] = []
    with Path(manifest_path).open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            p = Path(obj["path"])
            if base_dir is not None and not p.is_absolute():
                p = base_dir / p
            label = int(obj["label"])
            group = str(obj.get("group", p.stem))
            items.append(ManifestItem(path=p, label=label, group=group))
    return items


def build_from_manifest_items(
    manifest_items: List[ManifestItem],
    output_dir: Path,
    seed: int = 1234,
    xml_group: str = "staff_voice",
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    max_len_tokens: int = 6150,
):
    """
    Example for sequence classification transfer tasks:
      - train/valid/test.npy
      - train/valid/test.labels.npy
      - SimpleMono.pkl
      - metadata.jsonl
    """
    output_dir = Path(output_dir)
    ensure_dir_empty(output_dir)

    vocab = SimpleMonoVocab.build()

    if max_len_tokens is None or max_len_tokens <= 0:
        max_len_tokens = default_max_len_tokens()
    if max_len_tokens % 3 != 0:
        raise ValueError("max_len_tokens must be divisible by 3")
    max_events = max_len_tokens // 3

    splits_items = split_by_groups(
        [it.path for it in manifest_items],
        group_key=lambda p: next(x.group for x in manifest_items if x.path == p),  # naive; for big manifests build dict
        seed=seed,
        ratios=ratios,
    )

    # build a lookup for labels/groups
    path2label = {it.path: it.label for it in manifest_items}
    path2group = {it.path: it.group for it in manifest_items}

    writers = {
        sp: NpyAppendWriter(output_dir / f"{sp}.npy", dtype=np.int32, row_shape=(max_events, 3))
        for sp in ["train", "valid", "test"]
    }
    label_buf: Dict[str, List[int]] = {sp: [] for sp in ["train", "valid", "test"]}
    meta_f = (output_dir / "metadata.jsonl").open("w", encoding="utf-8")

    row_idx = {sp: 0 for sp in ["train", "valid", "test"]}

    for sp in ["train", "valid", "test"]:
        print(f"[INFO] Processing {sp} set...")
        for piece_path in tqdm(splits_items[sp], desc=f"Processing {sp} set", unit="piece", ascii=True, total=len(splits_items[sp]), smoothing=0.0):
            label = path2label[piece_path]
            group = path2group[piece_path]
            for triples, meta in iter_piece_triple_windows(
                piece_path=piece_path,
                input_dir=piece_path.parent.parent if piece_path.parent.parent.exists() else piece_path.parent,
                seed=seed,
                xml_group=xml_group,
            ):
                events = encode_triples_to_events(triples, vocab)
                padded, used_len = pad_or_truncate_events(events, max_events=max_events, vocab=vocab)

                writers[sp].append(padded)
                label_buf[sp].append(int(label))

                meta_f.write(json.dumps({
                    "split": sp,
                    "row_idx": row_idx[sp],
                    "label": int(label),
                    "group": group,
                    "source_file": str(piece_path),
                    "voice_id": meta.voice_id,
                    "voice_idx": meta.voice_idx,
                    "window_idx": meta.window_idx,
                    "used_len_events": int(used_len),
                }, ensure_ascii=False) + "\n")
                row_idx[sp] += 1

    for sp in writers:
        writers[sp].close()

    meta_f.close()

    # labels
    for sp in ["train", "valid", "test"]:
        np.save(output_dir / f"{sp}.labels.npy", np.asarray(label_buf[sp], dtype=np.int64))

    vocab.save_simplemono_pkl(output_dir / "SimpleMono.pkl")
    print("[DONE] manifest task dataset:", output_dir)

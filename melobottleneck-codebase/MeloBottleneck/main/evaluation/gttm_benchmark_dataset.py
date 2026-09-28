from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GTTMBackboneBenchmarkDataset(Dataset):
    """
    root/
      x.npy              [N,L,3] int
      len_x.npy          [N] int (optional)
      note_oldidx.npy    [N,L] int (optional)
      gold_depth.npy     [N,L] int16 (-1 for non-note/special/pad)
      gold_score.npy     [N,L] float32 (0 for non-note)
      gold_cut_masks.npy [N,C,L] uint8 (optional)
      pieces.jsonl       (optional)
      meta.json          (optional)
    """

    def __init__(self, root_dir: Union[str, Path], *, mmap: bool = True) -> None:
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"GTTM benchmark dir not found: {self.root}")

        def _load(name: str, required: bool = True):
            p = self.root / name
            if not p.exists():
                if required:
                    raise FileNotFoundError(f"Missing file: {p}")
                return None
            return np.load(p, mmap_mode="r" if mmap else None)

        self.x = _load("x.npy", required=True)
        self.len_x = _load("len_x.npy", required=False)
        self.note_oldidx = _load("note_oldidx.npy", required=False)

        self.gold_depth = _load("gold_depth.npy", required=True)
        self.gold_score = _load("gold_score.npy", required=True)
        self.gold_cut_masks = _load("gold_cut_masks.npy", required=False)

        if self.x.ndim != 3 or self.x.shape[-1] != 3:
            raise ValueError(f"x.npy must be [N,L,3], got {self.x.shape}")

        N, L, _ = self.x.shape
        if self.gold_depth.shape != (N, L):
            raise ValueError(f"gold_depth.npy shape mismatch: {self.gold_depth.shape} vs {(N, L)}")
        if self.gold_score.shape != (N, L):
            raise ValueError(f"gold_score.npy shape mismatch: {self.gold_score.shape} vs {(N, L)}")

        if self.len_x is not None and self.len_x.shape != (N,):
            raise ValueError(f"len_x.npy must be [N], got {self.len_x.shape}")

        if self.note_oldidx is not None and self.note_oldidx.shape != (N, L):
            raise ValueError(f"note_oldidx.npy shape mismatch: {self.note_oldidx.shape} vs {(N, L)}")

        if self.gold_cut_masks is not None:
            if self.gold_cut_masks.ndim != 3 or self.gold_cut_masks.shape[0] != N or self.gold_cut_masks.shape[2] != L:
                raise ValueError(f"gold_cut_masks.npy must be [N,C,L], got {self.gold_cut_masks.shape}")

        self.N = int(N)
        self.L = int(L)

        meta_path = self.root / "meta.json"
        self.meta: Dict[str, Any] = {}
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.x[idx])).long()  # [L,3]
        gold_depth = torch.from_numpy(np.asarray(self.gold_depth[idx])).long()  # [L]
        gold_score = torch.from_numpy(np.asarray(self.gold_score[idx])).float()  # [L]

        out: Dict[str, torch.Tensor] = {
            "x": x,
            "gold_depth": gold_depth,
            "gold_score": gold_score,
            "idx": torch.tensor(int(idx), dtype=torch.long),
        }

        if self.len_x is not None:
            out["len_x"] = torch.tensor(int(self.len_x[idx]), dtype=torch.long)

        if self.note_oldidx is not None:
            out["note_oldidx"] = torch.from_numpy(np.asarray(self.note_oldidx[idx])).long()

        if self.gold_cut_masks is not None:
            # uint8 -> uint8 tensor (eval 时可转 bool)
            out["gold_cut_masks"] = torch.from_numpy(np.asarray(self.gold_cut_masks[idx])).to(torch.uint8)

        return out


def make_gttm_benchmark_dataloader(
    root_dir: Union[str, Path],
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    mmap: bool = True,
) -> DataLoader:
    ds = GTTMBackboneBenchmarkDataset(root_dir, mmap=mmap)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
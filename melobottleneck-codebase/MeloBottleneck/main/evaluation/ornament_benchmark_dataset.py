from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class OrnamentToBackboneBenchmarkDataset(Dataset):
    """
    root/
      x.npy         [N,L,3]
      x_orn.npy     [N,L,3]
      pi.npy        [N,L]
      len_x.npy     [N] optional
      len_x_orn.npy [N] optional
      rho.npy       [N] optional
      meta.json     optional
    """

    def __init__(self, root_dir: Union[str, Path], *, mmap: bool = True) -> None:
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Benchmark dir not found: {self.root}")

        def _load(name: str, required: bool = True):
            p = self.root / name
            if not p.exists():
                if required:
                    raise FileNotFoundError(f"Missing file: {p}")
                return None
            return np.load(p, mmap_mode="r" if mmap else None)

        self.x = _load("x.npy", required=True)
        self.x_orn = _load("x_orn.npy", required=True)
        self.pi = _load("pi.npy", required=True)

        self.len_x = _load("len_x.npy", required=False)
        self.len_x_orn = _load("len_x_orn.npy", required=False)
        self.rho = _load("rho.npy", required=False)

        if self.x.ndim != 3 or self.x.shape[-1] != 3:
            raise ValueError(f"x.npy must be [N,L,3], got {self.x.shape}")
        if self.x_orn.shape != self.x.shape:
            raise ValueError("x_orn.npy shape mismatch.")
        if self.pi.ndim != 2 or self.pi.shape[:2] != self.x.shape[:2]:
            raise ValueError("pi.npy shape mismatch.")

        self.N = int(self.x.shape[0])
        self.L = int(self.x.shape[1])

        meta_path = self.root / "meta.json"
        self.meta: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(np.asarray(self.x[idx])).long()
        x_orn = torch.from_numpy(np.asarray(self.x_orn[idx])).long()
        pi = torch.from_numpy(np.asarray(self.pi[idx])).long()

        out = {
            "x": x,
            "x_orn": x_orn,
            "pi": pi,
            "idx": torch.tensor(int(idx), dtype=torch.long),
        }

        if self.len_x is not None:
            out["len_x"] = torch.tensor(int(self.len_x[idx]), dtype=torch.long)
        if self.len_x_orn is not None:
            out["len_x_orn"] = torch.tensor(int(self.len_x_orn[idx]), dtype=torch.long)
        if self.rho is not None:
            out["rho"] = torch.tensor(float(self.rho[idx]), dtype=torch.float32)

        return out


def make_ornament_benchmark_dataloader(
    root_dir: Union[str, Path],
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    mmap: bool = True,
) -> DataLoader:
    ds = OrnamentToBackboneBenchmarkDataset(root_dir, mmap=mmap)
    return DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=False,
    )
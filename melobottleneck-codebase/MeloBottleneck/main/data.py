from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .augment import MusicAugmentConfig, MusicAugmenter


class NpyMusicDataset(Dataset):
    """
    corpus_npy: shape [N, L, 3] int64/int32
    存储的是全局 ID，需要转换为局部 ID
    """
    def __init__(
        self,
        npy_path: str,
        global2local_pitch: Optional[np.ndarray] = None,
        global2local_duration: Optional[np.ndarray] = None,
        global2local_dt: Optional[np.ndarray] = None,
        mmap: bool = True,
        augment: bool = False,
        augment_seed: Optional[int] = None,
        augment_config: Optional[MusicAugmentConfig] = None,
    ):
        super().__init__()
        self.arr = np.load(npy_path, mmap_mode="r" if mmap else None)  # [N,L,3]
        if self.arr.ndim != 3 or self.arr.shape[-1] != 3:
            raise ValueError(f"Expected [N,L,3], got {self.arr.shape}")

        # 保存映射表
        self.global2local_pitch = global2local_pitch
        self.global2local_duration = global2local_duration
        self.global2local_dt = global2local_dt
        # 增强器
        self.augment_seed = augment_seed
        if augment:
            if augment_config is None:
                raise ValueError("augment=True requires augment_config with quantization tables from SimpleMono.pkl")
            self.augmenter = MusicAugmenter(augment_config)
        else:
            self.augmenter = None

        # per-process rng (safe under multi-worker)
        self._rng: Optional[np.random.Generator] = None
        self._rng_seed32: Optional[int] = None
        
    def _get_rng(self) -> np.random.Generator:
        """
        用 torch.initial_seed() 混合 augment_seed 来初始化 numpy Generator。
        优点：
        - DataLoader 多 worker 时，每个 worker 的 torch.initial_seed() 不同 -> rng 流不同
        - 如果你的 DataLoader 每个 epoch 会 reseed worker（常见），这里也会自动更新 rng
        """
        base = int(torch.initial_seed())  # 64-bit
        if self.augment_seed is not None:
            base = (base + int(self.augment_seed)) & 0xFFFFFFFFFFFFFFFF
        seed32 = base % (2**32)

        if self._rng is None or self._rng_seed32 != seed32:
            self._rng = np.random.default_rng(seed32)
            self._rng_seed32 = seed32
        return self._rng

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __getitem__(self, idx: int) -> torch.LongTensor:
        x = self.arr[idx]  # [L,3] 全局 ID
        # 注意：np.load mmap 返回的可能不是连续内存，这里转一下
        x = np.array(x, copy=True)  # [L, 3]

        # 如果提供了映射表，则进行转换
        if self.global2local_pitch is not None:
            # 转换为局部 ID
            x[:, 0] = self.global2local_pitch[x[:, 0]]  # pitch
            x[:, 1] = self.global2local_duration[x[:, 1]]  # duration
            x[:, 2] = self.global2local_dt[x[:, 2]]  # dt

        if self.augmenter is not None:
            self.augmenter.augment_inplace(x, rng=self._get_rng())

        return torch.as_tensor(x, dtype=torch.long)


class NpyMusicSequenceLabeledDataset(Dataset):
    """
    seqs_npy: shape [N, L, 3] int64/int32
    label_npy: shape [N] int64/int32
    存储的是全局 ID，需要转换为局部 ID
    """
    def __init__(
        self,
        npy_path: str,
        label_path: str,
        global2local_pitch: Optional[np.ndarray] = None,
        global2local_duration: Optional[np.ndarray] = None,
        global2local_dt: Optional[np.ndarray] = None,
        mmap: bool = True,
        augment: bool = False,
        augment_seed: Optional[int] = None,
        augment_config: Optional[MusicAugmentConfig] = None,
    ):
        super().__init__()
        self.arr = np.load(npy_path, mmap_mode="r" if mmap else None)  # [N,L,3]
        self.labels = np.load(label_path, mmap_mode="r" if mmap else None)  # [N]
        if self.arr.ndim != 3 or self.arr.shape[-1] != 3:
            raise ValueError(f"Expected [N,L,3], got {self.arr.shape}")
        if self.labels.ndim != 1:
            raise ValueError(f"Expected [N], got {self.labels.shape}")
        if self.arr.shape[0] != self.labels.shape[0]:
            raise ValueError(f"Expected same length, got {self.arr.shape[0]} and {self.labels.shape[0]}")

        # 保存映射表
        self.global2local_pitch = global2local_pitch
        self.global2local_duration = global2local_duration
        self.global2local_dt = global2local_dt
        # 增强器
        self.augment_seed = augment_seed
        self.augmenter = MusicAugmenter(augment_config or MusicAugmentConfig()) if augment else None
        # per-process rng (safe under multi-worker)
        self._rng: Optional[np.random.Generator] = None
        self._rng_seed32: Optional[int] = None

    def _get_rng(self) -> np.random.Generator:
        """
        用 torch.initial_seed() 混合 augment_seed 来初始化 numpy Generator。
        优点：
        - DataLoader 多 worker 时，每个 worker 的 torch.initial_seed() 不同 -> rng 流不同
        - 如果你的 DataLoader 每个 epoch 会 reseed worker（常见），这里也会自动更新 rng
        """
        base = int(torch.initial_seed())  # 64-bit
        if self.augment_seed is not None:
            base = (base + int(self.augment_seed)) & 0xFFFFFFFFFFFFFFFF
        seed32 = base % (2**32)

        if self._rng is None or self._rng_seed32 != seed32:
            self._rng = np.random.default_rng(seed32)
            self._rng_seed32 = seed32
        return self._rng

    def __len__(self) -> int:
        return int(self.arr.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        x = self.arr[idx]  # [L,3] 全局 ID
        y = self.labels[idx]  # [1] 标签
        # 注意：np.load mmap 返回的可能不是连续内存，这里转一下
        x = np.array(x, copy=True)  # [L, 3]

        # 如果提供了映射表，则进行转换
        if self.global2local_pitch is not None:
            # 转换为局部 ID
            x[:, 0] = self.global2local_pitch[x[:, 0]]  # pitch
            x[:, 1] = self.global2local_duration[x[:, 1]]  # duration
            x[:, 2] = self.global2local_dt[x[:, 2]]  # dt

        if self.augmenter is not None:
            self.augmenter.augment_inplace(x, rng=self._get_rng())
        
        return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)



def make_dataloader(
    npy_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    seq_labels_npy_path: Optional[str] = None,
    global2local_pitch: Optional[np.ndarray] = None,
    global2local_duration: Optional[np.ndarray] = None,
    global2local_dt: Optional[np.ndarray] = None,
    augment: bool = False,
    augment_seed: Optional[int] = None,
    augment_config: Optional[MusicAugmentConfig] = None,
    max_samples: Optional[int] = None,
) -> DataLoader:

    if seq_labels_npy_path is None:
        ds = NpyMusicDataset(
            npy_path=npy_path,
            global2local_pitch=global2local_pitch,
            global2local_duration=global2local_duration,
            global2local_dt=global2local_dt,
            mmap=True,
            augment=augment,
            augment_seed=augment_seed,
            augment_config=augment_config,
        )

    else:
        ds = NpyMusicSequenceLabeledDataset(
            npy_path=npy_path,
            label_path=seq_labels_npy_path,
            global2local_pitch=global2local_pitch,
            global2local_duration=global2local_duration,
            global2local_dt=global2local_dt,
            mmap=True,
            augment=augment,
            augment_seed=augment_seed,
            augment_config=augment_config,
        )

    if max_samples is not None and max_samples < len(ds):
        ds = Subset(ds, range(max_samples))

    loader_kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    return DataLoader(**loader_kwargs)

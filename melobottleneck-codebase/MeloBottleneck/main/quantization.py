# models/skeleton/quantization.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class MusicQuantizationTables:
    """
    这里假设：
    - duration local_id: [special_n .. special_n+dur_samples-1] -> code 0..dur_samples-1
    - dt local_id:
        pos/zero: [special_n .. special_n+dt_samples-1] -> code 0..dt_samples-1
        neg:      [special_n+dt_samples .. special_n+2*dt_samples-1] -> -1..-dt_samples
    """
    special_n: int = 5

    duration_code_to_pos: Optional[np.ndarray] = None
    duration_pos_to_code: Optional[np.ndarray] = None
    duration_min_code: int = 1

    deltatime_code_offset: Optional[int] = None     # == dt_samples
    deltatime_code_to_pos: Optional[np.ndarray] = None
    deltatime_pos_to_code: Optional[np.ndarray] = None


class DurationQuantizer(nn.Module):
    def __init__(
        self,
        *,
        special_n: int,
        code_to_pos: np.ndarray,
        pos_to_code: np.ndarray,
        min_code: int = 1,
    ):
        super().__init__()
        self.special_n = int(special_n)
        self.min_code = int(min_code)

        code_to_pos_t = torch.as_tensor(code_to_pos, dtype=torch.long)
        pos_to_code_t = torch.as_tensor(pos_to_code, dtype=torch.long)
        self.register_buffer("code_to_pos", code_to_pos_t)
        self.register_buffer("pos_to_code", pos_to_code_t)

        self.max_code = int(code_to_pos_t.numel() - 1)
        self.pos_max = int(pos_to_code_t.numel() - 1)

        if 0 <= self.min_code <= self.max_code:
            self.min_pos = int(code_to_pos_t[self.min_code].item())
        else:
            self.min_pos = 0

    def decode_local_to_pos(self, dur_local: torch.LongTensor) -> torch.LongTensor:
        """
        dur_local: [...], local id space
        return: [...], integer pos units
        """
        code = dur_local - self.special_n
        valid = code >= 0
        code = code.clamp(min=0, max=self.max_code)
        pos = self.code_to_pos[code]
        pos = pos * valid.to(pos.dtype)  # special -> 0
        return pos

    def encode_pos_to_local(self, pos: torch.LongTensor) -> torch.LongTensor:
        """
        pos: integer pos units, [...], >=0
        return: local id
        """
        pos = pos.clamp(min=self.min_pos, max=self.pos_max)
        code = self.pos_to_code[pos]
        code = code.clamp(min=self.min_code, max=self.max_code)
        return code + self.special_n


class DeltaTimeQuantizer(nn.Module):
    def __init__(
        self,
        *,
        special_n: int,
        code_offset: int,           # == dt_samples
        code_to_pos: np.ndarray,    # len == 2*dt_samples
        pos_to_code: np.ndarray,    # len == 2*dt_samples
    ):
        super().__init__()
        self.special_n = int(special_n)
        self.offset = int(code_offset)
        self.dt_samples = int(code_offset)
        code_to_pos_t = torch.as_tensor(code_to_pos, dtype=torch.long)
        if code_to_pos_t.numel() != 2 * self.dt_samples:
            raise ValueError(...)
        self.register_buffer("code_to_pos", code_to_pos_t)
        pos_to_code_t = torch.as_tensor(pos_to_code, dtype=torch.long)
        if pos_to_code_t.numel() != 2 * self.dt_samples:
            raise ValueError(
                f"dt pos_to_code length mismatch: got {pos_to_code_t.numel()}, expected {2*self.dt_samples}"
            )
        self.register_buffer("pos_to_code", pos_to_code_t)

    def decode_local_to_signed_code(self, dt_local: torch.LongTensor) -> torch.LongTensor:
        """
        dt_local: [...], local id space
        return: signed dt_code in [-dt_samples .. dt_samples-1]
        """
        code = torch.zeros_like(dt_local)

        pos_mask = (dt_local >= self.special_n) & (dt_local < self.special_n + self.dt_samples)
        code[pos_mask] = dt_local[pos_mask] - self.special_n  # 0..dt_samples-1

        neg_mask = (dt_local >= self.special_n + self.dt_samples) & (
            dt_local < self.special_n + 2 * self.dt_samples
        )
        mag = dt_local[neg_mask] - (self.special_n + self.dt_samples) + 1  # 1..dt_samples
        code[neg_mask] = -mag

        # other specials stay 0
        return code

    def decode_local_to_pos(self, dt_local: torch.LongTensor) -> torch.LongTensor:
        """
        dt_local -> signed code -> pos (may be negative in principle; table决定)
        """
        code = self.decode_local_to_signed_code(dt_local)
        idx = (code + self.offset).clamp(min=0, max=2 * self.dt_samples - 1)
        pos = self.code_to_pos[idx]

        # specials -> 0（即便 code=0 时本来也常是0，这里更稳）
        special = (dt_local < self.special_n) | (dt_local >= self.special_n + 2 * self.dt_samples)
        if special.any():
            pos = pos.clone()
            pos[special] = 0
        return pos
    
    def encode_pos_to_local(self, pos: torch.LongTensor) -> torch.LongTensor:
        """
        pos: signed integer pos units, [...]
        return: dt_local id in the same local vocab space
        """
        # clamp to supported signed pos range
        pos = pos.clamp(min=-self.dt_samples, max=self.dt_samples - 1)

        idx = (pos + self.offset).clamp(min=0, max=2 * self.dt_samples - 1)
        code = self.pos_to_code[idx]  # signed code in [-dt_samples .. dt_samples-1]
        code = code.clamp(min=-self.dt_samples, max=self.dt_samples - 1)

        local = torch.empty_like(code)

        pos_mask = code >= 0
        local[pos_mask] = code[pos_mask] + self.special_n

        neg_code = code[~pos_mask]           # negative
        mag = (-neg_code).clamp(min=1, max=self.dt_samples)   # 1..dt_samples
        local[~pos_mask] = (self.special_n + self.dt_samples) + (mag - 1)
        return local


def build_quantizers(q: MusicQuantizationTables) -> tuple[DurationQuantizer, DeltaTimeQuantizer]:
    if q.duration_code_to_pos is None or q.duration_pos_to_code is None:
        raise ValueError("duration_code_to_pos & duration_pos_to_code must be provided")
    if q.deltatime_code_offset is None or q.deltatime_code_to_pos is None:
        raise ValueError("deltatime_code_offset & deltatime_code_to_pos must be provided")
    if q.deltatime_pos_to_code is None:
        raise ValueError("deltatime_pos_to_code must be provided")

    dur = DurationQuantizer(
        special_n=q.special_n,
        code_to_pos=q.duration_code_to_pos,
        pos_to_code=q.duration_pos_to_code,
        min_code=q.duration_min_code,
    )
    dt = DeltaTimeQuantizer(
        special_n=q.special_n,
        code_offset=q.deltatime_code_offset,
        code_to_pos=q.deltatime_code_to_pos,
        pos_to_code=q.deltatime_pos_to_code,
    )
    return dur, dt

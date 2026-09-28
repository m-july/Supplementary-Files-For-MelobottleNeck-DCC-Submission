# models/skeleton/postprocess.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from ...quantization import DurationQuantizer, DeltaTimeQuantizer


class SkeletonForwardExtend(nn.Module):
    """
    Absolute-time re-encoding forward-extend:
    For each adjacent skeleton pair (cur_idx, next_idx):
      1) Compute src absolute onset/offset in pos units.
      2) Let offset_target = max(offset[k]) for k in [cur_idx, next_idx).
      3) dur_new = offset_target - onset[cur_idx]   (never shrink)
      4) dt_new  = onset[next_idx] - (onset[cur_idx] + dur_new)
    BOS special case (if cur_idx == 0):
      keep BOS.duration=0, set BOS.dt = onset[next] - onset[0].
    """

    def __init__(
        self,
        *,
        duration_q: DurationQuantizer,
        dt_q: DeltaTimeQuantizer,
        pad_id: int,
        eos_id: int,
    ):
        super().__init__()
        self.duration_q = duration_q
        self.dt_q = dt_q
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)

    @torch.no_grad()
    def forward(
        self,
        *,
        src_tokens: torch.LongTensor,     # [B,L,3]
        z_tokens: torch.LongTensor,       # [B,T,3]
        hard_indices: torch.LongTensor,   # [B,T]
        z_mask: torch.Tensor,             # [B,T] bool
    ) -> torch.LongTensor:
        device = src_tokens.device
        B, L, _ = src_tokens.shape
        T = z_tokens.size(1)
        z_final = z_tokens.clone()
        if T <= 1:
            return z_final
        # ---- src absolute time (pos units) ----
        src_mask = (src_tokens[..., 0] != self.pad_id)  # [B,L]
        dur_pos = self.duration_q.decode_local_to_pos(src_tokens[..., 1])  # [B,L]
        dt_pos  = self.dt_q.decode_local_to_pos(src_tokens[..., 2])        # [B,L]
        span_pos = dur_pos + dt_pos                                        # [B,L]
        # onset[l] = sum_{k<l} span[k]
        onset = torch.cumsum(span_pos, dim=1) - span_pos                   # [B,L]
        offset = onset + dur_pos                                           # [B,L]
        cur_idx  = hard_indices[:, :-1]   # [B,T-1]
        next_idx = hard_indices[:, 1:]    # [B,T-1]
        pair_mask = z_mask[:, :-1] & z_mask[:, 1:]  # [B,T-1]
        cur_onset  = onset.gather(1, cur_idx)   # [B,T-1]
        next_onset = onset.gather(1, next_idx)  # [B,T-1]
        # ---- max offset in each [cur, next) segment ----
        pos = torch.arange(L, device=device)[None, None, :]          # [1,1,L]
        cur = cur_idx.unsqueeze(-1)                                   # [B,T-1,1]
        nxt = next_idx.unsqueeze(-1)                                  # [B,T-1,1]

        seg_mask = (pos >= cur) & (pos < nxt)                         # [B,T-1,L]
        seg_mask = seg_mask & src_mask[:, None, :] & pair_mask.unsqueeze(-1)

        fill = -10**9
        max_offset = offset[:, None, :].masked_fill(~seg_mask, fill).max(dim=-1).values  # [B,T-1]
        # ---- new duration (absorb) ----
        dur_pos_new = (max_offset - cur_onset).clamp_min(0)
        dur_local_new = self.duration_q.encode_pos_to_local(dur_pos_new)
        # use quantized duration to compute dt, to keep onset exact under clipping/rounding
        dur_pos_q = self.duration_q.decode_local_to_pos(dur_local_new)
        dt_pos_new = next_onset - (cur_onset + dur_pos_q)
        dt_local_new = self.dt_q.encode_pos_to_local(dt_pos_new)
        # ---- BOS special case: keep BOS.duration=0, recompute dt with dur=0 ----
        bos_mask = (cur_idx == 0) & pair_mask
        if bos_mask.any():
            bos_dt_local = self.dt_q.encode_pos_to_local(next_onset - cur_onset)
            dt_local_new = torch.where(bos_mask, bos_dt_local, dt_local_new)
        pitch_old = z_final[:, :-1, 0]
        dur_old   = z_final[:, :-1, 1]
        dt_old    = z_final[:, :-1, 2]
        # duration: skip PAD/EOS/special-duration (e.g. BOS)
        update_dur = (
            pair_mask
            & (pitch_old != self.pad_id)
            & (pitch_old != self.eos_id)
            & (dur_old >= self.duration_q.special_n)
        )
        z_final[:, :-1, 1] = torch.where(update_dur, dur_local_new, dur_old)
        # dt: update for all non-PAD/EOS (including BOS)
        update_dt = pair_mask & (pitch_old != self.pad_id) & (pitch_old != self.eos_id)
        z_final[:, :-1, 2] = torch.where(update_dt, dt_local_new, dt_old)
        return z_final

# models/skeleton/ornament_invariance.py
from __future__ import annotations

import torch

from ...pointer_utils import marginal_importance_from_pointer_soft

def apply_duration_weight_to_importance(
    s: torch.Tensor,                    # [B,L]
    *,
    x_tokens: torch.LongTensor,         # [B,L,3] (原序列坐标系)
    duration_q,                         # DurationQuantizer
    alpha: float = 0.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    你提到的可选 duration 加权：
        s~_l = (dur(x_l)^alpha * s_l) / sum_j(...)
    alpha=0 表示关闭。
    """
    if alpha is None or float(alpha) == 0.0:
        return s

    alpha = float(alpha)
    dur_local = x_tokens[..., 1]
    dur_pos = duration_q.decode_local_to_pos(dur_local).to(torch.float32)

    special = dur_local < int(duration_q.special_n)
    w = torch.where(special, torch.ones_like(dur_pos), dur_pos.clamp_min(0.0))
    w = w.pow(alpha)

    sw = s.to(torch.float32) * w
    sw = sw / sw.sum(dim=1, keepdim=True).clamp_min(eps)
    return sw.to(dtype=s.dtype)


def aggregate_importance_by_pi(
    s_strong: torch.Tensor,     # [B,L_in]  (x'坐标)
    pi: torch.LongTensor,       # [B,L_in]  (x'->x)
    *,
    L_out: int,                 # 原序列长度维度(通常就是 max_seq_len)
) -> torch.Tensor:
    """
    s^{x'->x}_l = sum_{k:pi(k)=l} s'_k
    pi<0 的 (inserted/pad) 会被忽略。
    return: [B,L_out]
    """
    B, L_in = s_strong.shape
    pi = pi.to(torch.long)

    out = torch.zeros((B, int(L_out)), device=s_strong.device, dtype=s_strong.dtype)

    valid = (pi >= 0) & (pi < int(L_out))
    idx = pi.clamp(min=0, max=int(L_out) - 1)

    out.scatter_add_(dim=1, index=idx, src=s_strong * valid.to(s_strong.dtype))
    return out


def masked_mse_loss(
    a: torch.Tensor,            # [B,L]
    b: torch.Tensor,            # [B,L]
    mask: torch.Tensor,         # [B,L] bool/0-1
    eps: float = 1e-8,
) -> torch.Tensor:
    m = mask.to(torch.float32)
    diff2 = (a.to(torch.float32) - b.to(torch.float32)).pow(2) * m
    return diff2.sum() / m.sum().clamp_min(1.0)


def insertion_mass_loss(
    s_strong: torch.Tensor,      # [B,L_in]
    pi: torch.LongTensor,        # [B,L_in]
    *,
    inserted_value: int = -1,
) -> torch.Tensor:
    """
    L_ins = sum_{k:pi(k)=-1} s'_k  (再对 batch mean)
    """
    ins = (pi == int(inserted_value)).to(torch.float32)
    return (s_strong.to(torch.float32) * ins).sum(dim=1).mean()
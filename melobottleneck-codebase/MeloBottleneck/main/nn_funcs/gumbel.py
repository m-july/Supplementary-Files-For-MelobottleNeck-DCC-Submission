# models/skeleton/gumbel.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F


def sample_gumbel(shape, device, eps: float = 1e-12) -> torch.Tensor:
    # gumbel 始终 fp32 
    u = torch.rand(shape, device=device, dtype=torch.float32)
    u = u.clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(u))


def gumbel_softmax_st(
    logits: torch.Tensor,  # [B, V]
    tau: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor]:
    """
    return:
        y_st:   [B,V] straight-through prob
        y_soft: [B,V]
        y_hard: [B,V] onehot
        idx:    [B]
    """
    logits_f = logits.float()
    g = sample_gumbel(logits_f.shape, logits_f.device)
    y_soft = F.softmax((logits_f + g) / float(tau), dim=-1)
    idx = y_soft.argmax(dim=-1)
    y_hard = F.one_hot(idx, num_classes=logits.size(-1)).to(dtype=y_soft.dtype)
    y_st = (y_hard - y_soft).detach() + y_soft
    return y_st, y_soft.to(logits.dtype), y_hard, idx

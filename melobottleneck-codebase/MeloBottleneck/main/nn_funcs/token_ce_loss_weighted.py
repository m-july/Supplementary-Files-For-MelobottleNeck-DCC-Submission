# nn_funcs/token_ce_loss_weighted.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def multi_attribute_ce_loss_weighted(
    *,
    labels: torch.LongTensor,                # [B,T,3]
    pitch_logits: torch.Tensor,              # [B,T,Vp]
    duration_logits: torch.Tensor,           # [B,T,Vd]
    dt_logits: torch.Tensor,                 # [B,T,Vdt]
    attr_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    token_weights: Optional[torch.Tensor] = None,  # [B,T] float
    ignore_index: int = -100,
    reduction: str = "mean",                 # mean|sum|none
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    多属性 token CE（pitch/dur/dt），支持每个 token 的加权（例如 duration-pos 加权）。

    token_weights:
        - None: 等权（对非 ignore 的 token）
        - Tensor[B,T]: 权重（会自动把 ignore token 的权重置 0）

    reduction="mean" 时做 weighted mean：
        sum(loss * w) / sum(w)
    """
    if labels.ndim != 3 or labels.size(-1) != 3:
        raise ValueError(f"labels must be [B,T,3], got {tuple(labels.shape)}")

    B, T, _ = labels.shape
    w_p, w_d, w_dt = attr_weights

    # token mask：假设 padding 时 3 个属性都会被 mask_labels_with_ignore_index() 设为 ignore_index
    token_valid = (labels[..., 0] != ignore_index)  # [B,T]

    if token_weights is None:
        w_tok = token_valid.to(pitch_logits.dtype)
    else:
        w_tok = token_weights.to(pitch_logits.dtype) * token_valid.to(pitch_logits.dtype)

    # [B,T]
    loss_pitch = F.cross_entropy(
        pitch_logits.reshape(-1, pitch_logits.size(-1)),
        labels[..., 0].reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, T)

    loss_dur = F.cross_entropy(
        duration_logits.reshape(-1, duration_logits.size(-1)),
        labels[..., 1].reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, T)

    loss_dt = F.cross_entropy(
        dt_logits.reshape(-1, dt_logits.size(-1)),
        labels[..., 2].reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(B, T)

    # token loss
    token_loss = w_p * loss_pitch + w_d * loss_dur + w_dt * loss_dt  # [B,T]
    weighted = token_loss * w_tok

    if reduction == "none":
        total = weighted
    elif reduction == "sum":
        total = weighted.sum()
    elif reduction == "mean":
        denom = w_tok.sum().clamp_min(eps)
        total = weighted.sum() / denom
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # logging（都做 weighted mean 口径，方便看）
    denom = w_tok.sum().clamp_min(eps)
    loss_dict = {
        "loss_pitch": (loss_pitch * w_tok).sum() / denom,
        "loss_duration": (loss_dur * w_tok).sum() / denom,
        "loss_dt": (loss_dt * w_tok).sum() / denom,
    }

    return total, loss_dict

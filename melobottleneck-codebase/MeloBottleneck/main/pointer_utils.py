# main/pointer_utils.py
from __future__ import annotations

from typing import Optional

import torch


def compute_step_mask(
    z_mask: torch.Tensor,
    *,
    exclude_last_step: bool = True,
) -> torch.Tensor:
    """
    z_mask: [B,T] bool
    return: [B,T] bool
    """
    z_mask = z_mask.to(torch.bool)
    if z_mask.ndim != 2:
        raise ValueError(f"z_mask must be [B,T], got {tuple(z_mask.shape)}")

    if not exclude_last_step:
        return z_mask

    B, T = z_mask.shape
    z_len = z_mask.to(torch.int64).sum(dim=1)               # [B]
    t_ids = torch.arange(T, device=z_mask.device)[None, :]  # [1,T]
    return z_mask & (t_ids < (z_len[:, None] - 1))


def marginal_importance_from_pointer_soft(
    pointer_soft: torch.Tensor,          # [B,T,L]
    z_mask: Optional[torch.Tensor] = None,
    *,
    step_mask: Optional[torch.Tensor] = None,
    exclude_last_step: bool = True,
    temperature: float = 1.0,            # NEW
    eps: float = 1e-8,                   # NEW (for renorm)
) -> torch.Tensor:
    """
    s_l = mean_t p_soft[t,l] over valid decoding steps.

    允许两种调用方式：
      1) 传 z_mask，由函数内部构造 step_mask
      2) 直接传 step_mask
    """
    if pointer_soft.ndim != 3:
        raise ValueError(f"pointer_soft must be [B,T,L], got {tuple(pointer_soft.shape)}")

    if step_mask is None:
        if z_mask is None:
            raise ValueError("Either z_mask or step_mask must be provided.")
        step_mask = compute_step_mask(z_mask, exclude_last_step=exclude_last_step)

    step_mask = step_mask.to(torch.bool)
    if step_mask.shape != pointer_soft.shape[:2]:
        raise ValueError(
            f"step_mask shape mismatch: {tuple(step_mask.shape)} vs {tuple(pointer_soft.shape[:2])}"
        )

    p = pointer_soft.to(torch.float32)
    # NEW: temperature scaling on probabilities (for export / ranking)
    T = float(temperature)
    if abs(T - 1.0) > 1e-6:
        T = max(T, 1e-6)
        p = p.clamp_min(0.0).pow(1.0 / T)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = step_mask.to(torch.float32)
    denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (p * m.unsqueeze(-1)).sum(dim=1) / denom


def hard_indices_to_mask(
    hard_indices: torch.LongTensor,      # [B,T]
    step_mask: torch.Tensor,             # [B,T]
    *,
    L: int,
) -> torch.Tensor:
    """
    hard_indices + step_mask -> source-position boolean mask [B,L]
    """
    if hard_indices.ndim != 2 or step_mask.ndim != 2:
        raise ValueError("hard_indices and step_mask must be [B,T].")
    if hard_indices.shape != step_mask.shape:
        raise ValueError(f"shape mismatch: {hard_indices.shape} vs {step_mask.shape}")

    B, T = hard_indices.shape
    L = int(L)
    idx = hard_indices.clamp(min=0, max=max(L - 1, 0))

    acc = torch.zeros((B, L), device=hard_indices.device, dtype=torch.int32)
    acc.scatter_add_(dim=1, index=idx, src=step_mask.to(torch.int32))
    return acc > 0


def renormalize_scores_on_mask(
    scores: torch.Tensor,     # [B,L]
    mask: torch.Tensor,       # [B,L]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    对 mask 上的 score 重新归一化，使其和为 1。
    """
    scores = scores.to(torch.float32)
    mask = mask.to(torch.float32)
    masked = scores * mask
    denom = masked.sum(dim=1, keepdim=True).clamp_min(eps)
    return masked / denom

def masked_softmax(
    logits: torch.Tensor,     # [B,L]
    mask: torch.Tensor,       # [B,L] bool
    *,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    数值稳定的 masked softmax。
    - mask==False 的位置输出 0
    - mask==True 的位置按 softmax 归一化（若整行 mask 全 False，则整行输出 0）
    返回 float32。
    """
    if logits.ndim != 2 or mask.ndim != 2 or logits.shape != mask.shape:
        raise ValueError(f"logits/mask must be [B,L] and same shape, got {tuple(logits.shape)} vs {tuple(mask.shape)}")

    temp = max(float(temperature), 1e-6)
    x = logits.to(torch.float32) / temp
    m = mask.to(torch.bool)

    # 用一个足够小的负数（而不是 -inf），避免 “全 False” 行出现 NaN
    neg = x.new_full((), -1e4)
    x = torch.where(m, x, neg)

    # 减 max 防止 exp 溢出；全 False 行 max=-1e4，不会出 NaN
    x = x - x.max(dim=1, keepdim=True).values

    ex = torch.exp(x) * m.to(torch.float32)
    return ex / ex.sum(dim=1, keepdim=True).clamp_min(eps)


def importance_from_score_logits(
    score_logits: torch.Tensor,   # [B,L]
    score_mask: torch.Tensor,     # [B,L] bool
    *,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    encoder_topk 的“正确 soft importance”来源：
      s = masked_softmax(score_logits / temperature)  (sum=1, 非负)
    """
    return masked_softmax(score_logits, score_mask, temperature=temperature, eps=eps)
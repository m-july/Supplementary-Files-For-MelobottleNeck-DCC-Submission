from __future__ import annotations
from typing import Optional

import torch


def infer_attention_mask_from_tokens(
    tokens: torch.LongTensor,
    pad_id: int,
    pad_attr_idx: int = 0,
) -> torch.LongTensor:
    """
    tokens: [B, L, A]
    return: [B, L] (long) 1=valid, 0=pad
    默认用 tokens[..., 0] 判断 pad（与原代码 dec_attention_mask 的逻辑一致）。
    """
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be [B,L,A], got {tuple(tokens.shape)}")
    return (tokens[..., pad_attr_idx] != pad_id).to(dtype=torch.long)


def right_shift_tokens(
    tokens: torch.LongTensor,
    bos_id: int,
    pad_id: int,
) -> torch.LongTensor:
    """
    tokens: [B, L, A]
    return: [B, L, A]，out[:,0,:]=bos_id，out[:,1:,:]=tokens[:,:-1,:]
    """
    if tokens.dim() != 3:
        raise ValueError(f"tokens must be [B,L,A], got {tuple(tokens.shape)}")
    b, l, a = tokens.shape
    out = tokens.new_full((b, l, a), fill_value=pad_id)
    out[:, 0, :] = bos_id
    out[:, 1:, :] = tokens[:, :-1, :]
    return out


def mask_labels_with_ignore_index(
    labels: torch.LongTensor,
    attention_mask: torch.Tensor,
    ignore_index: int,
) -> torch.LongTensor:
    """
    labels: [B, L, A]
    attention_mask: [B, L] 1/0 or bool
    """
    if labels.dim() != 3:
        raise ValueError(f"labels must be [B,L,A], got {tuple(labels.shape)}")
    if attention_mask.dim() != 2:
        raise ValueError(f"attention_mask must be [B,L], got {tuple(attention_mask.shape)}")

    m = attention_mask.to(dtype=torch.bool)
    return labels.masked_fill((~m).unsqueeze(-1), ignore_index)

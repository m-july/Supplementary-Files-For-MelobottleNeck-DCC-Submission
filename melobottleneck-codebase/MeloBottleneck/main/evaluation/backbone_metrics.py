# main/evaluation/backbone_metrics.py
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def _ensure_2d_bool(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.bool)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D tensor, got shape={tuple(x.shape)}")
    return x


def binary_counts(
    pred: torch.Tensor,
    true: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = _ensure_2d_bool(pred)
    true = _ensure_2d_bool(true)
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: {pred.shape} vs {true.shape}")

    if mask is not None:
        mask = _ensure_2d_bool(mask)
        if mask.shape != pred.shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} vs {pred.shape}")
        pred = pred & mask
        true = true & mask

    tp = (pred & true).sum(dim=1).to(torch.float32)
    fp = (pred & ~true).sum(dim=1).to(torch.float32)
    fn = (~pred & true).sum(dim=1).to(torch.float32)
    return tp, fp, fn


def binary_scores_from_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tp = tp.to(torch.float32)
    fp = fp.to(torch.float32)
    fn = fn.to(torch.float32)

    precision = tp / (tp + fp).clamp_min(eps)
    recall = tp / (tp + fn).clamp_min(eps)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(eps)
    iou = tp / (tp + fp + fn).clamp_min(eps)
    return precision, recall, f1, iou


def precision_recall_f1_iou_from_masks(
    pred_mask: np.ndarray,
    gold_mask: np.ndarray,
    eps: float = 1e-8,
) -> tuple[float, float, float, float]:
    pred = np.asarray(pred_mask, dtype=bool)
    gold = np.asarray(gold_mask, dtype=bool)
    if pred.shape != gold.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gold.shape}")

    tp = float(np.logical_and(pred, gold).sum())
    fp = float(np.logical_and(pred, np.logical_not(gold)).sum())
    fn = float(np.logical_and(np.logical_not(pred), gold).sum())

    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    return float(precision), float(recall), float(f1), float(iou)


def topk_mask(
    scores: np.ndarray,
    eval_mask: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    只在 eval_mask=True 的 universe 上取 top-k。
    返回和 scores 同 shape 的 bool mask。
    """
    scores = np.asarray(scores, dtype=np.float64)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    if scores.shape != eval_mask.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {eval_mask.shape}")

    out = np.zeros_like(eval_mask, dtype=bool)
    idx = np.flatnonzero(eval_mask)
    n = int(idx.size)

    k = int(k)
    if n <= 0 or k <= 0:
        return out

    k = min(k, n)
    order = np.argsort(-scores[idx], kind="mergesort")
    out[idx[order[:k]]] = True
    return out


def average_precision_binary(
    scores: np.ndarray,
    labels: np.ndarray,
    eps: float = 1e-8,
) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)

    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]

    tp = np.cumsum(y)
    denom = np.arange(1, y.size + 1, dtype=np.float64)
    precision = tp / np.maximum(denom, eps)

    ap = precision[y.astype(bool)].sum() / float(n_pos)
    return float(ap)


def trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be 1D with the same length.")
    if x.shape[0] < 2:
        return 0.0
    return float(np.trapz(y, x))


def _apply_gain(rel: np.ndarray, gain: str) -> np.ndarray:
    rel = np.asarray(rel, dtype=np.float64)
    if gain == "identity":
        return rel
    if gain == "exp2":
        return np.power(2.0, rel) - 1.0
    raise ValueError(f"Unknown gain: {gain}")


def ndcg_at_k(
    scores: np.ndarray,
    rel: np.ndarray,
    k: int,
    *,
    gain: str = "identity",
    eps: float = 1e-8,
) -> float:
    """
    General graded NDCG@k.
    rel 可以是 binary 也可以是 graded。
    """
    scores = np.asarray(scores, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)

    n = int(rel.size)
    if n == 0:
        return float("nan")

    k = max(1, min(int(k), n))

    order = np.argsort(-scores, kind="mergesort")
    rel_sorted = rel[order][:k]
    gains = _apply_gain(rel_sorted, gain)
    denom = np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float((gains / denom).sum())

    ideal_rel = np.sort(rel)[::-1][:k]
    ideal_gains = _apply_gain(ideal_rel, gain)
    idcg = float((ideal_gains / denom).sum())

    if idcg <= eps:
        return float("nan")
    return float(dcg / idcg)


def ndcg_at_k_binary(
    scores: np.ndarray,
    labels: np.ndarray,
    k: int,
    eps: float = 1e-8,
) -> float:
    labels = np.asarray(labels, dtype=np.int32)
    if int(labels.sum()) == 0:
        return float("nan")
    return ndcg_at_k(scores, labels.astype(np.float64), k, gain="identity", eps=eps)
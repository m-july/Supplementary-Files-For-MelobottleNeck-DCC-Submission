from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch
from tqdm import tqdm

from ..pointer_utils import renormalize_scores_on_mask
from .backbone_metrics import (
    average_precision_binary,
    binary_counts,
    binary_scores_from_counts,
    ndcg_at_k,
    precision_recall_f1_iou_from_masks,
    topk_mask,
    trapz_auc,
)


def _default_cut_rhos() -> Tuple[float, ...]:
    return (
        1.0 / 3.0,
        5.0 / 12.0,
        1.0 / 2.0,
        7.0 / 12.0,
        2.0 / 3.0,
        3.0 / 4.0,
        5.0 / 6.0,
        11.0 / 12.0,
        1.0,
    )


@dataclass(frozen=True)
class GTTMBackboneEvalConfig:
    # reference rho for "single-number" metrics
    eval_rho: float = 2.0 / 3.0

    # AP positives are defined by top-k gold at ap_rho
    ap_rho: Optional[float] = None

    # If not None, evaluation will pass z_len to pointer models:
    #   k = ceil(force_z_len_rho * n_notes)
    #   z_len = k + 2 (BOS + EOS)
    # This is recommended for fair comparison and for baselines that don't have native z_len.
    force_z_len_rho: Optional[float] = 2.0 / 3.0

    exclude_last_step: bool = True
    ignore_special_tokens: bool = True

    # recommended: use gold_depth>=0 to define "note universe"
    eval_mask_from_gold_depth: bool = True

    renormalize_scores_on_eval_mask: bool = True

    # ranking
    ndcg_gain: str = "identity"   # "identity" or "exp2"
    compute_spearman: bool = True

    # cut curve
    compute_cut_curve: bool = True
    cut_rhos: Tuple[float, ...] = field(default_factory=_default_cut_rhos)

    # runtime
    max_batches: Optional[int] = None
    amp: bool = True
    show_progress: bool = True
    tau: Optional[float] = None

    # NEW: cut_rhos 从 dataset meta.json 读（preproc_gttm_benchmark.py 会写）
    cut_rhos_from_dataset_meta: bool = True
    # NEW: cut curve 时对每个 cut_rho 重新跑一次 adapter.predict，并强制 z_len 对应该 rho
    cut_curve_force_z_len_per_rho: bool = True
    # NEW: per-rho cut curve 用 hard_mask 还是 topk(scores)
    # - False: topk(scores)（对 ScoreArrayAdapter/MuDeP 也可用）
    # - True : 有 hard_mask 就用 hard_mask，否则 fallback topk(scores)
    cut_curve_use_hard_mask: bool = True


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def _finite_count(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64)
    return int(np.isfinite(x).sum())


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    """
    Simple rankdata with average ranks for ties (1..n).
    No scipy dependency.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return x.astype(np.float64)

    order = np.argsort(x, kind="mergesort")  # ascending
    ranks = np.empty((n,), dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)

    xs = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        if j - i > 1:
            avg = 0.5 * (i + (j - 1)) + 1.0
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("spearman_corr size mismatch")
    n = int(x.size)
    if n < 2:
        return float("nan")

    rx = _rankdata_average(x)
    ry = _rankdata_average(y)

    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom <= eps:
        return float("nan")
    return float((rx * ry).sum() / denom)


def evaluate_gttm_backbone(
    adapter,
    loader,
    *,
    device: torch.device,
    cfg: GTTMBackboneEvalConfig = GTTMBackboneEvalConfig(),
    prefix: str = "",
) -> Dict[str, float]:
    """
    adapter 协议（与 O2B 对齐）：
      - adapter.pad_id : int
      - adapter.special_n : int
      - adapter.predict(...) -> BackbonePrediction(scores, hard_mask(optional), ...)
      - adapter.module optional (nn.Module) for eval/train switching
    loader batch keys:
      - x: [B,L,3]
      - gold_depth: [B,L]
      - gold_score: [B,L]
      - idx: [B]
    """
    pad_id = int(adapter.pad_id)
    special_n = int(adapter.special_n)

    cut_rhos = np.asarray(cfg.cut_rhos, dtype=np.float64)
    if cfg.cut_rhos_from_dataset_meta:
        ds = getattr(loader, "dataset", None)
        meta = getattr(ds, "meta", None) if ds is not None else None
        if isinstance(meta, dict) and "cut_rhos" in meta:
            cut_rhos = np.asarray(meta["cut_rhos"], dtype=np.float64)
    cut_rhos = np.clip(cut_rhos, 0.0, 1.0)
    cut_rhos = np.unique(cut_rhos)
    cut_rhos.sort()

    ap_rho = float(cfg.eval_rho if cfg.ap_rho is None else cfg.ap_rho)

    buf: Dict[str, List[float]] = {
        "rho": [],
        "n_notes": [],

        # hard-path (if available)
        "precision_hard": [],
        "recall_hard": [],
        "f1_hard": [],
        "iou_hard": [],

        # hard-topk
        "precision_hard_topk": [],
        "recall_hard_topk": [],
        "f1_hard_topk": [],
        "iou_hard_topk": [],

        # soft-score
        "ap_soft": [],

        # soft-ranking
        "ndcg_soft": [],       # NDCG@k_ref (k_ref from eval_rho)
        "ndcg_full": [],       # NDCG@N_notes
        "spearman": [],

        # cut curve
        "cut_f1_auc": [],
        "cut_f1_auc_norm": [],
        "mean_ndcg_cut": [],
    }

    module = getattr(adapter, "module", None)
    was_training = None
    if isinstance(module, torch.nn.Module):
        was_training = bool(module.training)
        module.eval()

    try:
        use_amp = bool(cfg.amp and device.type == "cuda")
        it = loader
        if cfg.show_progress:
            it = tqdm(loader, desc=f"[Eval GTTM] {prefix}".strip(), dynamic_ncols=True, smoothing=0.0)

        with torch.inference_mode():
            for bi, batch in enumerate(it):
                if cfg.max_batches is not None and bi >= int(cfg.max_batches):
                    break

                x: torch.Tensor = batch["x"].to(device, non_blocking=True)  # [B,L,3]
                gold_depth: torch.Tensor = batch["gold_depth"].to(device, non_blocking=True)  # [B,L]
                gold_score: torch.Tensor = batch["gold_score"].to(device, non_blocking=True)  # [B,L]
                idx_t: Optional[torch.Tensor] = batch.get("idx", None)
                if idx_t is not None:
                    idx_t = idx_t.to(device, non_blocking=True)

                B, L, _ = x.shape
                src_mask = (x[..., 0] != pad_id)  # [B,L]
                pitch = x[..., 0]

                if cfg.eval_mask_from_gold_depth:
                    eval_mask = src_mask & (gold_depth >= 0)
                else:
                    eval_mask = src_mask & (pitch >= special_n)

                if cfg.ignore_special_tokens:
                    eval_mask = eval_mask & (pitch >= special_n)

                n_notes_t = eval_mask.to(torch.long).sum(dim=1)  # [B]

                # build z_len for pointer models (optional)
                z_len = None
                if cfg.force_z_len_rho is not None:
                    rho_force = float(cfg.force_z_len_rho)
                    k_force = torch.ceil(n_notes_t.to(torch.float32) * rho_force).to(torch.long)  # [B]
                    max_len = src_mask.to(torch.long).sum(dim=1).clamp_min(2)  # [B]
                    z_len = (k_force + 2).clamp_min(2)                         # 先处理 min
                    z_len = torch.minimum(z_len, max_len)                      # 再处理 max（tensor vs tensor）

                    # if a piece has 0 notes, z_len becomes 2 (BOS+EOS)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred = adapter.predict(
                        src_tokens=x,
                        src_attention_mask=src_mask,
                        z_len=z_len,
                        tau=cfg.tau,
                        exclude_last_step=cfg.exclude_last_step,
                        idx=idx_t,  # ScoreArrayAdapter needs it; PointerModelAdapter will ignore
                    )

                scores_raw = pred.scores.to(torch.float32)
                if scores_raw.shape != src_mask.shape:
                    raise ValueError(f"pred.scores shape mismatch: {tuple(scores_raw.shape)} vs {tuple(src_mask.shape)}")

                if cfg.renormalize_scores_on_eval_mask:
                    scores_eval = renormalize_scores_on_mask(scores_raw, eval_mask)
                else:
                    scores_eval = scores_raw * eval_mask.to(torch.float32)

                # move to cpu once per batch for per-piece metrics
                scores_cpu = scores_eval.detach().cpu().numpy().astype(np.float64)  # [B,L]
                gold_cpu = gold_score.detach().cpu().numpy().astype(np.float64)     # [B,L]
                eval_cpu = eval_mask.detach().cpu().numpy().astype(bool)            # [B,L]

                hard_cpu = None
                if pred.hard_mask is not None:
                    hard_cpu = (pred.hard_mask.to(torch.bool) & eval_mask).detach().cpu().numpy().astype(bool)

                # per-piece macro metrics
                for b in range(B):
                    m = eval_cpu[b]
                    n = int(m.sum())
                    buf["n_notes"].append(float(n))

                    if n <= 0:
                        # pad all metrics with nan
                        for k in (
                            "precision_hard", "recall_hard", "f1_hard", "iou_hard",
                            "precision_hard_topk", "recall_hard_topk", "f1_hard_topk", "iou_hard_topk",
                            "ap_soft", "ndcg_soft", "ndcg_full", "spearman",
                            "cut_f1_auc", "cut_f1_auc_norm", "mean_ndcg_cut",
                        ):
                            buf[k].append(float("nan"))
                        buf["rho"].append(float("nan"))
                        continue

                    s = scores_cpu[b][m]      # [n]
                    g = gold_cpu[b][m]        # [n]

                    def k_from_rho(rho: float) -> int:
                        kk = int(math.ceil(float(rho) * float(n)))
                        kk = max(1, min(kk, n))
                        return kk

                    k_ref = k_from_rho(float(cfg.eval_rho))
                    k_ap = k_from_rho(ap_rho)

                    buf["rho"].append(float(k_ref) / float(max(n, 1)))

                    # gold masks
                    gold_ref = topk_mask(g, np.ones((n,), dtype=bool), k=k_ref)
                    gold_ap = topk_mask(g, np.ones((n,), dtype=bool), k=k_ap)

                    # hard-path (optional)
                    if hard_cpu is not None:
                        hard_b = hard_cpu[b][m]
                        tp, fp, fn = (
                            float(np.logical_and(hard_b, gold_ref).sum()),
                            float(np.logical_and(hard_b, np.logical_not(gold_ref)).sum()),
                            float(np.logical_and(np.logical_not(hard_b), gold_ref).sum()),
                        )
                        prec = tp / max(tp + fp, 1e-8)
                        rec = tp / max(tp + fn, 1e-8)
                        f1 = (2.0 * tp) / max(2.0 * tp + fp + fn, 1e-8)
                        iou = tp / max(tp + fp + fn, 1e-8)
                        buf["precision_hard"].append(float(prec))
                        buf["recall_hard"].append(float(rec))
                        buf["f1_hard"].append(float(f1))
                        buf["iou_hard"].append(float(iou))
                    else:
                        buf["precision_hard"].append(float("nan"))
                        buf["recall_hard"].append(float("nan"))
                        buf["f1_hard"].append(float("nan"))
                        buf["iou_hard"].append(float("nan"))

                    # hard-topk @ k_ref
                    pred_ref = topk_mask(s, np.ones((n,), dtype=bool), k=k_ref)
                    p_tk, r_tk, f1_tk, iou_tk = precision_recall_f1_iou_from_masks(pred_ref, gold_ref)
                    buf["precision_hard_topk"].append(p_tk)
                    buf["recall_hard_topk"].append(r_tk)
                    buf["f1_hard_topk"].append(f1_tk)
                    buf["iou_hard_topk"].append(iou_tk)

                    # AP@k_ap
                    ap = average_precision_binary(s, gold_ap.astype(np.int32))
                    buf["ap_soft"].append(float(ap))

                    # NDCG@k_ref (graded relevance)
                    ndcg_ref = ndcg_at_k(s, g, k=k_ref, gain=str(cfg.ndcg_gain))
                    buf["ndcg_soft"].append(float(ndcg_ref))

                    # NDCG@N (full)
                    ndcg_full = ndcg_at_k(s, g, k=n, gain=str(cfg.ndcg_gain))
                    buf["ndcg_full"].append(float(ndcg_full))

                    # Spearman (optional)
                    if cfg.compute_spearman:
                        buf["spearman"].append(float(spearman_corr(s, g)))
                    else:
                        buf["spearman"].append(float("nan"))

                    # cut curve
                    if (not cfg.compute_cut_curve) or (cut_rhos.size < 2):
                        buf["cut_f1_auc"].append(float("nan"))
                        buf["cut_f1_auc_norm"].append(float("nan"))
                        buf["mean_ndcg_cut"].append(float("nan"))
                        continue

                    if cfg.cut_curve_force_z_len_per_rho:
                        # 先占位，等会儿按 batch 统一 per-rho forward 后再回填
                        buf["cut_f1_auc"].append(float("nan"))
                        buf["cut_f1_auc_norm"].append(float("nan"))
                        buf["mean_ndcg_cut"].append(float("nan"))
                        continue

                    f1_curve = np.full((cut_rhos.size,), np.nan, dtype=np.float64)
                    ndcg_curve = np.full((cut_rhos.size,), np.nan, dtype=np.float64)

                    for ri, rr in enumerate(cut_rhos):
                        k = k_from_rho(float(rr))
                        pred_k = topk_mask(s, np.ones((n,), dtype=bool), k=k)
                        gold_k = topk_mask(g, np.ones((n,), dtype=bool), k=k)
                        _, _, f1c, _ = precision_recall_f1_iou_from_masks(pred_k, gold_k)
                        f1_curve[ri] = float(f1c)

                        ndcg_curve[ri] = float(ndcg_at_k(s, g, k=k, gain=str(cfg.ndcg_gain)))

                    auc = trapz_auc(cut_rhos, f1_curve)
                    auc_norm = auc / max(float(cut_rhos[-1] - cut_rhos[0]), 1e-8)

                    buf["cut_f1_auc"].append(float(auc))
                    buf["cut_f1_auc_norm"].append(float(auc_norm))
                    buf["mean_ndcg_cut"].append(float(np.nanmean(ndcg_curve)))

                # -----------------------------------------
                # NEW: per-rho forward cut curve
                # -----------------------------------------
                if cfg.compute_cut_curve and cfg.cut_curve_force_z_len_per_rho and cut_rhos.size >= 2:
                    # 这一批在 buf 中的起始行号
                    row0 = len(buf["rho"]) - B

                    f1_curves = np.full((B, cut_rhos.size), np.nan, dtype=np.float64)
                    ndcg_curves = np.full((B, cut_rhos.size), np.nan, dtype=np.float64)

                    for ri, rr in enumerate(cut_rhos.tolist()):
                        # 1) 为这个 rr 构造 z_len_rr（对整个 batch）
                        rr = float(rr)
                        k_force = torch.ceil(n_notes_t.to(torch.float32) * rr).to(torch.long)  # [B]
                        max_len = src_mask.to(torch.long).sum(dim=1).clamp_min(2)              # [B]
                        z_len_rr = (k_force + 2).clamp_min(2)
                        z_len_rr = torch.minimum(z_len_rr, max_len)

                        # 2) forward
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            pred_rr = adapter.predict(
                                src_tokens=x,
                                src_attention_mask=src_mask,
                                z_len=z_len_rr,
                                tau=cfg.tau,
                                exclude_last_step=cfg.exclude_last_step,
                                idx=idx_t,
                            )

                        scores_rr = pred_rr.scores.to(torch.float32)  # [B,L]
                        if cfg.renormalize_scores_on_eval_mask:
                            scores_rr = renormalize_scores_on_mask(scores_rr, eval_mask)
                        else:
                            scores_rr = scores_rr * eval_mask.to(torch.float32)

                        scores_rr_cpu = scores_rr.detach().cpu().numpy().astype(np.float64)  # [B,L]

                        hard_rr_cpu = None
                        if cfg.cut_curve_use_hard_mask and (pred_rr.hard_mask is not None):
                            hard_rr_cpu = (pred_rr.hard_mask.to(torch.bool) & eval_mask).detach().cpu().numpy().astype(bool)

                        # 3) per piece 计算 F1(rr), NDCG@k(rr)
                        for b in range(B):
                            m = eval_cpu[b]
                            n = int(m.sum())
                            if n <= 0:
                                continue

                            s = scores_rr_cpu[b][m]  # [n]
                            g = gold_cpu[b][m]       # [n]

                            k = int(math.ceil(rr * n))
                            k = max(1, min(k, n))

                            gold_k = topk_mask(g, np.ones((n,), dtype=bool), k=k)

                            if hard_rr_cpu is not None:
                                pred_k = hard_rr_cpu[b][m]  # [n] bool
                            else:
                                pred_k = topk_mask(s, np.ones((n,), dtype=bool), k=k)

                            _, _, f1c, _ = precision_recall_f1_iou_from_masks(pred_k, gold_k)
                            f1_curves[b, ri] = float(f1c)
                            ndcg_curves[b, ri] = float(ndcg_at_k(s, g, k=k, gain=str(cfg.ndcg_gain)))

                    # 4) per piece 积分 + 回填 buf
                    x0 = float(cut_rhos[0])
                    x1 = float(cut_rhos[-1])
                    denom = max(x1 - x0, 1e-8)

                    for b in range(B):
                        n = int(eval_cpu[b].sum())
                        if n <= 0:
                            continue

                        auc = trapz_auc(cut_rhos, f1_curves[b])
                        buf["cut_f1_auc"][row0 + b] = float(auc)
                        buf["cut_f1_auc_norm"][row0 + b] = float(auc / denom)
                        buf["mean_ndcg_cut"][row0 + b] = float(np.nanmean(ndcg_curves[b]))

    finally:
        if isinstance(module, torch.nn.Module) and was_training is not None:
            module.train(was_training)

    # gap_f1 = f1_hard_topk - f1_hard (nan if either component is nan,
    # which covers the common case where the adapter provides no hard_mask)
    buf["gap_f1"] = (
        np.asarray(buf["f1_hard_topk"], dtype=np.float64)
        - np.asarray(buf["f1_hard"], dtype=np.float64)
    ).tolist()

    out: Dict[str, float] = {}
    out[prefix + "n"] = float(len(buf["rho"]))

    def add_mean(name: str):
        arr = np.asarray(buf[name], dtype=np.float64)
        out[prefix + name] = _nanmean(arr)
        out[prefix + name + "_n"] = float(_finite_count(arr))

    add_mean("rho")
    add_mean("n_notes")

    add_mean("precision_hard")
    add_mean("recall_hard")
    add_mean("f1_hard")
    add_mean("iou_hard")

    add_mean("precision_hard_topk")
    add_mean("recall_hard_topk")
    add_mean("f1_hard_topk")
    add_mean("iou_hard_topk")

    add_mean("gap_f1")

    add_mean("ap_soft")
    add_mean("ndcg_soft")
    add_mean("ndcg_full")
    add_mean("spearman")

    add_mean("cut_f1_auc")
    add_mean("cut_f1_auc_norm")
    add_mean("mean_ndcg_cut")

    # record config scalars (方便 wandb / json 里读)
    out[prefix + "eval_rho"] = float(cfg.eval_rho)
    out[prefix + "ap_rho"] = float(ap_rho)
    out[prefix + "force_z_len_rho"] = float(cfg.force_z_len_rho) if cfg.force_z_len_rho is not None else float("nan")

    return out
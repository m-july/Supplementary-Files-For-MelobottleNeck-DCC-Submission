# main/evaluation/ornament_to_backbone_eval.py
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
class OrnamentToBackboneEvalConfig:
    rho_bins: Optional[Tuple[float, ...]] = None
    # (
    #     1.0 / 3.0,
    #     1.0 / 2.0,
    #     2.0 / 3.0,
    #     5.0 / 6.0,
    #     1.0 + 1e-6,
    # )

    exclude_last_step: bool = True
    ignore_special_tokens: bool = True
    renormalize_scores_on_eval_mask: bool = True

    compute_cut_curve: bool = True
    cut_rhos: Tuple[float, ...] = field(default_factory=_default_cut_rhos)

    inserted_value: int = -1
    pad_pi_value: int = -2

    max_batches: Optional[int] = None
    amp: bool = True
    show_progress: bool = True

    tau: Optional[float] = None


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return float(np.nanmean(x))


def _finite_count(x: np.ndarray) -> int:
    x = np.asarray(x, dtype=np.float64)
    return int(np.isfinite(x).sum())


def evaluate_ornament_to_backbone(
    adapter,
    loader,
    *,
    device: torch.device,
    cfg: OrnamentToBackboneEvalConfig = OrnamentToBackboneEvalConfig(),
    prefix: str = "",
) -> Dict[str, float]:
    """
    adapter 必须暴露：
      - pad_id
      - special_n
      - predict(...)
      - 可选 module（nn.Module，用于 eval/train mode 切换）
    """
    pad_id = int(adapter.pad_id)
    special_n = int(adapter.special_n)

    buf: Dict[str, List[float]] = {
        "rho": [],
        "rho_note": [],

        # hard path (来自 adapter.hard_mask)
        "precision_hard": [],
        "recall_hard": [],
        "f1_hard": [],
        "iou_hard": [],

        # hard top-k from score
        "precision_hard_topk": [],
        "recall_hard_topk": [],
        "f1_hard_topk": [],
        "iou_hard_topk": [],

        # soft
        "ap_soft": [],
        "ndcg_soft": [],
        "ins_mass_soft": [],

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

    cut_rhos = np.asarray(cfg.cut_rhos, dtype=np.float64)
    cut_rhos = np.clip(cut_rhos, 0.0, 1.0)
    cut_rhos = np.unique(cut_rhos)
    cut_rhos.sort()

    try:
        use_amp = bool(cfg.amp and device.type == "cuda")
        it = loader
        if cfg.show_progress:
            total = len(loader) if cfg.max_batches is None else min(len(loader), cfg.max_batches)
            it = tqdm(loader, desc=f"[Eval OTB] {prefix}".strip(), dynamic_ncols=True, smoothing=0.0, total=total)

        with torch.inference_mode():
            for bi, batch in enumerate(it):
                if cfg.max_batches is not None and bi >= int(cfg.max_batches):
                    break

                x_orn: torch.Tensor = batch["x_orn"].to(device, non_blocking=True)  # [B,L,3]
                pi: torch.Tensor = batch["pi"].to(device, non_blocking=True)        # [B,L]
                B, L, _ = x_orn.shape

                src_mask = (x_orn[..., 0] != pad_id)  # [B,L]

                if "len_x" in batch:
                    z_len = batch["len_x"].to(device, non_blocking=True).to(torch.long)
                else:
                    z_len = src_mask.to(torch.long).sum(dim=1).clamp_min(1)

                if "rho" in batch:
                    rho = batch["rho"].detach().cpu().numpy().astype(np.float64)
                else:
                    if "len_x_orn" in batch:
                        len_orn = batch["len_x_orn"].detach().cpu().numpy().astype(np.float64)
                    else:
                        len_orn = src_mask.to(torch.long).sum(dim=1).detach().cpu().numpy().astype(np.float64)
                    len_x_np = z_len.detach().cpu().numpy().astype(np.float64)
                    rho = len_x_np / np.maximum(len_orn, 1.0)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred = adapter.predict(
                        src_tokens=x_orn,
                        src_attention_mask=src_mask,
                        z_len=z_len,
                        tau=cfg.tau,
                        exclude_last_step=cfg.exclude_last_step,
                    )

                scores_raw = pred.scores.to(torch.float32)  # [B,L]
                if scores_raw.shape != src_mask.shape:
                    raise ValueError(f"pred.scores shape mismatch: {scores_raw.shape} vs {src_mask.shape}")

                pitch = x_orn[..., 0]
                if cfg.ignore_special_tokens:
                    eval_mask = src_mask & (pitch >= special_n)
                else:
                    eval_mask = src_mask

                true_mask = (pi >= 0) & eval_mask  # O2B gold positives

                # -------- hard path metrics (如果 adapter 提供) --------
                if pred.hard_mask is not None:
                    hard_mask = pred.hard_mask.to(torch.bool) & eval_mask
                    tp, fp, fn = binary_counts(hard_mask, true_mask, mask=None)
                    prec, rec, f1, iou = binary_scores_from_counts(tp, fp, fn)

                    buf["precision_hard"].extend(prec.detach().cpu().numpy().astype(np.float64).tolist())
                    buf["recall_hard"].extend(rec.detach().cpu().numpy().astype(np.float64).tolist())
                    buf["f1_hard"].extend(f1.detach().cpu().numpy().astype(np.float64).tolist())
                    buf["iou_hard"].extend(iou.detach().cpu().numpy().astype(np.float64).tolist())
                else:
                    buf["precision_hard"].extend([float("nan")] * B)
                    buf["recall_hard"].extend([float("nan")] * B)
                    buf["f1_hard"].extend([float("nan")] * B)
                    buf["iou_hard"].extend([float("nan")] * B)

                # -------- insertion mass: 在 src valid universe 上归一化后再算 --------
                scores_src_mass = renormalize_scores_on_mask(scores_raw, src_mask)
                ins_mask = (pi == int(cfg.inserted_value)) & src_mask
                ins_mass = (scores_src_mass * ins_mask.to(torch.float32)).sum(dim=1)
                buf["ins_mass_soft"].extend(ins_mass.detach().cpu().numpy().astype(np.float64).tolist())

                # -------- note-level rho --------
                n_eval = eval_mask.to(torch.long).sum(dim=1).detach().cpu().numpy().astype(np.float64)
                n_true = true_mask.to(torch.long).sum(dim=1).detach().cpu().numpy().astype(np.float64)
                rho_note = n_true / np.maximum(n_eval, 1.0)

                buf["rho"].extend(rho.tolist())
                buf["rho_note"].extend(rho_note.tolist())

                # -------- soft metrics 使用 eval universe --------
                if cfg.renormalize_scores_on_eval_mask:
                    scores_eval = renormalize_scores_on_mask(scores_raw, eval_mask)
                else:
                    scores_eval = scores_raw * eval_mask.to(torch.float32)

                # piece-wise macro metrics
                for b in range(B):
                    m_b = eval_mask[b].detach().cpu().numpy().astype(bool)
                    if m_b.sum() == 0:
                        buf["precision_hard_topk"].append(float("nan"))
                        buf["recall_hard_topk"].append(float("nan"))
                        buf["f1_hard_topk"].append(float("nan"))
                        buf["iou_hard_topk"].append(float("nan"))
                        buf["ap_soft"].append(float("nan"))
                        buf["ndcg_soft"].append(float("nan"))
                        buf["cut_f1_auc"].append(float("nan"))
                        buf["cut_f1_auc_norm"].append(float("nan"))
                        buf["mean_ndcg_cut"].append(float("nan"))
                        continue

                    scores_b = scores_eval[b].detach().cpu().numpy().astype(np.float64)[m_b]
                    labels_b = true_mask[b].detach().cpu().numpy().astype(np.int32)[m_b]
                    labels_bool_b = labels_b.astype(bool)

                    k_true = int(labels_b.sum())

                    # ---- hard_topk ----
                    if k_true > 0:
                        pred_topk = topk_mask(
                            scores_b,
                            np.ones_like(labels_bool_b, dtype=bool),
                            k=k_true,
                        )
                        p_tk, r_tk, f1_tk, iou_tk = precision_recall_f1_iou_from_masks(pred_topk, labels_bool_b)
                    else:
                        p_tk = r_tk = f1_tk = iou_tk = float("nan")

                    buf["precision_hard_topk"].append(p_tk)
                    buf["recall_hard_topk"].append(r_tk)
                    buf["f1_hard_topk"].append(f1_tk)
                    buf["iou_hard_topk"].append(iou_tk)

                    # ---- AP / NDCG@K_true ----
                    ap = average_precision_binary(scores_b, labels_b)
                    ndcg = ndcg_at_k(
                        scores_b,
                        labels_b.astype(np.float64),
                        k=max(1, k_true) if k_true > 0 else 1,
                        gain="identity",
                    ) if k_true > 0 else float("nan")

                    buf["ap_soft"].append(ap)
                    buf["ndcg_soft"].append(ndcg)

                    # ---- cut curve (固定 gold set，改变 pred top-K) ----
                    if (not cfg.compute_cut_curve) or (cut_rhos.size < 2) or (k_true <= 0):
                        buf["cut_f1_auc"].append(float("nan"))
                        buf["cut_f1_auc_norm"].append(float("nan"))
                        buf["mean_ndcg_cut"].append(float("nan"))
                        continue

                    n_cand = int(scores_b.shape[0])

                    f1_curve = np.full((cut_rhos.size,), np.nan, dtype=np.float64)
                    ndcg_curve = np.full((cut_rhos.size,), np.nan, dtype=np.float64)

                    for ri, rr in enumerate(cut_rhos):
                        k = int(math.ceil(float(rr) * float(n_cand)))
                        k = max(1, min(k, n_cand))

                        pred_cut = topk_mask(
                            scores_b,
                            np.ones_like(labels_bool_b, dtype=bool),
                            k=k,
                        )
                        _, _, f1c, _ = precision_recall_f1_iou_from_masks(pred_cut, labels_bool_b)
                        f1_curve[ri] = f1c

                        ndcg_curve[ri] = ndcg_at_k(
                            scores_b,
                            labels_b.astype(np.float64),
                            k=k,
                            gain="identity",
                        )

                    auc = trapz_auc(cut_rhos, f1_curve)
                    auc_norm = auc / max(float(cut_rhos[-1] - cut_rhos[0]), 1e-8)

                    buf["cut_f1_auc"].append(float(auc))
                    buf["cut_f1_auc_norm"].append(float(auc_norm))
                    buf["mean_ndcg_cut"].append(float(np.nanmean(ndcg_curve)))

    finally:
        if isinstance(module, torch.nn.Module) and was_training is not None:
            module.train(was_training)

    # gap_f1 = f1_hard_topk - f1_hard (nan if either component is nan)
    buf["gap_f1"] = (
        np.asarray(buf["f1_hard_topk"], dtype=np.float64)
        - np.asarray(buf["f1_hard"], dtype=np.float64)
    ).tolist()

    out: Dict[str, float] = {}
    n_total = len(buf["rho"])
    out[prefix + "n"] = float(n_total)
    out[prefix + "rho"] = _nanmean(np.asarray(buf["rho"]))
    out[prefix + "rho_note"] = _nanmean(np.asarray(buf["rho_note"]))

    def add_mean(name: str):
        arr = np.asarray(buf[name], dtype=np.float64)
        out[prefix + name] = _nanmean(arr)
        out[prefix + name + "_n"] = float(_finite_count(arr))

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
    add_mean("ins_mass_soft")

    add_mean("cut_f1_auc")
    add_mean("cut_f1_auc_norm")
    add_mean("mean_ndcg_cut")

    # rho buckets
    if cfg.rho_bins is not None:
        rho_arr = np.asarray(buf["rho"], dtype=np.float64)
        bins = tuple(float(x) for x in cfg.rho_bins)
        if len(bins) >= 2:
            for i in range(len(bins) - 1):
                lo, hi = bins[i], bins[i + 1]
                m = (rho_arr >= lo) & (rho_arr < hi)
                key = f"rho_bin{i}_[{lo:.3f},{hi:.3f})/"

                out[prefix + key + "n"] = float(int(m.sum()))
                if m.sum() == 0:
                    for nm in (
                        "f1_hard",
                        "f1_hard_topk",
                        "gap_f1",
                        "iou_hard",
                        "iou_hard_topk",
                        "ap_soft",
                        "ndcg_soft",
                        "ins_mass_soft",
                        "cut_f1_auc_norm",
                        "mean_ndcg_cut",
                    ):
                        out[prefix + key + nm] = float("nan")
                    continue

                def bmean(name: str) -> float:
                    a = np.asarray(buf[name], dtype=np.float64)
                    return _nanmean(a[m])

                out[prefix + key + "f1_hard"] = bmean("f1_hard")
                out[prefix + key + "f1_hard_topk"] = bmean("f1_hard_topk")
                out[prefix + key + "gap_f1"] = bmean("gap_f1")
                out[prefix + key + "iou_hard"] = bmean("iou_hard")
                out[prefix + key + "iou_hard_topk"] = bmean("iou_hard_topk")
                out[prefix + key + "ap_soft"] = bmean("ap_soft")
                out[prefix + key + "ndcg_soft"] = bmean("ndcg_soft")
                out[prefix + key + "ins_mass_soft"] = bmean("ins_mass_soft")
                out[prefix + key + "cut_f1_auc_norm"] = bmean("cut_f1_auc_norm")
                out[prefix + key + "mean_ndcg_cut"] = bmean("mean_ndcg_cut")

    return out
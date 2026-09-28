from __future__ import annotations

import os
import shutil
import math
from dataclasses import dataclass, field, fields, asdict, replace
from typing import Optional, Tuple, Dict, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

import gc
import argparse
import warnings
import contextlib
import time
import json
import copy

import torch.nn.functional as F

try:
    import wandb
except Exception:
    wandb = None

from .vocab_utils import load_vocab_info
from .data import make_dataloader
from .augment import MusicAugmentConfig
from .quantization import MusicQuantizationTables

from .config import BartDenoiseConfig
from .denoise import BartStyleDenoiser

from .models.bart import MusicBartConfig, MusicBartBackboneConfig
from .nn_modules import MusicBartBackbone, MusicBartForSeq2SeqLM
from .nn_funcs.token_ce_loss import multi_attribute_ce_loss
from .utils import infer_attention_mask_from_tokens, mask_labels_with_ignore_index

from .models.skeleton.model import MusicSkeletonModelIII, MusicSkeletonIIIConfig
from .models.skeleton.lm_prior_decoder_only import MusicBartDecoderOnlyLM

from .ornament import MusicOrnamenter, MusicOrnamentConfig, PI_INSERTED
from .models.skeleton.ornament_invariance import (
    marginal_importance_from_pointer_soft,
    apply_duration_weight_to_importance,
    aggregate_importance_by_pi,
    masked_mse_loss,
    insertion_mass_loss,
)

from .evaluation import (
    make_ornament_benchmark_dataloader,
    OrnamentToBackboneEvalConfig,
    evaluate_ornament_to_backbone,
    build_pointer_adapter_from_model,
    make_gttm_benchmark_dataloader,
    GTTMBackboneEvalConfig,
    evaluate_gttm_backbone,
)

from .evaluation.music_prior_proxy_eval import MusicPriorProxyEvalConfig, evaluate_music_prior_proxy

from .pointer_utils import renormalize_scores_on_mask, importance_from_score_logits


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def warmup_cosine_lr_factor(step: int, total_steps: int, warmup_steps: int, final_lr_ratio: float, init_lr_ratio: float) -> float:
    if total_steps <= 1:
        return 1.0
    step = max(0, min(step, total_steps - 1))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    if warmup_steps > 0 and step < warmup_steps:
        return init_lr_ratio + (1 - init_lr_ratio) * float(step + 1) / float(warmup_steps)

    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return 1.0
    decay_step = step - warmup_steps
    t = decay_step / float(decay_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(final_lr_ratio + (1.0 - final_lr_ratio) * cosine)


def warmup_corrupt_factor(
    epoch: int,            # 当前 epoch（从 1 开始计）
    total_epochs: int,     # 总训练轮数
    warmup_epochs: int,    # warmup 持续的 epoch 数
    begin_epoch: int,      # 从第几个 epoch 开始 warmup（0-based，和 seqcls 一致）
    init_factor: float = 0.0,
    final_factor: float = 1.0,
) -> float:
    """
    计算一个乘到 base_corrupt_ratio 上的 factor，用于在训练早期逐步增加扰动比例。

    约定（与 train_pretrain_then_finetune_seqcls.py 保持一致）：
      - epoch < begin_epoch: 恒为 init_factor
      - begin_epoch <= epoch < begin_epoch + warmup_epochs:
          线性从 init_factor -> final_factor
          （第一个 warmup epoch 就比 init_factor 略大，最后一个 warmup epoch 为 final_factor）
      - epoch >= begin_epoch + warmup_epochs: 恒为 final_factor
    """
    epoch -= 1  # 1-based -> 0-based

    if total_epochs <= 1:
        return float(final_factor)

    epoch = max(0, min(epoch, total_epochs - 1))
    begin_epoch = max(0, min(begin_epoch, total_epochs))

    max_warmup = max(0, total_epochs - begin_epoch)
    warmup_epochs = max(0, min(warmup_epochs, max_warmup))

    if warmup_epochs == 0:
        return float(init_factor if epoch < begin_epoch else final_factor)

    warmup_start = begin_epoch
    warmup_end = begin_epoch + warmup_epochs

    if epoch < warmup_start:
        return float(init_factor)

    if epoch < warmup_end:
        warmup_step = epoch - warmup_start
        alpha = float(warmup_step + 1) / float(warmup_epochs)
        return float(init_factor + (final_factor - init_factor) * alpha)

    return float(final_factor)


def linear_warmup_step(
    step_0based: int,
    begin_step: int,
    warmup_steps: int,
    start: float,
    end: float,
) -> float:
    """
    0-based step 版线性 warmup。

    约定：
    - step < begin_step: 返回 start
    - begin_step <= step < begin_step + warmup_steps:
        线性从 start -> end
        且第一个 warmup step 就略大于 start（与旧 epoch 逻辑风格一致）
    - step >= begin_step + warmup_steps: 返回 end
    """
    begin_step = max(0, int(begin_step))
    warmup_steps = max(0, int(warmup_steps))

    if step_0based < begin_step:
        return float(start)

    if warmup_steps <= 0:
        return float(end)

    rel = min(max(step_0based - begin_step + 1, 0), warmup_steps)
    alpha = float(rel) / float(warmup_steps)
    return float(start + (end - start) * alpha)


def warmup_corrupt_factor_stepwise_from_epoch_anchors(
    step_0based: int,
    steps_per_epoch: int,
    begin_epoch: float,
    warmup_epochs: float,
    init_factor: float = 0.0,
    final_factor: float = 1.0,
) -> float:
    """
    配置仍然用 epoch 单位，但训练时按 step 平滑更新。
    """
    steps_per_epoch = max(1, int(steps_per_epoch))
    begin_step = max(0, int(round(float(begin_epoch) * steps_per_epoch)))
    warmup_steps = max(0, int(round(float(warmup_epochs) * steps_per_epoch)))

    return linear_warmup_step(
        step_0based=step_0based,
        begin_step=begin_step,
        warmup_steps=warmup_steps,
        start=init_factor,
        end=final_factor,
    )


def apply_lr_factor(optim: torch.optim.Optimizer, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


def set_dropout_p(model: nn.Module, p: float):
    """
    Make dropout effective for:
      1) nn.Dropout modules (m.p)
      2) HuggingFace-style float dropouts used via F.dropout(..., p=self.dropout, ...)
         e.g. BartEncoderLayer.dropout / activation_dropout, BartAttention.dropout, etc.
      3) (optional but recommended) patch transformers config fields too.
    """
    p = float(p)
    if p < 0.0:
        p = 0.0
    # p==1.0 often leads to inf scale (1/(1-p)) inside dropout kernels -> NaN risk.
    # If you really want "destroy" ablation, use 0.99/0.999 instead of 1.0.
    if p >= 1.0:
        p = 1.0 - 1e-6

    # Common dropout-related float attributes in HF transformer blocks
    float_dropout_attrs = (
        "dropout",             # BartEncoderLayer.dropout / BartAttention.dropout / etc.
        "attention_dropout",   # mostly on config; some models keep it as attr
        "activation_dropout",  # BartEncoderLayer.activation_dropout
        "classifier_dropout",  # heads (not used here but safe)
    )

    for m in model.modules():
        # 1) nn.Dropout modules
        if isinstance(m, nn.Dropout):
            m.p = p

        # 2) HF-style float attrs
        for name in float_dropout_attrs:
            if hasattr(m, name):
                v = getattr(m, name)
                if isinstance(v, (float, int)):
                    setattr(m, name, p)

        # 3) Patch HF config too (some code paths may read from config)
        cfg = getattr(m, "config", None)
        if cfg is not None:
            for name in float_dropout_attrs:
                if hasattr(cfg, name):
                    setattr(cfg, name, p)


def linear_warmup_epoch(epoch_1based: int, warmup_epochs: int, start: float, end: float) -> float:
    """
    epoch_1based: 1..E
    warmup_epochs: 在前 warmup_epochs 内线性从 start -> end
    """
    if warmup_epochs <= 0:
        return float(end)
    e = max(1, int(epoch_1based))
    if e >= warmup_epochs:
        alpha = 1.0
    else:
        alpha = float(e) / float(warmup_epochs)
    return float(start + (end - start) * alpha)


def build_music_bart_cfg(model_cfg, vocab_cfg, dropout: float) -> MusicBartConfig:
    backbone_cfg = MusicBartBackboneConfig(
        max_seq_len=model_cfg.max_seq_len,
        d_embed=model_cfg.d_embed,
        d_model=model_cfg.d_model,
        n_encoder_layers=model_cfg.n_encoder_layers,
        n_decoder_layers=model_cfg.n_decoder_layers,
        n_heads=model_cfg.n_heads,
        d_ff=model_cfg.d_ff,
        dropout=float(dropout),
    )
    return MusicBartConfig(vocab=vocab_cfg, backbone=backbone_cfg)


def save_ckpt(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)
    print(f"[Saved] {path}")

class ScalarMeanTracker:
    def __init__(self) -> None:
        self.sums: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.num_updates: int = 0

    def update(self, metrics: Dict[str, object]) -> None:
        if not metrics:
            return
        self.num_updates += 1
        for k, v in metrics.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() != 1:
                    raise ValueError(f"ScalarMeanTracker only accepts scalar tensors, got {k}: {tuple(v.shape)}")
                x = float(v.detach().cpu().item())
            else:
                x = float(v)

            if not math.isfinite(x):
                continue

            self.sums[k] = self.sums.get(k, 0.0) + x
            self.counts[k] = self.counts.get(k, 0) + 1

    def has(self, key: str) -> bool:
        return self.counts.get(key, 0) > 0

    def mean(self, key: str) -> float:
        return self.sums[key] / self.counts[key]

    def mean_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}{k}": self.sums[k] / self.counts[k]
            for k in sorted(self.sums.keys())
            if self.counts.get(k, 0) > 0
        }
    
def extract_stagec_final_window_batch_metrics(out) -> Dict[str, object]:
    metrics: Dict[str, object] = {}

    recon_loss_dict = getattr(out, "recon_loss_dict", None) or {}
    if "loss_pitch" in recon_loss_dict:
        metrics["recon_ce/pitch"] = recon_loss_dict["loss_pitch"]
    if "loss_duration" in recon_loss_dict:
        metrics["recon_ce/duration"] = recon_loss_dict["loss_duration"]
    if "loss_dt" in recon_loss_dict:
        metrics["recon_ce/dt"] = recon_loss_dict["loss_dt"]

    diag = getattr(out, "diag", None) or {}
    if "recon_wo_z/ratio_wo_over_w" in diag:
        metrics["recon_wo_z/ratio_wo_over_w"] = diag["recon_wo_z/ratio_wo_over_w"]

    return metrics

def summarize_stagec_final_window_metrics(
    tracker: ScalarMeanTracker,
    *,
    attr_weights: Tuple[float, float, float],
    requested_batches: int,
    total_steps: int,
    prefix: str = "C_lastN/",
    exp_clip: float = 20.0,
) -> Dict[str, float]:
    effective_batches = int(tracker.num_updates)

    out: Dict[str, float] = {
        prefix + "window_batches_requested": int(requested_batches),
        prefix + "window_batches_effective": effective_batches,
        prefix + "total_steps": int(total_steps),
        prefix + "start_step_0based": int(max(0, total_steps - effective_batches)),
        prefix + "end_step_0based_inclusive": int(total_steps - 1) if effective_batches > 0 else -1,
    }

    out.update(tracker.mean_dict(prefix=prefix))

    if tracker.has("recon_ce/pitch") and tracker.has("recon_ce/duration") and tracker.has("recon_ce/dt"):
        ce_p = tracker.mean("recon_ce/pitch")
        ce_d = tracker.mean("recon_ce/duration")
        ce_dt = tracker.mean("recon_ce/dt")

        def _safe_exp(x: float) -> float:
            return float(math.exp(min(float(x), float(exp_clip))))

        out[prefix + "recon_ppl/pitch"] = _safe_exp(ce_p)
        out[prefix + "recon_ppl/duration"] = _safe_exp(ce_d)
        out[prefix + "recon_ppl/dt"] = _safe_exp(ce_dt)

        w = [max(0.0, float(v)) for v in attr_weights]
        w_sum = sum(w)
        if w_sum <= 0.0:
            w = [1.0, 1.0, 1.0]
            w_sum = 3.0

        ce_weighted = (w[0] * ce_p + w[1] * ce_d + w[2] * ce_dt) / w_sum
        out[prefix + "recon_ce/weighted_mean"] = float(ce_weighted)
        out[prefix + "recon_ppl/weighted_geom"] = _safe_exp(ce_weighted)

    return out


# -------------------------
# Step-level timer (training profiler)
# -------------------------
class StepTimer:
    """
    Lightweight per-step wall-clock timer with optional CUDA synchronisation.

    Usage
    -----
    timer = StepTimer(enabled=True, device=device)

    # start/stop style — use when the timed region may exit via break/return:
    timer.start("data_load")
    x = next(it)              # may raise StopIteration -> break
    timer.stop("data_load")

    # context-manager style — cleaner for normal regions:
    with timer.measure("forward_main"):
        out = model(x)

    # Cumulative-average dict for wandb (mean seconds per step, all tags):
    log_dict.update(timer.wandb_log_dict())

    Notes
    -----
    * When enabled=False every method is a strict no-op (zero overhead).
    * When enabled=True, torch.cuda.synchronize() is called at every boundary
      so that GPU kernels finish before the clock is read.  This makes timings
      accurate but adds ~0.1-0.5 ms of overhead per boundary — only use in
      profiling runs.
    * Counters accumulate across epochs; wandb_log_dict() always returns the
      cumulative average from the moment the timer was created.
    * If a start() is never matched by stop() (e.g. because a StopIteration
      break fires), the orphaned entry in _starts is harmlessly overwritten on
      the next call to start() with the same tag.
    """

    def __init__(self, enabled: bool, device: torch.device) -> None:
        self.enabled = enabled
        self.device = device
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._starts: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------
    def start(self, tag: str) -> None:
        if not self.enabled:
            return
        self._sync()
        self._starts[tag] = time.perf_counter()

    def stop(self, tag: str) -> None:
        if not self.enabled:
            return
        if tag not in self._starts:
            return
        self._sync()
        elapsed = time.perf_counter() - self._starts.pop(tag)
        self.totals[tag] = self.totals.get(tag, 0.0) + elapsed
        self.counts[tag] = self.counts.get(tag, 0) + 1

    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def measure(self, tag: str):
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - t0
            self.totals[tag] = self.totals.get(tag, 0.0) + elapsed
            self.counts[tag] = self.counts.get(tag, 0) + 1

    # ------------------------------------------------------------------
    def mean(self, tag: str) -> float:
        c = self.counts.get(tag, 0)
        return self.totals.get(tag, 0.0) / c if c > 0 else 0.0

    def wandb_log_dict(self, prefix: str = "C_timers/") -> Dict[str, float]:
        """
        Returns a dict of {prefix + tag + "_s": mean_seconds_per_step} for
        every tag that has been measured at least once.  Tags are sorted
        alphabetically for stable wandb column ordering.
        """
        return {f"{prefix}{tag}_s": self.mean(tag) for tag in sorted(self.totals)}


# -------------------------
# Configs
# -------------------------
@dataclass
class ModelHyperConfig:
    max_seq_len: int = 514
    d_embed: int = 256
    d_model: int = 512
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 8
    d_ff: int = 2048


@dataclass
class StageASeq2SeqPretrainConfig:
    enable: bool = True
    epochs: int = 10
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 50
    dropout: float = 0.15
    amp: bool = True
    use_lr_schedule: bool = True
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.2
    # --- checkpoint saving ---
    save_epochs: Tuple[int, ...] = (5, 40, 200)  # 在这些 epoch 保存 checkpoint
    # --- noise curriculum (step-interpolated; epoch params are anchors) ---
    # 仍然用“epoch”为配置单位，方便直觉理解；
    # 但训练时会换算成 step，并按 batch / every-n-batches 更新。
    use_noise_curriculum: bool = True
    noise_curriculum_update_every_steps: int = 1  # 1 = 每个 batch 更新一次

    # masking
    masking_begin_epoch: float = 0.0
    masking_warmup_epochs: float = 4.0
    masking_init_factor: float = 0.1

    # deletion
    deletion_begin_epoch: float = 2.0
    deletion_warmup_epochs: float = 15.0
    deletion_init_factor: float = 0.0

    # rotation
    rotation_begin_epoch: float = 4.0
    rotation_warmup_epochs: float = 30.0
    rotation_init_factor: float = 0.0
    # ---- step-level timing / profiling ----
    enable_timers: bool = True


@dataclass
class StageBLMPriorTrainConfig:
    enable: bool = True
    epochs: int = 10
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 50
    dropout: float = 0.1
    amp: bool = True
    use_lr_schedule: bool = True
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.3
    attr_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    # --- checkpoint saving ---
    save_epochs: Tuple[int, ...] = (5, 80, 400)  # 在这些 epoch 保存 checkpoint


@dataclass
class StageCSkeletonTrainConfig:
    enable: bool = True
    epochs: int = 50
    batch_size: int = 16

    # 分组 LR：pointer head 通常需要更大学习率
    lr_backbone: float = 1e-5
    lr_pointer: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 20
    dropout: float = 0.15
    amp: bool = True

    use_lr_schedule: bool = True
    warmup_ratio: float = 0.3
    final_lr_ratio: float = 0.05
    init_lr_ratio: float = 0.05

    # ---- curricula ----
    rho_start: float = 2.0 / 3.0
    rho_warmup_ratio: float = 0.5

    tau_start: float = 1.5
    tau_final: float = 0.5
    tau_warmup_ratio: float = 0.5

    lambda_P_start: float = 0.8 # 0.001
    lambda_P_final: float = 0.2
    lambda_P_warmup_ratio: float = 0.5

    lambda_GA_start: float = 0.05 # 0.001
    lambda_GA_final: float = 0.0
    lambda_GA_warmup_ratio: float = 0.1

    lambda_L_start: float = 10.0 # 0.001
    lambda_L_final: float = 10.0
    lambda_L_warmup_ratio: float = 0.7

    lambda_sharp_start: float = 0.0
    lambda_sharp_final: float = 0.0
    lambda_sharp_warmup_ratio: float = 0.5

    lambda_cons_start: float = 6.0 # 0.001
    lambda_cons_final: float = 3.0
    lambda_cons_warmup_ratio: float = 0.5

    lambda_ins_start: float = 6.0 # 0.001
    lambda_ins_final: float = 3.0
    lambda_ins_warmup_ratio: float = 0.5

    # ---- reconstructor scheduled sampling (teacher forcing scheduling) ----
    ss_prob_start: float = 0.0
    ss_prob_final: float = 0.0
    ss_prob_warmup_ratio: float = 0.3

    # ---- reconstructor decoder-input mask dropout (scheduled by rho) ----
    recon_input_mask_prob_start: float = 0.001 # 0.001
    recon_input_mask_prob_final: float = 0.8
    recon_input_mask_prob_warmup_ratio: float = 0.5

    freeze_backbone_epochs: int = 0  # 0=不冻结

    # ---- ornament invariance (Compressor consistency) ----
    use_ornament_invariance: bool = True
    ornament_p_apply: float = 1.0  # strong view 通常设 1.0
    # teacher s(x) 的来源：
    # - "same_forward": 直接用 out.pointer_soft (含 dropout/gumbel 造成的噪声，但零额外算力)
    # - "eval_compressor": 额外用 eval 模式跑一次 compressor（更稳，但更慢）
    # - "ema"
    ornament_teacher_mode: str = "eval_compressor"
    # 计算 s_l 时是否排除最后一步(通常是强制 EOS)
    ornament_exclude_last_step: bool = True
    # duration re-weight: alpha=0 关闭；>0 强调长音
    ornament_dur_alpha: float = 0.0
    # student strong 分支是否强制使用 teacher 的 z_len（推荐 True，避免“长度不一致导致的假不变性/假不一致”）
    ornament_use_teacher_zlen: bool = True
    # 【注意】dynamic_rho=True 且不使用 teacher zlen 的话，需要你自己决定 z_len 策略
    # strong 分支为了稳定一致性，可选禁用 gumbel（不影响主分支的探索）
    ornament_student_disable_gumbel: bool = False

    ornament_tau: float = 1.0                 # NEW: 固定用于 ornament loss 的 tau（teacher+student）
    ornament_student_eval_mode: bool = True   # NEW: strong 分支用 eval() 跑 compressor（关 dropout/gumbel）
    ornament_cons_loss: str = "kl"            # NEW: "mse" or "kl"
    ornament_cons_renorm: bool = True         # NEW: 在 mask 上把 s 重新归一化
    ornament_cons_ignore_special: bool = True # NEW: consistency 比较时忽略 BOS/EOS 等 special（推荐）
    # ---- EMA teacher (optional) ----
    ornament_ema_momentum: float = 0.99            # teacher_mode="ema" 时用

    # ---- debug / fast iteration ----
    # 若非 None，则只使用数据集前 limit_batches 个 batch 的样本（即前 limit_batches*batch_size 条）。
    # 适合快速验证一次完整 epoch 的训练流程，不影响其他 stage。
    limit_batches: Optional[int] = None

    # ---- step-level timing / profiling ----
    # When True, each training step is profiled with CUDA-synchronised wall-clock
    # timers and results are reported to wandb under "C_timers/...".
    # The synchronisation adds a small overhead (~0.1-0.5 ms per boundary), so
    # keep False in production runs and only enable for profiling.
    enable_timers: bool = False

    # ---- ablation: w/o BART pretrain init ----
    # True: Stage C backbone 从随机权重训练（不加载 StageA 的 backbone_init_ckpt）
    wo_bart_pretrain_init: bool = False

    # ---- warm-start from O2B-Learner (keep baseline) ckpt ----
    # ckpt saved by train_skeleton_keep_baseline.py (e.g. keep_last.pt)
    keep_ckpt: Optional[str] = None # r".\ckpt\keep_last.pt"
    # What to load from keep_ckpt:
    #   - "none": disable (default)
    #   - "backbone": load backbone weights (NEW recommended behavior)
    #   - "score_head": load only compressor.keep_head (LEGACY behavior)
    #   - "both": load backbone + keep_head
    keep_warm_start: Literal["none", "backbone", "score_head", "both"] = "none"
    # When loading backbone from keep_ckpt:
    #   - "encoder_only": load token_embed + bart.encoder (safe default)
    #   - "full": load entire MusicBartBackbone (encoder+decoder+lm_head)
    keep_warm_start_backbone_scope: Literal["encoder_only", "full"] = "encoder_only"

    # ---- proxy evaluation (music prior sanity check) ----
    proxy_eval_pos_per_beat: int = 12
    proxy_eval_strong_period_beats: int = 2
    proxy_eval_use_cummax_onset: bool = True

    # ---- aggregate metrics on the final N training batches of Stage C ----
    # <= 0 means disabled
    final_window_num_batches: int = 50


@dataclass
class OTBRunnerConfig:
    enable: bool = False
    valid_bench_dir: str = ""
    test_bench_dir: str = ""
    batch_size: int = 128

    # 0 => 只在 epoch end eval；>0 => 每隔这么多 step eval 一次 valid
    eval_interval_steps: int = 0
    eval_at_epoch_end: bool = True

    # 评估时可只跑前 max_batches 个 batch（加速）
    max_valid_batches: Optional[int] = None
    max_test_batches: Optional[int] = None

    # 指标配置（分桶、cut_rhos 等）
    metrics: OrnamentToBackboneEvalConfig = field(default_factory=OrnamentToBackboneEvalConfig)


@dataclass
class GTTMRunnerConfig:
    enable: bool = False
    test_bench_dir: str = ""
    batch_size: int = 32
    # eval_rho is passed directly to GTTMBackboneEvalConfig; kept here as a convenience alias
    eval_rho: float = 2.0 / 3.0
    metrics: GTTMBackboneEvalConfig = field(default_factory=GTTMBackboneEvalConfig)


@dataclass
class TrainSkeletonE2EConfig:
    # data
    pretrain_npy: str = ""
    lm_npy: str = ""
    skeleton_npy: str = ""
    vocab_pkl: str = ""

    # runtime
    save_dir: str = "./ckpt_skeleton_e2e"
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 1234

    # wandb
    use_wandb: bool = False
    wandb_project: str = "music-skeleton-e2e"
    wandb_run_name: Optional[str] = None

    # model + stages
    model: ModelHyperConfig = field(default_factory=ModelHyperConfig)
    stageA: StageASeq2SeqPretrainConfig = field(default_factory=StageASeq2SeqPretrainConfig)
    stageB: StageBLMPriorTrainConfig = field(default_factory=StageBLMPriorTrainConfig)
    stageC_train: StageCSkeletonTrainConfig = field(default_factory=StageCSkeletonTrainConfig)
    eval_otb: OTBRunnerConfig = field(default_factory=OTBRunnerConfig)
    eval_tavern: OTBRunnerConfig = field(default_factory=OTBRunnerConfig)
    eval_jiugong: OTBRunnerConfig = field(default_factory=OTBRunnerConfig)
    eval_gttm: GTTMRunnerConfig = field(default_factory=GTTMRunnerConfig)

    # skeleton model config（核心超参都在这里）
    skeleton_model: MusicSkeletonIIIConfig = field(default_factory=MusicSkeletonIIIConfig)

    # ---- optional external ckpt overrides (skip training) ----
    pretrain_ckpt: Optional[str] = None  # stageA: must contain "backbone_state_dict"
    lm_ckpt: Optional[str] = None        # stageB: must contain "backbone_state_dict"


# -------------------------
# Stage A: Seq2Seq Denoising Pretrain
# -------------------------
def run_stageA_pretrain(cfg: TrainSkeletonE2EConfig, device: torch.device, music_bart_cfg: MusicBartConfig,
                       vocab, quant_tables: MusicQuantizationTables, global_step: int) -> tuple[str, int]:
    if (not cfg.stageA.enable) or (cfg.stageA.epochs <= 0) or (not cfg.pretrain_npy):
        print("[StageA] skipped.")
        return "", global_step

    print("\n========== Stage A: Seq2Seq Denoising Pretrain ==========")
    backbone = MusicBartBackbone(cfg=music_bart_cfg).to(device)
    set_dropout_p(backbone, cfg.stageA.dropout)

    model = MusicBartForSeq2SeqLM(backbone=backbone).to(device)

    denoiser = BartStyleDenoiser(config=BartDenoiseConfig(
        mask_token_id=vocab.mask_id,
        pad_token_id=vocab.pad_id,
    ))
    base_denoise_config = replace(denoiser.config)

    loader = make_dataloader(
        cfg.pretrain_npy,
        batch_size=cfg.stageA.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=True,
        augment_seed=cfg.seed,
        augment_config=MusicAugmentConfig(
            transpose_sigma=6.0,
            transpose_min=-12,
            transpose_max=12,
            p_time_scale_2x=0.30,
            p_time_scale_half=0.30,
            quantization_tables=quant_tables,
        ),
    )

    optim = AdamW(model.parameters(), lr=cfg.stageA.lr, weight_decay=cfg.stageA.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(cfg.stageA.amp and device.type == "cuda"))

    steps_per_epoch = max(1, len(loader))
    total_steps = cfg.stageA.epochs * len(loader)
    warmup_steps = int(cfg.stageA.warmup_ratio * total_steps)
    base_lrs = [pg["lr"] for pg in optim.param_groups]
    step = 0

    ckpt_dir = os.path.join(cfg.save_dir, "stageA_pretrain")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Step-level timer (no-op when enable_timers=False)
    timer = StepTimer(enabled=cfg.stageA.enable_timers, device=device)

    # Step-level / every-n-steps noise schedule
    noise_update_every = max(1, int(cfg.stageA.noise_curriculum_update_every_steps))
    next_noise_update_step = 0

    masking_factor = 1.0
    deletion_factor = 1.0
    rotation_factor = 1.0

    def _refresh_denoiser_noise(cur_step: int) -> None:
        nonlocal masking_factor, deletion_factor, rotation_factor

        if cfg.stageA.use_noise_curriculum:
            masking_factor = warmup_corrupt_factor_stepwise_from_epoch_anchors(
                step_0based=cur_step,
                steps_per_epoch=steps_per_epoch,
                begin_epoch=cfg.stageA.masking_begin_epoch,
                warmup_epochs=cfg.stageA.masking_warmup_epochs,
                init_factor=cfg.stageA.masking_init_factor,
            )
            deletion_factor = warmup_corrupt_factor_stepwise_from_epoch_anchors(
                step_0based=cur_step,
                steps_per_epoch=steps_per_epoch,
                begin_epoch=cfg.stageA.deletion_begin_epoch,
                warmup_epochs=cfg.stageA.deletion_warmup_epochs,
                init_factor=cfg.stageA.deletion_init_factor,
            )
            rotation_factor = warmup_corrupt_factor_stepwise_from_epoch_anchors(
                step_0based=cur_step,
                steps_per_epoch=steps_per_epoch,
                begin_epoch=cfg.stageA.rotation_begin_epoch,
                warmup_epochs=cfg.stageA.rotation_warmup_epochs,
                init_factor=cfg.stageA.rotation_init_factor,
            )
        else:
            masking_factor = 1.0
            deletion_factor = 1.0
            rotation_factor = 1.0

        denoiser.set_runtime_noise(
            masking_noise_density=masking_factor * base_denoise_config.masking_noise_density,
            deletion_prob=deletion_factor * base_denoise_config.deletion_prob,
            rotation_prob=rotation_factor * base_denoise_config.rotation_prob,
        )

    _refresh_denoiser_noise(0)

    for epoch in range(1, cfg.stageA.epochs + 1):
        model.train()
        running = 0.0
        n = 0

        # --- noise curriculum (epoch-based) ---
        masking_factor = 1.0
        deletion_factor = 1.0
        rotation_factor = 1.0

        pbar = tqdm(loader, desc=f"[StageA] epoch {epoch}/{cfg.stageA.epochs}", dynamic_ncols=True, smoothing=0.0)
        interval_loss = 0.0
        interval_steps = 0

        data_iter = iter(pbar)
        while True:
            # ------------------------------------------------------------
            # [1] Data loading
            # ------------------------------------------------------------
            timer.start("step_total")
            timer.start("data_load")
            try:
                tgt_tokens_cpu = next(data_iter)
            except StopIteration:
                break
            timer.stop("data_load")
            # ------------------------------------------------------------
            # [2] Host -> Device
            # ------------------------------------------------------------
            with timer.measure("to_device"):
                tgt_tokens = tgt_tokens_cpu.to(device, non_blocking=True)
                tgt_mask = (tgt_tokens[..., 0] != vocab.pad_id)
            if cfg.stageA.use_lr_schedule:
                factor = warmup_cosine_lr_factor(
                    step, total_steps, warmup_steps,
                    cfg.stageA.min_lr_ratio, cfg.stageA.min_lr_ratio
                )
                apply_lr_factor(optim, base_lrs, factor)
            # step-wise / every-n-steps noise curriculum update
            if step >= next_noise_update_step:
                _refresh_denoiser_noise(step)
                next_noise_update_step = step + noise_update_every
            # ------------------------------------------------------------
            # [3] Corrupt (denoiser)
            # ------------------------------------------------------------
            with timer.measure("corrupt"):
                src_tokens, src_mask = denoiser.corrupt(
                    input_tokens=tgt_tokens,
                    attention_mask=tgt_mask,
                    apply_masking=True,
                    apply_deletion=True,
                    apply_rotation=True,
                )
            optim.zero_grad(set_to_none=True)
            # ------------------------------------------------------------
            # [4] Forward
            # ------------------------------------------------------------
            with torch.amp.autocast(device_type=device.type, enabled=(cfg.stageA.amp and device.type == "cuda")):
                with timer.measure("forward"):
                    out = model(
                        src_tokens=src_tokens,
                        src_attention_mask=src_mask,
                        tgt_tokens=tgt_tokens,
                        tgt_attention_mask=tgt_mask,
                        attr_weights=(1.0, 1.0, 1.0),
                        ignore_index=-100,
                    )
                    loss = out.loss
            # ------------------------------------------------------------
            # [5] Backward
            # ------------------------------------------------------------
            with timer.measure("backward"):
                scaler.scale(loss).backward()
            # ------------------------------------------------------------
            # [6] Optimizer step
            # ------------------------------------------------------------
            with timer.measure("optim_step"):
                if cfg.stageA.grad_clip and cfg.stageA.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.stageA.grad_clip)
                scaler.step(optim)
                scaler.update()
            timer.stop("step_total")

            bsz = tgt_tokens.size(0)
            running += float(loss.item()) * bsz
            n += bsz
            step += 1
            global_step += 1

            interval_loss += float(loss.item())
            interval_steps += 1
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

            if cfg.use_wandb and wandb is not None and interval_steps >= cfg.stageA.log_interval:
                log_dict = {
                    "stage": "A",
                    "A/batch_loss": interval_loss / interval_steps,
                    "A/lr": optim.param_groups[0]["lr"],
                    "A/epoch": epoch,
                    "A/step": step,

                    "A/denoise/masking_noise_density": float(denoiser.config.masking_noise_density),
                    "A/denoise/deletion_prob": float(denoiser.config.deletion_prob),
                    "A/denoise/rotation_prob": float(denoiser.config.rotation_prob),
                    "A/denoise/masking_factor": float(masking_factor),
                    "A/denoise/deletion_factor": float(deletion_factor),
                    "A/denoise/rotation_factor": float(rotation_factor),

                    "global_step": global_step,
                }

                if cfg.stageA.enable_timers:
                    log_dict.update(timer.wandb_log_dict(prefix="A_timers/"))

                wandb.log(log_dict, step=global_step)
                interval_loss = 0.0
                interval_steps = 0

        epoch_loss = running / max(1, n)
        print(f"[StageA] epoch {epoch} loss={epoch_loss:.6f}")

        if cfg.use_wandb and wandb is not None:
            log_dict = {
                "stage": "A",
                "A/epoch_loss": epoch_loss,
                "A/epoch": epoch,
                "A/lr": optim.param_groups[0]["lr"],

                # ---- denoise params (actual, after curriculum) ----
                "A/denoise/masking_noise_density": float(denoiser.config.masking_noise_density),
                "A/denoise/deletion_prob": float(denoiser.config.deletion_prob),
                "A/denoise/rotation_prob": float(denoiser.config.rotation_prob),

                # ---- optional: curriculum factors (debug) ----
                "A/denoise/masking_factor": float(masking_factor),
                "A/denoise/deletion_factor": float(deletion_factor),
                "A/denoise/rotation_factor": float(rotation_factor),

                "global_step": global_step,
            }

            if cfg.stageA.enable_timers:
                log_dict.update(timer.wandb_log_dict(prefix="A_timers/"))

            wandb.log(log_dict, step=global_step)

        # 只在指定的 epoch 保存 checkpoint
        if epoch in cfg.stageA.save_epochs:
            save_ckpt(os.path.join(ckpt_dir, f"pretrain_epoch{epoch}.pt"), {
                "stage": "A",
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),  # backbone 内含 lm_head
                "optimizer_state_dict": optim.state_dict(),
                "config": asdict(cfg),
                "vocab_pkl": cfg.vocab_pkl,
                "epoch_loss": epoch_loss,
            })

    last_path = os.path.join(ckpt_dir, "backbone_pretrained_last.pt")
    save_ckpt(last_path, {
        "stage": "A",
        "backbone_state_dict": backbone.state_dict(),
        "config": asdict(cfg),
        "vocab_pkl": cfg.vocab_pkl,
    })
    return last_path, global_step


# -------------------------
# Stage B: Train LM prior (decoder-only)
# -------------------------
def run_stageB_lm(cfg: TrainSkeletonE2EConfig, device: torch.device, music_bart_cfg: MusicBartConfig,
                  vocab, backbone_init_ckpt: str, quant_tables: MusicQuantizationTables, global_step: int) -> tuple[str, int]:
    if (not cfg.stageB.enable) or (cfg.stageB.epochs <= 0) or (not cfg.lm_npy):
        print("[StageB] skipped.")
        return "", global_step

    if not backbone_init_ckpt:
        raise ValueError("StageB needs backbone_init_ckpt from StageA (or you provide a ckpt path).")

    print("\n========== Stage B: Train decoder-only LM prior ==========")
    ckpt = torch.load(backbone_init_ckpt, map_location="cpu")

    backbone = MusicBartBackbone(cfg=music_bart_cfg).to(device)
    backbone.load_state_dict(ckpt["backbone_state_dict"], strict=True)
    set_dropout_p(backbone, cfg.stageB.dropout)

    lm = MusicBartDecoderOnlyLM(backbone=backbone, freeze=False).to(device)

    loader = make_dataloader(
        cfg.lm_npy,
        batch_size=cfg.stageB.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=True,  # 你也可以关掉
        augment_seed=cfg.seed,
        augment_config=MusicAugmentConfig(
            transpose_sigma = 6.0,
            transpose_min = -12,
            transpose_max = 12,
            p_time_scale_2x = 0.50,
            quantization_tables=quant_tables
        ),
    )

    optim = AdamW(lm.parameters(), lr=cfg.stageB.lr, weight_decay=cfg.stageB.weight_decay, fused=(device.type == "cuda"))
    scaler = torch.amp.GradScaler(enabled=(cfg.stageB.amp and device.type == "cuda"))

    total_steps = cfg.stageB.epochs * len(loader)
    warmup_steps = int(cfg.stageB.warmup_ratio * total_steps)
    base_lrs = [pg["lr"] for pg in optim.param_groups]
    step = 0

    ckpt_dir = os.path.join(cfg.save_dir, "stageB_lm")
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, cfg.stageB.epochs + 1):
        lm.train()
        running = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"[StageB] epoch {epoch}/{cfg.stageB.epochs}", dynamic_ncols=True, smoothing=0.0)
        interval_loss = 0.0
        interval_steps = 0

        for tokens in pbar:
            tokens = tokens.to(device, non_blocking=True)
            attn = infer_attention_mask_from_tokens(tokens, pad_id=vocab.pad_id)

            if cfg.stageB.use_lr_schedule:
                factor = warmup_cosine_lr_factor(step, total_steps, warmup_steps, cfg.stageB.min_lr_ratio, cfg.stageB.min_lr_ratio)
                apply_lr_factor(optim, base_lrs, factor)

            optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(cfg.stageB.amp and device.type == "cuda")):
                logits = lm.logits(tokens=tokens, attention_mask=attn)
                labels = mask_labels_with_ignore_index(tokens.clone(), attn, ignore_index=-100)

                loss, _ = multi_attribute_ce_loss(
                    labels=labels,
                    pitch_logits=logits.pitch,
                    duration_logits=logits.duration,
                    dt_logits=logits.dt,
                    attr_weights=cfg.stageB.attr_weights,
                    ignore_index=-100,
                    reduction="mean",
                )

            scaler.scale(loss).backward()
            if cfg.stageB.grad_clip and cfg.stageB.grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(lm.parameters(), cfg.stageB.grad_clip)
            scaler.step(optim)
            scaler.update()

            bsz = tokens.size(0)
            running += float(loss.item()) * bsz
            n += bsz
            step += 1
            global_step += 1

            interval_loss += float(loss.item())
            interval_steps += 1
            pbar.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

            if cfg.use_wandb and wandb is not None and interval_steps >= cfg.stageB.log_interval:
                wandb.log({
                    "stage": "B",
                    "B/batch_loss": interval_loss / interval_steps,
                    "B/lr": optim.param_groups[0]["lr"],
                    "B/epoch": epoch,
                    "B/step": step,
                    "global_step": global_step,
                }, step=global_step)
                interval_loss = 0.0
                interval_steps = 0

        epoch_loss = running / max(1, n)
        print(f"[StageB] epoch {epoch} loss={epoch_loss:.6f}")

        if cfg.use_wandb and wandb is not None:
            wandb.log({
                "stage": "B",
                "B/epoch_loss": epoch_loss,
                "B/epoch": epoch,
                "B/lr": optim.param_groups[0]["lr"],
                "global_step": global_step,
            }, step=global_step)

        # 只在指定的 epoch 保存 checkpoint
        if epoch in cfg.stageB.save_epochs:
            save_ckpt(os.path.join(ckpt_dir, f"lm_epoch{epoch}.pt"), {
                "stage": "B",
                "epoch": epoch,
                "backbone_state_dict": backbone.state_dict(),  # 训练后的 LM prior backbone（含 lm_head）
                "optimizer_state_dict": optim.state_dict(),
                "config": asdict(cfg),
                "vocab_pkl": cfg.vocab_pkl,
                "epoch_loss": epoch_loss,
            })

    last_path = os.path.join(ckpt_dir, "lm_prior_backbone_last.pt")
    save_ckpt(last_path, {
        "stage": "B",
        "backbone_state_dict": backbone.state_dict(),
        "config": asdict(cfg),
        "vocab_pkl": cfg.vocab_pkl,
    })
    return last_path, global_step


# -------------------------
# Stage C: Train Skeleton model
# -------------------------
def run_stageC_skeleton(cfg: TrainSkeletonE2EConfig, device: torch.device, music_bart_cfg: MusicBartConfig,
                        vocab, backbone_init_ckpt: str, lm_prior_ckpt: str,
                        quant_tables: MusicQuantizationTables, global_step: int) -> tuple[str, int]:
    if (not cfg.stageC_train.enable) or (cfg.stageC_train.epochs <= 0) or (not cfg.skeleton_npy):
        print("[StageC] skipped.")
        return "", global_step

    use_pretrain_init = (not cfg.stageC_train.wo_bart_pretrain_init)
    if use_pretrain_init and (not backbone_init_ckpt):
        raise ValueError("StageC needs backbone_init_ckpt from StageA (or you provide a ckpt path).")

    # ---- sanity check: dynamic_rho + rho curriculum conflict ----
    # When dynamic_rho=True the model predicts rho per-sequence; the training loop
    # therefore passes rho_arg=None to model.forward(), so cur_rho (computed from
    # the rho curriculum) is never forwarded to the model and has no effect.
    sk_cfg_pre = cfg.skeleton_model
    if sk_cfg_pre.dynamic_rho:
        rho_curriculum_nontrivial = (
            abs(cfg.stageC_train.rho_start - sk_cfg_pre.rho) > 1e-6
            and cfg.stageC_train.rho_warmup_ratio > 0
        )
        if rho_curriculum_nontrivial:
            warnings.warn(
                f"[StageC] dynamic_rho=True but a non-trivial rho curriculum is also configured "
                f"(rho_start={cfg.stageC_train.rho_start}, rho_final(=sk_cfg.rho)={sk_cfg_pre.rho}, "
                f"rho_warmup_ratio={cfg.stageC_train.rho_warmup_ratio}). "
                "With dynamic_rho=True, cur_rho is computed each step but rho_arg=None is always "
                "passed to model.forward(), so the rho curriculum is dead code and has no effect. "
                "To suppress this warning, either set dynamic_rho=False or align rho_start with sk_cfg.rho.",
                stacklevel=2,
            )

    print("\n========== Stage C: Train MusicSkeletonModelIII ==========")

    def _pick_pointer_soft(obj):
        mode = getattr(cfg.skeleton_model, "pointer_soft_for_aux_losses", "train")
        if mode == "strict" and getattr(obj, "pointer_soft_strict", None) is not None:
            return obj.pointer_soft_strict
        return obj.pointer_soft
    
    def _importance_for_consistency(obj, *, tau: float, exclude_last_step: bool) -> torch.Tensor:
        """
        对 encoder_topk：用 score_logits -> s
        对 pointer_decoder：fallback 到 mean(pointer_soft)
        """
        score_logits = getattr(obj, "score_logits", None)
        score_mask = getattr(obj, "score_mask", None)
        if score_logits is not None and score_mask is not None:
            return importance_from_score_logits(score_logits, score_mask, temperature=tau)

        # fallback: old behavior
        ptr = _pick_pointer_soft(obj).to(torch.float32)
        z_mask = getattr(obj, "z_mask", None)
        if z_mask is None:
            raise ValueError("obj has no z_mask; cannot compute pointer-soft marginal importance.")
        return marginal_importance_from_pointer_soft(
            ptr,
            z_mask,
            exclude_last_step=exclude_last_step,
        )

    # ---- trainable backbone (for compressor+reconstructor) ----
    backbone = MusicBartBackbone(cfg=music_bart_cfg).to(device)

    if use_pretrain_init:
        ckptA = torch.load(backbone_init_ckpt, map_location="cpu")
        backbone.load_state_dict(ckptA["backbone_state_dict"], strict=True)
        print(f"[StageC] Backbone init: load StageA pretrained weights from: {backbone_init_ckpt}")
    else:
        print("[StageC] Backbone init: RANDOM (w/o BART pretrain init)")

    # ------------------------------------------------------------
    # NEW: warm-start from O2B-Learner (keep baseline) ckpt
    # ------------------------------------------------------------
    def _extract_prefixed(sd: dict, prefix: str) -> dict:
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
        return out

    keep_sd = None
    warm_mode = getattr(cfg.stageC_train, "keep_warm_start", "none")
    # backward compat: old flag now means "backbone" warm-start by default
    if warm_mode == "none" and getattr(cfg.stageC_train, "use_keep_score_ckpt", False):
        warm_mode = "backbone"

    if warm_mode != "none":
        keep_path = cfg.stageC_train.keep_ckpt
        if keep_path is None or str(keep_path).strip() == "":
            raise ValueError(f"[StageC] keep_warm_start={warm_mode} but keep_ckpt is empty.")
        print(f"[Warm-start] mode={warm_mode} ckpt={keep_path}")

        keep_ckpt_obj = torch.load(keep_path, map_location="cpu")
        # keep baseline ckpt format: {"model_state_dict": ...}
        if isinstance(keep_ckpt_obj, dict) and "model_state_dict" in keep_ckpt_obj:
            keep_sd = keep_ckpt_obj["model_state_dict"]
        elif isinstance(keep_ckpt_obj, dict):
            # fallback: assume it's already a state_dict
            keep_sd = keep_ckpt_obj
        else:
            raise ValueError(f"[Warm-start] Unrecognized keep_ckpt format: {type(keep_ckpt_obj)}")

        if warm_mode in ("backbone", "both"):
            bb_sd = _extract_prefixed(keep_sd, "backbone.")
            if not bb_sd:
                raise ValueError(
                    "[Warm-start] Cannot find 'backbone.*' keys in keep_ckpt['model_state_dict']. "
                    "Make sure the ckpt comes from train_skeleton_keep_baseline.py."
                )

            scope = getattr(cfg.stageC_train, "keep_warm_start_backbone_scope", "encoder_only")
            scope = str(scope)

            if scope == "full":
                inc = backbone.load_state_dict(bb_sd, strict=True)
                print(f"[Warm-start backbone/full] missing={inc.missing_keys} unexpected={inc.unexpected_keys}")

            elif scope == "encoder_only":
                te_sd  = _extract_prefixed(bb_sd, "token_embed.")
                enc_sd = _extract_prefixed(bb_sd, "bart.encoder.")

                inc_te  = backbone.token_embed.load_state_dict(te_sd, strict=True)
                inc_enc = backbone.bart.encoder.load_state_dict(enc_sd, strict=True)

                print(f"[Warm-start backbone/encoder_only] token_embed missing={inc_te.missing_keys} unexpected={inc_te.unexpected_keys}")
                print(f"[Warm-start backbone/encoder_only] encoder     missing={inc_enc.missing_keys} unexpected={inc_enc.unexpected_keys}")

            else:
                raise ValueError(f"[Warm-start] Unknown keep_warm_start_backbone_scope: {scope}")

    # 现在再设置 StageC dropout（state_dict 不包含 dropout，所以放前放后都行；放这里更直观）
    set_dropout_p(backbone, cfg.stageC_train.dropout)

    # after set_dropout_p(backbone, cfg.stageC_train.dropout), 确认确实改到了 HF Bart 的 float dropout
    enc0 = backbone.bart.encoder.layers[0]
    att0 = enc0.self_attn
    print("[Dropout sanity]",
        "enc0.dropout=", enc0.dropout,
        "enc0.activation_dropout=", getattr(enc0, "activation_dropout", None),
        "attn.dropout=", att0.dropout)

    # ---- frozen LM prior backbone ----
    lm_prior = None
    if cfg.skeleton_model.lambda_P != 0.0:
        if not lm_prior_ckpt:
            raise ValueError("lambda_P != 0 but lm_prior_ckpt is empty. Train StageB or set lambda_P=0.")
        ckptB = torch.load(lm_prior_ckpt, map_location="cpu")
        lm_backbone = MusicBartBackbone(cfg=music_bart_cfg).to(device)
        lm_backbone.load_state_dict(ckptB["backbone_state_dict"], strict=True)
        lm_backbone.eval()
        for p in lm_backbone.parameters():
            p.requires_grad = False
        lm_prior = MusicBartDecoderOnlyLM(backbone=lm_backbone, freeze=True).to(device)

    # ---- skeleton model ----
    sk_cfg = cfg.skeleton_model
    model = MusicSkeletonModelIII(
        backbone=backbone,
        quant=quant_tables,
        cfg=sk_cfg,
        lm_prior=lm_prior,
        use_bias_in_lm_head=True,
    ).to(device)

    # if device.type == "cuda":
    #     model.backbone.bart.encoder = torch.compile(
    #         model.backbone.bart.encoder, mode="max-autotune", fullgraph=False
    #     )
    #     model.backbone.bart.decoder = torch.compile(
    #         model.backbone.bart.decoder, mode="max-autotune", fullgraph=False
    #     )

    # ------------------------------------------------------------
    # Optional: warm-start the scorer head (legacy or "both")
    # ------------------------------------------------------------
    if warm_mode in ("score_head", "both"):
        assert keep_sd is not None, "keep_sd should have been loaded when warm_mode != 'none'."

        keep_head_sd = {
            k[len("keep_head."):]: v
            for k, v in keep_sd.items()
            if k.startswith("keep_head.")
        }
        if not keep_head_sd:
            raise ValueError("[Warm-start score_head] Cannot find 'keep_head.*' keys in keep ckpt.")

        inc = model.compressor.keep_head.load_state_dict(keep_head_sd, strict=True)
        print(f"[Warm-start score_head] missing={inc.missing_keys} unexpected={inc.unexpected_keys}")

    # 输出 model 的参数数量
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_bb_train = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    print(n_train, n_bb_train)

    otb_adapter = build_pointer_adapter_from_model(model)

    print(f"[StageC] params(total)={count_params(model):,} | params(backbone)={count_params(backbone):,}")

    loader = make_dataloader(
        cfg.skeleton_npy,
        batch_size=cfg.stageC_train.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=True,
        augment_seed=cfg.seed,
        augment_config=MusicAugmentConfig(quantization_tables=quant_tables),
        max_samples=(cfg.stageC_train.limit_batches * cfg.stageC_train.batch_size
                     if cfg.stageC_train.limit_batches is not None else None),
    )

    # -------------------------
    # OTB benchmark loaders (valid/test)
    # -------------------------
    otb_runner = cfg.eval_otb
    otb_valid_loader = None
    otb_test_loader = None

    if otb_runner.enable:
        if otb_runner.valid_bench_dir and os.path.isdir(otb_runner.valid_bench_dir):
            otb_valid_loader = make_ornament_benchmark_dataloader(
                otb_runner.valid_bench_dir,
                batch_size=otb_runner.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=cfg.pin_memory,
            )
            print(f"[OTB] valid loader: {otb_runner.valid_bench_dir}")
        if otb_runner.test_bench_dir and os.path.isdir(otb_runner.test_bench_dir):
            otb_test_loader = make_ornament_benchmark_dataloader(
                otb_runner.test_bench_dir,
                batch_size=otb_runner.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=cfg.pin_memory,
            )
            print(f"[OTB] test loader: {otb_runner.test_bench_dir}")

    def _run_bench_eval(bench_name: str, split: str, runner: OTBRunnerConfig, loader_eval, show_progress=False):
        if loader_eval is None:
            return None

        metric_cfg = replace(
            runner.metrics,
            max_batches=(runner.max_valid_batches if split == "valid" else runner.max_test_batches),
            show_progress=show_progress,
            amp=(device.type == "cuda"),
        )
        prefix = f"C_{split}/{bench_name}/"

        metrics = evaluate_ornament_to_backbone(
            otb_adapter,
            loader_eval,
            device=device,
            cfg=metric_cfg,
            prefix=prefix,
        )

        # stdout 摘要
        key_f1 = prefix + "f1_hard"
        key_f1_topk = prefix + "f1_hard_topk"
        key_gap = prefix + "gap_f1"
        key_ap = prefix + "ap_soft"
        key_rho = prefix + "rho"
        print(
            f"[{bench_name.upper()}-{split}] "
            f"f1_hard={metrics.get(key_f1, float('nan')):.4f} "
            f"f1_topk={metrics.get(key_f1_topk, float('nan')):.4f} "
            f"gap_f1={metrics.get(key_gap, float('nan')):.4f} "
            f"ap_soft={metrics.get(key_ap, float('nan')):.4f} "
            f"rho_mean={metrics.get(key_rho, float('nan')):.4f}"
        )

        if cfg.use_wandb and wandb is not None:
            metrics.update({
                "stage": "C",
                "C/epoch": epoch,
                "C/step": step,
                "global_step": global_step,
            })
            wandb.log(metrics, step=global_step)

        return metrics
    
    # -------------------------
    # TAVERN benchmark loader (test only)
    # -------------------------
    tavern_runner = cfg.eval_tavern
    tavern_test_loader = None

    if tavern_runner.enable:
        if tavern_runner.test_bench_dir and os.path.isdir(tavern_runner.test_bench_dir):
            tavern_test_loader = make_ornament_benchmark_dataloader(
                tavern_runner.test_bench_dir,
                batch_size=tavern_runner.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=cfg.pin_memory,
            )
            print(f"[TAVERN] test loader: {tavern_runner.test_bench_dir}")

    # -------------------------
    # Jiugong benchmark loader (test only)
    # -------------------------
    jiugong_runner = cfg.eval_jiugong
    jiugong_test_loader = None

    if jiugong_runner.enable:
        if jiugong_runner.test_bench_dir and os.path.isdir(jiugong_runner.test_bench_dir):
            jiugong_test_loader = make_ornament_benchmark_dataloader(
                jiugong_runner.test_bench_dir,
                batch_size=jiugong_runner.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=cfg.pin_memory,
            )
            print(f"[Jiugong] test loader: {jiugong_runner.test_bench_dir}")

    # -------------------------

    ornamenter = None
    orn_rng = None
    if cfg.stageC_train.use_ornament_invariance:
        pad_id_local = int(vocab.global2local_pitch[int(vocab.pad_id)])
        bos_id_local = int(vocab.global2local_pitch[int(vocab.bos_id)])
        eos_id_local = int(vocab.global2local_pitch[int(vocab.eos_id)])
        orn_cfg = MusicOrnamentConfig(
            enable=True,
            p_apply=float(cfg.stageC_train.ornament_p_apply),
            pad_id=pad_id_local,
            bos_id=bos_id_local,
            eos_id=eos_id_local,
            quantization_tables=quant_tables,
        )
        ornamenter = MusicOrnamenter(orn_cfg)
        orn_rng = np.random.default_rng(cfg.seed + 260312)  # 随便选个常数即可

    def build_ornament_batch(x_cpu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_cpu: [B,L,3] on CPU
        return:
          x_strong_cpu: [B,L,3]
          pi_cpu: [B,L] (int32->torch.long)
        """
        assert ornamenter is not None and orn_rng is not None
        x_cpu = x_cpu.contiguous()
        x_np = x_cpu.numpy()  # [B,L,3]
        B, L, A = x_np.shape
        x_aug = np.empty_like(x_np)
        pi = np.empty((B, L), dtype=np.int32)
        for b in range(B):
            xa, pib = ornamenter.augment(x_np[b], rng=orn_rng)
            x_aug[b] = xa
            pi[b] = pib
        return torch.from_numpy(x_aug).long(), torch.from_numpy(pi).long()

    # param groups: backbone vs pointer head
    pointer_params = []
    backbone_params = []
    for n0, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (
            n0.startswith("compressor.pointer.")
            or n0.startswith("compressor.keep_head.")
        ):
            pointer_params.append(p)
        else:
            backbone_params.append(p)

    optim = AdamW(
        [
            {"params": backbone_params, "lr": cfg.stageC_train.lr_backbone, "weight_decay": cfg.stageC_train.weight_decay},
            {"params": pointer_params, "lr": cfg.stageC_train.lr_pointer, "weight_decay": cfg.stageC_train.weight_decay},
        ]
    )
    scaler = torch.amp.GradScaler(enabled=(cfg.stageC_train.amp and device.type == "cuda"))

    total_steps = cfg.stageC_train.epochs * len(loader)
    warmup_steps = int(cfg.stageC_train.warmup_ratio * total_steps)
    base_lrs = [pg["lr"] for pg in optim.param_groups]
    step = 0

    final_window_num_batches = max(0, int(cfg.stageC_train.final_window_num_batches))
    final_window_start_step = (
        max(0, total_steps - final_window_num_batches)
        if final_window_num_batches > 0
        else total_steps + 1
    )
    final_window_tracker = ScalarMeanTracker()

    ckpt_dir = os.path.join(cfg.save_dir, "stageC_skeleton")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Step-level timer (no-op when enable_timers=False)
    timer = StepTimer(enabled=cfg.stageC_train.enable_timers, device=device)

    # curricula targets
    rho_start = float(cfg.stageC_train.rho_start)
    tau_start = float(cfg.stageC_train.tau_start)
    lambda_P_start = float(cfg.stageC_train.lambda_P_start)
    lambda_GA_start = float(cfg.stageC_train.lambda_GA_start)
    lambda_L_start = float(cfg.stageC_train.lambda_L_start)
    lambda_sharp_start = float(cfg.stageC_train.lambda_sharp_start)
    lambda_cons_start = float(cfg.stageC_train.lambda_cons_start)
    lambda_ins_start = float(cfg.stageC_train.lambda_ins_start)
    rho_final = float(sk_cfg.rho)
    tau_final = float(cfg.stageC_train.tau_final)
    lambda_P_final = float(cfg.stageC_train.lambda_P_final)
    lambda_GA_final = float(cfg.stageC_train.lambda_GA_final)
    lambda_L_final = float(cfg.stageC_train.lambda_L_final)
    lambda_sharp_final = float(cfg.stageC_train.lambda_sharp_final)
    lambda_cons_final = float(cfg.stageC_train.lambda_cons_final)
    lambda_ins_final = float(cfg.stageC_train.lambda_ins_final)
    rho_warmup_steps = int(cfg.stageC_train.rho_warmup_ratio * total_steps)
    tau_warmup_steps = int(cfg.stageC_train.tau_warmup_ratio * total_steps)
    lambda_P_warmup_steps = int(cfg.stageC_train.lambda_P_warmup_ratio * total_steps)
    lambda_GA_warmup_steps = int(cfg.stageC_train.lambda_GA_warmup_ratio * total_steps)
    lambda_L_warmup_steps = int(cfg.stageC_train.lambda_L_warmup_ratio * total_steps)
    lambda_sharp_warmup_steps = int(cfg.stageC_train.lambda_sharp_warmup_ratio * total_steps)
    lambda_cons_warmup_steps = int(cfg.stageC_train.lambda_cons_warmup_ratio * total_steps)
    lambda_ins_warmup_steps = int(cfg.stageC_train.lambda_ins_warmup_ratio * total_steps)
    ss_start = float(cfg.stageC_train.ss_prob_start)
    ss_final = float(cfg.stageC_train.ss_prob_final)
    ss_warmup_steps = int(cfg.stageC_train.ss_prob_warmup_ratio * total_steps)
    recon_input_mask_prob_start = float(cfg.stageC_train.recon_input_mask_prob_start)
    recon_input_mask_prob_final = float(cfg.stageC_train.recon_input_mask_prob_final)
    recon_input_mask_prob_warmup_steps = int(cfg.stageC_train.recon_input_mask_prob_warmup_ratio * total_steps)

    ema_compressor = None
    ema_param_pairs = None
    ema_buffer_pairs = None

    if cfg.stageC_train.use_ornament_invariance and cfg.stageC_train.ornament_teacher_mode == "ema":
        ema_compressor = copy.deepcopy(model.compressor).to(device)
        ema_compressor.eval()
        for p in ema_compressor.parameters():
            p.requires_grad_(False)

        ema_param_pairs = list(zip(ema_compressor.parameters(), model.compressor.parameters()))
        ema_buffer_pairs = list(zip(ema_compressor.buffers(), model.compressor.buffers()))

    for epoch in range(1, cfg.stageC_train.epochs + 1):
        model.train()

        # optional: freeze backbone by lr=0 (epoch-level)
        freeze_backbone = (epoch <= cfg.stageC_train.freeze_backbone_epochs)

        # running_loss = torch.zeros((), device=device, dtype=torch.float32)
        # running_recon = torch.zeros((), device=device, dtype=torch.float32)
        # running_prior = torch.zeros((), device=device, dtype=torch.float32)
        # running_guided = torch.zeros((), device=device, dtype=torch.float32)
        # running_len = torch.zeros((), device=device, dtype=torch.float32)
        # running_sharp = torch.zeros((), device=device, dtype=torch.float32)
        # running_zlen = torch.zeros((), device=device, dtype=torch.float32)
        # running_cons = torch.zeros((), device=device, dtype=torch.float32)
        # running_ins = torch.zeros((), device=device, dtype=torch.float32)
        # n = 0

        pbar = tqdm(loader, desc=f"[StageC] epoch {epoch}/{cfg.stageC_train.epochs}", dynamic_ncols=True, smoothing=0.0)
        interval_loss = torch.zeros((), device=device, dtype=torch.float32)
        interval_recon = torch.zeros((), device=device, dtype=torch.float32)
        interval_prior = torch.zeros((), device=device, dtype=torch.float32)
        interval_guided = torch.zeros((), device=device, dtype=torch.float32)
        interval_len = torch.zeros((), device=device, dtype=torch.float32)
        interval_sharp = torch.zeros((), device=device, dtype=torch.float32)
        interval_zlen = torch.zeros((), device=device, dtype=torch.float32)
        interval_cons = torch.zeros((), device=device, dtype=torch.float32)
        interval_ins = torch.zeros((), device=device, dtype=torch.float32)
        interval_steps = 0

        data_iter = iter(pbar)
        while True:
            # ------------------------------------------------------------------
            # [1] Data loading  (includes DataLoader collate + augment on workers)
            # ------------------------------------------------------------------
            timer.start("step_total")   # outer envelope — stop after optim_step
            timer.start("data_load")
            try:
                x_tokens_cpu = next(data_iter)
            except StopIteration:
                break                   # orphaned timer.start() calls are harmless
            timer.stop("data_load")

            # ------------------------------------------------------------------
            # [2] Ornament batch (CPU numpy augmentation)
            # ------------------------------------------------------------------
            x_strong_cpu = None
            pi_cpu = None
            if ornamenter is not None:
                with timer.measure("ornament_batch"):
                    x_strong_cpu, pi_cpu = build_ornament_batch(x_tokens_cpu)

            # ------------------------------------------------------------------
            # [3] Host→device transfer  (PCIe bandwidth)
            # ------------------------------------------------------------------
            with timer.measure("to_device"):
                x_tokens = x_tokens_cpu.to(device, non_blocking=True)
                x_mask = (x_tokens[..., 0] != vocab.pad_id)

                if x_strong_cpu is not None:
                    x_strong = x_strong_cpu.to(device, non_blocking=True)
                    x_strong_mask = (x_strong[..., 0] != vocab.pad_id)
                    pi = pi_cpu.to(device, non_blocking=True)

            if cfg.stageC_train.use_lr_schedule:
                factor = warmup_cosine_lr_factor(step, total_steps, warmup_steps, cfg.stageC_train.final_lr_ratio, cfg.stageC_train.init_lr_ratio)
                apply_lr_factor(optim, base_lrs, factor)

            # curricula update (step-level)
            cur_rho = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=rho_warmup_steps,
                start=rho_start, end=rho_final
            )
            cur_tau = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=tau_warmup_steps,
                start=tau_start, end=tau_final
            )
            cur_lambda_P = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=lambda_P_warmup_steps,
                start=lambda_P_start, end=lambda_P_final
            )
            cur_lambda_GA = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=lambda_GA_warmup_steps,
                start=lambda_GA_start, end=lambda_GA_final
            )
            cur_lambda_L = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=lambda_L_warmup_steps,
                start=lambda_L_start, end=lambda_L_final
            )
            cur_lambda_sharp = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=lambda_sharp_warmup_steps,
                start=lambda_sharp_start, end=lambda_sharp_final
            )
            cur_lambda_cons = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=lambda_cons_warmup_steps,
                start=lambda_cons_start, end=lambda_cons_final
            )
            cur_lambda_ins = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=lambda_ins_warmup_steps,
                start=lambda_ins_start, end=lambda_ins_final
            )
            cur_ss_prob = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=ss_warmup_steps,
                start=ss_start, end=ss_final
            )
            cur_recon_input_mask_prob = linear_warmup_step(
                step_0based=step, begin_step=0, warmup_steps=recon_input_mask_prob_warmup_steps,
                start=recon_input_mask_prob_start, end=recon_input_mask_prob_final
            )
            model.cfg.tau = float(cur_tau)
            model.cfg.lambda_P = float(cur_lambda_P)
            model.cfg.lambda_GA = float(cur_lambda_GA)
            model.cfg.lambda_L = float(cur_lambda_L)
            model.cfg.lambda_sharp = float(cur_lambda_sharp)
            model.cfg.recon_ss_prob = float(cur_ss_prob)
            model.cfg.recon_input_mask_prob = float(cur_recon_input_mask_prob)

            if freeze_backbone:
                optim.param_groups[0]["lr"] = 0.0  # backbone group
            else:
                # restore scheduled lr for backbone
                pass

            optim.zero_grad(set_to_none=True)

            need_log = (
                cfg.use_wandb and wandb is not None
                and (interval_steps + 1) >= cfg.stageC_train.log_interval
            )
            in_final_window = (
                final_window_num_batches > 0
                and step >= final_window_start_step
            )

            diagnostics = None
            if need_log or in_final_window:
                diagnostics = {
                    "recon_wo_z": bool(need_log or in_final_window),
                    "z_start_time": bool(need_log),
                    # "pointer_soft_gap": bool(need_log),
                }

            comp_strong = None

            # ------------------------------------------------------------------
            # [4] Main forward pass  (compressor + reconstructor)
            # ------------------------------------------------------------------
            with torch.amp.autocast(device_type=device.type, enabled=(cfg.stageC_train.amp and device.type == "cuda")):
                rho_arg = None if model.cfg.dynamic_rho else cur_rho
                with timer.measure("forward_main"):
                    out = model(x_tokens=x_tokens, x_attention_mask=x_mask, rho=rho_arg, diagnostics=diagnostics)
                loss_total = out.loss

                # ---- strong branch: only compressor ----
                if ornamenter is not None:

                    tau_orn = float(cfg.stageC_train.ornament_tau)

                    # 可选：strong 分支禁用 gumbel 以稳定一致性
                    orig_use_gumbel = model.compressor.use_gumbel_in_train
                    if cfg.stageC_train.ornament_student_disable_gumbel:
                        model.compressor.use_gumbel_in_train = False
                    try:

                        # --------------------------------------------------
                        # [5] Strong-branch compressor forward
                        # --------------------------------------------------
                        with timer.measure("forward_strong"):
                            # 让 strong 分支更“确定性”：关 dropout + 关 gumbel（eval 模式即可）
                            if cfg.stageC_train.ornament_student_eval_mode:
                                was_training = model.training
                                model.eval()
                            try:
                                if cfg.stageC_train.ornament_use_teacher_zlen:
                                    z_len_strong = out.z_len_hard  # [B] detached
                                    comp_strong = model.compressor(
                                        src_tokens=x_strong,
                                        src_attention_mask=x_strong_mask,
                                        z_len=z_len_strong,
                                        tau=tau_orn,
                                        return_pointer_logits=False,
                                        return_z_embeds_hard=False,
                                    )
                                else:
                                    if model.cfg.dynamic_rho:
                                        raise ValueError("dynamic_rho=True 时建议 ornament_use_teacher_zlen=True。")
                                    comp_strong = model.compressor(
                                        src_tokens=x_strong,
                                        src_attention_mask=x_strong_mask,
                                        rho=float(cur_rho),
                                        tau=tau_orn,
                                        return_pointer_logits=False,
                                        return_z_embeds_hard=False,
                                    )
                            finally:
                                if cfg.stageC_train.ornament_student_eval_mode:
                                    model.train(was_training)

                    finally:
                        model.compressor.use_gumbel_in_train = orig_use_gumbel

            loss_cons = torch.zeros((), device=device, dtype=torch.float32)
            loss_ins = torch.zeros((), device=device, dtype=torch.float32)

            if ornamenter is not None and comp_strong is not None:

                tau_orn = float(cfg.stageC_train.ornament_tau)

                # -------- teacher s(x) --------
                if cfg.stageC_train.ornament_teacher_mode == "same_forward":
                    ptr_teacher = _pick_pointer_soft(out)
                    zmask_teacher = out.z_mask

                elif cfg.stageC_train.ornament_teacher_mode == "eval_compressor":
                    with timer.measure("teacher_eval"):
                        with torch.no_grad():
                            was_training = model.training
                            model.eval()
                            try:
                                comp_teacher = model.compressor(
                                    src_tokens=x_tokens,
                                    src_attention_mask=x_mask,
                                    z_len=out.z_len_hard,
                                    tau=tau_orn,
                                )
                            finally:
                                model.train(was_training)
                    ptr_teacher = _pick_pointer_soft(comp_teacher).detach()
                    zmask_teacher = comp_teacher.z_mask

                elif cfg.stageC_train.ornament_teacher_mode == "ema":
                    assert ema_compressor is not None
                    with timer.measure("teacher_ema"):
                        with torch.no_grad():
                            comp_teacher = ema_compressor(
                                src_tokens=x_tokens,
                                src_attention_mask=x_mask,
                                z_len=out.z_len_hard,
                                tau=tau_orn,
                            )
                    ptr_teacher = _pick_pointer_soft(comp_teacher).detach()
                    zmask_teacher = comp_teacher.z_mask

                else:
                    raise ValueError(f"Unknown ornament_teacher_mode: {cfg.stageC_train.ornament_teacher_mode}")

                # ------------------------------------------------------------
                # [7] Ornament invariance losses
                #     (marginal importance, scatter-align, MSE, insertion loss)
                # ------------------------------------------------------------
                with timer.measure("ornament_losses"):

                    # teacher object
                    if cfg.stageC_train.ornament_teacher_mode == "same_forward":
                        teacher_obj = out
                    elif cfg.stageC_train.ornament_teacher_mode == "eval_compressor":
                        teacher_obj = comp_teacher   # 你上面算出来的那个
                    elif cfg.stageC_train.ornament_teacher_mode == "ema":
                        teacher_obj = comp_teacher
                    else:
                        raise ValueError(...)
                    s_teacher = _importance_for_consistency(
                        teacher_obj,
                        tau=tau_orn,
                        exclude_last_step=cfg.stageC_train.ornament_exclude_last_step,
                    ).to(torch.float32)
                    s_strong = _importance_for_consistency(
                        comp_strong,
                        tau=tau_orn,
                        exclude_last_step=cfg.stageC_train.ornament_exclude_last_step,
                    ).to(torch.float32)

                    # x' -> x 对齐聚合
                    s_to_x = aggregate_importance_by_pi(s_strong, pi, L_out=x_tokens.size(1))  # [B,L]

                    # 可选：duration reweight（在 x 坐标系上做，保证 teacher/student 用同一套权重）
                    alpha = float(cfg.stageC_train.ornament_dur_alpha)
                    if alpha != 0.0:
                        s_teacher = apply_duration_weight_to_importance(
                            s_teacher,
                            x_tokens=x_tokens,
                            duration_q=model.duration_q,
                            alpha=alpha,
                        )
                        s_to_x = apply_duration_weight_to_importance(
                            s_to_x,
                            x_tokens=x_tokens,
                            duration_q=model.duration_q,
                            alpha=alpha,
                        )

                    # 一致性：只在 x 的有效 token 上比较
                    mask_cons = x_mask
                    if cfg.stageC_train.ornament_cons_ignore_special:
                        # pitch local id < special_n 视作 special（BOS/EOS/PAD...）
                        mask_cons = mask_cons & (x_tokens[..., 0] >= int(model.duration_q.special_n))

                    if cfg.stageC_train.ornament_cons_renorm:
                        s_teacher = renormalize_scores_on_mask(s_teacher, mask_cons)
                        s_to_x    = renormalize_scores_on_mask(s_to_x, mask_cons)

                    if cfg.stageC_train.ornament_cons_loss == "kl":
                        eps = 1e-8
                        # KL(teacher || student)
                        # 注意：自己算 KL 比 F.kl_div 更不容易踩 0*log(0) 的 NaN
                        q = s_teacher.to(torch.float32)
                        p = s_to_x.to(torch.float32)
                        loss_cons = (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=1).mean()
                    else:
                        loss_cons = masked_mse_loss(s_teacher, s_to_x, mask_cons)

                    # 装饰音质量：inserted 的 mass 越小越好
                    loss_ins = insertion_mass_loss(s_strong, pi, inserted_value=PI_INSERTED)

                loss_total = loss_total.to(torch.float32)
                loss_total = loss_total + float(cur_lambda_cons) * loss_cons
                loss_total = loss_total + float(cur_lambda_ins) * loss_ins

            # ------------------------------------------------------------------
            # [8] Backward pass
            # ------------------------------------------------------------------
            with timer.measure("backward"):
                scaler.scale(loss_total).backward()

            # ------------------------------------------------------------------
            # [9] Optimizer step  (unscale + grad-clip + step + update)
            # ------------------------------------------------------------------
            with timer.measure("optim_step"):
                if cfg.stageC_train.grad_clip and cfg.stageC_train.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.stageC_train.grad_clip)
                scaler.step(optim)
                scaler.update()

            if ema_compressor is not None:
                m = float(cfg.stageC_train.ornament_ema_momentum)
                with torch.no_grad():
                    for ema_p, p in ema_param_pairs:
                        ema_p.data.mul_(m).add_(p.data, alpha=1.0 - m)
                    for ema_b, b in ema_buffer_pairs:
                        if ema_b.dtype.is_floating_point:
                            ema_b.data.mul_(m).add_(b.data, alpha=1.0 - m)
                        else:
                            ema_b.copy_(b)

            timer.stop("step_total")    # stop outer envelope here (excludes metrics/logging below)

            # bsz = x_tokens.size(0)
            # running_loss += float(loss_total.detach().float()) * bsz
            # running_recon += float(out.loss_recon.float()) * bsz
            # running_prior += float(out.loss_prior.float()) * bsz
            # running_guided += float(out.loss_guided_attn.float()) * bsz
            # running_len += float(out.loss_len.float()) * bsz
            # running_sharp += float(out.loss_ptr_sharp.float()) * bsz
            # if model.cfg.dynamic_rho:
            #     running_zlen += float(out.z_len_cont.mean().float()) * bsz
            # else:
            #     running_zlen += float(out.z_mask.sum(dim=1).float().mean().float()) * bsz
            # if ornamenter is not None:
            #     running_cons += float(loss_cons.detach().float()) * bsz
            #     running_ins += float(loss_ins.detach().float()) * bsz
            # n += bsz

            step += 1
            global_step += 1

            interval_loss += float(loss_total.detach().float())
            interval_recon += float(out.loss_recon.float())
            interval_prior += float(out.loss_prior.float())
            interval_guided += float(out.loss_guided_attn.float())
            interval_len += float(out.loss_len.float())
            interval_sharp += float(out.loss_ptr_sharp.float())
            interval_zlen += float(out.z_mask.sum(dim=1).float().mean().float())
            if ornamenter is not None:
                interval_cons += float(loss_cons.detach().float())
                interval_ins += float(loss_ins.detach().float())
            interval_steps += 1

            if in_final_window:
                batch_metrics = extract_stagec_final_window_batch_metrics(out)
                if batch_metrics:
                    final_window_tracker.update(batch_metrics)

            if cfg.use_wandb and wandb is not None and interval_steps >= cfg.stageC_train.log_interval:
                log_dict = {
                    "stage": "C",
                    "C/batch_loss": (interval_loss / interval_steps).item(),
                    "C/loss_recon": (interval_recon / interval_steps).item(),
                    "C/loss_prior": (interval_prior / interval_steps).item(),
                    "C/loss_guided_attn": (interval_guided / interval_steps).item(),
                    "C/loss_len": (interval_len / interval_steps).item(),
                    "C/loss_sharp": (interval_sharp / interval_steps).item(),
                    "C/loss_cons": (interval_cons / interval_steps).item(),
                    "C/loss_ins": (interval_ins / interval_steps).item(),
                    "C/avg_zlen": (interval_zlen / interval_steps).item(),
                    "C/lr_backbone": optim.param_groups[0]["lr"],
                    "C/lr_pointer": optim.param_groups[1]["lr"],
                    "C/rho": cur_rho,
                    "C/tau": cur_tau,
                    "C/lambda_P": cur_lambda_P,
                    "C/lambda_GA": cur_lambda_GA,
                    "C/lambda_L": cur_lambda_L,
                    "C/lambda_sharp": cur_lambda_sharp,
                    "C/lambda_cons": cur_lambda_cons,
                    "C/lambda_ins": cur_lambda_ins,
                    "C/recon_ss_prob": cur_ss_prob,
                    "C/recon_input_mask_prob": cur_recon_input_mask_prob,
                    "C/epoch": epoch,
                    "C/step": step,
                    "global_step": global_step,
                    "C/rho_pred_mean": float(out.rho_pred.mean().item()),
                    "C/rho_pred_std": float(out.rho_pred.std(unbiased=False).item()),
                    "C/z_len_cont_mean": float(out.z_len_cont.mean().item()),
                }

                loss_show = log_dict["C/batch_loss"]
                pbar.set_postfix_str(f"loss={loss_show:.6f}")

                # NEW: diagnostics
                if getattr(out, "diag", None):
                    for k, v in out.diag.items():
                        log_dict[f"C/diag/{k}"] = float(v.item())

                # Step timers: cumulative-average seconds/step since training start
                if cfg.stageC_train.enable_timers:
                    log_dict.update(timer.wandb_log_dict())

                wandb.log(log_dict, step=global_step)
                interval_loss.zero_()
                interval_recon.zero_()
                interval_prior.zero_()
                interval_guided.zero_()
                interval_len.zero_()
                interval_sharp.zero_()
                interval_zlen.zero_()
                interval_cons.zero_()
                interval_ins.zero_()
                interval_steps = 0

            if otb_valid_loader is not None and otb_runner.eval_interval_steps > 0:
                if global_step % int(otb_runner.eval_interval_steps) == 0:
                    # 释放本 step 仍被 Python 变量引用的显存，避免 eval 叠加峰值
                    del loss_total, loss_cons, loss_ins
                    del out, comp_strong
                    del x_tokens, x_mask
                    if ornamenter is not None:
                        del x_strong, x_strong_mask, pi

                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    _run_bench_eval("otb", "valid", otb_runner, otb_valid_loader, show_progress=True)

                    # 可选：只有你确实卡在 eval 点 OOM 时再开；频繁开会拖慢
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    if device.type == "cuda":
                        alloc = torch.cuda.memory_allocated() / 1024**2
                        reserv = torch.cuda.memory_reserved() / 1024**2
                        print(f"[mem] allocated={alloc:.1f}MB reserved={reserv:.1f}MB")

        # epoch_loss = running_loss.item() / max(1, n)
        # epoch_recon = running_recon.item() / max(1, n)
        # epoch_prior = running_prior.item() / max(1, n)
        # epoch_guided = running_guided.item() / max(1, n)
        # epoch_len = running_len.item() / max(1, n)
        # epoch_sharp = running_sharp.item() / max(1, n)
        # epoch_zlen = running_zlen.item() / max(1, n)
        # epoch_cons = running_cons.item() / max(1, n)
        # epoch_ins = running_ins.item() / max(1, n)

        # print(f"[StageC] epoch {epoch} loss={epoch_loss:.6f} loss_recon={epoch_recon:.6f} loss_prior={epoch_prior:.6f} avg_zlen={epoch_zlen:.2f}")

        # if cfg.use_wandb and wandb is not None:
        #     wandb.log({
        #         "stage": "C",
        #         "C/epoch_loss": epoch_loss,
        #         "C/epoch_loss_recon": epoch_recon,
        #         "C/epoch_loss_prior": epoch_prior,
        #         "C/epoch_loss_guided_attn": epoch_guided,
        #         "C/epoch_loss_len": epoch_len,
        #         "C/epoch_loss_sharp": epoch_sharp,
        #         "C/epoch_loss_cons": epoch_cons,
        #         "C/epoch_loss_ins": epoch_ins,
        #         "C/epoch_avg_zlen": epoch_zlen,
        #         "C/rho": cur_rho,
        #         "C/tau": cur_tau,
        #         "C/lambda_P": cur_lambdaP,
        #         "C/lambda_GA": cur_lambdaGA,
        #         "C/lambda_L": cur_lambdaL,
        #         "C/lr_backbone": optim.param_groups[0]["lr"],
        #         "C/lr_pointer": optim.param_groups[1]["lr"],
        #         "C/epoch": epoch,
        #         "global_step": global_step,
        #         "C/rho_pred_mean": float(out.rho_pred.mean().item()),
        #         "C/z_len_cont_mean": float(out.z_len_cont.mean().item()),
        #     }, step=global_step)

        # save (建议每 epoch 都存；你也可以改成 save_interval)
        state = model.state_dict()
        # inference 不需要 lm_prior，保存时丢掉，减少体积 + 避免加载麻烦
        state = {k: v for k, v in state.items() if not k.startswith("lm_prior.")}

        save_ckpt(os.path.join(ckpt_dir, f"skeleton_epoch{epoch}.pt"), {
            "stage": "C",
            "epoch": epoch,
            "model_state_dict": state,
            "optimizer_state_dict": optim.state_dict(),
            "config": asdict(cfg),
            "vocab_pkl": cfg.vocab_pkl,
        })

        if otb_valid_loader is not None and otb_runner.eval_at_epoch_end:
            # while 循环结束后，最后一个 step 的局部变量还在作用域里
            try:
                del loss_total, loss_cons, loss_ins
                del out, comp_strong
                del x_tokens, x_mask
                if ornamenter is not None:
                    del x_strong, x_strong_mask, pi
            except UnboundLocalError:
                pass

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            _run_bench_eval("otb", "valid", otb_runner, otb_valid_loader, show_progress=True)

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if final_window_num_batches > 0 and final_window_tracker.num_updates > 0:
        final_window_metrics = summarize_stagec_final_window_metrics(
            final_window_tracker,
            attr_weights=model.cfg.recon_attr_weights,
            requested_batches=final_window_num_batches,
            total_steps=total_steps,
            prefix="C_lastN/",
        )

        def _fm(k: str) -> float:
            return float(final_window_metrics.get(k, float("nan")))

        eff_n = int(final_window_metrics["C_lastN/window_batches_effective"])
        print(
            f"[StageC][Last {eff_n} batches] "
            f"recon_ppl={_fm('C_lastN/recon_ppl/weighted_geom'):.6f} "
            f"(p={_fm('C_lastN/recon_ppl/pitch'):.6f}, "
            f"d={_fm('C_lastN/recon_ppl/duration'):.6f}, "
            f"dt={_fm('C_lastN/recon_ppl/dt'):.6f}) "
            f"recon_wo_z/ratio_wo_over_w={_fm('C_lastN/recon_wo_z/ratio_wo_over_w'):.6f}"
        )

        with open(os.path.join(ckpt_dir, "stageC_last_n_batch_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(final_window_metrics, f, ensure_ascii=False, indent=2)

        if cfg.use_wandb and wandb is not None:
            wandb.log({
                **final_window_metrics,
                "stage": "C",
                "global_step": global_step,
            }, step=global_step)

    last_path = os.path.join(ckpt_dir, "skeleton_last.pt")
    state = model.state_dict()
    state = {k: v for k, v in state.items() if not k.startswith("lm_prior.")}
    save_ckpt(last_path, {
        "stage": "C",
        "model_state_dict": state,
        "config": asdict(cfg),
        "vocab_pkl": cfg.vocab_pkl,
    })

    if otb_test_loader is not None:

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        m = _run_bench_eval("otb", "test", otb_runner, otb_test_loader, show_progress=True)
        if m is not None:
            with open(os.path.join(ckpt_dir, "otb_test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)

        # -------------------------
        # NEW: proxy eval on full OTB test set
        # -------------------------
        proxy_cfg = MusicPriorProxyEvalConfig(
            pos_per_beat=cfg.stageC_train.proxy_eval_pos_per_beat,                 # 或者你想从 cfg.stageC_train 读（如果你加了字段）
            strong_period_beats=cfg.stageC_train.proxy_eval_strong_period_beats,
            use_cummax_onset=cfg.stageC_train.proxy_eval_use_cummax_onset,
            max_batches=None,                # <= 关键：None = 整个 test dataset
            amp=(device.type == "cuda"),
            show_progress=True,
            tau=otb_runner.metrics.tau,      # 跟 OTB eval 用同一个 tau 口径（可为 None）
            token_key="x_orn",
            z_len_key="len_x",
        )
        proxy_prefix = "C_test/otb/proxy/"

        m_proxy = evaluate_music_prior_proxy(
            otb_adapter,
            otb_test_loader,
            device=device,
            duration_q=model.duration_q,
            dt_q=model.dt_q,
            cfg=proxy_cfg,
            prefix=proxy_prefix,
        )

        # stdout 简要打印（你可按需增删）
        def _pg(k: str) -> float:
            return float(m_proxy.get(proxy_prefix + k, float("nan")))

        print(
            f"[OTB-test proxy] "
            f"JS(cnt)={_pg('pchist_js_cnt'):.4f} "
            f"JS(dur)={_pg('pchist_js_dur'):.4f} "
            f"JS_hard(dur)={_pg('pchist_js_hard_dur'):.4f} "
            f"Lift(strong)={_pg('lift_strong'):.3f} "
            f"Lift(dur)={_pg('lift_duration'):.3f} "
            f"Lift(ext)={_pg('lift_extrema'):.3f}"
        )

        with open(os.path.join(ckpt_dir, "otb_test_proxy_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m_proxy, f, ensure_ascii=False, indent=2)

        if cfg.use_wandb and wandb is not None:
            m_proxy.update({
                "stage": "C",
                "global_step": global_step,
            })
            wandb.log(m_proxy, step=global_step)
                
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if tavern_test_loader is not None:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        m = _run_bench_eval("tavern", "test", tavern_runner, tavern_test_loader, show_progress=True)
        if m is not None:
            with open(os.path.join(ckpt_dir, "tavern_test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if jiugong_test_loader is not None:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        m = _run_bench_eval("jiugong", "test", jiugong_runner, jiugong_test_loader, show_progress=True)
        if m is not None:
            with open(os.path.join(ckpt_dir, "jiugong_test_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False, indent=2)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -------------------------
    # GTTM benchmark evaluation (post-training, test only)
    # -------------------------
    gttm_runner = cfg.eval_gttm
    if gttm_runner.enable and gttm_runner.test_bench_dir and os.path.isdir(gttm_runner.test_bench_dir):
        print(f"\n[GTTM] Running GTTM evaluation on {gttm_runner.test_bench_dir} ...")
        gttm_loader = make_gttm_benchmark_dataloader(
            gttm_runner.test_bench_dir,
            batch_size=gttm_runner.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        # 用 metrics 作为唯一真源，避免 runner.eval_rho 和 metrics.eval_rho 两套打架
        gttm_metric_cfg = replace(
            gttm_runner.metrics,
            amp=(device.type == "cuda"),
            show_progress=True,
        )
        gttm_prefix = "C_test/gttm/"
        gttm_metrics = evaluate_gttm_backbone(
            otb_adapter,
            gttm_loader,
            device=device,
            cfg=gttm_metric_cfg,
            prefix=gttm_prefix,
        )

        def _g(k: str) -> float:
            return float(gttm_metrics.get(gttm_prefix + k, float("nan")))
        print(
            f"[GTTM-test] "
            f"f1_path={_g('f1_hard'):.4f} | "
            f"f1_topk={_g('f1_hard_topk'):.4f} | "
            f"gap_f1={_g('gap_f1'):.4f} | "
            f"ap={_g('ap_soft'):.4f} | "
            f"ndcg@ref={_g('ndcg_soft'):.4f} | "
            f"ndcg@full={_g('ndcg_full'):.4f} | "
            f"cut_auc={_g('cut_f1_auc_norm'):.4f} | "
            f"mean_ndcg_cut={_g('mean_ndcg_cut'):.4f} | "
            f"spearman={_g('spearman'):.4f} | "
            f"rho={_g('rho'):.4f}"
        )

        with open(os.path.join(ckpt_dir, "gttm_test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(gttm_metrics, f, ensure_ascii=False, indent=2)
        print(f"[GTTM] Saved metrics to {os.path.join(ckpt_dir, 'gttm_test_metrics.json')}")

        if cfg.use_wandb and wandb is not None:
            gttm_metrics.update({
                "stage": "C",
                "global_step": global_step,
            })
            wandb.log(gttm_metrics, step=global_step)

    return last_path, global_step


def _nonempty(p: Optional[str]) -> bool:
    return p is not None and str(p).strip() != ""
def _check_file(p: str, name: str):
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{name} not found: {p}")
    
def _stageA_last_ckpt(save_dir: str) -> str:
    return os.path.join(save_dir, "stageA_pretrain", "backbone_pretrained_last.pt")

def _stageB_last_ckpt(save_dir: str) -> str:
    return os.path.join(save_dir, "stageB_lm", "lm_prior_backbone_last.pt")

def _stageC_last_ckpt(save_dir: str) -> str:
    return os.path.join(save_dir, "stageC_skeleton", "skeleton_last.pt")


def _cleanup_between_runs():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if hasattr(torch, "clear_autocast_cache"):
            torch.clear_autocast_cache()
        torch.cuda.empty_cache()


def _seed_corpus_dir(data_root: str, seed: int, corpus_dirname: str) -> str:
    # e.g. .\preproc\output\seed10101\skeletion_unsup_corpus_...\...
    return os.path.join(data_root, f"seed{seed}", corpus_dirname)


def build_cfg_for_seed(
    base_cfg: TrainSkeletonE2EConfig,
    *,
    seed: int,
    data_root: str,
    corpus_dirname: str,
) -> TrainSkeletonE2EConfig:
    root = _seed_corpus_dir(data_root, seed, corpus_dirname)

    cfg = replace(
        base_cfg,
        seed=int(seed),

        pretrain_npy=os.path.join(root, "train.npy"),
        lm_npy=os.path.join(root, "train.npy"),
        skeleton_npy=os.path.join(root, "train.npy"),
        vocab_pkl=os.path.join(root, "SimpleMono.pkl"),

        # OTB split 需要 seed
        eval_otb=replace(
            base_cfg.eval_otb,
            valid_bench_dir=os.path.join(root, "otb_bench", "valid_ood"),
            test_bench_dir=os.path.join(root, "otb_bench", "test_ood"),
        ),
    )

    # 强校验（建议有，避免跑一半才发现路径错）
    _check_file(cfg.pretrain_npy, "pretrain_npy")
    _check_file(cfg.lm_npy, "lm_npy")
    _check_file(cfg.skeleton_npy, "skeleton_npy")
    _check_file(cfg.vocab_pkl, "vocab_pkl")
    if cfg.eval_otb.enable:
        if not os.path.isdir(cfg.eval_otb.valid_bench_dir):
            raise FileNotFoundError(f"eval_otb.valid_bench_dir not found: {cfg.eval_otb.valid_bench_dir}")
        if not os.path.isdir(cfg.eval_otb.test_bench_dir):
            raise FileNotFoundError(f"eval_otb.test_bench_dir not found: {cfg.eval_otb.test_bench_dir}")

    return cfg


def ensure_ab_ckpts(
    seed_cfg: TrainSkeletonE2EConfig,
    *,
    shared_dir: str,
    force_retrain: bool = False,
) -> tuple[str, str]:
    """
    Make sure StageA & StageB last ckpts exist under shared_dir.
    If both exist and not force_retrain -> reuse.
    Otherwise -> run main() with StageC disabled to produce them.
    """
    ckptA_expect = _stageA_last_ckpt(shared_dir)
    ckptB_expect = _stageB_last_ckpt(shared_dir)

    haveA = os.path.isfile(ckptA_expect)
    haveB = os.path.isfile(ckptB_expect)

    if (not force_retrain) and haveA and haveB:
        print(f"[AB cache] reuse:\n  A={ckptA_expect}\n  B={ckptB_expect}")
        return ckptA_expect, ckptB_expect

    # 如果 A 不存在，就别复用 B（B 依赖 A 的 init，避免不一致）
    useA = ckptA_expect if (haveA and (not force_retrain)) else None
    useB = ckptB_expect if (haveA and haveB and (not force_retrain)) else None

    ab_cfg = replace(
        seed_cfg,
        save_dir=shared_dir,
        wandb_run_name=f"ab-seed{seed_cfg.seed}",
        pretrain_ckpt=useA,
        lm_ckpt=useB,
        stageC_train=replace(seed_cfg.stageC_train, enable=False),  # 关键：只跑 A+B
    )

    out = main(ab_cfg)
    ckptA = out["ckptA"]
    ckptB = out["ckptB"]
    _check_file(ckptA, "StageA last ckpt")
    _check_file(ckptB, "StageB last ckpt")
    return ckptA, ckptB


# -------------------------
# Main
# -------------------------
def main(cfg: TrainSkeletonE2EConfig) -> dict:
    """
    Run a single experiment described by cfg.
    Returns a dict with ckpt paths so that an outer runner can chain stages.
    """
    ckptA = ""
    ckptB = ""
    ckptC = ""
    global_step = 0

    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device)
    print(f"[Device] {device}")

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    if cfg.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but use_wandb=True")
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    try:
        vocab = load_vocab_info(cfg.vocab_pkl)
        vocab_cfg = vocab.to_music_bart_vocab_config()

        quant_tables = MusicQuantizationTables(
            special_n=vocab.special_n,
            duration_code_to_pos=vocab.duration_code_to_pos,
            duration_pos_to_code=vocab.duration_pos_to_code,
            deltatime_code_offset=vocab.deltatime_code_offset,
            deltatime_code_to_pos=vocab.deltatime_code_to_pos,
            deltatime_pos_to_code=vocab.deltatime_pos_to_code,
        )

        music_bart_cfg = build_music_bart_cfg(cfg.model, vocab_cfg, dropout=cfg.stageA.dropout)

        # -----------------
        # Stage A
        # -----------------
        if _nonempty(cfg.pretrain_ckpt):
            ckptA = str(cfg.pretrain_ckpt).strip()
            _check_file(ckptA, "pretrain_ckpt")
            print(f"[StageA] skipped. Use pretrain_ckpt: {ckptA}")
        else:
            ckptA, global_step = run_stageA_pretrain(
                cfg, device, music_bart_cfg, vocab,
                quant_tables=quant_tables,
                global_step=global_step
            )

        # -----------------
        # Stage B
        # -----------------
        if _nonempty(cfg.lm_ckpt):
            ckptB = str(cfg.lm_ckpt).strip()
            _check_file(ckptB, "lm_ckpt")
            print(f"[StageB] skipped. Use lm_ckpt: {ckptB}")
        else:
            ckptB, global_step = run_stageB_lm(
                cfg, device, music_bart_cfg, vocab,
                backbone_init_ckpt=ckptA,
                quant_tables=quant_tables,
                global_step=global_step
            )

        # -----------------
        # Stage C
        # -----------------
        ckptC, global_step = run_stageC_skeleton(
            cfg, device, music_bart_cfg, vocab,
            backbone_init_ckpt=ckptA,
            lm_prior_ckpt=ckptB,
            quant_tables=quant_tables,
            global_step=global_step
        )

        print("\n========== Done ==========")
        print(f"StageA ckpt: {ckptA}")
        print(f"StageB ckpt: {ckptB}")
        print(f"StageC ckpt: {ckptC}")

        return {
            "ckptA": ckptA,
            "ckptB": ckptB,
            "ckptC": ckptC,
            "global_step": int(global_step),
        }

    finally:
        if cfg.use_wandb and wandb is not None:
            wandb.finish()


if __name__ == "__main__":
    # =========================================================
    # Ablation study
    # =========================================================

    SEEDS = [10101, 20202, 30303, 40404, 50505]
    WANDB_PROJECT = "skel-final-main"

    DATA_ROOT = r".\preproc\output"
    CORPUS_DIRNAME = "skeletion_unsup_corpus_v260411_with_ornamented_split"

    OUTPUT_ROOT = r".\ckpt\main-2604-final"
    FORCE_RETRAIN_AB = False
    FORCE_RERUN_C = False

    # ------------------------------------------------------------------
    # Shared model / stage configs used as a base for all experiments
    # ------------------------------------------------------------------
    _model_cfg = ModelHyperConfig(
        max_seq_len=514, d_embed=256, d_model=512,
        n_encoder_layers=6, n_decoder_layers=3, n_heads=8, d_ff=2048,
    )

    _stageA_cfg = StageASeq2SeqPretrainConfig(
        enable=True,
        epochs=100,
        batch_size=160,
        lr=5e-4,
        dropout=0.15,
    )

    _stageB_cfg = StageBLMPriorTrainConfig(
        enable=True,
        epochs=80,
        batch_size=200,
        lr=4e-4,
        dropout=0.15,
        attr_weights=(0.5, 0.3, 0.2),
    )

    # Base skeleton model config (experiment A)
    _base_sk = replace(
        MusicSkeletonIIIConfig(),
        max_z_len=514,
        tau=1.0,
        pointer_d_head=256,
        recon_z_mode="final", # "onfly" or "final"
        lambda_R=1.8,
        lambda_P=1.0,
        lambda_GA=1.0,  # initial value only; overridden per-step during training
        lambda_L=1.0,   # initial value only; overridden per-step during training
    )

    # Base stage-C training config (experiment A)
    # Note: lambda_GA_start=5.0, lambda_GA_final=20.0 are defaults in StageCSkeletonTrainConfig.
    # Note: lambda_P_start=0.0 (default) -> warmup from 0 to skeleton_model.lambda_P.
    _base_stageC = StageCSkeletonTrainConfig(
        enable=True,
        epochs=1,
        batch_size=28, # local capacity max is 8~12
        warmup_ratio=0.30,
        lr_backbone=3e-4,
        lr_pointer=4e-4,
        dropout=0.075,
        limit_batches=None # int(64000*4*8/38) # local capacity max is 8~12
    )

    _base_cfg_template = TrainSkeletonE2EConfig(

        pretrain_npy="",
        lm_npy="",
        skeleton_npy="",
        vocab_pkl="",
        save_dir="",                 # 每个 run 再填
        pretrain_ckpt=None,          # 先不提供，AB 会自己训练/缓存
        lm_ckpt=None,
        device="cuda",
        num_workers=0,
        pin_memory=True,
        seed=0,                      # 每个 seed 覆盖
        use_wandb=True,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=None,
        model=_model_cfg,
        stageA=_stageA_cfg,
        stageB=_stageB_cfg,
        skeleton_model=_base_sk,
        stageC_train=_base_stageC,
        # 注意：otb 的 bench_dir 也先空，seed 时填
        eval_otb=OTBRunnerConfig(
            enable=True,
            valid_bench_dir="",
            test_bench_dir="",
            eval_interval_steps=50,
            eval_at_epoch_end=False,
            max_valid_batches=8,
            max_test_batches=None,
            metrics=OrnamentToBackboneEvalConfig(
                exclude_last_step=True,
                ignore_special_tokens=True,
                compute_cut_curve=True,
            ),
        ),
        eval_tavern=OTBRunnerConfig(
            enable=True,
            valid_bench_dir="",  # 一般不给
            test_bench_dir=r".\preproc\output\tavern_silver_otb\test",
            batch_size=128,
            eval_interval_steps=0,      # IMPORTANT: 不要训练中 eval test
            eval_at_epoch_end=False,    # IMPORTANT: 不要 epoch end eval test
            max_valid_batches=None,
            max_test_batches=None,
            metrics=OrnamentToBackboneEvalConfig(
                exclude_last_step=True,
                ignore_special_tokens=True,
                compute_cut_curve=True,
            ),
        ),
        eval_jiugong=OTBRunnerConfig(
            enable=True,
            valid_bench_dir="",  # 一般不给
            test_bench_dir=r".\preproc\output\real_jiugongdacheng_otb_bench\test",
            batch_size=128,
            eval_interval_steps=0,      # IMPORTANT: 不要训练中 eval test
            eval_at_epoch_end=False,    # IMPORTANT: 不要 epoch end eval test
            max_valid_batches=None,
            max_test_batches=None,
            metrics=OrnamentToBackboneEvalConfig(
                exclude_last_step=True,
                ignore_special_tokens=True,
                compute_cut_curve=True,
            ),
        ),
        eval_gttm=GTTMRunnerConfig(
            enable=False,
            test_bench_dir=r".\preproc\output\gttm_bench_v1.2\test",
            eval_rho=2.0 / 3.0,
            metrics=GTTMBackboneEvalConfig(
                eval_rho=2.0 / 3.0,
                force_z_len_rho=2.0 / 3.0,
                compute_cut_curve=True,
            ),
        ),
    )

    ablation_experiments = [
        # (exp_tag, skeleton_model override, stageC_train override)
        (
            "def",
            {},
            {}
        ),
        # ---------------------------------------------
        (
            "no_ga",
            {},
            dict(
                lambda_GA_start=0.0,
                lambda_GA_final=0.0,
            )
        ),
        (
            "no_lamb_len",
            {},
            dict(
                lambda_L_start=0.0,
                lambda_L_final=0.0,
            )
        ),
        (
            "no_cons",
            {},
            dict(
                lambda_cons_start=0.0,
                lambda_cons_final=0.0,
            )
        ),
        (
            "no_ins",
            {},
            dict(
                lambda_ins_start=0.0,
                lambda_ins_final=0.0,
            )
        ),
        (
            "no_recon",
            dict(
                lambda_R=0.0,
            ),
            {},
        ),
        (
            "no_prior",
            {},
            dict(
                lambda_P_start=0.0,
                lambda_P_final=0.0,
            )
        ),
        (
            "wo_init",
            {},  # skeleton_model override
            dict(
                wo_bart_pretrain_init=True,
            ),
        ),
        (
            "wo_rhythmic_closure",
            dict(
                recon_z_mode="onfly",
            ),
            {},
        ),
    ]

    # ----------------------------------------------------------------
    # Pre-flight config validation
    # Catches bad field names / type mismatches before any run starts.
    # ----------------------------------------------------------------
    _preflight_errors: list[str] = []

    _seen_tags: set[str] = set()
    for _i, (_tag, _sk_ov, _stageC_ov) in enumerate(ablation_experiments):
        _prefix = f"[preflight #{_i} '{_tag}']"

        # 1. Duplicate exp_tag
        if _tag in _seen_tags:
            _preflight_errors.append(f"{_prefix} Duplicate exp_tag.")
        _seen_tags.add(_tag)

        # 2. replace() dry-run — catches unknown field names immediately
        try:
            _sk_dry = replace(_base_sk, **_sk_ov)
        except TypeError as _e:
            _preflight_errors.append(f"{_prefix} skeleton_model override error: {_e}")
            _sk_dry = None
        try:
            _stageC_dry = replace(_base_stageC, **_stageC_ov)
        except TypeError as _e:
            _preflight_errors.append(f"{_prefix} stageC_train override error: {_e}")
            _stageC_dry = None

        # 3. Type-annotation check for override values
        def _check_types(dry_obj, overrides: dict, label: str) -> None:
            if dry_obj is None:
                return
            _hints = {f.name: f.type for f in fields(dry_obj)}
            for _k, _v in overrides.items():
                _ann = _hints.get(_k)
                if _ann is None:
                    continue  # unknown field already caught above
                # For simple built-in types (int, float, bool, str) check isinstance
                try:
                    _origin = getattr(_ann, "__origin__", None)
                    if isinstance(_ann, type) and not _origin:
                        if not isinstance(_v, _ann):
                            _preflight_errors.append(
                                f"{_prefix} {label}.{_k}: expected {_ann.__name__}, "
                                f"got {type(_v).__name__} ({_v!r})"
                            )
                except Exception:
                    pass  # skip if annotation resolution fails

        _check_types(_sk_dry, _sk_ov, "skeleton_model")
        _check_types(_stageC_dry, _stageC_ov, "stageC_train")

    if _preflight_errors:
        print("\n" + "!" * 60)
        print("[PREFLIGHT FAILED] Fix the following config errors before running:")
        for _err in _preflight_errors:
            print(f"  {_err}")
        print("!" * 60 + "\n")
        raise SystemExit(1)

    print(f"[preflight] All {len(ablation_experiments)} experiment configs validated OK.\n")
    # ----------------------------------------------------------------

# -------------------------
# Run: seeds × ablations
# -------------------------
for seed in SEEDS:
    print("\n" + "#" * 80)
    print(f"[SEED] {seed}")
    print("#" * 80)

    seed_cfg = build_cfg_for_seed(
        _base_cfg_template,
        seed=seed,
        data_root=DATA_ROOT,
        corpus_dirname=CORPUS_DIRNAME,
    )

    seed_out_dir = os.path.join(OUTPUT_ROOT, f"seed{seed}")
    shared_dir = os.path.join(seed_out_dir, "_AB_shared")

    # 1) Stage A + B once per seed
    ckptA, ckptB = ensure_ab_ckpts(
        seed_cfg,
        shared_dir=shared_dir,
        force_retrain=FORCE_RETRAIN_AB,
    )
    _cleanup_between_runs()

    # 2) Stage C ablations
    for exp_tag, sk_overrides, stageC_overrides in ablation_experiments:
        run_name = f"c-{exp_tag}-seed{seed}"

        # 关键：输出目录必须带 seed（你提的需求）
        run_save_dir = os.path.join(seed_out_dir, exp_tag)

        # 断点续跑：如果 skeleton_last.pt 已存在则跳过
        lastC = _stageC_last_ckpt(run_save_dir)
        if (not FORCE_RERUN_C) and os.path.isfile(lastC):
            print(f"[Skip] {run_name} (found {lastC})")
            continue

        run_cfg = replace(
            seed_cfg,
            wandb_run_name=run_name,
            save_dir=run_save_dir,

            # 关键：复用该 seed 的 AB ckpt
            pretrain_ckpt=ckptA,
            lm_ckpt=ckptB,

            skeleton_model=replace(_base_sk, **sk_overrides),
            stageC_train=replace(_base_stageC, **stageC_overrides),

            # 可选：显式关掉 A/B（即使 ckpt 路径写错，也不会误跑到 ablation 目录里）
            stageA=replace(seed_cfg.stageA, enable=False),
            stageB=replace(seed_cfg.stageB, enable=False),
        )

        print("=" * 60)
        print(f"[RUN] {run_name}")
        print(f"      save_dir: {run_save_dir}")
        print(f"      ckptA: {ckptA}")
        print(f"      ckptB: {ckptB}")
        print("=" * 60)

        try:
            main(run_cfg)
        finally:
            del run_cfg
            _cleanup_between_runs()

# python -m main.train_skeleton_end2end
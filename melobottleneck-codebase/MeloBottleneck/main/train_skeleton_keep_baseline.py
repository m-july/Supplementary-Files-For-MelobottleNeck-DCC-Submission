# main/train_skeleton_keep_baseline.py
from __future__ import annotations

import os
import math
import json
import gc
import time
import copy
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Dict, Any, Tuple

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from .vocab_utils import load_vocab_info
from .data import make_dataloader
from .augment import MusicAugmentConfig
from .quantization import MusicQuantizationTables, build_quantizers

from .models.bart import MusicBartConfig, MusicBartBackboneConfig
from .nn_modules import MusicBartBackbone

from .ornament import MusicOrnamenter, MusicOrnamentConfig, PI_INSERTED, PI_PAD

from .evaluation.ornament_benchmark_dataset import make_ornament_benchmark_dataloader
from .evaluation.ornament_to_backbone_eval import (
    OrnamentToBackboneEvalConfig,
    evaluate_ornament_to_backbone,
)

from .evaluation.score_extraction import build_keep_adapter_from_model

from .evaluation.music_prior_proxy_eval import (
    MusicPriorProxyEvalConfig,
    evaluate_music_prior_proxy,
)

from .models.skeleton.baseline_keep_encoder import (
    MusicSkeletonKeepBaseline,
    MusicSkeletonKeepBaselineConfig,
)


# -------------------------
# Utils (copy from your style)
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def warmup_cosine_lr_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    final_lr_ratio: float,
    init_lr_ratio: float,
) -> float:
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


def apply_lr_factor(optim: torch.optim.Optimizer, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = base_lr * factor


def set_dropout_p(model: nn.Module, p: float):
    p = float(p)
    if p < 0.0:
        p = 0.0
    if p >= 1.0:
        p = 1.0 - 1e-6

    float_dropout_attrs = (
        "dropout",
        "attention_dropout",
        "activation_dropout",
        "classifier_dropout",
    )

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = p

        for name in float_dropout_attrs:
            if hasattr(m, name):
                v = getattr(m, name)
                if isinstance(v, (float, int)):
                    setattr(m, name, p)

        cfg = getattr(m, "config", None)
        if cfg is not None:
            for name in float_dropout_attrs:
                if hasattr(cfg, name):
                    setattr(cfg, name, p)


def save_ckpt(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
    print(f"[Saved] {path}")


def build_music_bart_cfg(model_cfg, vocab_cfg, dropout: float) -> MusicBartConfig:
    backbone_cfg = MusicBartBackboneConfig(
        max_seq_len=model_cfg.max_seq_len,
        d_embed=model_cfg.d_embed,
        d_model=model_cfg.d_model,
        n_encoder_layers=model_cfg.n_encoder_layers,
        n_decoder_layers=model_cfg.n_decoder_layers,  # even if unused, keep same for ckpt compatibility
        n_heads=model_cfg.n_heads,
        d_ff=model_cfg.d_ff,
        dropout=float(dropout),
    )
    return MusicBartConfig(vocab=vocab_cfg, backbone=backbone_cfg)


def _check_file(p: str, name: str):
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(f"{name} not found: {p}")


def _check_dir(p: str, name: str):
    if not p or not os.path.isdir(p):
        raise FileNotFoundError(f"{name} not found: {p}")


def _atomic_json_dump(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Optional[dict]):
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def _keep_ckpt_dir(save_dir: str) -> str:
    return os.path.join(save_dir, "keep_baseline")


def _keep_last_full_ckpt(save_dir: str) -> str:
    return os.path.join(_keep_ckpt_dir(save_dir), "keep_last_full.pt")


def _keep_last_weights_ckpt(save_dir: str) -> str:
    return os.path.join(_keep_ckpt_dir(save_dir), "keep_last.pt")


def _keep_otb_test_proxy_json(save_dir: str) -> str:
    return os.path.join(_keep_ckpt_dir(save_dir), "otb_test_proxy_metrics.json")


def _keep_success_marker(save_dir: str) -> str:
    return os.path.join(_keep_ckpt_dir(save_dir), "_SUCCESS.json")


def _run_finished(save_dir: str, *, require_otb_proxy: bool = False) -> bool:
    if not os.path.isfile(_keep_success_marker(save_dir)):
        return False
    if require_otb_proxy and not os.path.isfile(_keep_otb_test_proxy_json(save_dir)):
        return False
    return True


def _find_resume_ckpt(cfg: "TrainKeepBaselineConfig") -> Optional[str]:
    if cfg.resume_ckpt is not None and str(cfg.resume_ckpt).strip():
        p = str(cfg.resume_ckpt).strip()
        _check_file(p, "resume_ckpt")
        return p

    if cfg.auto_resume:
        p = _keep_last_full_ckpt(cfg.save_dir)
        if os.path.isfile(p):
            return p

    return None


def _cleanup_between_runs():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if hasattr(torch, "clear_autocast_cache"):
            torch.clear_autocast_cache()
        torch.cuda.empty_cache()


def load_train_state(
    resume_path: str,
    *,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
) -> tuple[int, int, int]:
    ckpt = torch.load(resume_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    if ckpt.get("optimizer_state_dict") is not None:
        optim.load_state_dict(ckpt["optimizer_state_dict"])

    if ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    restore_rng_state(ckpt.get("rng_state"))

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    step = int(ckpt.get("step", 0))
    global_step = int(ckpt.get("global_step", step))

    print(f"[Resume] loaded ckpt: {resume_path}")
    print(f"[Resume] next_epoch={start_epoch}, step={step}, global_step={global_step}")
    return start_epoch, step, global_step


# -------------------------
# Configs
# -------------------------
@dataclass
class ModelHyperConfig:
    max_seq_len: int = 514
    d_embed: int = 256
    d_model: int = 512
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4   # IMPORTANT: match StageA ckpt architecture
    n_heads: int = 8
    d_ff: int = 2048


@dataclass
class BaselineTrainConfig:
    epochs: int = 30
    batch_size: int = 64

    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.10
    amp: bool = True

    log_interval: int = 50

    use_lr_schedule: bool = True
    warmup_ratio: float = 0.1
    final_lr_ratio: float = 0.2
    init_lr_ratio: float = 0.1

    # class imbalance
    use_dynamic_neg_weight: bool = True
    neg_weight_cap: float = 10.0

    # debug
    limit_batches: Optional[int] = None

    # checkpoint
    save_every_epochs: int = 1


@dataclass
class OrnamentSupervisionConfig:
    """
    在线从 x 生成 (x_orn, pi) supervision
    """
    p_apply: float = 1.0
    rho_min: float = 1.0 / 3.0
    min_extra_tokens: int = 1
    max_tries: int = 6
    max_extra_tokens: int = 256

    ornament_json: str = ""  # optional overrides


@dataclass
class OTBRunnerConfig:
    enable: bool = True
    valid_bench_dir: str = ""
    test_bench_dir: str = ""
    batch_size: int = 128

    eval_interval_steps: int = 200
    eval_at_epoch_end: bool = True

    max_valid_batches: Optional[int] = 8
    max_test_batches: Optional[int] = None

    metrics: OrnamentToBackboneEvalConfig = field(default_factory=lambda: OrnamentToBackboneEvalConfig(
        exclude_last_step=True,
        ignore_special_tokens=True,
        compute_cut_curve=True,
    ))


@dataclass
class OTBProxyEvalConfig:
    enable: bool = True
    pos_per_beat: int = 12
    strong_period_beats: int = 2
    use_cummax_onset: bool = True


@dataclass
class TrainKeepBaselineConfig:
    # data
    train_npy: str = ""
    vocab_pkl: str = ""

    # runtime
    save_dir: str = "./ckpt_keep_baseline"
    device: str = "cuda"
    num_workers: int = 0
    pin_memory: bool = True
    seed: int = 1234

    # wandb
    use_wandb: bool = False
    wandb_project: str = "keep-baseline"
    wandb_run_name: Optional[str] = None

    # model
    model: ModelHyperConfig = field(default_factory=ModelHyperConfig)
    keep_model: MusicSkeletonKeepBaselineConfig = field(default_factory=MusicSkeletonKeepBaselineConfig)

    # init
    backbone_init_ckpt: Optional[str] = None   # StageA backbone ckpt
    wo_bart_pretrain_init: bool = False        # True => random init

    # resume / rerun
    auto_resume: bool = True
    resume_ckpt: Optional[str] = None
    skip_if_finished: bool = True

    # training + supervision + eval
    train: BaselineTrainConfig = field(default_factory=BaselineTrainConfig)
    ornament: OrnamentSupervisionConfig = field(default_factory=OrnamentSupervisionConfig)

    eval_otb: OTBRunnerConfig = field(default_factory=OTBRunnerConfig)
    eval_otb_proxy: OTBProxyEvalConfig = field(default_factory=OTBProxyEvalConfig)

    eval_tavern: OTBRunnerConfig = field(default_factory=OTBRunnerConfig)
    eval_jiugong: OTBRunnerConfig = field(default_factory=OTBRunnerConfig)


# -------------------------
# Ornament supervision helpers (ported from preproc_ornament_benchmark)
# -------------------------
def _seq_len_including_eos_local(x_local: np.ndarray, *, pad_id: int, eos_id: Optional[int]) -> int:
    """
    x_local: [L,3] local ids
    return valid length ending at EOS (inclusive) if EOS exists, else first PAD.
    """
    pitch = x_local[:, 0]
    L = int(pitch.shape[0])

    pad_idx = np.where(pitch == int(pad_id))[0]
    valid_len = int(pad_idx[0]) if pad_idx.size > 0 else L

    if eos_id is not None:
        eos_idx = np.where(pitch[:valid_len] == int(eos_id))[0]
        if eos_idx.size > 0:
            valid_len = int(eos_idx[0]) + 1
    return int(valid_len)


def _load_ornament_json_overrides(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("ornament_json must be a JSON dict.")
    return d


def build_ornament_supervision_batch(
    x_cpu: torch.Tensor,  # [B,L,3] local-id CPU
    *,
    ornamenter: MusicOrnamenter,
    rng: np.random.Generator,
    pad_id_local: int,
    eos_id_local: int,
    rho_min: float,
    min_extra_tokens: int,
    max_tries: int,
    max_extra_tokens_cfg: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    return:
      x_orn_cpu: [B,L,3] long
      pi_cpu:    [B,L] long
    """
    x_cpu = x_cpu.contiguous()
    x_np = x_cpu.numpy()  # int64
    B, L, _ = x_np.shape

    x_orn = np.empty_like(x_np)
    pi = np.empty((B, L), dtype=np.int32)

    for b in range(B):
        xb = x_np[b]

        len_x = _seq_len_including_eos_local(xb, pad_id=pad_id_local, eos_id=eos_id_local)
        pad_budget = max(0, int(L) - int(len_x))

        # enforce rho_min by capping extra tokens:
        # rho = len_x / (len_x + extra) >= rho_min  => extra <= len_x*(1/rho_min - 1)
        extra_cap_rho = int(math.floor(float(len_x) * (1.0 / float(rho_min) - 1.0) + 1e-9))
        extra_cap_rho = max(0, extra_cap_rho)

        extra_cap = min(int(max_extra_tokens_cfg), int(pad_budget), int(extra_cap_rho))

        best = None
        last = None
        for _ in range(int(max_tries)):
            xa, pib = ornamenter.augment(xb, rng=rng, max_extra_tokens=extra_cap)

            valid = (pib != int(PI_PAD))
            len_orn = int(valid.sum())
            extra = int(len_orn - len_x)
            rho = float(len_x) / float(max(1, len_orn))

            last = (xa, pib, extra, rho)
            if (rho + 1e-9) >= float(rho_min) and extra >= int(min_extra_tokens):
                best = last
                break

        if best is None:
            best = last  # may have 0 negatives in worst case

        xa, pib, _, _ = best
        x_orn[b] = xa
        pi[b] = pib

    return torch.from_numpy(x_orn).long(), torch.from_numpy(pi).long()


# -------------------------
# Training loss
# -------------------------
def keep_bce_loss(
    keep_logits: torch.Tensor,      # [B,L]
    pi: torch.LongTensor,           # [B,L]
    src_tokens: torch.LongTensor,   # [B,L,3] (this is x_orn)
    src_mask: torch.Tensor,         # [B,L] bool
    *,
    special_n: int,
    use_dynamic_neg_weight: bool,
    neg_weight_cap: float,
) -> torch.Tensor:
    """
    keep=1 if pi>=0 (original)
    keep=0 if pi==-1 (inserted)
    Only compute loss on note tokens (pitch>=special_n) within src_mask.
    """
    pitch = src_tokens[..., 0]
    note_mask = src_mask & (pitch >= int(special_n))
    y_keep = (pi >= 0).to(torch.float32)

    loss_elem = F.binary_cross_entropy_with_logits(
        keep_logits.to(torch.float32),
        y_keep,
        reduction="none",
    )  # [B,L]

    if use_dynamic_neg_weight:
        with torch.no_grad():
            pos = (note_mask & (y_keep > 0.5)).sum().to(torch.float32)
            neg = (note_mask & (y_keep < 0.5)).sum().to(torch.float32)
            w_neg = (pos / neg.clamp_min(1.0)).clamp(min=1.0, max=float(neg_weight_cap))
        w = torch.where(y_keep < 0.5, w_neg, 1.0)
    else:
        w = 1.0

    denom = note_mask.to(torch.float32).sum().clamp_min(1.0)
    return (loss_elem * note_mask.to(torch.float32) * w).sum() / denom


# -------------------------
# Main
# -------------------------
def main(cfg: TrainKeepBaselineConfig) -> dict:

    if cfg.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed but use_wandb=True")
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))

    try:

        _check_file(cfg.train_npy, "train_npy")
        _check_file(cfg.vocab_pkl, "vocab_pkl")
        if not cfg.wo_bart_pretrain_init:
            _check_file(str(cfg.backbone_init_ckpt), "backbone_init_ckpt")

        set_seed(cfg.seed)
        os.makedirs(cfg.save_dir, exist_ok=True)

        need_otb_proxy = bool(
            cfg.eval_otb_proxy.enable
            and cfg.eval_otb.enable
            and str(cfg.eval_otb.test_bench_dir).strip()
            and os.path.isdir(cfg.eval_otb.test_bench_dir)
        )

        if cfg.skip_if_finished and _run_finished(cfg.save_dir, require_otb_proxy=need_otb_proxy):
            print(f"[Skip] finished run found: {cfg.save_dir}")
            return {
                "finished": True,
                "last_full_ckpt": _keep_last_full_ckpt(cfg.save_dir),
                "last_weights_ckpt": _keep_last_weights_ckpt(cfg.save_dir),
                "success_marker": _keep_success_marker(cfg.save_dir),
            }

        device = torch.device(cfg.device)
        print(f"[Device] {device}")

        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

        # -------------------------
        # Vocab + quant tables
        # -------------------------
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

        dur_q, dt_q = build_quantizers(quant_tables)
        dur_q = dur_q.to(device)
        dt_q = dt_q.to(device)

        music_bart_cfg = build_music_bart_cfg(cfg.model, vocab_cfg, dropout=cfg.train.dropout)

        # -------------------------
        # Backbone init
        # -------------------------
        backbone = MusicBartBackbone(cfg=music_bart_cfg).to(device)

        use_pretrain_init = (not cfg.wo_bart_pretrain_init)
        if use_pretrain_init:
            if not str(cfg.backbone_init_ckpt).strip():
                raise ValueError("wo_bart_pretrain_init=False but backbone_init_ckpt is empty.")
            ckpt = torch.load(cfg.backbone_init_ckpt, map_location="cpu")
            backbone.load_state_dict(ckpt["backbone_state_dict"], strict=True)
            print(f"[Init] load StageA backbone weights: {cfg.backbone_init_ckpt}")
        else:
            print("[Init] RANDOM backbone (w/o BART pretrain init)")

        set_dropout_p(backbone, cfg.train.dropout)

        # -------------------------
        # Model
        # -------------------------
        model = MusicSkeletonKeepBaseline(
            backbone=backbone,
            cfg=cfg.keep_model,
        ).to(device)

        print(f"[Model] params={count_params(model):,} | backbone={count_params(backbone):,}")

        # -------------------------
        # Data loader (train only)
        # -------------------------
        loader = make_dataloader(
            cfg.train_npy,
            batch_size=cfg.train.batch_size,
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
            max_samples=(cfg.train.limit_batches * cfg.train.batch_size
                        if cfg.train.limit_batches is not None else None),
        )

        # -------------------------
        # Ornamenter (online supervision generator)
        # -------------------------
        pad_id_local = int(vocab.global2local_pitch[int(vocab.pad_id)])
        bos_id_local = int(vocab.global2local_pitch[int(vocab.bos_id)])
        eos_id_local = int(vocab.global2local_pitch[int(vocab.eos_id)])

        orn_cfg = MusicOrnamentConfig(
            enable=True,
            p_apply=float(cfg.ornament.p_apply),
            pad_id=pad_id_local,
            bos_id=bos_id_local,
            eos_id=eos_id_local,
            max_extra_tokens=int(cfg.ornament.max_extra_tokens),
            quantization_tables=quant_tables,
        )

        if str(cfg.ornament.ornament_json).strip():
            overrides = _load_ornament_json_overrides(cfg.ornament.ornament_json)
            legal = set(MusicOrnamentConfig.__dataclass_fields__.keys())
            overrides = {k: v for k, v in overrides.items() if k in legal}
            # forbid changing special/qt here
            overrides.pop("pad_id", None)
            overrides.pop("bos_id", None)
            overrides.pop("eos_id", None)
            overrides.pop("quantization_tables", None)
            orn_cfg = replace(orn_cfg, **overrides)
            print(f"[Ornament] loaded overrides from: {cfg.ornament.ornament_json}")

        ornamenter = MusicOrnamenter(orn_cfg)
        orn_rng = np.random.default_rng(cfg.seed + 260406)  # arbitrary constant

        # -------------------------
        # OTB loaders + adapter
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

        otb_adapter = build_keep_adapter_from_model(
            model,
            hard_policy="topk_zlen",   # makes f1_hard meaningful vs O2B
            threshold=0.5,
            zero_special_scores=True,
        )

        def _run_bench_eval(bench_name: str, split: str, runner: OTBRunnerConfig, loader_eval, show_progress: bool = True):
            if loader_eval is None:
                return None
            metric_cfg = replace(
                runner.metrics,
                max_batches=(runner.max_valid_batches if split == "valid" else runner.max_test_batches),
                show_progress=show_progress,
                amp=(device.type == "cuda"),
            )
            prefix = f"BL_{split}/{bench_name}/"
            metrics = evaluate_ornament_to_backbone(
                otb_adapter,
                loader_eval,
                device=device,
                cfg=metric_cfg,
                prefix=prefix,
            )

            # stdout summary
            key_f1 = prefix + "f1_hard"
            key_f1_topk = prefix + "f1_hard_topk"
            key_gap = prefix + "gap_f1"
            key_ap = prefix + "ap_soft"
            key_rho = prefix + "rho"
            print(
                f"[OTB-{split}] "
                f"f1_hard={metrics.get(key_f1, float('nan')):.4f} "
                f"f1_topk={metrics.get(key_f1_topk, float('nan')):.4f} "
                f"gap_f1={metrics.get(key_gap, float('nan')):.4f} "
                f"ap_soft={metrics.get(key_ap, float('nan')):.4f} "
                f"rho_mean={metrics.get(key_rho, float('nan')):.4f}"
            )

            if cfg.use_wandb and wandb is not None:
                metrics.update({
                    "stage": "keep_baseline",
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                })
                wandb.log(metrics, step=global_step)
            return metrics
        
        def _run_otb_proxy_eval(split: str, loader_eval, show_progress: bool = True):
            if loader_eval is None or not cfg.eval_otb_proxy.enable:
                return None

            proxy_cfg = MusicPriorProxyEvalConfig(
                pos_per_beat=int(cfg.eval_otb_proxy.pos_per_beat),
                strong_period_beats=int(cfg.eval_otb_proxy.strong_period_beats),
                use_cummax_onset=bool(cfg.eval_otb_proxy.use_cummax_onset),
                max_batches=None,              # 和 Stage C 一样：final proxy 跑 full test set
                amp=(device.type == "cuda"),
                show_progress=show_progress,
                tau=otb_runner.metrics.tau,    # keep adapter 当前会忽略 tau；保留口径一致性
                token_key="x_orn",
                z_len_key="len_x",
            )

            proxy_prefix = f"BL_{split}/otb/proxy/"

            m_proxy = evaluate_music_prior_proxy(
                otb_adapter,
                loader_eval,
                device=device,
                duration_q=dur_q,
                dt_q=dt_q,
                cfg=proxy_cfg,
                prefix=proxy_prefix,
            )

            def _pg(k: str) -> float:
                return float(m_proxy.get(proxy_prefix + k, float("nan")))

            print(
                f"[OTB-{split} proxy] "
                f"JS(cnt)={_pg('pchist_js_cnt'):.4f} "
                f"JS(dur)={_pg('pchist_js_dur'):.4f} "
                f"JS_hard(dur)={_pg('pchist_js_hard_dur'):.4f} "
                f"Lift(strong)={_pg('lift_strong'):.3f} "
                f"Lift(dur)={_pg('lift_duration'):.3f} "
                f"Lift(ext)={_pg('lift_extrema'):.3f}"
            )

            if cfg.use_wandb and wandb is not None:
                m_proxy.update({
                    "stage": "keep_baseline",
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                })
                wandb.log(m_proxy, step=global_step)

            return m_proxy

        # -------------------------
        # TAVERN loader
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
        # Jiugong loader
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
        # Optim
        # -------------------------
        optim = AdamW(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            fused=(device.type == "cuda"),
        )
        scaler = torch.amp.GradScaler(enabled=(cfg.train.amp and device.type == "cuda"))

        total_steps = cfg.train.epochs * len(loader)
        warmup_steps = int(cfg.train.warmup_ratio * total_steps)
        base_lrs = [pg["lr"] for pg in optim.param_groups]

        # -------------------------
        # Train loop
        # -------------------------
        ckpt_dir = _keep_ckpt_dir(cfg.save_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        epoch = 0
        global_step = 0
        step = 0
        start_epoch = 1

        resume_path = _find_resume_ckpt(cfg)
        if resume_path is not None:
            start_epoch, step, global_step = load_train_state(
                resume_path,
                model=model,
                optim=optim,
                scaler=scaler,
            )
            epoch = start_epoch - 1

        if start_epoch > cfg.train.epochs:
            print("[Resume] all training epochs already completed; skip train loop and only run final save/eval.")

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            model.train()

            pbar = tqdm(loader, desc=f"[KeepBL] epoch {epoch}/{cfg.train.epochs}", dynamic_ncols=True, smoothing=0.0)
            interval_loss = 0.0
            interval_steps = 0

            for bi, x_tokens_cpu in enumerate(pbar):
                # --------------- (1) build supervision on CPU ---------------
                x_orn_cpu, pi_cpu = build_ornament_supervision_batch(
                    x_tokens_cpu,
                    ornamenter=ornamenter,
                    rng=orn_rng,
                    pad_id_local=pad_id_local,
                    eos_id_local=eos_id_local,
                    rho_min=cfg.ornament.rho_min,
                    min_extra_tokens=cfg.ornament.min_extra_tokens,
                    max_tries=cfg.ornament.max_tries,
                    max_extra_tokens_cfg=int(orn_cfg.max_extra_tokens),
                )

                # --------------- (2) to device ---------------
                x_orn = x_orn_cpu.to(device, non_blocking=True)
                pi = pi_cpu.to(device, non_blocking=True)
                src_mask = (x_orn[..., 0] != pad_id_local)

                # --------------- (3) LR schedule ---------------
                if cfg.train.use_lr_schedule:
                    factor = warmup_cosine_lr_factor(
                        step=step,
                        total_steps=total_steps,
                        warmup_steps=warmup_steps,
                        final_lr_ratio=cfg.train.final_lr_ratio,
                        init_lr_ratio=cfg.train.init_lr_ratio,
                    )
                    apply_lr_factor(optim, base_lrs, factor)

                optim.zero_grad(set_to_none=True)

                # --------------- (4) forward + loss ---------------
                with torch.amp.autocast(device_type=device.type, enabled=(cfg.train.amp and device.type == "cuda")):
                    keep_logits = model(src_tokens=x_orn, src_attention_mask=src_mask)  # [B,L]
                    loss = keep_bce_loss(
                        keep_logits=keep_logits,
                        pi=pi,
                        src_tokens=x_orn,
                        src_mask=src_mask,
                        special_n=int(model.special_n),
                        use_dynamic_neg_weight=cfg.train.use_dynamic_neg_weight,
                        neg_weight_cap=cfg.train.neg_weight_cap,
                    )

                # --------------- (5) backward + step ---------------
                scaler.scale(loss).backward()
                if cfg.train.grad_clip and cfg.train.grad_clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optim)
                scaler.update()

                step += 1
                global_step += 1

                # --------------- (6) logging ---------------
                loss_f = float(loss.detach().float().item())
                interval_loss += loss_f
                interval_steps += 1
                pbar.set_postfix(loss=f"{loss_f:.4f}", lr=f"{optim.param_groups[0]['lr']:.2e}")

                if cfg.use_wandb and wandb is not None and interval_steps >= cfg.train.log_interval:
                    wandb.log({
                        "stage": "keep_baseline",
                        "BL/batch_loss": interval_loss / max(1, interval_steps),
                        "BL/lr": optim.param_groups[0]["lr"],
                        "BL/epoch": epoch,
                        "BL/step": step,
                        "global_step": global_step,
                    }, step=global_step)
                    interval_loss = 0.0
                    interval_steps = 0

                # --------------- (7) periodic OTB valid eval ---------------
                if otb_valid_loader is not None and otb_runner.eval_interval_steps > 0:
                    if global_step % int(otb_runner.eval_interval_steps) == 0:
                        # free peak memory before eval
                        del x_orn, pi, keep_logits, loss
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        _run_bench_eval("otb", "valid", otb_runner, otb_valid_loader, show_progress=True)

                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

            # --------------- save per-epoch ckpt ---------------
            full_ckpt = {
                "stage": "keep_baseline",
                "epoch": int(epoch),
                "step": int(step),
                "global_step": int(global_step),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optim.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "rng_state": capture_rng_state(),
                "config": asdict(cfg),
                "vocab_pkl": cfg.vocab_pkl,
            }

            if epoch % max(1, cfg.train.save_every_epochs) == 0:
                save_ckpt(os.path.join(ckpt_dir, f"keep_epoch{epoch}.pt"), full_ckpt)

            save_ckpt(_keep_last_full_ckpt(cfg.save_dir), full_ckpt)

            # --------------- epoch-end OTB valid eval ---------------
            if otb_valid_loader is not None and otb_runner.eval_at_epoch_end:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                _run_bench_eval("otb", "valid", otb_runner, otb_valid_loader, show_progress=True)

        # --------------- save last ---------------
        last_path = _keep_last_weights_ckpt(cfg.save_dir)
        save_ckpt(last_path, {
            "stage": "keep_baseline",
            "epoch": int(epoch),
            "step": int(step),
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "config": asdict(cfg),
            "vocab_pkl": cfg.vocab_pkl,
        })

        # --------------- final test eval ---------------
        if otb_test_loader is not None:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            m = _run_bench_eval("otb", "test", otb_runner, otb_test_loader, show_progress=True)
            if m is not None:
                _atomic_json_dump(os.path.join(ckpt_dir, "otb_test_metrics.json"), m)

            if cfg.eval_otb_proxy.enable:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                m_proxy = _run_otb_proxy_eval("test", otb_test_loader, show_progress=True)
                if m_proxy is not None:
                    _atomic_json_dump(_keep_otb_test_proxy_json(cfg.save_dir), m_proxy)

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

        if jiugong_test_loader is not None:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            m = _run_bench_eval("jiugong", "test", jiugong_runner, jiugong_test_loader, show_progress=True)
            if m is not None:
                with open(os.path.join(ckpt_dir, "jiugong_test_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, indent=2)

        _atomic_json_dump(_keep_success_marker(cfg.save_dir), {
            "finished": True,
            "seed": int(cfg.seed),
            "wo_bart_pretrain_init": bool(cfg.wo_bart_pretrain_init),
            "save_dir": cfg.save_dir,
            "last_full_ckpt": _keep_last_full_ckpt(cfg.save_dir),
            "last_weights_ckpt": last_path,
            "epochs": int(cfg.train.epochs),
            "global_step": int(global_step),
        })

        print("[Done] keep baseline training finished.")
        print(f"[Saved last] {last_path}")

        return {
            "finished": True,
            "last_full_ckpt": _keep_last_full_ckpt(cfg.save_dir),
            "last_weights_ckpt": last_path,
            "success_marker": _keep_success_marker(cfg.save_dir),
        }

    finally:
        if cfg.use_wandb and wandb is not None:
            wandb.finish()



def _seed_corpus_dir(data_root: str, seed: int, corpus_dirname: str) -> str:
    return os.path.join(data_root, f"seed{seed}", corpus_dirname)


def _stageA_backbone_ckpt_for_seed(stageA_root: str, seed: int) -> str:
    """
    根据你之前 StageA 的实际输出结构改这里即可。
    下面这个写法假设你之前的 end2end 脚本把每个 seed 的 A/B 放在：
      {stageA_root}/seed{seed}/_AB_shared/stageA_pretrain/backbone_pretrained_last.pt
    """
    return os.path.join(
        stageA_root,
        f"seed{seed}",
        "_AB_shared",
        "stageA_pretrain",
        "backbone_pretrained_last.pt",
    )


def build_keep_cfg_for_seed_and_variant(
    base_cfg: TrainKeepBaselineConfig,
    *,
    seed: int,
    variant_tag: str,
    wo_bart_pretrain_init: bool,
    data_root: str,
    corpus_dirname: str,
    output_root: str,
    stageA_root: Optional[str],
) -> TrainKeepBaselineConfig:
    corpus_dir = _seed_corpus_dir(data_root, seed, corpus_dirname)

    backbone_init_ckpt = None
    if not wo_bart_pretrain_init:
        if stageA_root is None or str(stageA_root).strip() == "":
            raise ValueError("stageA_root is required when wo_bart_pretrain_init=False")
        backbone_init_ckpt = _stageA_backbone_ckpt_for_seed(stageA_root, seed)
        _check_file(backbone_init_ckpt, "backbone_init_ckpt")

    cfg = replace(
        base_cfg,
        seed=int(seed),
        train_npy=os.path.join(corpus_dir, "train.npy"),
        vocab_pkl=os.path.join(corpus_dir, "SimpleMono.pkl"),
        save_dir=os.path.join(output_root, f"seed{seed}", variant_tag),
        wandb_run_name=f"keep-{variant_tag}-seed{seed}",
        backbone_init_ckpt=backbone_init_ckpt,
        wo_bart_pretrain_init=wo_bart_pretrain_init,
        eval_otb=replace(
            base_cfg.eval_otb,
            valid_bench_dir=os.path.join(corpus_dir, "otb_bench", "valid_ood"),
            test_bench_dir=os.path.join(corpus_dir, "otb_bench", "test_ood"),
        ),
    )

    _check_file(cfg.train_npy, "train_npy")
    _check_file(cfg.vocab_pkl, "vocab_pkl")
    if cfg.eval_otb.enable:
        _check_dir(cfg.eval_otb.valid_bench_dir, "eval_otb.valid_bench_dir")
        _check_dir(cfg.eval_otb.test_bench_dir, "eval_otb.test_bench_dir")

    return cfg


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Multi-seed / multi-config runner
    # ------------------------------------------------------------
    SEEDS = [10101, 20202, 30303, 40404, 50505]

    DATA_ROOT = r".\preproc\output"
    CORPUS_DIRNAME = "skeletion_unsup_corpus_v260411_with_ornamented_split"

    # 这里请改成你 StageA ckpt 的父目录
    # 若你沿用了之前 end2end 脚本建议的目录结构，则会去找：
    #   {STAGEA_ROOT}\seed{seed}\_AB_shared\stageA_pretrain\backbone_pretrained_last.pt
    STAGEA_ROOT = r".\ckpt\2604-final"

    OUTPUT_ROOT = r".\ckpt\keep-baseline-2604-final"
    WANDB_PROJECT = "skel-keep-2604-final"

    VARIANTS = [
        ("w_bart_init", False),
        ("wo_bart_init", True),
    ]

    base_cfg = TrainKeepBaselineConfig(
        train_npy="",
        vocab_pkl="",
        save_dir="",
        device="cuda",
        num_workers=0,
        pin_memory=True,
        seed=0,

        use_wandb=True,
        wandb_project=WANDB_PROJECT,
        wandb_run_name=None,

        model=ModelHyperConfig(
            max_seq_len=514, d_embed=256, d_model=512,
            n_encoder_layers=6, n_decoder_layers=3, n_heads=8, d_ff=2048,
        ),

        backbone_init_ckpt=None,
        wo_bart_pretrain_init=False,

        auto_resume=True,
        resume_ckpt=None,
        skip_if_finished=True,

        keep_model=MusicSkeletonKeepBaselineConfig(
            head_hidden=256,
            head_dropout=0.1,
        ),

        train=BaselineTrainConfig(
            epochs=5,
            batch_size=64*4,
            lr=5e-4,
            weight_decay=0.01,
            grad_clip=1.0,
            dropout=0.10,
            amp=True,
            log_interval=50,
            use_lr_schedule=True,
            warmup_ratio=0.1,
            final_lr_ratio=0.2,
            init_lr_ratio=0.1,
            use_dynamic_neg_weight=True,
            neg_weight_cap=10.0,
            limit_batches=None,
            save_every_epochs=1,
        ),

        ornament=OrnamentSupervisionConfig(
            p_apply=1.0,
            rho_min=1.0 / 3.0,
            min_extra_tokens=1,
            max_tries=6,
            max_extra_tokens=256,
            ornament_json="",
        ),

        eval_otb=OTBRunnerConfig(
            enable=True,
            valid_bench_dir="",   # seed-specific; build_keep_cfg_for_seed_and_variant() 里填
            test_bench_dir="",    # seed-specific; build_keep_cfg_for_seed_and_variant() 里填
            batch_size=128,
            eval_interval_steps=200,
            eval_at_epoch_end=True,
            max_valid_batches=8,
            max_test_batches=None,
            metrics=OrnamentToBackboneEvalConfig(
                exclude_last_step=True,
                ignore_special_tokens=True,
                compute_cut_curve=True,
            ),
        ),

        # 这些不 split，不需要 seed
        eval_tavern=OTBRunnerConfig(
            enable=True,
            test_bench_dir=r".\preproc\output\tavern_silver_otb\test",
            batch_size=128,
            eval_interval_steps=0,
            eval_at_epoch_end=False,
            max_test_batches=None,
            metrics=OrnamentToBackboneEvalConfig(
                exclude_last_step=True,
                ignore_special_tokens=True,
                compute_cut_curve=True,
            ),
        ),

        eval_jiugong=OTBRunnerConfig(
            enable=True,
            test_bench_dir=r".\preproc\output\real_jiugongdacheng_otb_bench\test",
            batch_size=128,
            eval_interval_steps=0,
            eval_at_epoch_end=False,
            max_test_batches=None,
            metrics=OrnamentToBackboneEvalConfig(
                exclude_last_step=True,
                ignore_special_tokens=True,
                compute_cut_curve=True,
            ),
        ),
    )

    # -------------------------
    # Preflight: check seed data paths
    # -------------------------
    for seed in SEEDS:
        corpus_dir = _seed_corpus_dir(DATA_ROOT, seed, CORPUS_DIRNAME)
        _check_file(os.path.join(corpus_dir, "train.npy"), f"seed{seed}/train.npy")
        _check_file(os.path.join(corpus_dir, "SimpleMono.pkl"), f"seed{seed}/SimpleMono.pkl")
        _check_dir(os.path.join(corpus_dir, "otb_bench", "valid_ood"), f"seed{seed}/otb valid")
        _check_dir(os.path.join(corpus_dir, "otb_bench", "test_ood"), f"seed{seed}/otb test")

    # -------------------------
    # Run all seeds × variants
    # -------------------------
    for seed in SEEDS:
        for variant_tag, wo_bart_pretrain_init in VARIANTS:
            try:
                run_cfg = build_keep_cfg_for_seed_and_variant(
                    base_cfg,
                    seed=seed,
                    variant_tag=variant_tag,
                    wo_bart_pretrain_init=wo_bart_pretrain_init,
                    data_root=DATA_ROOT,
                    corpus_dirname=CORPUS_DIRNAME,
                    output_root=OUTPUT_ROOT,
                    stageA_root=STAGEA_ROOT,
                )
            except FileNotFoundError as e:
                print(f"[Skip] seed={seed}, variant={variant_tag}: {e}")
                continue

            print("=" * 80)
            print(f"[RUN] seed={seed} | variant={variant_tag}")
            print(f"      save_dir={run_cfg.save_dir}")
            print(f"      wo_bart_pretrain_init={run_cfg.wo_bart_pretrain_init}")
            print(f"      backbone_init_ckpt={run_cfg.backbone_init_ckpt}")
            print("=" * 80)

            try:
                main(run_cfg)
            finally:
                del run_cfg
                _cleanup_between_runs()

# usage
# python -m main.train_skeleton_keep_baseline
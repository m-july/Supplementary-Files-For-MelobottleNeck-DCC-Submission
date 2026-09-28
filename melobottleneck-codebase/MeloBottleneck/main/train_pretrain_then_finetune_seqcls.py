from __future__ import annotations

import os
import math
import shutil
from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from .vocab_utils import load_vocab_info
from .data import NpyMusicDataset, NpyMusicSequenceLabeledDataset, make_dataloader

from .config import BartDenoiseConfig
from .denoise import BartStyleDenoiser

from .models.bart import (
    MusicBartConfig,
    MusicBartBackboneConfig,
)
from .nn_modules import MusicBartBackbone, MusicBartForSeq2SeqLM, MusicBartForSequenceClassification

from .augment import MusicAugmentConfig

from .quantization import MusicQuantizationTables

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_float(x) -> float:
    if x is None:
        return 0.0
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def warmup_cosine_lr_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """
    返回一个乘到 base_lr 上的 factor：
      - warmup: 线性从 1/warmup_steps -> 1
      - cosine: 从 1 -> min_lr_ratio
    """
    if total_steps <= 1:
        return 1.0

    step = max(0, min(step, total_steps - 1))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))

    # warmup
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)

    # cosine decay
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return 1.0

    decay_step = step - warmup_steps  # 0 ... decay_steps-1
    t = decay_step / float(decay_steps - 1)  # 0 ... 1
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def apply_lr_factor(optim: torch.optim.Optimizer, base_lrs, factor: float):
    for pg, base_lr in zip(optim.param_groups, base_lrs):
        pg["lr"] = base_lr * factor

def warmup_corrupt_factor(
    epoch: int,            # 当前 epoch（从 1 开始计）
    total_epochs: int,     # 总训练轮数
    warmup_epochs: int,    # warmup 持续的 epoch 数
    begin_epoch: int,      # 从第几个 epoch 开始 warmup
    init_factor: float = 0.0,
    final_factor: float = 1.0,
) -> float:
    """
    计算一个可以乘到 base_corrupt_ratio 上的 factor，用于在训练早期逐步增加扰动比例。

    约定（与 warmup_cosine_lr_factor 的风格保持一致）：
      - epoch < begin_epoch:
          因子恒为 init_factor
      - begin_epoch <= epoch < begin_epoch + warmup_epochs:
          在 warmup_epochs 个 epoch 内，线性从 init_factor -> final_factor
          （第一个 warmup epoch 就比 init_factor 略大，最后一个 warmup epoch 为 final_factor）
      - epoch >= begin_epoch + warmup_epochs:
          因子恒为 final_factor
    """
    epoch -= 1 # 1-based -> 0-based

    # 退化情况：总共只有 0 或 1 个 epoch，就直接用 final_factor
    if total_epochs <= 1:
        return float(final_factor)

    # 将 epoch 限制在合法范围 [0, total_epochs - 1]
    epoch = max(0, min(epoch, total_epochs - 1))

    # begin_epoch 也裁剪到 [0, total_epochs]（允许等于 total_epochs，表示永远不进入 warmup）
    begin_epoch = max(0, min(begin_epoch, total_epochs))

    # warmup_epochs 至少为 0，且不能超过 [begin_epoch, total_epochs] 这段长度
    max_warmup = max(0, total_epochs - begin_epoch)
    warmup_epochs = max(0, min(warmup_epochs, max_warmup))

    # 如果没有 warmup（=0），则在 begin_epoch 之前用 init_factor，之后直接跳到 final_factor
    if warmup_epochs == 0:
        return float(init_factor if epoch < begin_epoch else final_factor)

    warmup_start = begin_epoch
    warmup_end = begin_epoch + warmup_epochs  # 不包含在 warmup 中的右边界

    # 1) warmup 之前
    if epoch < warmup_start:
        return float(init_factor)

    # 2) warmup 之中：线性从 init_factor -> final_factor
    if epoch < warmup_end:
        # 第一个 warmup epoch: warmup_step = 0
        # 最后一个 warmup epoch: warmup_step = warmup_epochs - 1
        warmup_step = epoch - warmup_start
        # 与你的 warmup_cosine_lr_factor 的线性部分一致：
        # 从 1 / warmup_epochs -> 1
        alpha = float(warmup_step + 1) / float(warmup_epochs)
        return float(init_factor + (final_factor - init_factor) * alpha)

    # 3) warmup 之后
    return float(final_factor)


@torch.no_grad()
def confusion_matrix_from_preds(
    preds: torch.LongTensor, labels: torch.LongTensor, num_classes: int
) -> torch.LongTensor:
    """
    preds, labels: [N]
    return: [C, C] where rows=true, cols=pred
    """
    preds = preds.view(-1)
    labels = labels.view(-1)
    k = labels * num_classes + preds
    cm = torch.bincount(k, minlength=num_classes * num_classes)
    return cm.view(num_classes, num_classes)


def macro_f1_from_confusion(cm: torch.Tensor, eps: float = 1e-12) -> float:
    """
    cm: [C, C], rows=true, cols=pred
    """
    cm = cm.to(torch.float64)
    tp = torch.diag(cm)
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # 如果你想忽略“valid/test 中根本没出现的类”，可用 support>0 做 mask
    return float(f1.mean().item())

@torch.no_grad()
def eval_seqcls(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    num_classes: int,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0
    total_correct = 0

    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for tokens, labels in loader:
        tokens = tokens.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(amp and device.type == "cuda")):
            out = model(tokens=tokens, labels=labels)

        loss = out.loss
        logits = out.logits

        bsz = tokens.size(0)
        total_loss += float(loss.item()) * bsz
        total_n += bsz

        preds = logits.argmax(dim=-1)
        total_correct += int((preds == labels).sum().item())

        cm += confusion_matrix_from_preds(preds.detach().cpu(), labels.detach().cpu(), num_classes)

    avg_loss = total_loss / max(1, total_n)
    acc = total_correct / max(1, total_n)
    macro_f1 = macro_f1_from_confusion(cm)

    return {
        "loss": avg_loss,
        "acc": acc,
        "macro_f1": macro_f1,
        "cm": cm,  # 可能用于 wandb confusion matrix
    }

# -------------------------
# Config
# -------------------------
@dataclass
class PretrainPhaseConfig:
    epochs: int = 4
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 20
    dropout: float = 0.15
    # --- lr scheduling ---
    use_lr_schedule: bool = True
    warmup_ratio: float = 0.05
    min_lr_ratio: float = 0.1  # final lr = lr * min_lr_ratio
    # --- noise curriculum (epoch-based) ---
    use_noise_curriculum: bool = True
    # masking
    masking_begin_epoch: int = 0
    masking_warmup_epochs: int = 4
    masking_init_factor: float = 0.2
    # deletion
    deletion_begin_epoch: int = 7
    deletion_warmup_epochs: int = 16
    deletion_init_factor: float = 0.0
    # rotation
    rotation_begin_epoch: int = 16
    rotation_warmup_epochs: int = 32
    rotation_init_factor: float = 0.0


@dataclass
class FinetunePhaseConfig:
    epochs: int = 30
    batch_size: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_interval: int = 20

    early_stop_patience: int = 5
    early_stop_min_delta: float = 0.0

    pooling: str = "awal"
    dropout: float = 0.1

    # --- discriminative LR + scheduling ---
    head_lr_mult: float = 2.5
    use_lr_schedule: bool = True
    warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.2


@dataclass
class ModelHyperConfig:
    max_seq_len: int = 514  # 应该 >= 你的 npy 序列长度(512+2=514)，除非保证实际上序列长度都小于你给出的数
    d_embed: int = 256
    d_model: int = 512
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 8
    d_ff: int = 2048


@dataclass
class TransferTrainConfig:
    # data
    pretrain_npy: str = ""
    train_npy: str = ""
    valid_npy: str = ""
    test_npy: str = ""
    train_labels_npy: str = ""
    valid_labels_npy: str = ""
    test_labels_npy: str = ""
    vocab_pkl: str = ""

    # runtime
    save_dir: str = "./ckpt_pretrain_then_finetune"
    device: str = "cuda"
    num_workers: int = 0  # Windows: 建议 0
    pin_memory: bool = True
    amp: bool = True
    seed: int = 1234

    # wandb
    use_wandb: bool = True
    wandb_project: str = "pianobart-pretrain-finetune"
    wandb_run_name: Optional[str] = None

    # sub configs
    model: ModelHyperConfig = field(default_factory=ModelHyperConfig)
    pretrain: PretrainPhaseConfig = field(default_factory=PretrainPhaseConfig)
    finetune: FinetunePhaseConfig = field(default_factory=FinetunePhaseConfig)


# -------------------------
# Main
# -------------------------
def main(cfg: TransferTrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.save_dir, "pretrain"), exist_ok=True)
    os.makedirs(os.path.join(cfg.save_dir, "finetune"), exist_ok=True)

    device = torch.device(cfg.device)
    print(f"[Device] {device}")

    # ============ wandb ============
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=asdict(cfg),
        )

    # ============ vocab ============
    vocab = load_vocab_info(cfg.vocab_pkl)
    music_bart_vocab_config = vocab.to_music_bart_vocab_config()

    quantization_tables = MusicQuantizationTables(
        special_n=vocab.special_n,
        duration_code_to_pos=vocab.duration_code_to_pos,
        duration_pos_to_code=vocab.duration_pos_to_code,
        deltatime_code_offset=vocab.deltatime_code_offset,
        deltatime_code_to_pos=vocab.deltatime_code_to_pos,
        deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    )

    print(f"[Vocab] n_pitch={vocab.n_pitch}, n_dur={vocab.n_duration}, n_dt={vocab.n_dt}")
    print(f"[Special] pad={vocab.pad_id}, bos={vocab.bos_id}, eos={vocab.eos_id}, mask={vocab.mask_id}")

    # ============ model configs ============
    backbone_cfg = MusicBartBackboneConfig(
        max_seq_len=cfg.model.max_seq_len,
        d_embed=cfg.model.d_embed,
        d_model=cfg.model.d_model,
        n_encoder_layers=cfg.model.n_encoder_layers,
        n_decoder_layers=cfg.model.n_decoder_layers,
        n_heads=cfg.model.n_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.pretrain.dropout,
    )
    music_bart_cfg = MusicBartConfig(vocab=music_bart_vocab_config, backbone=backbone_cfg)

    # ============ build backbone ============
    backbone = MusicBartBackbone(cfg=music_bart_cfg).to(device)

    if cfg.use_wandb:
        wandb.config.update({
            "num_params_backbone": count_params(backbone),
        }, allow_val_change=True)

    # ============================================================
    # Stage A: Pretrain (Denoising Seq2Seq LM)
    # ============================================================
    print("\n========== Stage A: Pretrain (Denoising Seq2Seq LM) ==========")

    # --- model ---
    pretrain_model = MusicBartForSeq2SeqLM(backbone=backbone).to(device)

    # --- denoiser ---
    denoiser = BartStyleDenoiser(config=BartDenoiseConfig(
        mask_token_id=vocab.mask_id,
        pad_token_id=vocab.pad_id,
    ))

    # --- load pretrain data ---
    pretrain_loader = make_dataloader(
        cfg.pretrain_npy,
        batch_size=cfg.pretrain.batch_size,
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
            transpose_sigma = 6.0,
            transpose_min = -12,
            transpose_max = 12,
            p_time_scale_2x = 0.30,
            p_time_scale_half = 0.30,
            quantization_tables=quantization_tables,
        ),
    )

    # --- optimizer ---
    pretrain_optim = AdamW(
        pretrain_model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.weight_decay,
    )
    pretrain_scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # --- pretrain lr schedule init ---
    pretrain_total_steps = cfg.pretrain.epochs * len(pretrain_loader)
    pretrain_warmup_steps = int(cfg.pretrain.warmup_ratio * pretrain_total_steps)
    pretrain_base_lrs = [pg["lr"] for pg in pretrain_optim.param_groups]
    pretrain_step = 0

    print(f"[PretrainSchedule] total_steps={pretrain_total_steps}, warmup_steps={pretrain_warmup_steps}, "
          f"peak_lr={pretrain_base_lrs[0]:.2e}, min_lr={pretrain_base_lrs[0]*cfg.pretrain.min_lr_ratio:.2e}")
    
    # --- corruption curriculum init ---
    base_denoise_config = replace(denoiser.config) # copy

    # ---
    global_step = 0
    for epoch in range(1, cfg.pretrain.epochs + 1):
        pretrain_model.train()
        running_loss = 0.0
        running_n = 0

        interval_loss = 0.0
        interval_steps = 0

        pbar = tqdm(pretrain_loader, desc=f"[Pretrain] Epoch {epoch}/{cfg.pretrain.epochs}", dynamic_ncols=True)
        for batch in pbar:
            tgt_tokens = batch.to(device, non_blocking=True)  # [B,L,3]
            tgt_attention_mask = (tgt_tokens[..., 0] != vocab.pad_id)  # [B,L] bool

            # --- step-based lr schedule (apply BEFORE optimizer step) ---
            if cfg.pretrain.use_lr_schedule:
                factor = warmup_cosine_lr_factor(
                    step=pretrain_step,
                    total_steps=pretrain_total_steps,
                    warmup_steps=pretrain_warmup_steps,
                    min_lr_ratio=cfg.pretrain.min_lr_ratio,
                )
                apply_lr_factor(pretrain_optim, pretrain_base_lrs, factor)

            # --- noise curriculum (epoch-based switches) ---
            if cfg.pretrain.use_noise_curriculum:
                denoiser.config.masking_noise_density = warmup_corrupt_factor(
                    epoch=epoch,
                    total_epochs=cfg.pretrain.epochs,
                    warmup_epochs=cfg.pretrain.masking_warmup_epochs,
                    begin_epoch=cfg.pretrain.masking_begin_epoch,
                    init_factor=cfg.pretrain.masking_init_factor,
                ) * base_denoise_config.masking_noise_density
                denoiser.config.deletion_prob = warmup_corrupt_factor(
                    epoch=epoch,
                    total_epochs=cfg.pretrain.epochs,
                    warmup_epochs=cfg.pretrain.deletion_warmup_epochs,
                    begin_epoch=cfg.pretrain.deletion_begin_epoch,
                    init_factor=cfg.pretrain.deletion_init_factor,
                ) * base_denoise_config.deletion_prob
                denoiser.config.rotation_prob = warmup_corrupt_factor(
                    epoch=epoch,
                    total_epochs=cfg.pretrain.epochs,
                    warmup_epochs=cfg.pretrain.rotation_warmup_epochs,
                    begin_epoch=cfg.pretrain.rotation_begin_epoch,
                    init_factor=cfg.pretrain.rotation_init_factor,
                ) * base_denoise_config.rotation_prob
                denoiser.config.enable_masking = denoiser.config.masking_noise_density > 1e-3
                denoiser.config.enable_deletion = denoiser.config.deletion_prob > 1e-3
                denoiser.config.enable_rotation = denoiser.config.rotation_prob > 1e-3

            # --- corrupt src tokens ---
            src_tokens, src_attention_mask = denoiser.corrupt(
                input_tokens=tgt_tokens,
                attention_mask=tgt_attention_mask,
                apply_masking=True,
                apply_deletion=True,
                apply_rotation=True,
            )
            src_tokens = src_tokens.to(device)
            src_attention_mask = src_attention_mask.to(device)

            pretrain_optim.zero_grad(set_to_none=True)

            # --- forward pass ---
            with torch.amp.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                out = pretrain_model(
                    src_tokens=src_tokens,
                    src_attention_mask=src_attention_mask,
                    tgt_tokens=tgt_tokens,
                    tgt_attention_mask=tgt_attention_mask,
                    attr_weights=(1.0, 1.0, 1.0),
                    ignore_index=-100,
                )

            # --- backward pass ---
            loss = out.loss

            pretrain_scaler.scale(loss).backward()
            if cfg.pretrain.grad_clip and cfg.pretrain.grad_clip > 0:
                pretrain_scaler.unscale_(pretrain_optim)
                torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), cfg.pretrain.grad_clip)

            pretrain_scaler.step(pretrain_optim)
            pretrain_scaler.update()
            pretrain_step += 1

            bsz = tgt_tokens.size(0)
            running_loss += float(loss.item()) * bsz
            running_n += bsz

            interval_loss += float(loss.item())
            interval_steps += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

            if cfg.use_wandb and interval_steps >= cfg.pretrain.log_interval:
                wandb.log({
                    "stage": "pretrain",
                    "pretrain/batch_loss": interval_loss / interval_steps,
                    "pretrain/lr": pretrain_optim.param_groups[0]["lr"],
                    "global_step": global_step,
                    "pretrain/epoch": epoch,
                }, step=global_step)
                interval_loss = 0.0
                interval_steps = 0

        epoch_loss = running_loss / max(1, running_n)
        print(f"[Pretrain] Epoch {epoch} loss = {epoch_loss:.6f}")

        if cfg.use_wandb:
            wandb.log({
                "stage": "pretrain",
                "pretrain/denoise/masking_noise_density": denoiser.config.masking_noise_density,
                "pretrain/denoise/deletion_prob": denoiser.config.deletion_prob,
                "pretrain/denoise/rotation_prob": denoiser.config.rotation_prob,
                "pretrain/epoch_loss": epoch_loss,
                "pretrain/epoch": epoch,
                "global_step": global_step,
            }, step=global_step)

        # save checkpoint (保存 LM 与 backbone 都行；这里都存)
        ckpt_path = os.path.join(cfg.save_dir, "pretrain", f"pretrain_epoch{epoch}.pt")
        torch.save({
            "stage": "pretrain",
            "epoch": epoch,
            "model_state_dict": pretrain_model.state_dict(),
            "backbone_state_dict": backbone.state_dict(),
            "optimizer_state_dict": pretrain_optim.state_dict(),
            "config": asdict(cfg),
            "vocab_pkl": cfg.vocab_pkl,
            "epoch_loss": epoch_loss,
        }, ckpt_path)
        print(f"[Saved] {ckpt_path}")

    # 额外保存一个“只含 backbone”的便捷文件
    backbone_ckpt = os.path.join(cfg.save_dir, "pretrain", "backbone_pretrained_last.pt")
    torch.save({
        "backbone_state_dict": backbone.state_dict(),
        "config": asdict(cfg),
        "vocab_pkl": cfg.vocab_pkl,
    }, backbone_ckpt)
    print(f"[Saved] {backbone_ckpt}")

    # 释放 LM head 显存（可选）
    del pretrain_model
    torch.cuda.empty_cache()

    # ============================================================
    # Stage B: Finetune (Sequence Classification, early stopping)
    # ============================================================
    print("\n========== Stage B: Finetune (Sequence Classification) ==========")

    # infer num_classes
    y_train = np.load(cfg.train_labels_npy, mmap_mode="r")
    y_valid = np.load(cfg.valid_labels_npy, mmap_mode="r")
    y_test = np.load(cfg.test_labels_npy, mmap_mode="r")
    num_classes = int(max(y_train.max(), y_valid.max(), y_test.max()) + 1)
    print(f"[Finetune] num_classes = {num_classes}")

    if cfg.use_wandb:
        wandb.config.update({"num_classes": num_classes}, allow_val_change=True)

    # adjust dropout rate
    for module in backbone.modules():
        if isinstance(module, nn.Dropout):
            module.p = cfg.finetune.dropout

    # --- model ---
    finetune_model = MusicBartForSequenceClassification(
        backbone=backbone,
        num_classes=num_classes,
        dropout=cfg.finetune.dropout,
        pooling=cfg.finetune.pooling,
    ).to(device)

    if cfg.use_wandb:
        wandb.config.update({
            "num_params_total_finetune_model": count_params(finetune_model),
        }, allow_val_change=True)

    # --- load finetune data ---
    train_loader = make_dataloader(
        cfg.train_npy, seq_labels_npy_path=cfg.train_labels_npy,
        batch_size=cfg.finetune.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=True,
        augment_seed=cfg.seed,
        augment_config=MusicAugmentConfig(quantization_tables=quantization_tables),
    )
    valid_loader = make_dataloader(
        cfg.valid_npy, seq_labels_npy_path=cfg.valid_labels_npy,
        batch_size=cfg.finetune.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=False,
    )
    test_loader = make_dataloader(
        cfg.test_npy, seq_labels_npy_path=cfg.test_labels_npy,
        batch_size=cfg.finetune.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=False,
    )

    # --- discriminative LR (backbone vs head) ---
    backbone_params, head_params = [], []
    for n, p in finetune_model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("backbone."):
            backbone_params.append(p)
        else:
            head_params.append(p)

    # --- optimizer ---
    finetune_optim = AdamW(
        [
            {"params": backbone_params, "lr": cfg.finetune.lr, "weight_decay": cfg.finetune.weight_decay},
            {"params": head_params, "lr": cfg.finetune.lr * cfg.finetune.head_lr_mult, "weight_decay": cfg.finetune.weight_decay},
        ]
    )
    finetune_scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # --- finetune lr schedule init ---
    finetune_total_steps = cfg.finetune.epochs * len(train_loader)
    finetune_warmup_steps = int(cfg.finetune.warmup_ratio * finetune_total_steps)
    finetune_base_lrs = [pg["lr"] for pg in finetune_optim.param_groups]
    finetune_step = 0

    print(f"[FinetuneSchedule] total_steps={finetune_total_steps}, warmup_steps={finetune_warmup_steps}, "
          f"backbone_peak_lr={finetune_base_lrs[0]:.2e}, head_peak_lr={finetune_base_lrs[1]:.2e}")

    # ---
    best_valid_macro_f1 = 0.0
    best_epoch = 0
    patience = 0
    best_ckpt_path = os.path.join(cfg.save_dir, "finetune", "finetune_best.pt")
    for epoch in range(1, cfg.finetune.epochs + 1):
        finetune_model.train()
        running_loss = 0.0
        running_n = 0

        interval_loss = 0.0
        interval_steps = 0

        pbar = tqdm(train_loader, desc=f"[Finetune] Epoch {epoch}/{cfg.finetune.epochs}", dynamic_ncols=True)
        for tokens, labels in pbar:
            tokens = tokens.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # --- step-based lr schedule ---
            if cfg.finetune.use_lr_schedule:
                factor = warmup_cosine_lr_factor(
                    step=finetune_step,
                    total_steps=finetune_total_steps,
                    warmup_steps=finetune_warmup_steps,
                    min_lr_ratio=cfg.finetune.min_lr_ratio,
                )
                apply_lr_factor(finetune_optim, finetune_base_lrs, factor)

            finetune_optim.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                out = finetune_model(tokens=tokens, labels=labels)

            loss = out.loss
            finetune_scaler.scale(loss).backward()

            if cfg.finetune.grad_clip and cfg.finetune.grad_clip > 0:
                finetune_scaler.unscale_(finetune_optim)
                torch.nn.utils.clip_grad_norm_(finetune_model.parameters(), cfg.finetune.grad_clip)

            finetune_scaler.step(finetune_optim)
            finetune_scaler.update()
            finetune_step += 1

            bsz = tokens.size(0)
            running_loss += float(loss.item()) * bsz
            running_n += bsz

            interval_loss += float(loss.item())
            interval_steps += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

            if cfg.use_wandb and interval_steps >= cfg.finetune.log_interval:
                wandb.log({
                    "stage": "finetune",
                    "finetune/train_batch_loss": interval_loss / interval_steps,
                    "finetune/lr_backbone": finetune_optim.param_groups[0]["lr"],
                    "finetune/lr_head": finetune_optim.param_groups[1]["lr"],
                    "finetune/epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)
                interval_loss = 0.0
                interval_steps = 0

        train_epoch_loss = running_loss / max(1, running_n)

        # ----- valid -----
        valid_metrics = eval_seqcls(
            finetune_model, valid_loader, device=device, amp=cfg.amp, num_classes=num_classes
        )
        valid_loss = float(valid_metrics["loss"])

        print(f"[Finetune] Epoch {epoch} train_loss={train_epoch_loss:.6f} | valid_loss={valid_loss:.6f} | valid_acc={valid_metrics['acc']:.4f} | valid_macro_f1={valid_metrics['macro_f1']:.4f}")

        if cfg.use_wandb:
            wandb.log({
                "stage": "finetune",
                "finetune/train_epoch_loss": train_epoch_loss,
                "finetune/valid_loss": valid_loss,
                "finetune/valid_acc": valid_metrics["acc"],
                "finetune/valid_macro_f1": valid_metrics["macro_f1"],
                "finetune/epoch": epoch,
                "global_step": global_step,
            }, step=global_step)

        # ----- early stopping -----
        valid_macro_f1 = valid_metrics["macro_f1"]
        improved = (valid_macro_f1 - best_valid_macro_f1) > cfg.finetune.early_stop_min_delta
        if improved:
            best_valid_macro_f1 = valid_macro_f1
            best_epoch = epoch
            patience = 0

            torch.save({
                "stage": "finetune",
                "epoch": epoch,
                "model_state_dict": finetune_model.state_dict(),
                "backbone_state_dict": backbone.state_dict(),
                "optimizer_state_dict": finetune_optim.state_dict(),
                "best_valid_macro_f1": best_valid_macro_f1,
                "num_classes": num_classes,
                "config": asdict(cfg),
                "vocab_pkl": cfg.vocab_pkl,
            }, best_ckpt_path)
            print(f"[Saved BEST] {best_ckpt_path}")

            if cfg.use_wandb:
                wandb.log({
                    "stage": "finetune",
                    "finetune/best_valid_macro_f1": best_valid_macro_f1,
                    "finetune/best_epoch": best_epoch,
                    "global_step": global_step,
                }, step=global_step)
        else:
            patience += 1
            print(f"[EarlyStop] no improvement. patience={patience}/{cfg.finetune.early_stop_patience}")

            if patience >= cfg.finetune.early_stop_patience:
                print(f"[EarlyStop] Stop at epoch {epoch}. Best epoch={best_epoch}, best_valid_macro_f1={best_valid_macro_f1:.6f}")
                break

    # ============================================================
    # Load best + Test evaluation
    # ============================================================
    print("\n========== Stage C: Test (Load best finetune checkpoint) ==========")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        finetune_model.load_state_dict(ckpt["model_state_dict"])
        print(f"[Load] {best_ckpt_path} (best_epoch={ckpt.get('epoch')}, best_valid_macro_f1={ckpt.get('best_valid_macro_f1')})")
    else:
        print("[Warn] best checkpoint not found, using last finetune weights.")

    test_metrics = eval_seqcls(
        finetune_model, test_loader, device=device, amp=cfg.amp, num_classes=num_classes
    )

    print(f"[Test] loss={test_metrics['loss']:.6f} | acc={test_metrics['acc']:.4f} | macro_f1={test_metrics['macro_f1']:.4f}")

    if cfg.use_wandb:
        # 可选：记录 confusion matrix（需要把 cm 转成 python list）
        cm = test_metrics["cm"].cpu().numpy()
        wandb.log({
            "stage": "test",
            "test/loss": test_metrics["loss"],
            "test/acc": test_metrics["acc"],
            "test/macro_f1": test_metrics["macro_f1"],
            "test/best_epoch": best_epoch,
            "test/best_valid_macro_f1": best_valid_macro_f1,
            "global_step": global_step,
        }, step=global_step)

        # 如果你想要 wandb 的可视化 confusion matrix：
        # wandb.log({
        #     "test/confusion_matrix": wandb.plot.confusion_matrix(
        #         probs=None,
        #         y_true=None,
        #         preds=None,
        #         class_names=[str(i) for i in range(num_classes)],
        #     )
        # }, step=global_step)
        #
        # 上面那个接口通常更依赖 y_true/preds 原始列表；
        # 如果你想我帮你把 preds/labels 全量收集并画图，也可以继续说。

        wandb.finish()

    # ============================================================
    # Cleanup: Remove all checkpoint files for this run
    # ============================================================
    print("\n========== Cleanup: Removing checkpoint files ==========")
    if os.path.exists(cfg.save_dir):
        try:
            shutil.rmtree(cfg.save_dir)
            print(f"[Cleaned] Removed directory: {cfg.save_dir}")
        except Exception as e:
            print(f"[Warn] Failed to remove directory {cfg.save_dir}: {e}")


if __name__ == "__main__":
    base_cfg = TransferTrainConfig(
        pretrain_npy=r"..\preproc\output\bart_pretrain_corpus_v260110\train.npy",
        train_npy=r"..\preproc\output\anthology_v251218_lyrics_included\train.npy",
        valid_npy=r"..\preproc\output\anthology_v251218_lyrics_included\valid.npy",
        test_npy=r"..\preproc\output\anthology_v251218_lyrics_included\test.npy",
        train_labels_npy=r"..\preproc\output\anthology_v251218_lyrics_included\train.labels.npy",
        valid_labels_npy=r"..\preproc\output\anthology_v251218_lyrics_included\valid.labels.npy",
        test_labels_npy=r"..\preproc\output\anthology_v251218_lyrics_included\test.labels.npy",

        # 这里必须填：与你生成这些 npy 时一致的 vocab pkl（包含 token2id / event2word / special）
        vocab_pkl=r"..\preproc\output\bart_pretrain_corpus_v260110\SimpleMono.pkl",

        save_dir=r"..\ckpt\pretrain_then_finetune_seqcls",

        # 你也可以按显存修改这些
        # max_seq_len 应该 >= 你的 npy 序列长度(512+2=514)，除非保证实际上序列长度都小于你给出的数
        model=ModelHyperConfig(
            max_seq_len=514, d_embed=256, d_model=512,
            n_encoder_layers=4, n_decoder_layers=4, n_heads=8, d_ff=2048
        ),
        pretrain=PretrainPhaseConfig(
            epochs=2, batch_size=(32+4)*6, log_interval=30, dropout=0.15,
            # lr scheduling
            lr=5e-4, use_lr_schedule=True, warmup_ratio=0.1, min_lr_ratio=0.3,
            # noise curriculum
            use_noise_curriculum=True,
            # weight decay
            weight_decay=0.01, grad_clip=1.0,
            # masking
            masking_begin_epoch = 0,
            masking_warmup_epochs = 4 * 3,
            masking_init_factor = 0.2,
            # deletion
            deletion_begin_epoch = 7 * 3,
            deletion_warmup_epochs = 16 * 3,
            deletion_init_factor = 0.0,
            # rotation
            rotation_begin_epoch = 16 * 3,
            rotation_warmup_epochs = 32 * 3,
            rotation_init_factor = 0.0,
        ),
        finetune=FinetunePhaseConfig(
            epochs=400, batch_size=(76)*6, log_interval=10000, pooling="awal", dropout=0.1,
            # lr scheduling
            lr=5e-5, head_lr_mult=2.5, use_lr_schedule=True, warmup_ratio=0.1, min_lr_ratio=0.3,
            # head lr multiplier
            weight_decay=0.01, grad_clip=1.0,
            # early stopping
            early_stop_patience=200, early_stop_min_delta=0.0,
        ),

        use_wandb=False,
        wandb_project="pianobart-pretrain-finetune-exp260111_2152",
        wandb_run_name=None,
        device="cuda",
        num_workers=0,
        pin_memory=True,
        amp=True,
        seed=12345,
    )

    # main(base_cfg)

    seeds = [12345, 23456, 34567, 45678, 56789, 67890]
    pretrain_epochs = [200] # 10~12 hours for 500 epochs

    for pretrain_epoch in pretrain_epochs:

        pretrain_cfg = replace(
            base_cfg.pretrain,
            epochs=pretrain_epoch,
        )

        for seed in seeds:
            run_cfg = replace(
                base_cfg,
                pretrain=pretrain_cfg,
                seed=seed,
                save_dir=os.path.join(
                    base_cfg.save_dir,
                    f"fast5_pretrain_epoch{pretrain_epoch}_seed{seed}",
                ),
                wandb_run_name=(
                    f"fast5_pretrain_epoch{pretrain_epoch}_seed{seed}"
                    if base_cfg.use_wandb else None
                ),
            )
            print("=" * 50)
            print(f"[RUN] pretrain_epoch={pretrain_epoch}, seed={seed}")

            main(run_cfg)

# python train_pretrain_then_finetune_seqcls.py


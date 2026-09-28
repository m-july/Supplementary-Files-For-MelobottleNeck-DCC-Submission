# Note: This task does not contains data augmentation.

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import wandb

from .data import make_dataloader
from .vocab_utils import load_vocab_info

from .config import BartDenoiseConfig
from .denoise import BartStyleDenoiser

from .models.bart import (
    MusicBartConfig, MusicBartBackboneConfig
)

from .nn_modules import MusicBartBackbone, MusicBartForSeq2SeqLM

from .quantization import MusicQuantizationTables

@dataclass
class TrainConfig:
    corpus_npy: str
    vocab_pkl: str
    save_dir: str = "./ckpt_pretrain"
    device: str = "cuda"

    epochs: int = 4
    batch_size: int = 128+96
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    max_seq_len: int = 2050 # check constant.py in data processing module
    d_embed: int = 256
    d_model: int = 512

    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 8
    d_ff: int = 2048

    num_workers: int = 0  # Windows 上建议默认 0
    pin_memory: bool = True

    amp: bool = True
    seed: int = 1234

    # wandb settings
    use_wandb: bool = False
    wandb_project: str = "pianobart-pretrain"
    wandb_run_name: Optional[str] = None
    log_interval: int = 20  # 每多少个 batch 汇报一次平均损失


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device(cfg.device)
    print(f"[Device] {device}")

    # === Initialize wandb ===
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=cfg.__dict__,
        )
        print(f"[Wandb] Initialized with project={cfg.wandb_project}")

    vocab = load_vocab_info(cfg.vocab_pkl)
    print(f"[Vocab] n_pitch={vocab.n_pitch}, n_dur={vocab.n_duration}, n_dt={vocab.n_dt}")
    print(f"[Special] pad={vocab.pad_id}, bos={vocab.bos_id}, eos={vocab.eos_id}, mask={vocab.mask_id}")

    # quantization_tables = MusicQuantizationTables(
    #     special_n=vocab.special_n,
    #     duration_code_to_pos=vocab.duration_code_to_pos,
    #     duration_pos_to_code=vocab.duration_pos_to_code,
    #     deltatime_code_offset=vocab.deltatime_code_offset,
    #     deltatime_code_to_pos=vocab.deltatime_code_to_pos,
    #     deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    # )

    music_bart_vocab_config = vocab.to_music_bart_vocab_config()

    # === Dataloader ===
    loader = make_dataloader(
        npy_path=cfg.corpus_npy,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
    )

    # === Dataloader TEST ===
    print("[Step] dataloader...", flush=True)
    print("  len(dataset) =", len(loader.dataset), flush=True)
    print("  len(loader)  =", len(loader), flush=True)

    print("[Step] fetch first batch...", flush=True)
    b0 = next(iter(loader))
    print("  b0.shape =", tuple(b0.shape), "dtype =", b0.dtype, flush=True)
    print("  b0 min/max =", int(b0.min()), int(b0.max()), flush=True)

    # 逐列检查范围（在 CPU 上做，能更早暴露问题）
    p = b0[..., 0]
    d = b0[..., 1]
    dt = b0[..., 2]
    print("  pitch min/max =", int(p.min()), int(p.max()), flush=True)
    print("  dur   min/max =", int(d.min()), int(d.max()), flush=True)
    print("  dt    min/max =", int(dt.min()), int(dt.max()), flush=True)
    # === Dataloader TEST END ===

    # === Denoiser ===
    # 这里你按自己的 BartDenoiseConfig 来构造
    # denoise_cfg = BartDenoiseConfig(...)
    # denoiser = BartStyleDenoiser(denoise_cfg)
    denoiser = BartStyleDenoiser(config=BartDenoiseConfig(
        mask_token_id=vocab.mask_id,  # 使用正确的 mask_id=4
        pad_token_id=vocab.pad_id,     # 使用正确的 pad_id=0
    ))

    music_bart_backbone_config = MusicBartBackboneConfig(
        max_seq_len=cfg.max_seq_len,
        d_embed=cfg.d_embed,
        d_model=cfg.d_model,
        n_encoder_layers=cfg.n_encoder_layers,
        n_decoder_layers=cfg.n_decoder_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
    )

    music_bart_config = MusicBartConfig(
        vocab=music_bart_vocab_config,
        backbone=music_bart_backbone_config,
    )

    print("[Step] build model...", flush=True)
    backbone = MusicBartBackbone(cfg=music_bart_config)
    model = MusicBartForSeq2SeqLM(backbone=backbone)
    print("[Step] move to device...", flush=True)
    backbone.to(device)
    model.to(device)
    print("[Step] moved.", flush=True)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # === Train ===
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0

        # For logging interval
        interval_loss = 0.0
        interval_pitch_loss = 0.0
        interval_duration_loss = 0.0
        interval_dt_loss = 0.0
        interval_steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.epochs}", dynamic_ncols=True)
        for batch in pbar:
            # batch: [B,L,3]
            tgt_tokens = batch.to(device, non_blocking=True)

            # clean attention mask（以 pitch 维 pad 判断；也可以改成三维全等 pad 判断）
            tgt_attention_mask = (tgt_tokens[..., 0] != vocab.pad_id)  # [B,L] bool

            # corrupt on-device（denoiser 内部 @no_grad）
            src_tokens, src_attention_mask = denoiser.corrupt(
                input_tokens=tgt_tokens,
                attention_mask=tgt_attention_mask,
                apply_masking=True,
                apply_deletion=True,
                apply_rotation=True,
            )
            src_tokens = src_tokens.to(device)
            src_attention_mask = src_attention_mask.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(cfg.amp and device.type == "cuda")):
                out = model(
                    src_tokens=src_tokens,
                    src_attention_mask=src_attention_mask,
                    tgt_tokens=tgt_tokens,
                    tgt_attention_mask=tgt_attention_mask,
                    attr_weights=(1.0, 1.0, 1.0),
                    ignore_index=-100,
                )

            loss = out.loss
            loss_dict = out.loss_dict

            scaler.scale(loss).backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            batch_loss = float(loss.item())
            running_loss += batch_loss
            n_steps += 1
            global_step += 1

            # Accumulate for interval logging
            interval_loss += batch_loss
            interval_pitch_loss += float(loss_dict.get('loss_pitch', torch.tensor(0.)).item())
            interval_duration_loss += float(loss_dict.get('loss_duration', torch.tensor(0.)).item())
            interval_dt_loss += float(loss_dict.get('loss_dt', torch.tensor(0.)).item())
            interval_steps += 1

            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "lp": f"{loss_dict.get('pitch_loss', torch.tensor(0.)).item():.4f}",
                "ld": f"{loss_dict.get('duration_loss', torch.tensor(0.)).item():.4f}",
                "ldt": f"{loss_dict.get('dt_loss', torch.tensor(0.)).item():.4f}",
            })

            # Log to wandb every log_interval steps
            if cfg.use_wandb and interval_steps >= cfg.log_interval:
                avg_interval_loss = interval_loss / interval_steps
                avg_interval_pitch = interval_pitch_loss / interval_steps
                avg_interval_duration = interval_duration_loss / interval_steps
                avg_interval_dt = interval_dt_loss / interval_steps

                wandb.log({
                    "train/batch_loss": avg_interval_loss,
                    "train/pitch_loss": avg_interval_pitch,
                    "train/duration_loss": avg_interval_duration,
                    "train/dt_loss": avg_interval_dt,
                    "epoch": epoch,
                    "global_step": global_step,
                })

                # Reset interval counters
                interval_loss = 0.0
                interval_pitch_loss = 0.0
                interval_duration_loss = 0.0
                interval_dt_loss = 0.0
                interval_steps = 0

        train_loss = running_loss / max(1, n_steps)
        print(f"[Epoch {epoch}] train_loss = {train_loss:.6f}")

        # Log epoch-level metrics to wandb
        if cfg.use_wandb:
            wandb.log({
                "train/epoch_loss": train_loss,
                "epoch": epoch,
            })

        # === Save checkpoint ===
        ckpt_path = os.path.join(cfg.save_dir, f"pretrain_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "train_loss": train_loss,
                "vocab_pkl": cfg.vocab_pkl,
                "config": cfg.__dict__,
            },
            ckpt_path,
        )
        print(f"[Saved] {ckpt_path}")

    # Finish wandb run
    if cfg.use_wandb:
        wandb.finish()
        print("[Wandb] Run finished")


if __name__ == "__main__":
    # 你也可以换成 argparse，这里先给最简可改版本
    cfg = TrainConfig(
        corpus_npy=r"J:\ACADEMIC\GRADPROJ\251215 PianoBART Learning\preproc\output\bart_pretrain_corpus_v260110\train.npy",
        vocab_pkl=r"J:\ACADEMIC\GRADPROJ\251215 PianoBART Learning\preproc\output\bart_pretrain_corpus_v260110\SimpleMono.pkl",
        save_dir=r"J:\ACADEMIC\GRADPROJ\251215 PianoBART Learning\ckpts\pretrain",
    )
    main(cfg)

# python ./train_pretrain.py

# cd bart && python train_pretrain.py

# python -c "import torch; print(torch.cuda.is_available())"
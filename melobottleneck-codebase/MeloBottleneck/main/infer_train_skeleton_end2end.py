# infer_train_skeleton_end2end.py
from __future__ import annotations

import argparse
import os
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .vocab_utils import load_vocab_info
from .augment import MusicAugmentConfig
from .quantization import MusicQuantizationTables

from .models.bart import MusicBartConfig, MusicBartBackboneConfig
from .nn_modules import MusicBartBackbone

# 和训练脚本保持一致的 import 路径：
from .models.skeleton.model import MusicSkeletonModelIII, MusicSkeletonIIIConfig
# 如果你项目里是 model_iii.py，则改为：
# from models.skeleton.model_iii import MusicSkeletonModelIII, MusicSkeletonIIIConfig

from .data import make_dataloader


def _filter_kwargs_for_dataclass(dc_cls, d: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(dc_cls)}
    return {k: v for k, v in (d or {}).items() if k in allowed}


def _resolve_path(path: str, base_dir: str) -> str:
    if not path:
        return path
    path = os.path.expanduser(path)
    if os.path.isabs(path) and os.path.exists(path):
        return path
    if os.path.exists(path):
        return path
    cand = os.path.join(base_dir, path)
    if os.path.exists(cand):
        return cand
    return path


def build_quant_tables_from_vocab(vocab) -> MusicQuantizationTables:
    return MusicQuantizationTables(
        special_n=vocab.special_n,
        duration_code_to_pos=vocab.duration_code_to_pos,
        duration_pos_to_code=vocab.duration_pos_to_code,
        deltatime_code_offset=vocab.deltatime_code_offset,
        deltatime_code_to_pos=vocab.deltatime_code_to_pos,
        deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    )


def _invert_global2local(global2local: np.ndarray, n_local: int) -> np.ndarray:
    """
    从 global_id → local_id 的映射数组，构建 local_id → global_id 的反向查找表。
    global2local[global_id] = local_id（-1 表示该 global_id 不在 local 词表中）
    返回: local2global[local_id] = global_id（-1 表示未映射）
    """
    g2l = np.asarray(global2local, dtype=np.int64)
    local2global = np.full(n_local, -1, dtype=np.int64)
    valid = (g2l >= 0) & (g2l < n_local)
    local2global[g2l[valid]] = np.nonzero(valid)[0]
    return local2global


def build_model_from_ckpt(
    ckpt_path: str,
    *,
    vocab_pkl: Optional[str] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Tuple[MusicSkeletonModelIII, Any, Dict[str, Any]]:
    """
    返回: (model, vocab, cfg_dict_from_ckpt)
    """
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg_dict = ckpt.get("config", {}) or {}
    model_h = cfg_dict.get("model", {}) or {}
    sk_h = cfg_dict.get("skeleton_model", {}) or {}

    # vocab path：优先用命令行覆盖，否则读 ckpt 里记录的路径
    vocab_pkl = vocab_pkl or ckpt.get("vocab_pkl") or cfg_dict.get("vocab_pkl")
    if vocab_pkl is None:
        raise ValueError("vocab_pkl is not provided and not found in checkpoint.")
    vocab_pkl = _resolve_path(str(vocab_pkl), base_dir=ckpt_dir)
    if not os.path.isfile(vocab_pkl):
        raise FileNotFoundError(f"vocab_pkl not found: {vocab_pkl}")

    vocab = load_vocab_info(vocab_pkl)
    vocab_cfg = vocab.to_music_bart_vocab_config()

    # backbone config：必须和训练时一致（尤其 max_seq_len / d_model / layers 等）
    backbone_cfg = MusicBartBackboneConfig(
        max_seq_len=int(model_h["max_seq_len"]),
        d_embed=int(model_h["d_embed"]),
        d_model=int(model_h["d_model"]),
        n_encoder_layers=int(model_h["n_encoder_layers"]),
        n_decoder_layers=int(model_h["n_decoder_layers"]),
        n_heads=int(model_h["n_heads"]),
        d_ff=int(model_h["d_ff"]),
        dropout=0.0,  # 推理时 eval 会关 dropout，p 不重要；但结构参数必须对
    )
    music_bart_cfg = MusicBartConfig(vocab=vocab_cfg, backbone=backbone_cfg)

    quant_tables = build_quant_tables_from_vocab(vocab)

    backbone = MusicBartBackbone(cfg=music_bart_cfg)
    sk_cfg = MusicSkeletonIIIConfig(**_filter_kwargs_for_dataclass(MusicSkeletonIIIConfig, sk_h))

    model = MusicSkeletonModelIII(
        backbone=backbone,
        quant=quant_tables,
        cfg=sk_cfg,
        lm_prior=None,              # inference 提取骨干音不需要 lm_prior
        use_bias_in_lm_head=True,
    ).to(device)

    state = ckpt.get("model_state_dict", None) or ckpt.get("state_dict", None)
    if state is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict' (or 'state_dict').")

    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, vocab, cfg_dict


def _find_eos_pos(
    tokens: torch.LongTensor,        # [B,L,3] local ids
    attn_mask: torch.Tensor,         # [B,L] bool/0-1
    eos_id: int,
) -> torch.LongTensor:
    pitch = tokens[..., 0]
    m = attn_mask.to(torch.bool)
    is_eos = (pitch == eos_id) & m
    has = is_eos.any(dim=1)
    first = is_eos.to(torch.long).argmax(dim=1)
    last_valid = m.to(torch.long).sum(dim=1).clamp_min(1) - 1
    return torch.where(has, first, last_valid)


def _greedy_tokens_from_logits_sync_special(
    *,
    pitch_logits: torch.Tensor,   # [B,L,Vp]
    dur_logits: torch.Tensor,     # [B,L,Vd]
    dt_logits: torch.Tensor,      # [B,L,Vt]
    special_n: int,
) -> torch.LongTensor:
    """
    Greedy decoding per position, with a constraint:
      - if pitch is special (<special_n), force dur/dt to be the same special id
      - if pitch is normal (>=special_n), forbid dur/dt from being special
    return: tokens [B,L,3] (local ids)
    """
    pitch_ids = pitch_logits.argmax(dim=-1)  # [B,L]

    if special_n <= 0:
        dur_ids = dur_logits.argmax(dim=-1)
        dt_ids = dt_logits.argmax(dim=-1)
        return torch.stack([pitch_ids, dur_ids, dt_ids], dim=-1)

    is_special = pitch_ids < int(special_n)   # [B,L]
    normal = ~is_special

    # forbid special ids on dur/dt where pitch is normal
    neg = torch.finfo(dur_logits.dtype).min
    dur2 = dur_logits
    dt2 = dt_logits
    if normal.any():
        dur2 = dur_logits.clone()
        dt2 = dt_logits.clone()
        dur2[..., :special_n] = dur2[..., :special_n].masked_fill(normal.unsqueeze(-1), neg)
        dt2[..., :special_n] = dt2[..., :special_n].masked_fill(normal.unsqueeze(-1), neg)

    dur_ids = dur2.argmax(dim=-1)
    dt_ids = dt2.argmax(dim=-1)

    # sync special
    dur_ids = torch.where(is_special, pitch_ids, dur_ids)
    dt_ids = torch.where(is_special, pitch_ids, dt_ids)

    return torch.stack([pitch_ids, dur_ids, dt_ids], dim=-1)


def _predict_dynamic_rho_and_zlen(
    *,
    model: MusicSkeletonModelIII,
    x_tokens: torch.LongTensor,      # [B,L,3] local ids
    x_mask: torch.Tensor,            # [B,L] bool/0-1
):
    """
    Replicate MusicSkeletonModelIII.forward() dynamic_rho part:
      encoder -> pooled -> rho_pred -> z_len_cont -> z_len_hard
    """
    cfg = model.cfg
    if not cfg.dynamic_rho:
        raise RuntimeError("Called _predict_dynamic_rho_and_zlen but cfg.dynamic_rho is False.")
    if model.rho_pred_head is None:
        raise RuntimeError("cfg.dynamic_rho=True but model.rho_pred_head is None (ckpt/code mismatch?).")

    eos_pos = _find_eos_pos(x_tokens, x_mask, eos_id=int(model.backbone.eos_id))
    L_x = eos_pos + 1  # [B]

    # encoder pooled
    src_embeds = model.backbone.embed(x_tokens)  # [B,L,D]
    enc_out = model.backbone.encode(src_embeds=src_embeds, src_attention_mask=x_mask)
    memory = enc_out.last_hidden_state          # [B,L,D]

    m = x_mask.to(memory.dtype).unsqueeze(-1)   # [B,L,1]
    pooled = (memory * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # [B,D]

    rmin, rmax = float(cfg.rho_min), float(cfg.rho_max)
    logits = model.rho_pred_head(pooled).squeeze(-1)                # [B]
    rho_pred = rmin + (rmax - rmin) * torch.sigmoid(logits)         # [B]

    z_len_cont = (L_x.to(torch.float32) * rho_pred).clamp(
        min=float(cfg.z_len_min),
        max=float(cfg.max_z_len),
    )  # [B] float

    z_len_hard = torch.ceil(z_len_cont).to(torch.long).clamp(
        min=int(cfg.z_len_min),
        max=int(cfg.max_z_len),
    )  # [B] long

    return rho_pred, z_len_cont, z_len_hard, L_x


def _fixed_rho_and_zlen(
    *,
    model: MusicSkeletonModelIII,
    x_tokens: torch.LongTensor,  # [B,L,3]
    x_mask: torch.Tensor,        # [B,L] bool/0-1
    rho_fixed: float,
):
    cfg = model.cfg
    device = x_tokens.device

    eos_pos = _find_eos_pos(x_tokens, x_mask, eos_id=int(model.backbone.eos_id))
    L_x = eos_pos + 1  # [B]

    rho_fixed = float(rho_fixed)
    rho_tensor = torch.full((x_tokens.size(0),), rho_fixed, device=device, dtype=torch.float32)

    z_len_cont = (L_x.to(torch.float32) * rho_fixed).clamp(
        min=float(getattr(cfg, "z_len_min", 1)),
        max=float(getattr(cfg, "max_z_len", 516)),
    )
    z_len_hard = torch.ceil(z_len_cont).to(torch.long).clamp(
        min=int(getattr(cfg, "z_len_min", 1)),
        max=int(getattr(cfg, "max_z_len", 516)),
    )
    return rho_tensor, z_len_cont, z_len_hard, L_x


def _prepend_bos_with_onset_anchor_for_export(
    *,
    model: MusicSkeletonModelIII,
    src_tokens: torch.LongTensor,      # [B,L,3] local ids
    z_tokens: torch.LongTensor,        # [B,T,3] local ids (the sequence you plan to SAVE)
    z_mask: torch.Tensor,              # [B,T] bool (valid steps of z_tokens BEFORE prepending)
    hard_indices: torch.LongTensor,    # [B,T] long (src indices aligned with z_tokens BEFORE prepending)
) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """
    Export-only fix for SimpleMono convention:
      BOS token exists, and BOS.dt encodes a clipped absolute onset anchor
      for the first NOTE token in the sequence.

    Returns:
      z_tokens2: [B,T+1,3]
      z_mask2:   [B,T+1]
      hard_idx2: [B,T+1] (BOS idx = 0)
      z_len2:    [B] (= old_z_len + 1, computed from z_mask)
    """
    device = src_tokens.device
    B, L, _ = src_tokens.shape
    T = z_tokens.size(1)

    # ---- compute absolute onset[pos] for every src position (same as SkeletonForwardExtend) ----
    dur_pos = model.duration_q.decode_local_to_pos(src_tokens[..., 1])  # [B,L] long
    dt_pos  = model.dt_q.decode_local_to_pos(src_tokens[..., 2])        # [B,L] long
    span_pos = dur_pos + dt_pos                                         # [B,L]
    onset = torch.cumsum(span_pos, dim=1) - span_pos                    # [B,L]

    # ---- find the first NOTE step in z (pitch >= special_n) ----
    special_n = int(model.duration_q.special_n)
    pitch_z = z_tokens[..., 0]
    is_note_step = z_mask.to(torch.bool) & (pitch_z >= special_n)       # [B,T]
    has_note = is_note_step.any(dim=1)                                  # [B]
    t0 = is_note_step.to(torch.long).argmax(dim=1)                      # [B] (0 if none)

    # ---- map that step to src index, then get its absolute onset ----
    src_idx0 = hard_indices.gather(1, t0[:, None]).squeeze(1)           # [B]
    src_idx0 = src_idx0.clamp(min=0, max=L - 1)

    anchor_pos = onset.gather(1, src_idx0[:, None]).squeeze(1)          # [B] long
    anchor_pos = torch.where(has_note, anchor_pos, anchor_pos.new_zeros((B,)))

    # ---- encode (with clipping) into dt local id ----
    bos_dt_local = model.dt_q.encode_pos_to_local(anchor_pos)           # [B]

    # ---- build BOS token: (bos, bos, dt_anchor) ----
    bos_id = int(model.backbone.bos_id)
    bos_tok = torch.full((B, 1, 3), bos_id, device=device, dtype=torch.long)
    bos_tok[:, 0, 2] = bos_dt_local

    # ---- prepend ----
    z_tokens2 = torch.cat([bos_tok, z_tokens], dim=1)                   # [B,T+1,3]
    z_mask2 = torch.cat(
        [torch.ones((B, 1), device=device, dtype=torch.bool), z_mask.to(torch.bool)],
        dim=1,
    )
    hard_idx2 = torch.cat(
        [torch.zeros((B, 1), device=device, dtype=torch.long), hard_indices],
        dim=1,
    )

    # sanity: must fit max_z_len
    max_z_len = int(model.cfg.max_z_len)
    if z_tokens2.size(1) > max_z_len:
        raise RuntimeError(
            f"[BOS-anchor export] z_len+1 exceeds max_z_len: "
            f"{z_tokens2.size(1)} > {max_z_len}. "
            f"Either disable this export fix or increase cfg.max_z_len."
        )

    z_len2 = z_mask2.to(torch.long).sum(dim=1)  # [B]
    return z_tokens2, z_mask2, hard_idx2, z_len2


@torch.inference_mode()
def run_inference(
    *,
    model: MusicSkeletonModelIII,
    vocab,
    input_npy: str,
    output_npy: str,
    batch_size: int = 64,
    rho_mode: str = "auto",
    rho: Optional[float] = None,
    skeleton_mode: str = "forward_extend",
    device: str = "cuda",
    num_workers: int = 0,
    pin_memory: bool = True,
    export_bos_anchor: bool = True,
    fix_recon_bos_anchor: bool = True,
    save_mask_npy: Optional[str] = None,
    save_len_npy: Optional[str] = None,
    save_indices_npy: Optional[str] = None,
    save_rho_npy: Optional[str] = None,
    save_z_len_cont_npy: Optional[str] = None,
    save_lx_npy: Optional[str] = None,
    save_recon_npy: Optional[str] = None,
):
    """
    输出:
      - output_npy: z_tokens_final padded to [N, max_z_len, 3]，token IDs 为 global IDs
                    与 preproc 阶段生成的 .npy 格式一致，可直接用 decode_npy_to_midi.py 解码
      - save_recon_npy: x_hat (reconstructor greedy TF) padded to [N, L_in, 3] global IDs
      - 可选：mask / len / hard_indices
    """

    rho_mode = str(rho_mode).lower().strip()
    ckpt_dyn = bool(getattr(model.cfg, "dynamic_rho", False))

    if rho_mode == "predict":
        if not ckpt_dyn:
            raise ValueError("rho_mode='predict' but checkpoint cfg.dynamic_rho is False.")
        use_dyn_rho = True
        rho_fixed = None

    elif rho_mode == "fixed":
        use_dyn_rho = False
        if rho is None:
            raise ValueError("rho_mode='fixed' but rho is None.")
        rho_fixed = float(rho)

    elif rho_mode == "auto":
        use_dyn_rho = ckpt_dyn and (rho is None)
        rho_fixed = None if use_dyn_rho else float(model.cfg.rho if rho is None else rho)

    else:
        raise ValueError(f"Unknown rho_mode: {rho_mode}")
    
    skeleton_mode = str(skeleton_mode).lower().strip()
    if skeleton_mode not in {"forward_extend", "hard_subseq"}:
        raise ValueError(
            f"Unknown skeleton_mode: {skeleton_mode}. "
            f"Expected one of ['forward_extend', 'hard_subseq']."
        )

    max_z_len = int(model.cfg.max_z_len)

    pad_id_global = int(vocab.pad_id)
    pad_id_local = int(model.backbone.pad_id)

    # 为了兼容 make_dataloader 的签名，这里即使 augment=False 也传一个 config
    quant_tables = build_quant_tables_from_vocab(vocab)
    aug_cfg = MusicAugmentConfig(quantization_tables=quant_tables)

    loader = make_dataloader(
        input_npy,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=False,
        augment_seed=0,
        augment_config=aug_cfg,
    )

    try:
        N = len(loader.dataset)
    except Exception:
        N = None

    x_arr = np.load(input_npy, mmap_mode="r")
    L_in = int(x_arr.shape[1])

    # 预分配（推荐，省内存碎片）；填充值用 global pad_id
    out_tokens = None
    out_mask = None
    out_len = None
    out_idx = None
    out_rho = None
    out_z_len_cont = None
    out_lx = None
    out_recon = None
    list_recon = None

    if N is not None:
        out_tokens = np.full((N, max_z_len, 3), pad_id_global, dtype=np.int64)
        if save_mask_npy is not None:
            out_mask = np.zeros((N, max_z_len), dtype=np.bool_)
        if save_len_npy is not None:
            out_len = np.zeros((N,), dtype=np.int32)
        if save_indices_npy is not None:
            out_idx = np.full((N, max_z_len), -1, dtype=np.int32)
        if save_rho_npy is not None:
            out_rho = np.zeros((N,), dtype=np.float32)
        if save_z_len_cont_npy is not None:
            out_z_len_cont = np.zeros((N,), dtype=np.float32)
        if save_lx_npy is not None:
            out_lx = np.zeros((N,), dtype=np.int32)
        if save_recon_npy is not None:
            out_recon = np.full((N, L_in, 3), pad_id_global, dtype=np.int64)
    else:
        list_rho, list_zlen_cont, list_lx, list_recon = [], [], [], []
        if save_recon_npy is not None:
            list_recon = []

    # 构建 local → global 反向查找表
    # DataLoader 将 global IDs 转为 local IDs 后喂给模型，模型输出也在 local 空间。
    # 保存前必须还原为 global IDs，才能被 decode_npy_to_midi.py 正确解码。
    local2global_pitch = _invert_global2local(vocab.global2local_pitch, vocab.n_pitch)
    local2global_dur = _invert_global2local(vocab.global2local_duration, vocab.n_duration)
    local2global_dt = _invert_global2local(vocab.global2local_dt, vocab.n_dt)

    write_pos = 0
    list_tokens, list_mask, list_len, list_idx = [], [], [], []

    pbar = tqdm(loader, desc="[Infer] extracting z_tokens_final", dynamic_ncols=True)
    for x_tokens in pbar:
        x_tokens = x_tokens.to(device, non_blocking=True)
        x_mask = (x_tokens[..., 0] != pad_id_local)

        if use_dyn_rho:
            rho_to_save, zlen_cont_to_save, z_len_hard, lx_to_save = _predict_dynamic_rho_and_zlen(
                model=model,
                x_tokens=x_tokens,
                x_mask=x_mask,
            )
            comp = model.compressor(
                src_tokens=x_tokens,
                src_attention_mask=x_mask,
                z_len=z_len_hard,
                tau=model.cfg.tau,
            )
            pbar.set_postfix(rho=f"{float(rho_to_save.mean().item()):.3f}", T=f"{int(z_len_hard.max().item())}")

        else:
            rho_to_save, zlen_cont_to_save, z_len_hard, lx_to_save = _fixed_rho_and_zlen(
                model=model,
                x_tokens=x_tokens,
                x_mask=x_mask,
                rho_fixed=rho_fixed,
            )
            comp = model.compressor(
                src_tokens=x_tokens,
                src_attention_mask=x_mask,
                z_len=z_len_hard,   # 关键：不用 rho=...，直接把长度喂进去
                tau=model.cfg.tau,
            )
            pbar.set_postfix(rho=f"{float(rho_fixed):.3f}", T=f"{int(z_len_hard.max().item())}")

        # -------------------------------------------------
        # choose which skeleton sequence to SAVE
        #   - forward_extend : old behavior
        #   - hard_subseq    : raw hard-selected subsequence before forward-extend
        # -------------------------------------------------
        recon_mode = str(getattr(model.cfg, "recon_z_mode", "final")).lower().strip()

        need_final_tokens = (
            skeleton_mode == "forward_extend"
            or (save_recon_npy is not None and recon_mode == "final")
        )

        z_tokens_final_local = None
        if need_final_tokens:
            z_tokens_final_local = model.forward_extend(
                src_tokens=x_tokens,
                z_tokens=comp.z_tokens_hard,
                hard_indices=comp.hard_indices,
                z_mask=comp.z_mask,
            )  # [B,T,3], local IDs

        if skeleton_mode == "hard_subseq":
            # raw hard subsequence, no forward-extend
            z_tokens_out_local = comp.z_tokens_hard
        else:
            # old behavior
            z_tokens_out_local = z_tokens_final_local

        if z_tokens_out_local is None:
            raise RuntimeError("z_tokens_out_local is None unexpectedly.")
        
        # -------------------------------------------------
        # NEW (export-only): restore SimpleMono BOS onset anchor
        # -------------------------------------------------
        if export_bos_anchor:
            z_tokens_out_local, z_mask_out, z_idx_out, z_len_out = _prepend_bos_with_onset_anchor_for_export(
                model=model,
                src_tokens=x_tokens,
                z_tokens=z_tokens_out_local,
                z_mask=comp.z_mask,
                hard_indices=comp.hard_indices,
            )
        else:
            z_mask_out = comp.z_mask
            z_idx_out = comp.hard_indices
            z_len_out = comp.z_len

        B, T, _ = z_tokens_out_local.shape
        z_tokens_local = z_tokens_out_local.cpu().numpy().astype(np.int64)
        z_mask_cpu = z_mask_out.cpu().numpy()
        z_len_cpu = z_len_out.cpu().numpy()
        z_idx_cpu = z_idx_out.cpu().numpy()

        # local → global 转换（向量化）
        z_tokens_cpu = np.empty_like(z_tokens_local)
        z_tokens_cpu[..., 0] = local2global_pitch[z_tokens_local[..., 0]]
        z_tokens_cpu[..., 1] = local2global_dur[z_tokens_local[..., 1]]
        z_tokens_cpu[..., 2] = local2global_dt[z_tokens_local[..., 2]]
        # padding 位置（z_mask == False）统一填 global pad_id
        z_tokens_cpu[~z_mask_cpu] = pad_id_global
            
        rho_cpu = rho_to_save.detach().cpu().numpy().astype(np.float32)
        zlen_cont_cpu = zlen_cont_to_save.detach().cpu().numpy().astype(np.float32)
        lx_cpu = lx_to_save.detach().cpu().numpy().astype(np.int32)

        # -----------------------------
        # Optional: reconstruct x_hat and save
        # -----------------------------
        if save_recon_npy is not None:
            # 1) choose reconstructor source embeds consistent with training cfg
            if recon_mode == "onfly":
                z_src_embeds = comp.z_embeds_st  # [B,T,D]
            else:
                # "final"
                if z_tokens_final_local is None:
                    raise RuntimeError("recon_z_mode='final' but z_tokens_final_local was not computed.")
                z_src_embeds = model.backbone.embed(z_tokens_final_local)

            z_attn = comp.z_mask.to(dtype=torch.long)  # [B,T]

            # 2) (optional) apply dynamic-rho soft gate like model.forward()
            if bool(getattr(model.cfg, "dynamic_rho", False)):
                Tz = z_src_embeds.size(1)
                t_ids = (torch.arange(Tz, device=z_src_embeds.device, dtype=torch.float32) + 0.5)[None, :]
                temp = max(float(getattr(model.cfg, "rho_gate_temp", 0.35)), 1e-3)
                gate = torch.sigmoid((zlen_cont_to_save[:, None].to(torch.float32) - t_ids) / temp)
                gate = gate.to(z_src_embeds.dtype) * comp.z_mask.to(z_src_embeds.dtype)
                z_src_embeds = z_src_embeds * gate.unsqueeze(-1)

            # 3) reconstructor forward (teacher-forcing) -> logits
            rec = model.reconstructor(
                src_embeds=z_src_embeds,
                src_attention_mask=z_attn,
                tgt_tokens=x_tokens,
                tgt_attention_mask=x_mask,
                attr_weights=tuple(getattr(model.cfg, "recon_attr_weights", (1.0, 1.0, 1.0))),
                token_weights=x_mask.to(torch.float32),     # 安全：即使 weighted-loss 实现不支持 None 也不会炸
                ignore_index=int(getattr(model.cfg, "ignore_index", -100)),
                return_hidden=False,
            )

            sN = int(getattr(model.backbone.cfg.vocab, "special_n", 0))
            x_hat_local = _greedy_tokens_from_logits_sync_special(
                pitch_logits=rec.logits.pitch,
                dur_logits=rec.logits.duration,
                dt_logits=rec.logits.dt,
                special_n=sN,
            )  # [B,L_in,3] local

            # pad outside x_mask
            pad_tok_local = x_hat_local.new_full((1, 1, 3), pad_id_local)
            x_hat_local = torch.where(x_mask.unsqueeze(-1), x_hat_local, pad_tok_local)

            if fix_recon_bos_anchor:
                # enforce BOS token and keep the original anchor from input x_tokens
                bos_id = int(model.backbone.bos_id)
                x_hat_local[:, 0, 0] = bos_id
                x_hat_local[:, 0, 1] = bos_id
                x_hat_local[:, 0, 2] = x_tokens[:, 0, 2]

            # to numpy + local->global
            x_hat_local_np = x_hat_local.detach().cpu().numpy().astype(np.int64)
            x_mask_np = x_mask.detach().cpu().numpy().astype(np.bool_)

            x_hat_global = np.empty_like(x_hat_local_np)
            x_hat_global[..., 0] = local2global_pitch[x_hat_local_np[..., 0]]
            x_hat_global[..., 1] = local2global_dur[x_hat_local_np[..., 1]]
            x_hat_global[..., 2] = local2global_dt[x_hat_local_np[..., 2]]
            x_hat_global[~x_mask_np] = pad_id_global

            # write
            if N is not None:
                out_recon[write_pos:write_pos + B, :, :] = x_hat_global
            else:
                list_recon.append(x_hat_global)

        if N is not None:
            out_tokens[write_pos:write_pos + B, :T, :] = z_tokens_cpu
            if out_mask is not None:
                out_mask[write_pos:write_pos + B, :T] = z_mask_cpu
            if out_len is not None:
                out_len[write_pos:write_pos + B] = z_len_cpu
            if out_idx is not None:
                out_idx[write_pos:write_pos + B, :T] = z_idx_cpu
            if out_rho is not None:
                out_rho[write_pos:write_pos + B] = rho_cpu
            if out_z_len_cont is not None:
                out_z_len_cont[write_pos:write_pos + B] = zlen_cont_cpu
            if out_lx is not None:
                out_lx[write_pos:write_pos + B] = lx_cpu
            write_pos += B
        else:
            list_tokens.append(z_tokens_cpu)
            if save_mask_npy is not None:
                list_mask.append(z_mask_cpu)
            if save_len_npy is not None:
                list_len.append(z_len_cpu)
            if save_indices_npy is not None:
                list_idx.append(z_idx_cpu)
            if save_rho_npy is not None:
                list_rho.append(rho_cpu)
            if save_z_len_cont_npy is not None:
                list_zlen_cont.append(zlen_cont_cpu)
            if save_lx_npy is not None:
                list_lx.append(lx_cpu)

    if N is None:
        out_tokens = np.concatenate(list_tokens, axis=0)
        if save_mask_npy is not None:
            out_mask = np.concatenate(list_mask, axis=0)
        if save_len_npy is not None:
            out_len = np.concatenate(list_len, axis=0)
        if save_indices_npy is not None:
            out_idx = np.concatenate(list_idx, axis=0)
        if save_rho_npy is not None:
            out_rho = np.concatenate(list_rho, axis=0)
        if save_z_len_cont_npy is not None:
            out_z_len_cont = np.concatenate(list_zlen_cont, axis=0)
        if save_lx_npy is not None:
            out_lx = np.concatenate(list_lx, axis=0)

    out_dir = os.path.dirname(os.path.abspath(output_npy))
    os.makedirs(out_dir, exist_ok=True)

    np.save(output_npy, out_tokens)
    print(f"[Saved] z_tokens_final -> {output_npy} | shape={out_tokens.shape}")

    if save_mask_npy is not None:
        np.save(save_mask_npy, out_mask)
        print(f"[Saved] z_mask -> {save_mask_npy} | shape={out_mask.shape}")

    if save_len_npy is not None:
        np.save(save_len_npy, out_len)
        print(f"[Saved] z_len -> {save_len_npy} | shape={out_len.shape}")

    if save_indices_npy is not None:
        np.save(save_indices_npy, out_idx)
        print(f"[Saved] hard_indices -> {save_indices_npy} | shape={out_idx.shape}")

    if save_rho_npy is not None:
        np.save(save_rho_npy, out_rho)
        print(f"[Saved] rho_pred -> {save_rho_npy} | shape={out_rho.shape}")

    if save_z_len_cont_npy is not None:
        np.save(save_z_len_cont_npy, out_z_len_cont)
        print(f"[Saved] z_len_cont -> {save_z_len_cont_npy} | shape={out_z_len_cont.shape}")

    if save_lx_npy is not None:
        np.save(save_lx_npy, out_lx)
        print(f"[Saved] L_x -> {save_lx_npy} | shape={out_lx.shape}")

    if save_recon_npy is not None:
        if N is None:
            out_recon = np.concatenate(list_recon, axis=0)

        recon_dir = os.path.dirname(os.path.abspath(save_recon_npy))
        os.makedirs(recon_dir, exist_ok=True)

        np.save(save_recon_npy, out_recon)
        print(f"[Saved] x_hat (reconstructed) -> {save_recon_npy} | shape={out_recon.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="StageC skeleton checkpoint (*.pt)")
    ap.add_argument("--input_npy", type=str, required=True)
    ap.add_argument("--output_npy", type=str, required=True)

    ap.add_argument("--vocab_pkl", type=str, default=None, help="Override vocab pkl path if ckpt records a stale path")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--rho_mode",
        type=str,
        default="auto",
        choices=["auto", "predict", "fixed"],
        help=(
            "Compression mode. "
            "auto: use rho predictor when ckpt.dynamic_rho=True and --rho not set; "
            "predict: force rho predictor; "
            "fixed: force fixed rho (use --rho if given else ckpt cfg.rho)."
        ),
    )
    ap.add_argument("--rho", type=float, default=None, help="Override compression ratio (default: cfg.skeleton_model.rho)")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Use strict=True when loading state_dict")
    ap.add_argument("--non_strict", action="store_true", help="Use strict=False when loading state_dict")

    ap.add_argument("--save_mask_npy", type=str, default=None)
    ap.add_argument("--save_len_npy", type=str, default=None)
    ap.add_argument("--save_indices_npy", type=str, default=None)
    ap.add_argument("--save_rho_npy", type=str, default=None)
    ap.add_argument("--save_z_len_cont_npy", type=str, default=None)
    ap.add_argument("--save_lx_npy", type=str, default=None)
    ap.add_argument("--save_recon_npy", type=str, default=None,
                    help="Optional: save reconstructor TF-greedy reconstruction x_hat to .npy (global IDs, [N,L,3])")

    ap.add_argument(
        "--skeleton_mode",
        type=str,
        default="forward_extend",
        choices=["forward_extend", "hard_subseq"],
        help=(
            "Which skeleton sequence to save. "
            "forward_extend: current behavior, save post-processed self-contained skeleton; "
            "hard_subseq: save raw hard-selected subsequence before forward-extend "
            "(recommended for retrieval ablation)."
        ),
    )

    args = ap.parse_args()
    strict = True
    if args.non_strict:
        strict = False
    if args.strict:
        strict = True

    model, vocab, _ = build_model_from_ckpt(
        args.ckpt,
        vocab_pkl=args.vocab_pkl,
        device=args.device,
        strict=strict,
    )

    run_inference(
        model=model,
        vocab=vocab,
        input_npy=args.input_npy,
        output_npy=args.output_npy,
        batch_size=args.batch_size,
        rho_mode=args.rho_mode,
        rho=args.rho,
        skeleton_mode=args.skeleton_mode,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        save_mask_npy=args.save_mask_npy,
        save_len_npy=args.save_len_npy,
        save_indices_npy=args.save_indices_npy,
        save_rho_npy=args.save_rho_npy,
        save_z_len_cont_npy=args.save_z_len_cont_npy,
        save_lx_npy=args.save_lx_npy,
        save_recon_npy=args.save_recon_npy,
    )


if __name__ == "__main__":
    main()

# python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\skel-260401\2def\stageC_skeleton\skeleton_last.pt --input_npy .\preproc\output\anthology_v251218_lyrics_included\test.npy --output_npy .\preproc\output\anthology_v251218_lyrics_included\test_skeleton.npy --save_recon_npy .\preproc\output\anthology_v251218_lyrics_included\test_recon.npy --device cuda --batch_size 128

# python -m main.infer_train_skeleton_end2end --ckpt .\ckpt\skel-260401\2def\stageC_skeleton\skeleton_last.pt --input_npy .\preproc\output\anthology_v251218_lyrics_included\test.npy --output_npy .\preproc\output\anthology_v251218_lyrics_included\test_skeleton.npy --save_recon_npy .\preproc\output\anthology_v251218_lyrics_included\test_recon.npy --device cuda --batch_size 128 --rho_mode fixed --rho 0.7
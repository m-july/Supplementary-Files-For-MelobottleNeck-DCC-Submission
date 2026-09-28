# main/infer_skeleton_keep_baseline.py
from __future__ import annotations

import argparse
import os
import math
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .vocab_utils import load_vocab_info
from .augment import MusicAugmentConfig
from .quantization import MusicQuantizationTables, build_quantizers

from .models.bart import MusicBartConfig, MusicBartBackboneConfig
from .nn_modules import MusicBartBackbone

from .models.skeleton.baseline_keep_encoder import (
    MusicSkeletonKeepBaseline,
    MusicSkeletonKeepBaselineConfig,
)

from .data import make_dataloader


# -------------------------
# Small utils (copy style from infer_train_skeleton_end2end.py)
# -------------------------
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
    g2l = np.asarray(global2local, dtype=np.int64)
    local2global = np.full(n_local, -1, dtype=np.int64)
    valid = (g2l >= 0) & (g2l < n_local)
    local2global[g2l[valid]] = np.nonzero(valid)[0]
    return local2global


def _find_eos_pos(
    tokens: torch.LongTensor,        # [B,L,3] local
    attn_mask: torch.Tensor,         # [B,L] bool/0-1
    eos_id: int,
) -> Tuple[torch.LongTensor, torch.Tensor]:
    pitch = tokens[..., 0]
    m = attn_mask.to(torch.bool)
    is_eos = (pitch == eos_id) & m
    has = is_eos.any(dim=1)
    first = is_eos.to(torch.long).argmax(dim=1)
    last_valid = m.to(torch.long).sum(dim=1).clamp_min(1) - 1
    eos_pos = torch.where(has, first, last_valid)
    return eos_pos, has


# -------------------------
# Build model from keep-baseline ckpt
# -------------------------
def build_keep_baseline_from_ckpt(
    ckpt_path: str,
    *,
    vocab_pkl: Optional[str] = None,
    device: str = "cuda",
    strict: bool = True,
) -> Tuple[MusicSkeletonKeepBaseline, Any, Dict[str, Any]]:
    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg_dict = ckpt.get("config", {}) or {}
    model_h = cfg_dict.get("model", {}) or {}
    keep_h = cfg_dict.get("keep_model", {}) or {}

    # vocab path
    vocab_pkl = vocab_pkl or ckpt.get("vocab_pkl") or cfg_dict.get("vocab_pkl")
    if vocab_pkl is None:
        raise ValueError("vocab_pkl is not provided and not found in checkpoint.")
    vocab_pkl = _resolve_path(str(vocab_pkl), base_dir=ckpt_dir)
    if not os.path.isfile(vocab_pkl):
        raise FileNotFoundError(f"vocab_pkl not found: {vocab_pkl}")

    vocab = load_vocab_info(vocab_pkl)
    vocab_cfg = vocab.to_music_bart_vocab_config()

    # backbone cfg must match training
    if not model_h:
        raise KeyError("Checkpoint config has no 'model' section; cannot rebuild backbone hyperparams.")
    backbone_cfg = MusicBartBackboneConfig(
        max_seq_len=int(model_h["max_seq_len"]),
        d_embed=int(model_h["d_embed"]),
        d_model=int(model_h["d_model"]),
        n_encoder_layers=int(model_h["n_encoder_layers"]),
        n_decoder_layers=int(model_h["n_decoder_layers"]),
        n_heads=int(model_h["n_heads"]),
        d_ff=int(model_h["d_ff"]),
        dropout=0.0,  # eval() disables dropout anyway
    )
    music_bart_cfg = MusicBartConfig(vocab=vocab_cfg, backbone=backbone_cfg)

    backbone = MusicBartBackbone(cfg=music_bart_cfg)

    keep_cfg = MusicSkeletonKeepBaselineConfig(
        **_filter_kwargs_for_dataclass(MusicSkeletonKeepBaselineConfig, keep_h)
    )

    model = MusicSkeletonKeepBaseline(backbone=backbone, cfg=keep_cfg).to(device)

    state = ckpt.get("model_state_dict", None) or ckpt.get("state_dict", None)
    if state is None:
        raise KeyError("Checkpoint does not contain 'model_state_dict' (or 'state_dict').")

    model.load_state_dict(state, strict=strict)
    model.eval()
    return model, vocab, cfg_dict


# -------------------------
# Inference
# -------------------------
@torch.inference_mode()
def run_inference_keep_baseline(
    *,
    model: MusicSkeletonKeepBaseline,
    vocab,
    input_npy: str,
    output_npy: str,
    batch_size: int = 128,
    rho_mode: str = "auto",          # auto/predict/fixed
    rho: Optional[float] = None,     # used when fixed
    threshold: float = 0.5,          # used when predict
    device: str = "cuda",
    num_workers: int = 0,
    pin_memory: bool = True,
    export_bos_anchor: bool = True,  # you want this on
    min_keep_notes: int = 1,         # avoid empty skeleton
    save_mask_npy: Optional[str] = None,
    save_len_npy: Optional[str] = None,
    save_indices_npy: Optional[str] = None,
    dt_mode: str = "relink",   # "raw" or "relink"
):
    rho_mode = str(rho_mode).lower().strip()
    if rho_mode not in {"auto", "predict", "fixed"}:
        raise ValueError("rho_mode must be in {auto,predict,fixed}")

    if rho_mode == "auto":
        rho_mode = "fixed" if (rho is not None) else "predict"

    if rho_mode == "fixed":
        if rho is None:
            raise ValueError("rho_mode='fixed' but rho is None.")
        rho = float(rho)
        if not (0.0 < rho <= 1.0):
            raise ValueError(f"rho must be in (0,1], got {rho}")

    threshold = float(threshold)
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0,1), got {threshold}")
    
    dt_mode = str(dt_mode).lower().strip()
    if dt_mode not in {"raw", "relink"}:
        raise ValueError("dt_mode must be 'raw' or 'relink'")

    device_t = torch.device(device)

    # --------- load input to get N ---------
    x_arr = np.load(input_npy, mmap_mode="r")
    if x_arr.ndim != 3 or x_arr.shape[-1] != 3:
        raise ValueError(f"input_npy must be [N,L,3], got {x_arr.shape}")
    N = int(x_arr.shape[0])

    max_len = int(model.backbone.cfg.backbone.max_seq_len)

    pad_id_global = int(vocab.pad_id)
    pad_id_local = int(vocab.global2local_pitch[int(vocab.pad_id)])
    bos_id_local = int(vocab.global2local_pitch[int(vocab.bos_id)])
    eos_id_local = int(vocab.global2local_pitch[int(vocab.eos_id)])

    # --------- quantizers for BOS anchor fix ---------
    quant_tables = build_quant_tables_from_vocab(vocab)
    dur_q, dt_q = build_quantizers(quant_tables)
    dur_q = dur_q.to(device_t)
    dt_q = dt_q.to(device_t)

    dt0_local = dt_q.encode_pos_to_local(torch.zeros((1,), device=device_t, dtype=torch.long))[0]  # scalar tensor

    # for dataloader signature
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

    # local->global
    local2global_pitch = _invert_global2local(vocab.global2local_pitch, vocab.n_pitch)
    local2global_dur = _invert_global2local(vocab.global2local_duration, vocab.n_duration)
    local2global_dt = _invert_global2local(vocab.global2local_dt, vocab.n_dt)

    # outputs
    out_tokens = np.full((N, max_len, 3), pad_id_global, dtype=np.int64)
    out_mask = np.zeros((N, max_len), dtype=np.bool_) if save_mask_npy else None
    out_len = np.zeros((N,), dtype=np.int32) if save_len_npy else None
    out_idx = np.full((N, max_len), -1, dtype=np.int32) if save_indices_npy else None

    special_n = int(model.special_n)

    write_pos = 0
    pbar = tqdm(loader, desc="[Infer KeepBL] extracting skeleton", dynamic_ncols=True)

    amp = (device_t.type == "cuda")

    for x_tokens_cpu in pbar:
        x_tokens = x_tokens_cpu.to(device_t, non_blocking=True)
        B, L, _ = x_tokens.shape

        if L > max_len:
            raise RuntimeError(f"Input seq len {L} exceeds model max_seq_len {max_len}.")

        x_mask = (x_tokens[..., 0] != pad_id_local)

        eos_pos, has_eos = _find_eos_pos(x_tokens, x_mask, eos_id=eos_id_local)

        # note candidates: valid & before EOS & pitch is normal note
        l_ids = torch.arange(L, device=device_t)[None, :].expand(B, L)
        pitch = x_tokens[..., 0]
        note_mask = x_mask & (l_ids < eos_pos[:, None]) & (pitch >= special_n)

        # keep logits
        with torch.amp.autocast(device_type=device_t.type, enabled=amp):
            keep_logits = model(src_tokens=x_tokens, src_attention_mask=x_mask)  # [B,L]
        keep_logits = keep_logits.to(torch.float32)

        # compute absolute onset[pos] for BOS anchor fix
        dur_pos = dur_q.decode_local_to_pos(x_tokens[..., 1])  # [B,L]
        dt_pos = dt_q.decode_local_to_pos(x_tokens[..., 2])    # [B,L]
        span_pos = dur_pos + dt_pos
        onset = torch.cumsum(span_pos, dim=1) - span_pos       # [B,L]

        # build padded local output for this batch
        z_local = x_tokens.new_full((B, max_len, 3), pad_id_local)
        z_mask = torch.zeros((B, max_len), device=device_t, dtype=torch.bool)
        z_idx = torch.full((B, max_len), -1, device=device_t, dtype=torch.long)
        z_len = torch.zeros((B,), device=device_t, dtype=torch.long)

        for b in range(B):
            cand = torch.nonzero(note_mask[b], as_tuple=False).squeeze(1)  # [Nc]
            Nc = int(cand.numel())

            if Nc == 0:
                idx_keep = cand  # empty
            else:
                scores = keep_logits[b, cand]  # [Nc]

                if rho_mode == "fixed":
                    L_x = int(eos_pos[b].item()) + 1  # include EOS
                    # z_len includes EOS but excludes BOS; keep_count = z_len-1
                    z_len_target = int(math.ceil(float(L_x) * float(rho)))
                    z_len_target = max(2, min(z_len_target, Nc + 1))
                    k = int(z_len_target - 1)
                    k = max(int(min_keep_notes), k)

                    topk = torch.topk(scores, k=min(k, Nc), largest=True).indices
                    idx_keep = cand[topk].sort().values

                else:
                    # predict mode: threshold on prob
                    prob = torch.sigmoid(scores)
                    idx_keep = cand[prob >= threshold]
                    if idx_keep.numel() < int(min_keep_notes):
                        # force at least one
                        top1 = scores.argmax().view(1)
                        idx_keep = cand[top1]
                    idx_keep = idx_keep.sort().values

            # anchor = onset of first kept note
            if export_bos_anchor and idx_keep.numel() > 0:
                first_idx = idx_keep[0].clamp(min=0, max=L - 1)
                anchor_pos = onset[b, first_idx]  # scalar long
                bos_dt_local = dt_q.encode_pos_to_local(anchor_pos)
            else:
                # fallback: keep original BOS anchor if exists, else 0
                bos_dt_local = x_tokens[b, 0, 2] if (L > 0) else torch.tensor(0, device=device_t)

            # BOS token: (bos,bos,dt_anchor)
            bos_tok = x_tokens.new_full((3,), bos_id_local)
            bos_tok[2] = bos_dt_local.to(dtype=bos_tok.dtype)

            # EOS token
            if bool(has_eos[b].item()):
                eos_tok = x_tokens[b, int(eos_pos[b].item())]
                eos_src_idx = int(eos_pos[b].item())
            else:
                eos_tok = x_tokens.new_full((3,), eos_id_local)
                eos_src_idx = int(eos_pos[b].item())

            if idx_keep.numel() > 0:
                note_tok = x_tokens[b, idx_keep].clone()  # [K,3] clone because we will edit dt

                # ---- NEW: relink dt to preserve absolute onsets of kept notes ----
                if dt_mode == "relink":
                    K = int(idx_keep.numel())
                    if K >= 2:
                        cur = idx_keep[:-1]   # [K-1]
                        nxt = idx_keep[1:]    # [K-1]

                        # dt_pos_new = onset[nxt] - (onset[cur] + dur[cur])
                        dt_pos_new = onset[b, nxt] - (onset[b, cur] + dur_pos[b, cur])  # [K-1], signed long

                        dt_local_new = dt_q.encode_pos_to_local(dt_pos_new)             # [K-1], local ids
                        note_tok[:-1, 2] = dt_local_new.to(note_tok.dtype)

                    # last kept note: dt = 0 (SimpleMono convention)
                    note_tok[-1, 2] = dt0_local.to(note_tok.dtype)

                seq = torch.cat([bos_tok[None, :], note_tok, eos_tok[None, :]], dim=0)
                idx_seq = torch.cat(
                    [
                        torch.zeros((1,), device=device_t, dtype=torch.long),  # BOS src idx = 0
                        idx_keep.to(torch.long),
                        torch.tensor([eos_src_idx], device=device_t, dtype=torch.long),
                    ],
                    dim=0,
                )
            else:
                seq = torch.cat([bos_tok[None, :], eos_tok[None, :]], dim=0)
                idx_seq = torch.tensor([0, eos_src_idx], device=device_t, dtype=torch.long)

            tlen = int(seq.size(0))
            if tlen > max_len:
                raise RuntimeError(f"Output len {tlen} > max_len {max_len} (unexpected).")

            z_local[b, :tlen, :] = seq
            z_mask[b, :tlen] = True
            z_idx[b, :tlen] = idx_seq
            z_len[b] = tlen

        # local -> global (vectorized on CPU numpy)
        z_local_np = z_local.cpu().numpy().astype(np.int64)
        z_mask_np = z_mask.cpu().numpy().astype(np.bool_)
        z_idx_np = z_idx.cpu().numpy().astype(np.int32)
        z_len_np = z_len.cpu().numpy().astype(np.int32)

        z_global_np = np.empty_like(z_local_np)
        z_global_np[..., 0] = local2global_pitch[z_local_np[..., 0]]
        z_global_np[..., 1] = local2global_dur[z_local_np[..., 1]]
        z_global_np[..., 2] = local2global_dt[z_local_np[..., 2]]
        z_global_np[~z_mask_np] = pad_id_global

        out_tokens[write_pos:write_pos + B, :, :] = z_global_np
        if out_mask is not None:
            out_mask[write_pos:write_pos + B, :] = z_mask_np
        if out_len is not None:
            out_len[write_pos:write_pos + B] = z_len_np
        if out_idx is not None:
            out_idx[write_pos:write_pos + B, :] = z_idx_np

        write_pos += B

    os.makedirs(os.path.dirname(os.path.abspath(output_npy)), exist_ok=True)
    np.save(output_npy, out_tokens)
    print(f"[Saved] keep-baseline skeleton -> {output_npy} | shape={out_tokens.shape}")

    if save_mask_npy is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_mask_npy)), exist_ok=True)
        np.save(save_mask_npy, out_mask)
        print(f"[Saved] z_mask -> {save_mask_npy} | shape={out_mask.shape}")

    if save_len_npy is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_len_npy)), exist_ok=True)
        np.save(save_len_npy, out_len)
        print(f"[Saved] z_len -> {save_len_npy} | shape={out_len.shape}")

    if save_indices_npy is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_indices_npy)), exist_ok=True)
        np.save(save_indices_npy, out_idx)
        print(f"[Saved] src_indices -> {save_indices_npy} | shape={out_idx.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="keep baseline checkpoint (*.pt), e.g. keep_last.pt")
    ap.add_argument("--input_npy", type=str, required=True)
    ap.add_argument("--output_npy", type=str, required=True)

    ap.add_argument("--vocab_pkl", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument(
        "--rho_mode",
        type=str,
        default="auto",
        choices=["auto", "predict", "fixed"],
        help="auto: fixed if --rho is set else predict; predict: threshold; fixed: topk by rho",
    )
    ap.add_argument("--rho", type=float, default=None, help="Used when rho_mode=fixed")
    ap.add_argument("--threshold", type=float, default=0.5, help="Used when rho_mode=predict")
    ap.add_argument("--min_keep_notes", type=int, default=1)

    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--non_strict", action="store_true")

    ap.add_argument("--save_mask_npy", type=str, default=None)
    ap.add_argument("--save_len_npy", type=str, default=None)
    ap.add_argument("--save_indices_npy", type=str, default=None)

    ap.add_argument(
        "--dt_mode",
        type=str,
        default="relink",
        choices=["raw", "relink"],
        help="raw: keep original dt; relink: recompute dt so kept-note onsets match the source sequence",
    )

    args = ap.parse_args()
    strict = True
    if args.non_strict:
        strict = False
    if args.strict:
        strict = True

    model, vocab, _ = build_keep_baseline_from_ckpt(
        args.ckpt,
        vocab_pkl=args.vocab_pkl,
        device=args.device,
        strict=strict,
    )

    run_inference_keep_baseline(
        model=model,
        vocab=vocab,
        input_npy=args.input_npy,
        output_npy=args.output_npy,
        batch_size=args.batch_size,
        rho_mode=args.rho_mode,
        rho=args.rho,
        threshold=args.threshold,
        min_keep_notes=args.min_keep_notes,
        device=args.device,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        save_mask_npy=args.save_mask_npy,
        save_len_npy=args.save_len_npy,
        save_indices_npy=args.save_indices_npy,
    )


if __name__ == "__main__":
    main()
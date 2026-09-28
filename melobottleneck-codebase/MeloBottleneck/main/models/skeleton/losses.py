# models/skeleton/losses.py
from __future__ import annotations
from typing import Tuple

import torch
import torch.nn.functional as F

from ...outputs import MultiAttrLogits
from ...quantization import DurationQuantizer, DeltaTimeQuantizer

from ...pointer_utils import compute_step_mask


def build_recon_token_weights_from_duration(
    *,
    tgt_tokens: torch.LongTensor,      # [B,T,3]
    tgt_attention_mask: torch.Tensor,  # [B,T]
    duration_q: DurationQuantizer,
    mode: str = "duration",
) -> torch.Tensor:
    """
    生成重建 loss 的 token 权重：
      - 普通音符 token: weight = dur_pos
      - special token (dur_local < special_n): weight = 1（保证 EOS 等也能学）
      - pad: weight = 0
    """
    dur_local = tgt_tokens[..., 1]
    dur_pos = duration_q.decode_local_to_pos(dur_local).to(torch.float32)

    special = dur_local < duration_q.special_n

    if mode == "uniform":
        w = torch.ones_like(dur_pos)
    elif mode == "duration":
        w = dur_pos
    elif mode == "sqrt_duration":
        w = dur_pos.clamp_min(1.0).sqrt()
    elif mode == "log_duration":
        w = torch.log1p(dur_pos.clamp_min(0.0))
    else:
        raise ValueError(f"Unknown recon token weight mode: {mode}")

    w = torch.where(special, torch.ones_like(w), w)
    w = w * tgt_attention_mask.to(torch.float32)
    w = torch.clamp(w, min=0.0)
    return w


def pointer_soft_to_attr_distributions(
    *,
    pointer_soft: torch.Tensor,  # [B,T,L]
    src_tokens: torch.LongTensor,# [B,L,3]
    n_pitch: int,
    n_duration: int,
    n_dt: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    r_t(v) = sum_{l: x_l=v} p_soft(t,l)

    返回:
      r_pitch: [B,T,n_pitch]
      r_dur:   [B,T,n_duration]
      r_dt:    [B,T,n_dt]
    """
    B, T, L = pointer_soft.shape
    device = pointer_soft.device
    dtype = pointer_soft.dtype

    pitch_ids = src_tokens[..., 0]  # [B,L]
    dur_ids = src_tokens[..., 1]
    dt_ids = src_tokens[..., 2]

    pitch_idx = pitch_ids[:, None, :].expand(B, T, L)
    dur_idx = dur_ids[:, None, :].expand(B, T, L)
    dt_idx = dt_ids[:, None, :].expand(B, T, L)

    r_pitch = torch.zeros((B, T, n_pitch), device=device, dtype=dtype)
    r_dur = torch.zeros((B, T, n_duration), device=device, dtype=dtype)
    r_dt = torch.zeros((B, T, n_dt), device=device, dtype=dtype)

    r_pitch.scatter_add_(dim=-1, index=pitch_idx, src=pointer_soft)
    r_dur.scatter_add_(dim=-1, index=dur_idx, src=pointer_soft)
    r_dt.scatter_add_(dim=-1, index=dt_idx, src=pointer_soft)

    return r_pitch, r_dur, r_dt


def kl_divergence_r_to_logits(
    *,
    r: torch.Tensor,         # [B,T,V]
    logits: torch.Tensor,    # [B,T,V]
    mask: torch.Tensor,      # [B,T] bool
    eps: float = 1e-8,
) -> torch.Tensor:
    # force fp32 to make eps meaningful and avoid fp16 underflow issues
    r_f = r.to(torch.float32)
    logits_f = logits.to(torch.float32)
    m = mask.to(torch.float32)

    # (optional but recommended) renormalize to be safe
    r_f = r_f / r_f.sum(dim=-1, keepdim=True).clamp_min(eps)

    log_p = torch.nn.functional.log_softmax(logits_f, dim=-1)   # fp32
    log_r = torch.log(r_f.clamp_min(eps))                       # fp32, finite

    kl = (r_f * (log_r - log_p)).sum(dim=-1)                    # [B,T]

    denom = m.sum().clamp_min(1.0)
    out = (kl * m).sum() / denom
    return out


def lm_prior_kl_loss(
    *,
    pointer_soft: torch.Tensor,  # [B,T,L]
    src_tokens: torch.LongTensor,# [B,L,3]
    lm_logits: MultiAttrLogits,  # [B,T,V*]
    z_mask: torch.Tensor,        # [B,T] bool
    n_pitch: int,
    n_duration: int,
    n_dt: int,
    attr_weights: Tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> torch.Tensor:
    """
    L_P = sum_t KL(r_t || p_LM(.|z_<t))
    r_t 由 pointer_soft 聚合得到。
    """
    r_pitch, r_dur, r_dt = pointer_soft_to_attr_distributions(
        pointer_soft=pointer_soft,
        src_tokens=src_tokens,
        n_pitch=n_pitch,
        n_duration=n_duration,
        n_dt=n_dt,
    )

    w_p, w_d, w_dt = attr_weights
    loss = 0.0
    if w_p != 0:
        loss = loss + float(w_p) * kl_divergence_r_to_logits(r=r_pitch, logits=lm_logits.pitch, mask=z_mask)
    if w_d != 0:
        loss = loss + float(w_d) * kl_divergence_r_to_logits(r=r_dur, logits=lm_logits.duration, mask=z_mask)
    if w_dt != 0:
        loss = loss + float(w_dt) * kl_divergence_r_to_logits(r=r_dt, logits=lm_logits.dt, mask=z_mask)
    return loss

# models/skeleton/losses.py
def guided_attention_loss(
    *,
    pointer_soft: torch.Tensor,          # [B,T,L]
    z_mask: torch.Tensor,                # [B,T] bool
    src_attention_mask: torch.Tensor,    # [B,L] (0/1 or bool)
    eos_pos: torch.LongTensor,           # [B]
    z_len: torch.LongTensor,             # [B]
    sigma: float = 0.2,
    use_time_pos: bool = True,
    src_tokens: torch.LongTensor | None = None,  # [B,L,3], only needed if use_time_pos
    duration_q: DurationQuantizer | None = None,
    dt_q: DeltaTimeQuantizer | None = None,
    form: str = "quad",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Guided attention loss (ESPnet/DCTTS-style):
      W_{t,l} = 1 - exp(-((pos_l - pos_t)^2)/(2*sigma^2))
      L = mean_t E_{p_t}[W_{t,l}]  (masked by z_mask)
    """
    B, T, L = pointer_soft.shape
    device = pointer_soft.device

    # ---- build encoder normalized positions l_norm: [B,1,L] in [0,1]
    if use_time_pos:
        assert src_tokens is not None and duration_q is not None and dt_q is not None
        m = src_attention_mask.to(torch.float32)  # [B,L]

        dur = duration_q.decode_local_to_pos(src_tokens[..., 1]).to(torch.float32) * m
        dt  = dt_q.decode_local_to_pos(src_tokens[..., 2]).to(torch.float32) * m

        span = dur + dt
        prefix = torch.cumsum(span, dim=1)     # prefix[l] = onset_{l+1}
        onset = prefix - span                  # onset_l   = onset_{l+1} - (dur_l+dt_l) = sum_{k<l} span_k

        s_eos = onset.gather(1, eos_pos[:, None]).squeeze(1).clamp_min(1.0)  # [B]
        l_norm = (onset[:, None, :] / s_eos[:, None, None])                  # [B,1,L]
    else:
        l_ids = torch.arange(L, device=device, dtype=torch.float32)[None, None, :]  # [1,1,L]
        den = eos_pos.to(torch.float32).clamp_min(1.0)                               # [B]
        l_norm = l_ids / den[:, None, None]                                          # [B,1,L]

    # ---- decoder normalized positions t_norm: [B,T,1] in [0,1]
    t_ids = torch.arange(T, device=device, dtype=torch.float32)[None, :, None]       # [1,T,1]
    den_t = (z_len - 1).to(torch.float32).clamp_min(1.0)                             # [B]
    t_norm = t_ids / den_t[:, None, None]                                            # [B,T,1]

    # ---- guided mask & loss
    sigma = float(sigma)
    sigma2 = max(sigma * sigma, 1e-8)

    diff = l_norm - t_norm
    dist2 = diff * diff

    if form == "quad":
        W = dist2 / (2.0 * sigma2)
    elif form == "bounded_exp":
        W = 1.0 - torch.exp(-dist2 / (2.0 * sigma2))
    else:
        raise ValueError(f"Unknown guided attention form: {form}")

    p = pointer_soft.to(torch.float32)
    p = p * src_attention_mask[:, None, :].to(torch.float32)  # safety

    penalty_t = (p * W).sum(dim=-1)               # [B,T]
    m = z_mask.to(torch.float32)
    return (penalty_t * m).sum() / m.sum().clamp_min(eps)

def pointer_soft_sharpness_loss(
    *,
    pointer_soft: torch.Tensor,         # [B,T,L]
    hard_indices: torch.LongTensor,     # [B,T]
    z_mask: torch.Tensor,               # [B,T] bool
    kind: str = "hard_ce",
    temperature: float = 1.0,
    exclude_last_step: bool = True,
    entropy_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns a scalar loss (fp32).
    - hard_ce:  E[-log p_soft[t, hard_idx]]
    - entropy:  E[H(p_soft[t])]
    - hard_ce+entropy: sum
    temperature: 对 pointer_soft 做额外温度缩放 (p^(1/T))，只影响此 loss。
    """
    if pointer_soft.ndim != 3:
        raise ValueError(f"pointer_soft must be [B,T,L], got {tuple(pointer_soft.shape)}")
    if hard_indices.shape != pointer_soft.shape[:2]:
        raise ValueError("hard_indices shape mismatch.")
    if z_mask.shape != pointer_soft.shape[:2]:
        raise ValueError("z_mask shape mismatch.")

    step_mask = compute_step_mask(z_mask, exclude_last_step=exclude_last_step)  # [B,T] bool
    m = step_mask.to(torch.float32)
    denom = m.sum().clamp_min(1.0)

    p = pointer_soft.to(torch.float32).clamp_min(0.0)  # [B,T,L]

    # ----- temperature scaling on probs: p_T ∝ p^(1/T) -----
    T = float(temperature)
    if abs(T - 1.0) > 1e-6:
        T = max(T, 1e-6)
        p = p.pow(1.0 / T)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)

    kind = str(kind)
    loss = torch.zeros((), device=p.device, dtype=torch.float32)

    if kind in ("hard_ce", "hard_ce+entropy"):
        L = p.size(-1)
        idx = hard_indices.clamp(min=0, max=L - 1)
        p_at = p.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # [B,T]
        ce = (-torch.log(p_at.clamp_min(eps)) * m).sum() / denom
        loss = loss + ce

    if kind in ("entropy", "hard_ce+entropy"):
        ent = -(p * torch.log(p.clamp_min(eps))).sum(dim=-1)  # [B,T]
        ent = (ent * m).sum() / denom
        loss = loss + float(entropy_weight) * ent

    if kind not in ("hard_ce", "entropy", "hard_ce+entropy"):
        raise ValueError(f"Unknown pointer_soft_reg kind: {kind}")

    return loss
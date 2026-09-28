# main/evaluation/music_prior_proxy_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Any, Union

import numpy as np
import torch
from tqdm import tqdm

from ..pointer_utils import renormalize_scores_on_mask
from ..quantization import DurationQuantizer, DeltaTimeQuantizer


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class MusicPriorProxyEvalConfig:
    # rhythmic grid
    pos_per_beat: int = 12
    strong_period_beats: int = 2          # 4/4: beats 1&3 -> period=2 beats
    use_cummax_onset: bool = True         # handle occasional negative dt / overlap

    # numerical
    eps: float = 1e-8

    # eval-loop
    max_batches: Optional[int] = None
    amp: bool = True
    show_progress: bool = True
    tau: Optional[float] = None

    # batch dict keys (for OTB benchmark)
    token_key: str = "x_orn"
    z_len_key: Optional[str] = "len_x"


# -------------------------
# Small helpers
# -------------------------
def _nanmean(xs: List[float]) -> float:
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return float("nan")
    return float(np.nanmean(a))


def _finite_count(xs: List[float]) -> int:
    a = np.asarray(xs, dtype=np.float64)
    return int(np.isfinite(a).sum())


def compute_onset_pos(
    *,
    tokens: torch.LongTensor,   # [B,L,3]
    src_mask: torch.Tensor,     # [B,L] bool
    duration_q: DurationQuantizer,
    dt_q: DeltaTimeQuantizer,
    use_cummax: bool = True,
) -> torch.Tensor:
    """
    onset[l] = sum_{k<l}(dur_k + dt_k) in pos units.
    Returns float32 [B,L].
    """
    m = src_mask.to(torch.float32)
    dur = duration_q.decode_local_to_pos(tokens[..., 1]).to(torch.float32) * m
    dt = dt_q.decode_local_to_pos(tokens[..., 2]).to(torch.float32) * m
    span = dur + dt
    prefix = torch.cumsum(span, dim=1)
    onset = prefix - span
    if use_cummax:
        onset = torch.cummax(onset, dim=1).values
    return onset


def local_extrema_feature(
    pitch_midi: torch.Tensor,  # [B,L] long/float
    note_mask: torch.Tensor,   # [B,L] bool
) -> torch.Tensor:
    """
    Strict 3-point local extrema on token order:
      extrema at l if p[l]>p[l-1] and p[l]>p[l+1], or p[l]<p[l-1] and p[l]<p[l+1]
    Returns float32 [B,L] in {0,1}.
    """
    p = pitch_midi.to(torch.float32)
    B, L = p.shape
    out = torch.zeros((B, L), device=p.device, dtype=torch.float32)
    if L < 3:
        return out

    prev = p[:, :-2]
    cur = p[:, 1:-1]
    nxt = p[:, 2:]

    m = note_mask[:, :-2] & note_mask[:, 1:-1] & note_mask[:, 2:]
    ext = ((cur > prev) & (cur > nxt)) | ((cur < prev) & (cur < nxt))

    out[:, 1:-1] = (m & ext).to(torch.float32)
    return out


def _pitch_class_hist(
    pc: torch.LongTensor,     # [B,L]
    w: torch.Tensor,          # [B,L] float
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      hist:  [B,12] float32, row-normalized
      valid: [B] bool, whether sum(w)>eps
    """
    pc = pc.clamp(min=0, max=11)
    w = w.to(torch.float32)

    B, L = pc.shape
    hist = torch.zeros((B, 12), device=w.device, dtype=torch.float32)
    hist.scatter_add_(dim=1, index=pc, src=w)

    s = hist.sum(dim=1, keepdim=True)
    valid = (s.squeeze(1) > eps)
    hist = hist / s.clamp_min(eps)
    return hist, valid


def _kl(p: torch.Tensor, q: torch.Tensor, *, eps: float) -> torch.Tensor:
    p = p.to(torch.float32)
    q = q.to(torch.float32)
    return (p * (torch.log(p.clamp_min(eps)) - torch.log(q.clamp_min(eps)))).sum(dim=1)


def _js(p: torch.Tensor, q: torch.Tensor, *, eps: float) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * _kl(p, m, eps=eps) + 0.5 * _kl(q, m, eps=eps)


def _lift(
    *,
    scores: torch.Tensor,     # [B,L]
    feature: torch.Tensor,    # [B,L]
    note_mask: torch.Tensor,  # [B,L] bool
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Lift(f) = E_{s}[f] / E_{uniform}[f]
    Returns: (lift, E_s[f], E_u[f]) all [B] float32; invalid rows -> nan in lift.
    """
    s = scores.to(torch.float32)
    f = feature.to(torch.float32)
    m = note_mask.to(torch.bool)

    # re-norm on note mask (important if some mass is on BOS/EOS/etc.)
    s_note = renormalize_scores_on_mask(s, m, eps=eps)

    # validity: must have at least one note AND s has some mass on notes
    n_note = m.to(torch.int64).sum(dim=1)
    s_mass = (s * m.to(torch.float32)).sum(dim=1)
    valid_s = (n_note > 0) & (s_mass > eps)

    e_s = (s_note * f).sum(dim=1)
    e_u = (f * m.to(torch.float32)).sum(dim=1) / m.to(torch.float32).sum(dim=1).clamp_min(1.0)

    lift = e_s / e_u.clamp_min(eps)
    lift = torch.where(valid_s & (e_u > eps), lift, torch.full_like(lift, float("nan")))
    return lift, e_s, e_u


# -------------------------
# Core: per-sample proxy metrics
# -------------------------
def compute_music_prior_proxy_per_sample(
    *,
    src_tokens: torch.LongTensor,            # [B,L,3]
    src_mask: torch.Tensor,                  # [B,L] bool
    scores: torch.Tensor,                    # [B,L] float
    duration_q: DurationQuantizer,
    dt_q: DeltaTimeQuantizer,
    pos_per_beat: int = 12,
    strong_period_beats: int = 2,
    use_cummax_onset: bool = True,
    hard_mask: Optional[torch.Tensor] = None,  # [B,L] bool (optional)
    z_tokens: Optional[torch.LongTensor] = None, # [B,T,3] (optional)
    z_mask: Optional[torch.Tensor] = None,       # [B,T] bool (optional)
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Returns a dict of per-sample tensors ([B]) so caller can do macro averaging.
    """
    device = src_tokens.device
    B, L, _ = src_tokens.shape
    special_n = int(duration_q.special_n)

    pitch = src_tokens[..., 0]
    note_mask = src_mask.to(torch.bool) & (pitch >= special_n)

    # ---- decode pitch ----
    # assumption: pitch_local - special_n == midi_pitch (0..127)
    pitch_midi = (pitch.to(torch.long) - special_n).clamp(min=0, max=127)  # [B,L]
    pc = torch.remainder(pitch_midi, 12).to(torch.long)                    # [B,L]

    # ---- decode dur (pos units) ----
    dur_pos = duration_q.decode_local_to_pos(src_tokens[..., 1]).to(torch.float32)  # [B,L]
    dur_pos = dur_pos * note_mask.to(torch.float32)

    # ---- onset -> strong beat feature ----
    onset = compute_onset_pos(
        tokens=src_tokens,
        src_mask=src_mask,
        duration_q=duration_q,
        dt_q=dt_q,
        use_cummax=use_cummax_onset,
    )  # [B,L] float
    period = int(pos_per_beat) * int(strong_period_beats)
    onset_i = onset.round().to(torch.long)
    f_strong = (torch.remainder(onset_i, period) == 0).to(torch.float32) * note_mask.to(torch.float32)

    # ---- extrema feature ----
    f_ext = local_extrema_feature(pitch_midi, note_mask) * note_mask.to(torch.float32)

    # ---- lift metrics ----
    lift_strong, mass_strong, rate_strong = _lift(
        scores=scores, feature=f_strong, note_mask=note_mask, eps=eps
    )
    lift_dur, exp_dur, mean_dur = _lift(
        scores=scores, feature=dur_pos, note_mask=note_mask, eps=eps
    )
    lift_ext, mass_ext, rate_ext = _lift(
        scores=scores, feature=f_ext, note_mask=note_mask, eps=eps
    )

    # ---- pitch-class hist (x) ----
    x_hist_cnt, x_ok_cnt = _pitch_class_hist(pc, note_mask.to(torch.float32), eps=eps)
    x_hist_dur, x_ok_dur = _pitch_class_hist(pc, dur_pos, eps=eps)

    # ---- pitch-class hist (soft selection) ----
    s_note = renormalize_scores_on_mask(scores.to(torch.float32), note_mask, eps=eps)  # [B,L]
    s_mass_ok = (s_note.sum(dim=1) > eps)

    sel_hist_cnt, sel_ok_cnt = _pitch_class_hist(pc, s_note, eps=eps)
    sel_hist_dur, sel_ok_dur = _pitch_class_hist(pc, s_note * dur_pos, eps=eps)

    # ---- divergences: x vs soft-selection ----
    nan = torch.full((B,), float("nan"), device=device, dtype=torch.float32)

    ok_cnt = x_ok_cnt & sel_ok_cnt & s_mass_ok
    ok_dur = x_ok_dur & sel_ok_dur & s_mass_ok

    kl_x_sel_cnt = torch.where(ok_cnt, _kl(x_hist_cnt, sel_hist_cnt, eps=eps), nan)
    kl_sel_x_cnt = torch.where(ok_cnt, _kl(sel_hist_cnt, x_hist_cnt, eps=eps), nan)
    js_cnt = torch.where(ok_cnt, _js(x_hist_cnt, sel_hist_cnt, eps=eps), nan)

    kl_x_sel_dur = torch.where(ok_dur, _kl(x_hist_dur, sel_hist_dur, eps=eps), nan)
    kl_sel_x_dur = torch.where(ok_dur, _kl(sel_hist_dur, x_hist_dur, eps=eps), nan)
    js_dur = torch.where(ok_dur, _js(x_hist_dur, sel_hist_dur, eps=eps), nan)

    # ---- hard selection hist (optional) ----
    js_hard_cnt = nan
    js_hard_dur = nan

    if z_tokens is not None and z_mask is not None:
        # use final skeleton tokens (after forward-extend) if provided
        z_pitch = z_tokens[..., 0]
        z_note_mask = z_mask.to(torch.bool) & (z_pitch >= special_n)
        z_pitch_midi = (z_pitch.to(torch.long) - special_n).clamp(min=0, max=127)
        z_pc = torch.remainder(z_pitch_midi, 12).to(torch.long)

        z_dur = duration_q.decode_local_to_pos(z_tokens[..., 1]).to(torch.float32) * z_note_mask.to(torch.float32)

        z_hist_cnt, z_ok_cnt = _pitch_class_hist(z_pc, z_note_mask.to(torch.float32), eps=eps)
        z_hist_dur, z_ok_dur = _pitch_class_hist(z_pc, z_dur, eps=eps)

        ok_hc = x_ok_cnt & z_ok_cnt
        ok_hd = x_ok_dur & z_ok_dur
        js_hard_cnt = torch.where(ok_hc, _js(x_hist_cnt, z_hist_cnt, eps=eps), nan)
        js_hard_dur = torch.where(ok_hd, _js(x_hist_dur, z_hist_dur, eps=eps), nan)

    elif hard_mask is not None:
        hm = hard_mask.to(torch.bool) & note_mask
        hm_cnt = hm.to(torch.float32)
        hm_dur = hm.to(torch.float32) * dur_pos

        h_hist_cnt, h_ok_cnt = _pitch_class_hist(pc, hm_cnt, eps=eps)
        h_hist_dur, h_ok_dur = _pitch_class_hist(pc, hm_dur, eps=eps)

        ok_hc = x_ok_cnt & h_ok_cnt
        ok_hd = x_ok_dur & h_ok_dur
        js_hard_cnt = torch.where(ok_hc, _js(x_hist_cnt, h_hist_cnt, eps=eps), nan)
        js_hard_dur = torch.where(ok_hd, _js(x_hist_dur, h_hist_dur, eps=eps), nan)

    return {
        # lift
        "lift_strong": lift_strong,
        "mass_strong": mass_strong,
        "rate_strong": rate_strong,

        "lift_duration": lift_dur,
        "exp_duration": exp_dur,
        "mean_duration": mean_dur,

        "lift_extrema": lift_ext,
        "mass_extrema": mass_ext,
        "rate_extrema": rate_ext,

        # pitch-class distance (soft)
        "pchist_kl_x_sel_cnt": kl_x_sel_cnt,
        "pchist_kl_sel_x_cnt": kl_sel_x_cnt,
        "pchist_js_cnt": js_cnt,

        "pchist_kl_x_sel_dur": kl_x_sel_dur,
        "pchist_kl_sel_x_dur": kl_sel_x_dur,
        "pchist_js_dur": js_dur,

        # pitch-class distance (hard skeleton if provided)
        "pchist_js_hard_cnt": js_hard_cnt,
        "pchist_js_hard_dur": js_hard_dur,
    }


def compute_music_prior_proxy_metrics_batch(
    *,
    src_tokens: torch.LongTensor,
    src_mask: torch.Tensor,
    scores: torch.Tensor,
    duration_q: DurationQuantizer,
    dt_q: DeltaTimeQuantizer,
    pos_per_beat: int = 12,
    strong_period_beats: int = 2,
    use_cummax_onset: bool = True,
    hard_mask: Optional[torch.Tensor] = None,
    z_tokens: Optional[torch.LongTensor] = None,
    z_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    prefix: str = "",
) -> Dict[str, float]:
    per = compute_music_prior_proxy_per_sample(
        src_tokens=src_tokens,
        src_mask=src_mask,
        scores=scores,
        duration_q=duration_q,
        dt_q=dt_q,
        pos_per_beat=pos_per_beat,
        strong_period_beats=strong_period_beats,
        use_cummax_onset=use_cummax_onset,
        hard_mask=hard_mask,
        z_tokens=z_tokens,
        z_mask=z_mask,
        eps=eps,
    )

    out: Dict[str, float] = {}
    B = int(src_tokens.size(0))
    out[prefix + "n"] = float(B)

    for k, v in per.items():
        v = v.detach().to(torch.float32)
        m = torch.isfinite(v)
        out[prefix + k + "_n"] = float(int(m.sum().item()))
        if bool(m.any().item()):
            out[prefix + k] = float(v[m].mean().item())
        else:
            out[prefix + k] = float("nan")
    return out


# -------------------------
# Full evaluation loop on a loader (macro avg per piece)
# -------------------------
def evaluate_music_prior_proxy(
    adapter,
    loader,
    *,
    device: torch.device,
    duration_q: DurationQuantizer,
    dt_q: DeltaTimeQuantizer,
    cfg: MusicPriorProxyEvalConfig = MusicPriorProxyEvalConfig(),
    prefix: str = "",
) -> Dict[str, float]:
    pad_id = int(adapter.pad_id)

    module = getattr(adapter, "module", None)
    was_training = None
    if isinstance(module, torch.nn.Module):
        was_training = bool(module.training)
        module.eval()

    buf: Dict[str, List[float]] = {}

    try:
        use_amp = bool(cfg.amp and device.type == "cuda")
        it = loader
        if cfg.show_progress:
            total = len(loader) if cfg.max_batches is None else min(len(loader), int(cfg.max_batches))
            it = tqdm(it, desc=f"[Eval Proxy] {prefix}".strip(), dynamic_ncols=True, smoothing=0.0, total=total)

        with torch.inference_mode():
            for bi, batch in enumerate(it):
                if cfg.max_batches is not None and bi >= int(cfg.max_batches):
                    break

                # batch could be dict (OTB) or tensor (plain corpus)
                if isinstance(batch, dict):
                    tokens = batch[cfg.token_key]
                    z_len = batch[cfg.z_len_key] if (cfg.z_len_key is not None and cfg.z_len_key in batch) else None
                else:
                    tokens = batch
                    z_len = None

                src_tokens = tokens.to(device, non_blocking=True)  # [B,L,3]
                src_mask = (src_tokens[..., 0] != pad_id)

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    pred = adapter.predict(
                        src_tokens=src_tokens,
                        src_attention_mask=src_mask,
                        z_len=z_len.to(device) if z_len is not None else None,
                        tau=cfg.tau,
                        exclude_last_step=True,
                    )

                # ---- try build z_tokens_final (forward-extend) if possible ----
                z_tokens_final = None
                z_mask_final = None

                # PointerModelAdapter: adapter.model is the real MusicSkeletonModelIII
                m = getattr(adapter, "model", None)
                if (
                    m is not None
                    and hasattr(m, "forward_extend")
                    and pred.hard_indices is not None
                    and pred.z_mask is not None
                ):
                    # reconstruct z_tokens_hard from src + hard_indices
                    B, L, _ = src_tokens.shape
                    T = pred.hard_indices.size(1)
                    gather_idx = pred.hard_indices.unsqueeze(-1).expand(B, T, 3)
                    z_tokens_hard = src_tokens.gather(dim=1, index=gather_idx)  # [B,T,3]

                    # forward-extend -> z_tokens_final (dur/dt updated)
                    z_tokens_final = m.forward_extend(
                        src_tokens=src_tokens,
                        z_tokens=z_tokens_hard,
                        hard_indices=pred.hard_indices,
                        z_mask=pred.z_mask,
                    )
                    z_mask_final = pred.z_mask

                per = compute_music_prior_proxy_per_sample(
                    src_tokens=src_tokens,
                    src_mask=src_mask,
                    scores=pred.scores,
                    duration_q=duration_q,
                    dt_q=dt_q,
                    pos_per_beat=cfg.pos_per_beat,
                    strong_period_beats=cfg.strong_period_beats,
                    use_cummax_onset=cfg.use_cummax_onset,

                    # 这里同时传 hard_mask（fallback）+ z_tokens_final（优先）
                    hard_mask=pred.hard_mask,
                    z_tokens=z_tokens_final,
                    z_mask=z_mask_final,

                    eps=cfg.eps,
                )

                for k, v in per.items():
                    xs = v.detach().cpu().numpy().astype(np.float64).tolist()
                    buf.setdefault(k, []).extend(xs)

    finally:
        if isinstance(module, torch.nn.Module) and was_training is not None:
            module.train(was_training)

    out: Dict[str, float] = {}
    n_total = len(next(iter(buf.values()))) if buf else 0
    out[prefix + "n"] = float(n_total)

    for k, xs in buf.items():
        out[prefix + k] = _nanmean(xs)
        out[prefix + k + "_n"] = float(_finite_count(xs))

    return out
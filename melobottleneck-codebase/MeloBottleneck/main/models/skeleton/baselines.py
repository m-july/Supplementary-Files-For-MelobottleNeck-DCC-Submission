# main/models/skeleton/baselines.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# Minimal "model interface" required by evaluate_ornament_to_backbone()
# ============================================================

@dataclass(frozen=True)
class DummyVocab:
    special_n: int


@dataclass(frozen=True)
class DummyBackbone:
    pad_id: int
    bos_id: int
    eos_id: int


@dataclass(frozen=True)
class DummyCfg:
    tau: float = 1.0


@dataclass
class BaselineCompressorOutput:
    """
    Minimal output fields used by evaluation/ornament_to_backbone_eval.py
    """
    pointer_soft: torch.Tensor        # [B,T,L]
    hard_indices: torch.LongTensor    # [B,T]
    z_mask: torch.Tensor              # [B,T] bool
    z_len: torch.LongTensor           # [B]
    eos_pos: torch.LongTensor         # [B]


class OTBBaselineModel(nn.Module):
    """
    A fake 'skeleton model' that only exposes:
      - backbone.{pad_id,bos_id,eos_id}
      - vocab.special_n
      - cfg.tau
      - compressor(...)
    So it can be directly evaluated by evaluate_ornament_to_backbone().
    """

    def __init__(
        self,
        *,
        compressor: nn.Module,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        special_n: int,
        tau: float = 1.0,
    ):
        super().__init__()
        self.compressor = compressor
        self.backbone = DummyBackbone(pad_id=int(pad_id), bos_id=int(bos_id), eos_id=int(eos_id))
        self.vocab = DummyVocab(special_n=int(special_n))
        self.cfg = DummyCfg(tau=float(tau))


# ============================================================
# Shared helpers (decode duration/dt, find BOS/EOS, etc.)
# ============================================================

class _OTBNaiveCompressorBase(nn.Module):
    """
    Base class for naive baselines on Ornament-to-Backbone benchmark.

    Assumptions (consistent with your current codebase):
      - tokens are local-id triples: (pitch_local, dur_local, dt_local)
      - note pitch tokens satisfy pitch_local >= special_n
      - BOS/EOS/PAD pitch are special ids (< special_n)
      - duration_local = special_n + dur_code
      - dt_local encoding follows your ornament.py scheme:
          >= special_n and < special_n + dt_samples     -> non-negative dt code
          >= special_n + dt_samples and < special_n + 2*dt_samples -> negative dt code (magnitude coding)
    """

    def __init__(
        self,
        *,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        special_n: int,
        duration_code_to_pos: np.ndarray,
        deltatime_code_to_pos: np.ndarray,
        deltatime_code_offset: int,
        seed: int = 1234,
    ):
        super().__init__()
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.special_n = int(special_n)

        self.dt_offset = int(deltatime_code_offset)
        if int(deltatime_code_to_pos.shape[0]) != 2 * self.dt_offset:
            raise ValueError("Bad deltatime_code_to_pos length (expected 2*offset).")

        # Register as buffers so .to(device) works
        self.register_buffer("_dur_code_to_pos", torch.as_tensor(duration_code_to_pos, dtype=torch.int32), persistent=False)
        self.register_buffer("_dt_code_to_pos", torch.as_tensor(deltatime_code_to_pos, dtype=torch.int32), persistent=False)

        # RNG (only used by the random baseline)
        self._np_rng = np.random.default_rng(int(seed))

    # -------------------------
    # Decode helpers (local-id -> pos)
    # -------------------------
    def _decode_dur_pos(self, dur_local: torch.Tensor) -> torch.Tensor:
        """
        dur_local: [...]
        return int32 pos in [0..]
        """
        code = dur_local.to(torch.int64) - int(self.special_n)
        n = int(self._dur_code_to_pos.numel())
        valid = (code >= 0) & (code < n)
        code = code.clamp(0, max(n - 1, 0))
        pos = self._dur_code_to_pos[code]  # int32
        pos = pos * valid.to(torch.int32)
        return pos

    def _decode_dt_pos(self, dt_local: torch.Tensor) -> torch.Tensor:
        """
        dt_local: [...]
        return int32 signed pos (already mapped by table)
        """
        sN = int(self.special_n)
        dt_samples = int(self.dt_offset)

        dt = dt_local.to(torch.int64)

        # non-negative codes
        m_pos = (dt >= sN) & (dt < sN + dt_samples)
        code_pos = dt - sN  # 0..dt_samples-1

        # negative codes: <2-!{mag}>
        m_neg = (dt >= sN + dt_samples) & (dt < sN + 2 * dt_samples)
        mag = dt - (sN + dt_samples) + 1  # 1..dt_samples
        code_neg = -mag

        code = torch.zeros_like(dt)
        code = torch.where(m_pos, code_pos, code)
        code = torch.where(m_neg, code_neg, code)

        idx = (code + dt_samples).clamp(0, 2 * dt_samples - 1)
        pos = self._dt_code_to_pos[idx]
        return pos.to(torch.int32)

    # -------------------------
    # BOS/EOS helpers
    # -------------------------
    def _find_first_token_pos(self, pitch_1d: torch.Tensor, mask_1d: torch.Tensor, token_id: int) -> Optional[int]:
        m = (pitch_1d == int(token_id)) & mask_1d
        if bool(m.any().item()):
            return int(m.to(torch.int64).argmax().item())  # first True
        return None

    def _find_eos_pos_batch(self, pitch: torch.Tensor, src_mask: torch.Tensor) -> torch.LongTensor:
        """
        pitch: [B,L]
        src_mask: [B,L] bool
        """
        eos_mask = (pitch == int(self.eos_id)) & src_mask
        has = eos_mask.any(dim=1)
        first = eos_mask.to(torch.int64).argmax(dim=1)
        last_valid = src_mask.to(torch.int64).sum(dim=1).clamp_min(1) - 1
        eos_pos = torch.where(has, first, last_valid)
        return eos_pos.to(torch.long)

    def _find_bos_pos_batch(self, pitch: torch.Tensor, src_mask: torch.Tensor) -> torch.LongTensor:
        bos_mask = (pitch == int(self.bos_id)) & src_mask
        has = bos_mask.any(dim=1)
        first = bos_mask.to(torch.int64).argmax(dim=1)
        bos_pos = torch.where(has, first, torch.zeros_like(first))
        return bos_pos.to(torch.long)

    # -------------------------
    # Selection API (to be implemented by subclasses)
    # -------------------------
    def _select_note_positions(
        self,
        *,
        note_pos: torch.Tensor,     # [N] long, increasing in src index
        k: int,
        onset_note: torch.Tensor,   # [N] (float/int), aligned with note_pos
        dur_note: torch.Tensor,     # [N] int, aligned with note_pos
    ) -> torch.Tensor:
        raise NotImplementedError

    def _build_soft_scores(
        self,
        *,
        L: int,
        z_len_b: int,
        bos_pos: int,
        pitch_b: torch.Tensor,      # [L]
        src_mask_b: torch.Tensor,   # [L] bool
        dur_pos_b: torch.Tensor,    # [L] int64
        selected_note_pos: torch.Tensor,  # [k] long
        device: torch.device,
    ) -> torch.Tensor:
        """
        Default: uniform mass over (BOS + selected_note_pos), excluding EOS.
        This makes scores sum to 1 and matches step_mask semantics (exclude last step).
        """
        denom = max(1, int(z_len_b) - 1)  # steps excluding EOS
        scores = torch.zeros((L,), device=device, dtype=torch.float32)

        ids = torch.empty((denom,), device=device, dtype=torch.long)
        ids[0] = int(bos_pos)
        if denom > 1:
            # denom-1 == z_len_b-2 == k
            ids[1:] = selected_note_pos

        scores.scatter_add_(
            dim=0,
            index=ids,
            src=torch.full((denom,), 1.0 / float(denom), device=device, dtype=torch.float32),
        )
        return scores

    # -------------------------
    # Main forward
    # -------------------------
    def forward(
        self,
        *,
        src_tokens: torch.LongTensor,                      # [B,L,3]
        src_attention_mask: Optional[torch.Tensor] = None, # [B,L]
        rho: Optional[float] = None,                       # optional fallback
        z_len: Optional[torch.LongTensor] = None,          # [B]
        tau: Optional[float] = None,                       # ignored (for interface compat)
    ) -> BaselineCompressorOutput:
        device = src_tokens.device
        B, L, _ = src_tokens.shape

        pitch = src_tokens[..., 0]
        if src_attention_mask is None:
            src_mask = (pitch != int(self.pad_id))
        else:
            src_mask = src_attention_mask.to(torch.bool)

        eos_pos = self._find_eos_pos_batch(pitch, src_mask)  # [B]
        bos_pos = self._find_bos_pos_batch(pitch, src_mask)  # [B]

        # Determine z_len if not provided
        if z_len is None:
            if rho is None:
                raise ValueError("Baseline compressor requires z_len or rho.")
            # L_x ≈ eos_pos+1
            L_x = eos_pos + 1
            z_len = torch.ceil(L_x.to(torch.float32) * float(rho)).to(torch.long).clamp(min=1, max=L)
        else:
            z_len = z_len.to(torch.long).clamp(min=1, max=L)

        T = int(z_len.detach().cpu().max().item())
        t_ids = torch.arange(T, device=device)[None, :]
        z_mask = (t_ids < z_len[:, None])  # [B,T] bool

        # Decode duration/dt for onset computation / duration baseline
        dur_pos = self._decode_dur_pos(src_tokens[..., 1]).to(torch.int64)  # [B,L]
        dt_pos = self._decode_dt_pos(src_tokens[..., 2]).to(torch.int64)    # [B,L]

        span = (dur_pos + dt_pos) * src_mask.to(torch.int64)                # [B,L]
        prefix = torch.cumsum(span, dim=1)
        onset = prefix - span                                               # [B,L] int64

        hard_indices = torch.zeros((B, T), device=device, dtype=torch.long)
        scores = torch.zeros((B, L), device=device, dtype=torch.float32)

        z_len_cpu = z_len.detach().cpu().tolist()
        bos_cpu = bos_pos.detach().cpu().tolist()
        eos_cpu = eos_pos.detach().cpu().tolist()

        for b in range(B):
            zlb = int(z_len_cpu[b])
            bosb = int(bos_cpu[b])
            eosb = int(eos_cpu[b])

            # note candidates: (pitch >= special_n) within (bos, eos)
            note_mask_b = src_mask[b] & (pitch[b] >= int(self.special_n))
            note_pos = torch.nonzero(note_mask_b, as_tuple=False).squeeze(1)
            if note_pos.numel() > 0:
                note_pos = note_pos[(note_pos > bosb) & (note_pos < eosb)]

            k = max(0, zlb - 2)  # intermediate notes count
            if k <= 0 or note_pos.numel() == 0:
                selected = note_pos.new_empty((0,))
            else:
                # Gather onset/duration for candidate notes
                onset_note = onset[b, note_pos].to(torch.float32)
                # enforce non-decreasing to be safe under overlaps
                onset_note = torch.cummax(onset_note, dim=0).values
                dur_note = dur_pos[b, note_pos].to(torch.int64)

                selected = self._select_note_positions(
                    note_pos=note_pos,
                    k=int(k),
                    onset_note=onset_note,
                    dur_note=dur_note,
                )

                # Safety: if anything weird happens, clamp length
                if selected.numel() != k:
                    if selected.numel() > k:
                        selected = selected[:k]
                    else:
                        # pad by repeating last (should be rare)
                        if selected.numel() == 0:
                            selected = note_pos[:1].repeat(k)
                        else:
                            pad = selected[-1:].repeat(k - selected.numel())
                            selected = torch.cat([selected, pad], dim=0)

            # Fill hard_indices: [BOS] + selected + [EOS]
            hard_indices[b].fill_(eosb)
            if zlb >= 1:
                hard_indices[b, 0] = bosb
            if zlb >= 2:
                if zlb > 2:
                    hard_indices[b, 1:zlb - 1] = selected
                hard_indices[b, zlb - 1] = eosb

            # Soft scores (marginal importance distribution)
            scores[b] = self._build_soft_scores(
                L=L,
                z_len_b=zlb,
                bos_pos=bosb,
                pitch_b=pitch[b],
                src_mask_b=src_mask[b],
                dur_pos_b=dur_pos[b],
                selected_note_pos=selected,
                device=device,
            )

        # pointer_soft is a broadcast view; eval will reduce over steps anyway
        pointer_soft = scores[:, None, :].expand(B, T, L)

        return BaselineCompressorOutput(
            pointer_soft=pointer_soft,
            hard_indices=hard_indices,
            z_mask=z_mask,
            z_len=z_len,
            eos_pos=eos_pos,
        )


# ============================================================
# Baseline 1: Random downsampling (random but roughly uniform over note index)
# ============================================================

class OTBRandomDownsampleCompressor(_OTBNaiveCompressorBase):
    """
    Random downsampling: stratified random sampling over the candidate note indices
    (roughly equal spacing, but random within each segment).
    """

    def _select_note_positions(
        self,
        *,
        note_pos: torch.Tensor,
        k: int,
        onset_note: torch.Tensor,
        dur_note: torch.Tensor,
    ) -> torch.Tensor:
        n = int(note_pos.numel())
        if k <= 0 or n <= 0:
            return note_pos.new_empty((0,))
        if k >= n:
            return note_pos[:k]

        # Stratified segments on [0..n)
        picks: List[int] = []
        for i in range(k):
            lo = (i * n) // k
            hi = ((i + 1) * n) // k - 1
            if hi < lo:
                hi = lo
            j = int(self._np_rng.integers(lo, hi + 1))
            picks.append(j)

        idx = torch.as_tensor(picks, device=note_pos.device, dtype=torch.long)
        return note_pos[idx]


# ============================================================
# Baseline 2: Uniform downsampling on time grid (onset-pos)
# ============================================================

class OTBUniformTimeDownsampleCompressor(_OTBNaiveCompressorBase):
    """
    Uniform downsampling: pick notes whose onset positions are closest to an evenly-spaced
    time grid between the first and last candidate note.
    """

    def _select_note_positions(
        self,
        *,
        note_pos: torch.Tensor,
        k: int,
        onset_note: torch.Tensor,
        dur_note: torch.Tensor,
    ) -> torch.Tensor:
        n = int(note_pos.numel())
        if k <= 0 or n <= 0:
            return note_pos.new_empty((0,))
        if k >= n:
            return note_pos[:k]

        on = onset_note  # [n], float32, non-decreasing
        t0 = float(on[0].item())
        t1 = float(on[-1].item())
        if abs(t1 - t0) < 1e-6:
            # fallback: uniform by note index
            jj = torch.linspace(0, n - 1, steps=k, device=note_pos.device).round().to(torch.long)
            jj = jj.clamp(0, n - 1)
            return note_pos[jj]

        targets = torch.linspace(t0, t1, steps=k, device=note_pos.device, dtype=torch.float32)

        # searchsorted -> nearest
        j = torch.searchsorted(on, targets, right=False).clamp(0, n - 1)  # [k]
        j0 = (j - 1).clamp(0, n - 1)
        choose_prev = (j > 0) & ((targets - on[j0]).abs() <= (on[j] - targets).abs())
        j = torch.where(choose_prev, j0, j).to(torch.long)

        # Rare case: duplicates (e.g., repeated onsets). Fix on CPU only if needed.
        if bool((j[1:] <= j[:-1]).any().item()):
            j_cpu = j.detach().cpu().numpy().astype(np.int64)

            # enforce strictly increasing
            for i in range(1, k):
                if j_cpu[i] <= j_cpu[i - 1]:
                    j_cpu[i] = j_cpu[i - 1] + 1

            # shift back if overflow
            overflow = j_cpu[-1] - (n - 1)
            if overflow > 0:
                j_cpu -= overflow
                for i in range(1, k):
                    if j_cpu[i] <= j_cpu[i - 1]:
                        j_cpu[i] = j_cpu[i - 1] + 1

            j_cpu = np.clip(j_cpu, 0, n - 1)
            j = torch.from_numpy(j_cpu).to(note_pos.device, dtype=torch.long)

        return note_pos[j]


# ============================================================
# Baseline 3: Top-K duration (hard: pick longest notes; soft: duration-weighted scores)
# ============================================================

class OTBTopKDurationCompressor(_OTBNaiveCompressorBase):
    """
    Top-K duration baseline:
      - Hard: pick K notes with largest duration_pos
      - Soft: scores proportional to duration_pos (plus a tiny BOS mass to mimic step semantics)
    """

    def _select_note_positions(
        self,
        *,
        note_pos: torch.Tensor,
        k: int,
        onset_note: torch.Tensor,
        dur_note: torch.Tensor,
    ) -> torch.Tensor:
        n = int(note_pos.numel())
        if k <= 0 or n <= 0:
            return note_pos.new_empty((0,))
        if k >= n:
            return note_pos[:k]

        dur = dur_note.to(torch.float32)
        top = torch.topk(dur, k=int(k), largest=True, sorted=False).indices
        sel = note_pos[top]
        sel, _ = torch.sort(sel)  # keep monotonic order
        return sel

    def _build_soft_scores(
        self,
        *,
        L: int,
        z_len_b: int,
        bos_pos: int,
        pitch_b: torch.Tensor,
        src_mask_b: torch.Tensor,
        dur_pos_b: torch.Tensor,
        selected_note_pos: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # note universe
        note_mask = src_mask_b & (pitch_b >= int(self.special_n))
        w = dur_pos_b.to(torch.float32) * note_mask.to(torch.float32)

        w_sum = float(w.sum().item())
        if w_sum <= 1e-8:
            # fallback to uniform over notes
            w = note_mask.to(torch.float32)
            w_sum = float(w.sum().item())
            if w_sum <= 1e-8:
                # extreme fallback: uniform over valid tokens
                w = src_mask_b.to(torch.float32)
                w_sum = float(w.sum().item())
                if w_sum <= 1e-8:
                    out = torch.zeros((L,), device=device, dtype=torch.float32)
                    out[int(bos_pos)] = 1.0
                    return out

        p_notes = w / max(w_sum, 1e-8)

        # mimic that one non-EOS step is "BOS-ish": give BOS a small fixed mass
        denom = max(1, int(z_len_b) - 1)
        bos_mass = 1.0 / float(denom)

        out = p_notes * (1.0 - bos_mass)
        out[int(bos_pos)] = out[int(bos_pos)] + bos_mass

        # ensure sum to 1 numerically
        out = out / out.sum().clamp_min(1e-8)
        return out
    
# ============================================================
# Baseline 4: AMR-NoHarmony
# ============================================================


def _amr_onset_type_torch(onset_pos: torch.Tensor, *, nbpm: int, pos_per_beat: int) -> torch.Tensor:
    """
    Generalized version of AMR compute_onset_type() for arbitrary pos_per_beat.
    onset_pos: [N] int/float (pos units)
    returns: [N] long in {0,1,2,3}
      0: strong (beat 1/3 in 4/4)  -> coef 0.85
      1: weak   (beat 2/4)        -> coef 0.95
      2: offbeat (eighth)         -> coef 1.05
      3: others (finer grid)      -> coef 1.15
    """
    x = onset_pos.round().to(torch.long)

    # AMR: half-measure grid (2 beats in 4/4, 3 beats in 3/4)
    half = pos_per_beat * (2 if nbpm == 4 else 3)
    quarter = pos_per_beat
    eighth = max(1, pos_per_beat // 2)

    is_half = (x.remainder(half) == 0)
    is_quarter = (x.remainder(half) != 0) & (x.remainder(quarter) == 0)
    is_eighth = (x.remainder(quarter) != 0) & (x.remainder(eighth) == 0)
    is_sixteenth = ~(is_half | is_quarter | is_eighth)

    out = torch.zeros_like(x, dtype=torch.long)
    out[is_half] = 0
    out[is_quarter] = 1
    out[is_eighth] = 2
    out[is_sixteenth] = 3
    return out

def _amr_onset_coef_torch(onset_pos: torch.Tensor, *, nbpm: int, pos_per_beat: int) -> torch.Tensor:
    t = _amr_onset_type_torch(onset_pos, nbpm=nbpm, pos_per_beat=pos_per_beat)
    coef = onset_pos.new_zeros((t.numel(),), dtype=torch.float32)
    coef[t == 0] = 0.85
    coef[t == 1] = 0.95
    coef[t == 2] = 1.05
    coef[t == 3] = 1.15
    return coef

def _amr_duration_type_torch(dur_pos: torch.Tensor, *, nbpm: int, pos_per_beat: int) -> torch.Tensor:
    """
    Duration type bins like AMR but generalized.
    Type mapping matches AMR return_duration_score():
      0: long (>= 2 beats in 4/4) -> 0.9
      1: >= 1 beat               -> 0.95
      2: >= 0.5 beat             -> 1.05
      3: <  0.5 beat             -> 1.1
    """
    d = dur_pos.round().to(torch.long)
    half = pos_per_beat * (2 if nbpm == 4 else 3)
    quarter = pos_per_beat
    eighth = max(1, pos_per_beat // 2)

    is0 = (d >= half)
    is1 = (d < half) & (d >= quarter)
    is2 = (d < quarter) & (d >= eighth)
    is3 = (d < eighth)

    out = torch.zeros_like(d, dtype=torch.long)
    out[is0] = 0
    out[is1] = 1
    out[is2] = 2
    out[is3] = 3
    return out

def _amr_duration_coef_torch(dur_pos: torch.Tensor, *, nbpm: int, pos_per_beat: int) -> torch.Tensor:
    t = _amr_duration_type_torch(dur_pos, nbpm=nbpm, pos_per_beat=pos_per_beat)
    coef = dur_pos.new_zeros((t.numel(),), dtype=torch.float32)
    coef[t == 0] = 0.90
    coef[t == 1] = 0.95
    coef[t == 2] = 1.05
    coef[t == 3] = 1.10
    return coef

def _amr_pitch_coef_torch(pitch_midi: torch.Tensor) -> torch.Tensor:
    """
    Torch version of AMR return_pitch_score().
    Note: in AMR, extremes get slightly smaller coef -> cheaper -> more preferred.
    """
    p = pitch_midi.to(torch.float32)
    hi = float(p.max().item())
    lo = float(p.min().item())
    if abs(hi - lo) < 1e-6:
        return torch.ones_like(p, dtype=torch.float32)

    mid = 0.5 * (hi + lo)
    denom = max(hi - mid, 1e-6)  # == (hi-lo)/2
    ratio = (p - mid).abs() / float(denom)  # in [0,1]
    coef = (0.5 - ratio) * 0.1 + 1.0
    return coef.to(torch.float32)

def _amr_build_weight_noharmony(
    *,
    onset_pos: torch.Tensor,     # [N] float/int (pos)
    pitch_midi: torch.Tensor,    # [N] long (0..127)
    dur_pos: torch.Tensor,       # [N] float/int (pos)
    nbpm: int,
    pos_per_beat: int,
    bar_thresh: int,
    dist_eta: float,
    lambda_time: float,
    time_subdiv: int,            # 4 -> 16th-note units
    rhy_param: float,
    pitch_param: float,
    dur_param: float,
    rel_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      w: [N,N] float32, w[i,j]=cost if i<j else +inf
      node_coef: [N] float32 (destination importance product), useful for start cost
    """
    device = onset_pos.device
    N = int(onset_pos.numel())
    if N <= 0:
        w = torch.empty((0, 0), device=device, dtype=torch.float32)
        node_coef = torch.empty((0,), device=device, dtype=torch.float32)
        return w, node_coef

    # --- per-node importance coefs (destination node j) ---
    onset_c = _amr_onset_coef_torch(onset_pos, nbpm=nbpm, pos_per_beat=pos_per_beat).pow(float(rhy_param))
    pitch_c = _amr_pitch_coef_torch(pitch_midi).pow(float(pitch_param))
    dur_c = _amr_duration_coef_torch(dur_pos, nbpm=nbpm, pos_per_beat=pos_per_beat).pow(float(dur_param))
    node_coef = (onset_c * pitch_c * dur_c).to(torch.float32)  # [N]

    # --- upper-tri mask (i<j) ---
    idx = torch.arange(N, device=device)
    upper = (idx[:, None] < idx[None, :])

    # --- bar id (only for relation restriction, optional) ---
    # shift onset to start at 0 to mimic AMR phrase offset removal
    # onset0 = onset_pos - onset_pos[:1]
    # bar_len = int(pos_per_beat * nbpm)
    # bar_id = (onset0.round().to(torch.long) // max(bar_len, 1)).to(torch.long)
    # bar_diff = bar_id[None, :] - bar_id[:, None]  # [N,N]
    # close = (bar_diff < int(bar_thresh)) & upper

    close = upper

    # --- relation score matrix (no AE) ---
    UE = 3.0
    rel = torch.full((N, N), UE, device=device, dtype=torch.float32)

    diff = (pitch_midi[:, None] - pitch_midi[None, :]).to(torch.long)
    absdiff = diff.abs()
    mod12 = diff.remainder(12)
    absmod12 = absdiff.remainder(12)

    cond_same = (diff == 0)
    cond_oct = (mod12 == 0) & (~cond_same)
    cond_step = (absdiff >= 1) & (absdiff <= 2)
    cond_ile = (absmod12 == 1) | (absmod12 == 2) | (absmod12 == 10) | (absmod12 == 11)

    # priority: same > oct > step > ile > UE
    rel[close & cond_ile] = 1.3
    rel[close & cond_step] = 0.3
    rel[close & cond_oct] = 1.0
    rel[close & cond_same] = 0.1

    # --- temporal cost on time gap (16th units by default) ---
    # Δ16 = Δpos / (pos_per_beat / 4)  == Δbeats * 4
    pos_unit = max(1.0, float(pos_per_beat) / float(max(1, int(time_subdiv))))
    onset_f = onset_pos.to(torch.float32)
    delta_pos = (onset_f[None, :] - onset_f[:, None]).clamp_min(0.0)
    delta_u = delta_pos / float(pos_unit)
    delta_u = torch.clamp(delta_u, min=1.0)  # mimic AMR: moving forward has at least unit cost
    dist = float(lambda_time) * delta_u.pow(float(dist_eta))

    base = dist + float(rel_weight) * rel
    w = base * node_coef[None, :]
    w = torch.where(upper, w, torch.full_like(w, float("inf")))
    return w, node_coef

def _amr_exact_k_shortest_path(
    *,
    w: torch.Tensor,          # [N,N], inf for i>=j
    start_cost: torch.Tensor, # [N]
    k: int,
    end_cost: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Exact-K shortest path in DAG (increasing indices).
    Returns path indices in [0..N-1], shape [k], strictly increasing.
    """
    device = w.device
    N = int(w.size(0))
    k = int(k)
    if k <= 0 or N <= 0:
        return torch.empty((0,), device=device, dtype=torch.long)
    if k >= N:
        return torch.arange(N, device=device, dtype=torch.long)

    inf = float("inf")
    dp = w.new_full((k, N), inf)
    prev = torch.full((k, N), -1, device=device, dtype=torch.long)

    # feasibility ranges:
    # at step t (0-index), selected t+1 nodes, end index j must satisfy:
    #   j_min=t, j_max=N-k+t
    dp0 = start_cost.to(torch.float32).clone()
    j_max0 = N - k
    if j_max0 + 1 < N:
        dp0[j_max0 + 1:] = inf
    dp[0] = dp0

    for t in range(1, k):
        # cost[i,j] = dp[t-1,i] + w[i,j]
        cost = dp[t - 1].unsqueeze(1) + w  # [N,N]
        dp_t, argmin = cost.min(dim=0)     # [N]

        j_min = t
        j_max = N - k + t
        dp_t[:j_min] = inf
        if j_max + 1 < N:
            dp_t[j_max + 1:] = inf
        argmin[:j_min] = -1
        if j_max + 1 < N:
            argmin[j_max + 1:] = -1

        dp[t] = dp_t
        prev[t] = argmin

    last = dp[k - 1]
    if end_cost is not None:
        last = last + end_cost.to(last.dtype)

    end_j = int(last.argmin().item())
    if not torch.isfinite(last[end_j]):
        # fallback
        return torch.arange(k, device=device, dtype=torch.long)

    path = torch.empty((k,), device=device, dtype=torch.long)
    j = end_j
    for t in range(k - 1, 0, -1):
        path[t] = j
        j = int(prev[t, j].item())
        if j < 0:
            # fallback if something unexpected happens
            return torch.arange(k, device=device, dtype=torch.long)
    path[0] = j
    return path

class OTBAMRNoHarmonyCompressor(_OTBNaiveCompressorBase):
    """
    AMR-inspired baseline:
      - no harmony
      - no postprocess
      - free ends
      - exact K (from z_len)
      - temporal cost uses time gap (onset) not index gap
    """

    def __init__(
        self,
        *,
        pad_id: int,
        bos_id: int,
        eos_id: int,
        special_n: int,
        duration_code_to_pos: np.ndarray,
        deltatime_code_to_pos: np.ndarray,
        deltatime_code_offset: int,
        # AMR-like params
        nbpm: int = 4,
        pos_per_beat: int = 12,   # POS_RESOLUTION
        bar_thresh: int = 2,
        dist_eta: float = 1.6,
        lambda_time: float = 1.0,
        time_subdiv: int = 1,
        rhy_param: float = 0.0,
        pitch_param: float = 1.0,
        dur_param: float = 4.0,
        start_bias: float = 1.0,  # add a node-like cost for the first selected note
        rel_weight: float = 0.1,
        boundary_lambda: float = 1.0,
        boundary_eta: Optional[float] = 1.6,
        seed: int = 1234,
    ):
        super().__init__(
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            special_n=special_n,
            duration_code_to_pos=duration_code_to_pos,
            deltatime_code_to_pos=deltatime_code_to_pos,
            deltatime_code_offset=deltatime_code_offset,
            seed=seed,
        )
        self.nbpm = int(nbpm)
        self.pos_per_beat = int(pos_per_beat)
        self.bar_thresh = int(bar_thresh)
        self.dist_eta = float(dist_eta)
        self.lambda_time = float(lambda_time)
        self.time_subdiv = int(time_subdiv)
        self.rhy_param = float(rhy_param)
        self.pitch_param = float(pitch_param)
        self.dur_param = float(dur_param)
        self.start_bias = float(start_bias)
        self.rel_weight = float(rel_weight)
        self.boundary_lambda = float(boundary_lambda)
        self.boundary_eta = float(dist_eta if boundary_eta is None else boundary_eta)

    def _select_amr_exact_k(
        self,
        *,
        pitch_local: torch.Tensor,   # [N] local ids
        onset_pos: torch.Tensor,     # [N] float
        dur_pos: torch.Tensor,       # [N] long
        k: int,
    ) -> torch.Tensor:
        # decode pitch to midi for correct mod12
        pitch_midi = (pitch_local.to(torch.long) - int(self.special_n)).clamp(min=0, max=127)

        w, node_coef = _amr_build_weight_noharmony(
            onset_pos=onset_pos,
            pitch_midi=pitch_midi,
            dur_pos=dur_pos.to(torch.float32),
            nbpm=self.nbpm,
            pos_per_beat=self.pos_per_beat,
            bar_thresh=self.bar_thresh,
            dist_eta=self.dist_eta,
            lambda_time=self.lambda_time,
            time_subdiv=self.time_subdiv,
            rhy_param=self.rhy_param,
            pitch_param=self.pitch_param,
            dur_param=self.dur_param,
            rel_weight=self.rel_weight,
        )

        # ---- soft coverage penalty (free ends but discourage skipping too much time) ----
        pos_unit = max(1e-6, float(self.pos_per_beat) / float(max(1, int(self.time_subdiv))))
        on = onset_pos.to(torch.float32)

        skip_start_u = ((on - on[0]).clamp_min(0.0) / pos_unit)
        skip_end_u = ((on[-1] - on).clamp_min(0.0) / pos_unit)

        boundary_eta = float(self.boundary_eta)
        start_cost = (
            float(self.start_bias) * node_coef.to(torch.float32)
            + float(self.boundary_lambda) * skip_start_u.pow(boundary_eta)
        )
        end_cost = float(self.boundary_lambda) * skip_end_u.pow(boundary_eta)

        path_idx = _amr_exact_k_shortest_path(
            w=w,
            start_cost=start_cost,
            end_cost=end_cost,
            k=int(k),
        )
        return path_idx

    def forward(
        self,
        *,
        src_tokens: torch.LongTensor,                      # [B,L,3]
        src_attention_mask: Optional[torch.Tensor] = None, # [B,L]
        rho: Optional[float] = None,
        z_len: Optional[torch.LongTensor] = None,          # [B]
        tau: Optional[float] = None,
    ) -> BaselineCompressorOutput:
        device = src_tokens.device
        B, L, _ = src_tokens.shape

        pitch = src_tokens[..., 0]
        if src_attention_mask is None:
            src_mask = (pitch != int(self.pad_id))
        else:
            src_mask = src_attention_mask.to(torch.bool)

        eos_pos = self._find_eos_pos_batch(pitch, src_mask)
        bos_pos = self._find_bos_pos_batch(pitch, src_mask)

        if z_len is None:
            if rho is None:
                raise ValueError("AMR baseline requires z_len or rho.")
            L_x = eos_pos + 1
            z_len = torch.ceil(L_x.to(torch.float32) * float(rho)).to(torch.long).clamp(min=1, max=L)
        else:
            z_len = z_len.to(torch.long).clamp(min=1, max=L)

        T = int(z_len.detach().cpu().max().item())
        t_ids = torch.arange(T, device=device)[None, :]
        z_mask = (t_ids < z_len[:, None])

        dur_pos = self._decode_dur_pos(src_tokens[..., 1]).to(torch.int64)  # [B,L]
        dt_pos = self._decode_dt_pos(src_tokens[..., 2]).to(torch.int64)    # [B,L]

        span = (dur_pos + dt_pos) * src_mask.to(torch.int64)
        prefix = torch.cumsum(span, dim=1)
        onset = prefix - span  # [B,L] int64

        hard_indices = torch.zeros((B, T), device=device, dtype=torch.long)
        scores = torch.zeros((B, L), device=device, dtype=torch.float32)

        z_len_cpu = z_len.detach().cpu().tolist()
        bos_cpu = bos_pos.detach().cpu().tolist()
        eos_cpu = eos_pos.detach().cpu().tolist()

        for b in range(B):
            zlb = int(z_len_cpu[b])
            bosb = int(bos_cpu[b])
            eosb = int(eos_cpu[b])

            note_mask_b = src_mask[b] & (pitch[b] >= int(self.special_n))
            note_pos = torch.nonzero(note_mask_b, as_tuple=False).squeeze(1)
            if note_pos.numel() > 0:
                note_pos = note_pos[(note_pos > bosb) & (note_pos < eosb)]

            k = max(0, zlb - 2)
            if k <= 0 or note_pos.numel() == 0:
                selected = note_pos.new_empty((0,))
            else:
                onset_note = onset[b, note_pos].to(torch.float32)
                onset_note = torch.cummax(onset_note, dim=0).values
                dur_note = dur_pos[b, note_pos].to(torch.int64)
                pitch_note_local = pitch[b, note_pos].to(torch.long)

                n = int(note_pos.numel())
                if k >= n:
                    selected = note_pos
                else:
                    path_idx = self._select_amr_exact_k(
                        pitch_local=pitch_note_local,
                        onset_pos=onset_note,
                        dur_pos=dur_note,
                        k=int(k),
                    )
                    selected = note_pos[path_idx]

            # Fill hard_indices: [BOS] + selected + [EOS]
            hard_indices[b].fill_(eosb)
            if zlb >= 1:
                hard_indices[b, 0] = bosb
            if zlb >= 2:
                if zlb > 2:
                    hard_indices[b, 1:zlb - 1] = selected
                hard_indices[b, zlb - 1] = eosb

            scores[b] = self._build_soft_scores(
                L=L,
                z_len_b=zlb,
                bos_pos=bosb,
                pitch_b=pitch[b],
                src_mask_b=src_mask[b],
                dur_pos_b=dur_pos[b],
                selected_note_pos=selected,
                device=device,
            )

        pointer_soft = scores[:, None, :].expand(B, T, L)

        return BaselineCompressorOutput(
            pointer_soft=pointer_soft,
            hard_indices=hard_indices,
            z_mask=z_mask,
            z_len=z_len,
            eos_pos=eos_pos,
        )
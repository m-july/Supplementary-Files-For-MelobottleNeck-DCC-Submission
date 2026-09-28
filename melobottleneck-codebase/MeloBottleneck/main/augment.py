# music_augment.py
from __future__ import annotations
from typing import Optional
from dataclasses import dataclass
import numpy as np

from .quantization import MusicQuantizationTables


@dataclass(frozen=True)
class MusicAugmentConfig:
    # -------- transpose (pitch shift) --------
    enable_transpose: bool = True
    transpose_mu: float = 0.0
    transpose_sigma: float = 4.0
    transpose_min: int = -8
    transpose_max: int = 8

    # -------- time scaling --------
    enable_time_scale: bool = True
    p_time_scale_2x: float = 0.20
    p_time_scale_half: float = 0.20  # 0.5x
    # remaining prob -> 1.0x

    # -------- local token layout assumptions --------
    # Each column (pitch/dur/dt) is in its own "local id" space:
    #   0..special_n-1 : special tokens
    #
    # Pitch column:
    #   [special_n .. special_n+127]  -> pitches 0..127
    #
    # Duration column (NEW uniform):
    #   [special_n .. special_n+(dur_samples-1)] -> duration codes 0..dur_samples-1  (in pos units)
    #
    # DeltaTime column (NEW uniform, signed):
    #   positive/zero:
    #     [special_n .. special_n+(dt_samples-1)] -> dt codes 0..dt_samples-1
    #   negative:
    #     [special_n+dt_samples .. special_n+(2*dt_samples-1)] -> magnitudes 1..dt_samples (token <2-!mag>)
    #     i.e. local_id = (special_n+dt_samples) + (mag-1)
    special_n: int = 5
    pitch_min: int = 0
    pitch_max: int = 127

    # -------- NEW uniform quantization params (MUST match preprocessing) --------
    dur_samples: int = 96   # duration codes: 0..95
    dt_samples: int = 96    # dt signed codes: -96..95

    # ===== NEW: quantization tables (from SimpleMono.pkl) =====
    quantization_tables: MusicQuantizationTables = None


class MusicAugmenter:
    """
    在 local id 空间做增强（推荐）。
    x_local shape: (T, 3) with columns [pitch_local, dur_local, dt_local].
    """
    def __init__(self, cfg: MusicAugmentConfig = MusicAugmentConfig()):
        self.cfg = cfg

        # ------------------------------
        # Load quantization tables from cfg (must come from SimpleMono.pkl)
        # ------------------------------
        if cfg.quantization_tables.duration_code_to_pos is None or cfg.quantization_tables.duration_pos_to_code is None:
            raise ValueError("augment_config must include duration_code_to_pos & duration_pos_to_code from SimpleMono.pkl")
        if cfg.quantization_tables.deltatime_code_offset is None or cfg.quantization_tables.deltatime_code_to_pos is None or cfg.quantization_tables.deltatime_pos_to_code is None:
            raise ValueError("augment_config must include deltatime_* tables from SimpleMono.pkl")

        self._dur_code_to_pos = np.asarray(cfg.quantization_tables.duration_code_to_pos, dtype=np.int32)
        self._dur_pos_to_code = np.asarray(cfg.quantization_tables.duration_pos_to_code, dtype=np.int32)
        self._dur_samples = int(self._dur_code_to_pos.shape[0])
        self._dur_max = self._dur_samples - 1
        self._dur_pos_max = int(self._dur_pos_to_code.shape[0] - 1)
        self._dur_min_code = int(cfg.quantization_tables.duration_min_code)

        self._dt_offset = int(cfg.quantization_tables.deltatime_code_offset)
        self._dt_samples = int(self._dt_offset)
        self._dt_min = -self._dt_samples
        self._dt_max = self._dt_samples - 1

        self._dt_code_to_pos = np.asarray(cfg.quantization_tables.deltatime_code_to_pos, dtype=np.int32)
        self._dt_pos_to_code = np.asarray(cfg.quantization_tables.deltatime_pos_to_code, dtype=np.int32)

        if self._dt_code_to_pos.shape[0] != 2 * self._dt_samples:
            raise ValueError(f"deltatime_code_to_pos length mismatch: got {self._dt_code_to_pos.shape[0]}, expected {2*self._dt_samples}")
        if self._dt_pos_to_code.shape[0] != 2 * self._dt_samples:
            raise ValueError(f"deltatime_pos_to_code length mismatch: got {self._dt_pos_to_code.shape[0]}, expected {2*self._dt_samples}")

        self._dt_pos_min = int(self._dt_code_to_pos[0])
        self._dt_pos_max = int(self._dt_code_to_pos[-1])

        # -

        # Precompute code maps for 0.5x / 2.0x (vectorized)
        self._code_map_dur = {
            0.5: self._build_dur_code_map(scale=0.5),
            2.0: self._build_dur_code_map(scale=2.0),
        }
        self._code_map_dt_signed = {
            0.5: self._build_dt_signed_map(scale=0.5),
            2.0: self._build_dt_signed_map(scale=2.0),
        }

    def _build_dur_code_map(self, scale: float) -> np.ndarray:
        """
        Use pkl tables:
        dur_code -> pos -> scaled_pos -> dur_code
        """
        codes = np.arange(self._dur_samples, dtype=np.int32)  # 0..dur_max

        # decode
        pos = self._dur_code_to_pos[codes]
        new_pos = np.rint(pos.astype(np.float32) * float(scale)).astype(np.int32)

        # clamp in pos space (keep duration valid)
        min_code = max(0, min(self._dur_min_code, self._dur_max))
        min_pos = int(self._dur_code_to_pos[min_code])
        new_pos = np.clip(new_pos, min_pos, self._dur_pos_max)

        # encode (pos is integer-indexable under current scheme)
        new_pos_idx = np.clip(new_pos, 0, self._dur_pos_max)
        new_code = self._dur_pos_to_code[new_pos_idx].astype(np.int32)

        # safety clamp in code space
        new_code = np.clip(new_code, min_code, self._dur_max)
        return new_code.astype(np.int32)


    def _build_dt_signed_map(self, scale: float) -> np.ndarray:
        """
        Use pkl tables:
        dt_code_signed -> pos -> scaled_pos -> dt_code_signed
        Table is indexed by (dt_code_signed + dt_offset).
        """
        dt_codes = np.arange(self._dt_min, self._dt_max + 1, dtype=np.int32)  # len=2*dt_samples

        # decode
        dt_pos = self._dt_code_to_pos[dt_codes + self._dt_offset]
        new_pos = np.rint(dt_pos.astype(np.float32) * float(scale)).astype(np.int32)

        # clamp in pos space to representable region
        new_pos = np.clip(new_pos, self._dt_pos_min, self._dt_pos_max)

        # encode (pos is integer-indexable under current scheme)
        idx = new_pos + self._dt_offset
        idx = np.clip(idx, 0, 2 * self._dt_samples - 1)
        new_code = self._dt_pos_to_code[idx].astype(np.int32)

        # safety clamp
        new_code = np.clip(new_code, self._dt_min, self._dt_max)
        return new_code.astype(np.int32)


    # ---------------- sampling ----------------
    def sample_time_scale(self, rng: np.random.Generator) -> float:
        if not self.cfg.enable_time_scale:
            return 1.0
        r = float(rng.random())
        if r < float(self.cfg.p_time_scale_2x):
            return 2.0
        if r < float(self.cfg.p_time_scale_2x + self.cfg.p_time_scale_half):
            return 0.5
        return 1.0

    def sample_transpose_semitones(self, pitch_local: np.ndarray, rng: np.random.Generator) -> int:
        """
        k = round(N(mu, sigma))
        先做 [-8,8] 的 rejection sampling（reroll，不 clamp），
        同时保证对当前序列移调后不会越界到 <0-(-x)> / <0-(128+)>。
        """
        if not self.cfg.enable_transpose:
            return 0

        sN = self.cfg.quantization_tables.special_n
        lo_id = sN + self.cfg.pitch_min
        hi_id = sN + self.cfg.pitch_max

        mask = (pitch_local >= lo_id) & (pitch_local <= hi_id)
        if not np.any(mask):
            return 0

        p = pitch_local[mask].astype(np.int32) - sN  # 0..127
        pmin = int(p.min())
        pmax = int(p.max())

        allowed_lo = max(self.cfg.transpose_min, -pmin)
        allowed_hi = min(self.cfg.transpose_max, self.cfg.pitch_max - pmax)
        if allowed_lo > allowed_hi:
            return 0

        for _ in range(256):  # safety cap
            k = int(np.rint(rng.normal(self.cfg.transpose_mu, self.cfg.transpose_sigma)))
            if allowed_lo <= k <= allowed_hi:
                return k

        return int(rng.integers(allowed_lo, allowed_hi + 1))

    # ---------------- inplace transforms ----------------
    def transpose_pitch_inplace(self, x_local: np.ndarray, semitones: int) -> np.ndarray:
        if semitones == 0:
            return x_local
        sN = self.cfg.quantization_tables.special_n
        lo_id = sN + self.cfg.pitch_min
        hi_id = sN + self.cfg.pitch_max
        pitch = x_local[:, 0]
        mask = (pitch >= lo_id) & (pitch <= hi_id)
        if np.any(mask):
            pitch[mask] = pitch[mask] + semitones
            x_local[:, 0] = pitch
        return x_local

    def scale_time_inplace(self, x_local: np.ndarray, scale: float) -> np.ndarray:
        """
        Time scaling under NEW uniform quantization.
        If out of range after scaling -> clamp to min/max duration/dt code.
        """
        if scale == 1.0:
            return x_local
        if scale not in (0.5, 2.0):
            raise ValueError(f"Only 0.5/1.0/2.0 supported, got {scale}")

        sN = int(self.cfg.quantization_tables.special_n)

        # ----- duration column -----
        dur_tok_start = sN
        dur_tok_end = sN + (self._dur_samples - 1)

        dur = x_local[:, 1]
        mask_dur = (dur >= dur_tok_start) & (dur <= dur_tok_end)
        if np.any(mask_dur):
            dur_code = (dur[mask_dur] - dur_tok_start).astype(np.int32)  # 0..dur_max
            new_code = self._code_map_dur[scale][dur_code]
            dur[mask_dur] = (new_code + dur_tok_start).astype(dur.dtype)
            x_local[:, 1] = dur

        # ----- dt column -----
        dt = x_local[:, 2]

        dt_pos_tok_start = sN
        dt_pos_tok_end = sN + (self._dt_samples - 1)

        dt_neg_tok_start = sN + self._dt_samples
        dt_neg_tok_end = dt_neg_tok_start + (self._dt_samples - 1)

        dt_map = self._code_map_dt_signed[scale]  # indexed by (dt_signed + offset)

        # (A) dt >= 0 tokens
        mask_pos = (dt >= dt_pos_tok_start) & (dt <= dt_pos_tok_end)
        if np.any(mask_pos):
            dt_signed = (dt[mask_pos] - dt_pos_tok_start).astype(np.int32)  # 0..dt_samples-1
            new_signed = dt_map[dt_signed + self._dt_offset]                # clamp already in table

            out = np.empty_like(new_signed, dtype=dt.dtype)
            pos_m = (new_signed >= 0)
            out[pos_m] = (dt_pos_tok_start + new_signed[pos_m]).astype(dt.dtype)
            out[~pos_m] = (dt_neg_tok_start + (-new_signed[~pos_m] - 1)).astype(dt.dtype)
            dt[mask_pos] = out

        # (B) dt negative tokens (magnitudes 1..dt_samples)
        mask_neg = (dt >= dt_neg_tok_start) & (dt <= dt_neg_tok_end)
        if np.any(mask_neg):
            mag = (dt[mask_neg] - dt_neg_tok_start).astype(np.int32) + 1  # 1..dt_samples
            dt_signed = -mag                                              # -1..-dt_samples
            new_signed = dt_map[dt_signed + self._dt_offset]

            out = np.empty_like(new_signed, dtype=dt.dtype)
            pos_m = (new_signed >= 0)
            out[pos_m] = (dt_pos_tok_start + new_signed[pos_m]).astype(dt.dtype)
            out[~pos_m] = (dt_neg_tok_start + (-new_signed[~pos_m] - 1)).astype(dt.dtype)
            dt[mask_neg] = out

        x_local[:, 2] = dt
        return x_local

    def augment_inplace(self, x_local: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        # 1) transpose
        if self.cfg.enable_transpose:
            k = self.sample_transpose_semitones(x_local[:, 0], rng)
            if k != 0:
                self.transpose_pitch_inplace(x_local, k)

        # 2) time scaling
        if self.cfg.enable_time_scale:
            s = self.sample_time_scale(rng)
            if s != 1.0:
                self.scale_time_inplace(x_local, s)

        return x_local

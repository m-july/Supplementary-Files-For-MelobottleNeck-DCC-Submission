# main/retrieval/corrupt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from preproc.simplemono_preproc.decoder_midi import SimpleMonoDecodingConfig
from .simplemono_rel import infer_used_len_events


@dataclass(frozen=True)
class MusicCorruptionConfig:
    enable: bool = True
    p_apply: float = 1.0

    # keep at least this many notes (NOT events) after structural ops
    min_notes: int = 8

    # -----------------
    # (A) structural: merge adjacent notes (under-segmentation)
    # -----------------
    p_merge: float = 0.06
    merge_max_ops: int = 4
    merge_max_gap_pos: int = 2          # allow dt_i up to this (dt_i >0 means a gap)
    merge_max_overlap_pos: int = 3      # allow dt_i down to -this (overlap)
    merge_pick: str = "longer"          # "first" | "second" | "longer"
    merge_strict_range: bool = True     # out-of-range => skip (True) else clamp (False)

    # -----------------
    # (B) structural: delete note (missed note)
    # -----------------
    p_delete: float = 0.04
    delete_max_ops: int = 3
    delete_strict_range: bool = True    # dt overflow => skip (True) else clamp (False)

    # -----------------
    # (C) boundary shift (duration<->gap exchange), keep next onset fixed
    # -----------------
    p_boundary_shift: float = 0.10
    boundary_shift_max_pos: int = 3
    boundary_shift_require_dt_nonneg: bool = True

    # -----------------
    # (D) pitch jitter
    # -----------------
    p_pitch_jitter: float = 0.12
    pitch_jitter_max_semitones: int = 2
    pitch_jitter_sigma: float = 1.0

    # -----------------
    # (E) small quantization noise (duration / dt)
    # -----------------
    p_dur_jitter: float = 0.10
    dur_jitter_max_delta: int = 2
    p_dt_jitter: float = 0.05
    dt_jitter_max_delta: int = 2

    # -----------------
    # (F) free rhythm (rubato / time drift): allow total time change
    # -----------------
    p_free_rhythm_seq: float = 0.25

    p_dur_stretch: float = 0.12
    dur_stretch_sigma: float = 0.40
    dur_stretch_min: float = 0.6
    dur_stretch_max: float = 1.8
    dur_stretch_expand_only: bool = True

    p_gap_stretch: float = 0.06
    gap_stretch_sigma: float = 0.60
    gap_stretch_min: float = 0.5
    gap_stretch_max: float = 2.5
    gap_stretch_expand_only: bool = True

    max_ops_per_seq: int = 999999


class MusicCorruptor:
    """
    Operate in GLOBAL-ID events space ([L,3]) but manipulate semantic values:
      pitch: 0..127
      dur_code (==pos): 0..max_dur_code (we clamp to >=1 for notes)
      dt_code_signed (==pos): [-dt_samples .. dt_samples-1]
    """
    def __init__(self, cfg: MusicCorruptionConfig, dec_cfg: SimpleMonoDecodingConfig):
        self.cfg = cfg
        self.dec = dec_cfg

        self.pitch_min = 0
        self.pitch_max = int(dec_cfg.max_pitch)

        self.dur_min = 1
        self.dur_max = int(dec_cfg.max_dur_code)

        self.dt_min = -int(dec_cfg.dt_samples)
        self.dt_max = int(dec_cfg.dt_samples) - 1

        self.pitch_offset = int(dec_cfg.pitch_offset)
        self.dur_offset = int(dec_cfg.dur_offset)
        self.dt_pos_offset = int(dec_cfg.dt_pos_offset)
        self.dt_neg_offset = int(dec_cfg.dt_neg_offset)

        # pitch tokens live in [pitch_offset, dur_offset)
        self.pitch_gid_lo = self.pitch_offset
        self.pitch_gid_hi = self.dur_offset - 1

    # -------------------------
    # encode/decode helpers
    # -------------------------
    def _decode_dt_signed_vec(self, dt_gid: np.ndarray) -> np.ndarray:
        dt_gid = np.asarray(dt_gid, dtype=np.int32)
        out = np.zeros_like(dt_gid, dtype=np.int32)

        pos = (dt_gid >= self.dt_pos_offset) & (dt_gid < self.dt_neg_offset)
        out[pos] = dt_gid[pos] - self.dt_pos_offset

        neg = (dt_gid >= self.dt_neg_offset) & (dt_gid < (self.dt_neg_offset + int(self.dec.dt_samples)))
        mag = dt_gid[neg] - self.dt_neg_offset + 1
        out[neg] = -mag

        return out

    def _encode_dt_gid(self, dt_code: int) -> int:
        dt_code = int(np.clip(dt_code, self.dt_min, self.dt_max))
        if dt_code >= 0:
            return int(self.dt_pos_offset + dt_code)
        mag = int(np.clip(-dt_code, 1, int(self.dec.dt_samples)))
        return int(self.dt_neg_offset + (mag - 1))

    def _encode_dur_gid(self, dur_code: int) -> int:
        dur_code = int(np.clip(dur_code, 0, self.dur_max))
        return int(self.dur_offset + dur_code)

    def _encode_pitch_gid(self, pitch: int) -> int:
        pitch = int(np.clip(pitch, self.pitch_min, self.pitch_max))
        return int(self.pitch_offset + pitch)

    def _sample_nonzero_offset(self, rng: np.random.Generator, m: int, sigma: float) -> int:
        m = int(m)
        if m <= 0:
            return 0
        for _ in range(8):
            k = int(np.rint(rng.normal(0.0, float(sigma))))
            k = int(np.clip(k, -m, m))
            if k != 0:
                return k
        k = int(rng.integers(1, m + 1))
        if float(rng.random()) < 0.5:
            k = -k
        return k

    # -------------------------
    # main api
    # -------------------------
    def corrupt(
        self,
        events_global: np.ndarray,
        *,
        rng: np.random.Generator,
        used_len_events: Optional[int] = None,
    ) -> np.ndarray:
        cfg = self.cfg
        ev = np.asarray(events_global, dtype=np.int32)
        L = int(ev.shape[0])

        if (not cfg.enable) or (float(rng.random()) > float(cfg.p_apply)):
            return ev.copy()

        if used_len_events is None:
            used_len_events = infer_used_len_events(ev, self.dec)
        used_len_events = int(used_len_events)
        if used_len_events < 4:
            return ev.copy()

        # note rows are [1 : used_len-1)
        st = 1
        ed = max(st, used_len_events - 1)
        rows = ev[st:ed]  # [N,3]
        if rows.shape[0] < max(2, int(cfg.min_notes)):
            # too short => still allow small jitter but skip structural
            pass

        # keep only valid pitch rows (defensive)
        p_gid = rows[:, 0].astype(np.int32, copy=False)
        m_note = (p_gid >= self.pitch_gid_lo) & (p_gid <= self.pitch_gid_hi)
        rows = rows[m_note]
        if rows.shape[0] < 2:
            return ev.copy()

        pitch = (rows[:, 0] - self.pitch_offset).astype(np.int32, copy=False)
        dur = (rows[:, 1] - self.dur_offset).astype(np.int32, copy=False)
        dur = np.clip(dur, self.dur_min, self.dur_max).astype(np.int32, copy=False)
        dt = self._decode_dt_signed_vec(rows[:, 2])

        # last dt not really used; keep it 0 for stability later
        dt[-1] = 0

        # -------------------------
        # decide free-rhythm mode
        # -------------------------
        free_rhythm = (float(rng.random()) < float(cfg.p_free_rhythm_seq))

        ops = 0

        # -------------------------
        # (A) merge adjacent notes
        # keep onset of note_{i+2} unchanged (preserve total span)
        # -------------------------
        if rows.shape[0] >= 3:
            i = 0
            merged = 0
            while i < pitch.size - 1 and merged < int(cfg.merge_max_ops) and ops < int(cfg.max_ops_per_seq):
                if pitch.size <= int(cfg.min_notes):
                    break

                if float(rng.random()) >= float(cfg.p_merge):
                    i += 1
                    continue

                dt_i = int(dt[i])
                if dt_i > int(cfg.merge_max_gap_pos) or dt_i < -int(cfg.merge_max_overlap_pos):
                    i += 1
                    continue

                d_i = int(dur[i])
                d_j = int(dur[i + 1])
                dt_j = int(dt[i + 1])  # gap after second note

                # merged duration should cover both notes (esp. when overlap exists)
                block_end = d_i + dt_i + d_j          # end of note(i+1) relative to onset_i
                d_m = max(d_i, block_end)
                dt_m = (d_i + dt_i + d_j + dt_j) - d_m  # so (d_m + dt_m) preserves total span to onset_{i+2}

                in_range = (self.dur_min <= d_m <= self.dur_max) and (self.dt_min <= dt_m <= self.dt_max)

                if (not in_range) and bool(cfg.merge_strict_range):
                    i += 1
                    continue

                if not in_range:
                    d_m = int(np.clip(d_m, self.dur_min, self.dur_max))
                    dt_m = int(np.clip(dt_m, self.dt_min, self.dt_max))

                # pick pitch
                pick = str(cfg.merge_pick).lower().strip()
                if pick == "second":
                    p_m = int(pitch[i + 1])
                elif pick == "longer":
                    p_m = int(pitch[i] if d_i >= d_j else pitch[i + 1])
                else:
                    p_m = int(pitch[i])

                # apply merge: replace i, delete i+1
                pitch[i] = p_m
                dur[i] = d_m
                dt[i] = dt_m

                pitch = np.delete(pitch, i + 1)
                dur = np.delete(dur, i + 1)
                dt = np.delete(dt, i + 1)

                dt[-1] = 0
                merged += 1
                ops += 1
                # do not advance i (allow chaining)

            # end while

        # -------------------------
        # (B) delete notes (missed note), keep timing by pushing span into previous dt
        # -------------------------
        if pitch.size >= 3:
            deleted = 0
            # iterate from 1..n-2 (avoid deleting first/last note)
            i = 1
            while i < pitch.size - 1 and deleted < int(cfg.delete_max_ops) and ops < int(cfg.max_ops_per_seq):
                if pitch.size <= int(cfg.min_notes):
                    break

                if float(rng.random()) >= float(cfg.p_delete):
                    i += 1
                    continue

                span = int(dur[i] + dt[i])           # from onset_i to onset_{i+1}
                new_dt_prev = int(dt[i - 1] + span)

                in_range = (self.dt_min <= new_dt_prev <= self.dt_max)
                if (not in_range) and bool(cfg.delete_strict_range):
                    i += 1
                    continue
                if not in_range:
                    new_dt_prev = int(np.clip(new_dt_prev, self.dt_min, self.dt_max))

                dt[i - 1] = new_dt_prev

                pitch = np.delete(pitch, i)
                dur = np.delete(dur, i)
                dt = np.delete(dt, i)

                dt[-1] = 0
                deleted += 1
                ops += 1
                # do not advance i (index i now points to next note)

        # -------------------------
        # (C) boundary shift: exchange duration <-> gap (keep next onset fixed)
        # -------------------------
        if pitch.size >= 2 and int(cfg.boundary_shift_max_pos) > 0:
            for i in range(0, pitch.size - 1):
                if ops >= int(cfg.max_ops_per_seq):
                    break
                if float(rng.random()) >= float(cfg.p_boundary_shift):
                    continue

                if bool(cfg.boundary_shift_require_dt_nonneg) and int(dt[i]) < 0:
                    continue

                maxs = int(cfg.boundary_shift_max_pos)
                delta = int(rng.integers(-maxs, maxs + 1))
                if delta == 0:
                    continue

                d2 = int(dur[i] + delta)
                t2 = int(dt[i] - delta)

                if d2 < self.dur_min or d2 > self.dur_max:
                    continue
                if bool(cfg.boundary_shift_require_dt_nonneg) and t2 < 0:
                    continue
                if t2 < self.dt_min or t2 > self.dt_max:
                    continue

                dur[i] = d2
                dt[i] = t2
                ops += 1

        # -------------------------
        # (D) pitch jitter
        # -------------------------
        m = int(cfg.pitch_jitter_max_semitones)
        if m > 0:
            for i in range(pitch.size):
                if ops >= int(cfg.max_ops_per_seq):
                    break
                if float(rng.random()) < float(cfg.p_pitch_jitter):
                    off = self._sample_nonzero_offset(rng, m=m, sigma=float(cfg.pitch_jitter_sigma))
                    pitch[i] = int(np.clip(int(pitch[i]) + off, self.pitch_min, self.pitch_max))
                    ops += 1

        # -------------------------
        # (E) small duration/dt jitter (quantization noise)
        # -------------------------
        dj = int(cfg.dur_jitter_max_delta)
        if dj > 0:
            for i in range(dur.size):
                if ops >= int(cfg.max_ops_per_seq):
                    break
                if float(rng.random()) < float(cfg.p_dur_jitter):
                    delta = int(rng.integers(-dj, dj + 1))
                    if delta != 0:
                        dur[i] = int(np.clip(int(dur[i]) + delta, self.dur_min, self.dur_max))
                        ops += 1

        tj = int(cfg.dt_jitter_max_delta)
        if tj > 0:
            for i in range(dt.size - 1):  # ignore last dt (keep 0)
                if ops >= int(cfg.max_ops_per_seq):
                    break
                if float(rng.random()) < float(cfg.p_dt_jitter):
                    delta = int(rng.integers(-tj, tj + 1))
                    if delta != 0:
                        dt[i] = int(np.clip(int(dt[i]) + delta, self.dt_min, self.dt_max))
                        ops += 1

        # -------------------------
        # (F) free rhythm: stretch duration / gaps, allow total time drift
        # -------------------------
        if free_rhythm:
            # stretch duration
            for i in range(dur.size):
                if ops >= int(cfg.max_ops_per_seq):
                    break
                if float(rng.random()) < float(cfg.p_dur_stretch):
                    z = float(rng.normal(0.0, float(cfg.dur_stretch_sigma)))
                    if bool(cfg.dur_stretch_expand_only):
                        z = abs(z)
                    factor = float(np.exp(z))
                    factor = float(np.clip(factor, float(cfg.dur_stretch_min), float(cfg.dur_stretch_max)))
                    d2 = int(np.rint(float(dur[i]) * factor))
                    dur[i] = int(np.clip(d2, self.dur_min, self.dur_max))
                    ops += 1

            # stretch gaps (dt>=0 only, to avoid crazy overlaps)
            for i in range(dt.size - 1):
                if ops >= int(cfg.max_ops_per_seq):
                    break
                if int(dt[i]) < 0:
                    continue
                if float(rng.random()) < float(cfg.p_gap_stretch):
                    z = float(rng.normal(0.0, float(cfg.gap_stretch_sigma)))
                    if bool(cfg.gap_stretch_expand_only):
                        z = abs(z)
                    factor = float(np.exp(z))
                    factor = float(np.clip(factor, float(cfg.gap_stretch_min), float(cfg.gap_stretch_max)))
                    t2 = int(np.rint(float(dt[i]) * factor))
                    dt[i] = int(np.clip(t2, 0, self.dt_max))  # keep non-negative in free-rhythm gap stretch
                    ops += 1

        dt[-1] = 0

        # -------------------------
        # repack to [L,3] global events
        # -------------------------
        out = np.full((L, 3), int(self.dec.pad_id), dtype=np.int32)
        out[0] = ev[0]  # keep BOS row as-is (including start_pos dt)

        n_notes = int(pitch.size)
        # ensure there is space for EOS
        n_notes = min(n_notes, L - 2)

        for i in range(n_notes):
            out[1 + i, 0] = self._encode_pitch_gid(int(pitch[i]))
            out[1 + i, 1] = self._encode_dur_gid(int(dur[i]))
            out[1 + i, 2] = self._encode_dt_gid(int(dt[i]))

        eos = int(self.dec.eos_id)
        out[1 + n_notes, :] = eos

        return out
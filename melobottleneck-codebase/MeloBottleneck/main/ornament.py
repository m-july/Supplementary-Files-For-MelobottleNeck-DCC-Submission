# ornament.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from .quantization import MusicQuantizationTables


PI_INSERTED = -1  # 装饰/插入音
PI_PAD = -2       # pad 区


@dataclass(frozen=True)
class MusicOrnamentConfig:
    # -----------------
    # enable + strength
    # -----------------
    enable: bool = True
    p_apply: float = 0.8  # 每条序列是否启用加花（强视图里你可以设 1.0）

    # -----------------
    # token ids
    # -----------------
    pad_id: int = 0
    bos_id: Optional[int] = None
    eos_id: Optional[int] = None

    # -----------------
    # global constraints
    # -----------------
    max_extra_tokens: int = 128        # 最多额外插入多少个 token（避免 pad 很多时无限加花）
    max_ops_per_seq: int = 999999     # 限制操作次数（debug 用）

    # -----------------
    # pitch jitter
    # -----------------
    pitch_min: int = 0
    pitch_max: int = 127
    pitch_jitter_max_semitones: int = 2
    pitch_jitter_sigma: float = 1.0  # 用 N(0,sigma) 取整产生抖动（偏好小抖动）

    # -----------------
    # requested: split into pre/post grace note
    # -----------------
    p_pre_grace: float = 0.15
    p_post_grace: float = 0.15

    min_note_dur_pos_for_split: int = 6   # “较长的音”阈值（pos单位）
    min_seg_dur_pos: int = 1              # 任意切分片段最短 duration（pos）

    grace_min_dur_pos: int = 1
    grace_max_ratio: float = 0.40         # 装饰音最长不超过原 duration 的某个比例
    grace_overlap_prob: float = 0.2       # 允许“负 dt 轻微重叠”的概率（更像 acciaccatura）
    grace_overlap_max_pos: int = 3        # 重叠最多多少 pos（避免太多 polyphony）

    # -----------------
    # optional: scattered insert that can EXTEND total time (rubato-ish)
    # -----------------
    p_between_insert: float = 0.15            # 在 (note_i, note_{i+1}) 之间插音
    p_between_run: float = 0.0                # 插入 2~K 个音形成 run（passing notes）
    p_enable_time_extend_seq: float = 0.1    # 只有部分序列启用“可增长总时值”的模式（你要求：默认开但只抽一部分序列）
    between_min_gap_pos: int = 2              # 非增长模式下，必须有至少这么长的 gap 才插
    between_ins_min_dur_pos: int = 1
    between_ins_max_dur_pos: int = 4
    between_run_min_notes: int = 2
    between_run_max_notes: int = 4
    between_run_ins_min_dur_pos: int = 1
    between_run_ins_max_dur_pos: int = 2
    between_run_pitch_jitter_prob: float = 0.15
    between_pitch_step_choices: Tuple[int, ...] = (1, 2)
    between_use_next_pitch_prob: float = 0.5  # 插入音更像“趋向下一个音”的概率

    # -----------------
    # extra ideas: trill / multi-split / re-articulation / turn
    # -----------------
    p_trill: float = 0.10
    trill_min_dur_pos: int = 6
    trill_max_segments: int = 5  # 3..max；会尽量取 odd（更自然 alternating）

    p_rearticulation: float = 0.10
    reartic_min_dur_pos: int = 6
    reartic_max_repeats: int = 4
    reartic_gap_max_pos: int = 1  # 每次重发音之间允许插一个极短 gap（pos）

    p_turn: float = 0.0
    turn_min_dur_pos: int = 6
    turn_segments: int = 4        # 3 或 4（简单起见默认 4）

    p_pair_repeat: float = 0.0
    pair_repeat_min_note_dur_pos: int = 6
    pair_repeat_max_pairs: int = 4      # n_pairs 上限（总token=2*n_pairs）
    pair_repeat_allow_overlap_dt: bool = False  # 是否允许 A->B 的 dt 为负（重叠）时仍做该op

    # -----------------
    # quantization tables
    # -----------------
    quantization_tables: MusicQuantizationTables = None


class MusicOrnamenter:
    """
    在 local-id 空间对 (pitch, dur, dt) token 序列做“加花”增强：
      - 前装饰音 / 后装饰音（分裂一个长音为 2 个音，整体时值不变）
      - 零散插入次要音（可选：允许把整体时值拉长）
      - 额外：trill、多段切分、re-articulation、turn 等

    返回：
      x_aug_local: [L,3]
      pi: [L]，0-based 对齐；插入音为 -1；pad 为 -2
    """
    def __init__(self, cfg: MusicOrnamentConfig):
        self.cfg = cfg
        qt = cfg.quantization_tables
        if qt is None:
            raise ValueError("MusicOrnamentConfig.quantization_tables must be provided.")

        # ---- tables (numpy) ----
        self.special_n = int(qt.special_n)

        # duration tables
        self._dur_code_to_pos = np.asarray(qt.duration_code_to_pos, dtype=np.int32)
        self._dur_pos_to_code = np.asarray(qt.duration_pos_to_code, dtype=np.int32)
        self._dur_samples = int(self._dur_code_to_pos.shape[0])
        self._dur_max_code = self._dur_samples - 1
        self._dur_pos_max = int(self._dur_pos_to_code.shape[0] - 1)
        self._dur_min_code = int(getattr(qt, "duration_min_code", 1))
        self._dur_min_code = max(0, min(self._dur_min_code, self._dur_max_code))
        self._dur_min_pos = int(self._dur_code_to_pos[self._dur_min_code])

        # dt tables
        self._dt_offset = int(qt.deltatime_code_offset)   # == dt_samples
        self._dt_samples = int(self._dt_offset)
        self._dt_code_to_pos = np.asarray(qt.deltatime_code_to_pos, dtype=np.int32)
        self._dt_pos_to_code = np.asarray(qt.deltatime_pos_to_code, dtype=np.int32)
        if self._dt_code_to_pos.shape[0] != 2 * self._dt_samples:
            raise ValueError("Bad deltatime_code_to_pos length.")
        if self._dt_pos_to_code.shape[0] != 2 * self._dt_samples:
            raise ValueError("Bad deltatime_pos_to_code length.")

        # pitch local-id range (note tokens)
        self._pitch_lo = self.special_n + int(cfg.pitch_min)
        self._pitch_hi = self.special_n + int(cfg.pitch_max)

    # -------------------------
    # helpers: token type checks
    # -------------------------
    def _is_note_pitch(self, pitch_local: int) -> bool:
        return (pitch_local >= self._pitch_lo) and (pitch_local <= self._pitch_hi)

    # -------------------------
    # helpers: dur encode/decode (pos domain)
    # -------------------------
    def _decode_dur_pos(self, dur_local: int) -> int:
        code = int(dur_local) - self.special_n
        if code < 0 or code > self._dur_max_code:
            return 0
        return int(self._dur_code_to_pos[code])

    def _encode_dur_local(self, pos: int) -> int:
        pos = int(pos)
        pos = max(self._dur_min_pos, min(pos, self._dur_pos_max))
        code = int(self._dur_pos_to_code[pos])
        code = max(self._dur_min_code, min(code, self._dur_max_code))
        return int(code + self.special_n)

    # -------------------------
    # helpers: dt encode/decode (pos domain, signed)
    # -------------------------
    def _decode_dt_pos(self, dt_local: int) -> int:
        dt_local = int(dt_local)
        sN = self.special_n
        if sN <= dt_local < sN + self._dt_samples:
            code = dt_local - sN  # 0..dt_samples-1
        elif sN + self._dt_samples <= dt_local < sN + 2 * self._dt_samples:
            mag = dt_local - (sN + self._dt_samples) + 1  # 1..dt_samples
            code = -mag
        else:
            code = 0

        idx = code + self._dt_offset
        idx = max(0, min(idx, 2 * self._dt_samples - 1))
        return int(self._dt_code_to_pos[idx])

    def _encode_dt_local(self, pos: int) -> int:
        # pos expected in [-dt_samples .. dt_samples-1] (uniform scheme)
        pos = int(pos)
        pos = max(-self._dt_samples, min(pos, self._dt_samples - 1))
        idx = pos + self._dt_offset
        idx = max(0, min(idx, 2 * self._dt_samples - 1))

        code = int(self._dt_pos_to_code[idx])  # signed code in [-dt_samples .. dt_samples-1]
        code = max(-self._dt_samples, min(code, self._dt_samples - 1))

        if code >= 0:
            return int(self.special_n + code)
        mag = -code
        mag = max(1, min(mag, self._dt_samples))
        return int((self.special_n + self._dt_samples) + (mag - 1))

    # -------------------------
    # helpers: pitch jitter
    # -------------------------
    def _sample_nonzero_offset(self, rng: np.random.Generator) -> int:
        cfg = self.cfg
        m = int(cfg.pitch_jitter_max_semitones)
        if m <= 0:
            return 0

        # prefer small offsets
        for _ in range(8):
            k = int(np.rint(rng.normal(0.0, float(cfg.pitch_jitter_sigma))))
            k = int(np.clip(k, -m, m))
            if k != 0:
                return k

        # fallback: uniform non-zero
        k = int(rng.integers(1, m + 1))
        if float(rng.random()) < 0.5:
            k = -k
        return k

    def _jitter_pitch_local(self, pitch_local: int, rng: np.random.Generator) -> int:
        cfg = self.cfg
        if not self._is_note_pitch(pitch_local):
            return int(pitch_local)

        midi = int(pitch_local) - self.special_n
        off = self._sample_nonzero_offset(rng)
        midi2 = int(np.clip(midi + off, int(cfg.pitch_min), int(cfg.pitch_max)))
        return int(self.special_n + midi2)

    # -------------------------
    # helpers: random partition of integer duration
    # -------------------------
    @staticmethod
    def _random_partition(total: int, n: int, min_each: int, rng: np.random.Generator) -> List[int]:
        total = int(total)
        n = int(n)
        min_each = int(min_each)
        if n <= 0:
            return []
        if total < n * min_each:
            raise ValueError("total < n*min_each")
        rem = total - n * min_each
        if rem == 0:
            return [min_each] * n
        extra = rng.multinomial(rem, np.ones(n, dtype=np.float64) / float(n)).astype(np.int32)
        out = (extra + min_each).tolist()
        return [int(x) for x in out]

    # -------------------------
    # core: find valid_len / eos_pos
    # -------------------------
    def _find_valid_len(self, x_local: np.ndarray) -> int:
        pitch = x_local[:, 0]
        pad = int(self.cfg.pad_id)
        idx = np.nonzero(pitch != pad)[0]
        return int(idx[-1] + 1) if idx.size > 0 else 0

    def _find_eos_pos(self, x_local, valid_len):
        if self.cfg.eos_id is None:
            return None
        eos = int(self.cfg.eos_id)
        pitch = x_local[:valid_len, 0]
        idx = np.where(pitch == eos)[0]
        return int(idx[-1]) if idx.size > 0 else None

    # -------------------------
    # note-level ops
    # -------------------------
    def _op_pre_grace(self, tok: np.ndarray, dur_pos: int, rng: np.random.Generator, orig_idx: int) -> Optional[Tuple[List[np.ndarray], List[int]]]:
        cfg = self.cfg
        if dur_pos < int(cfg.min_note_dur_pos_for_split):
            return None

        grace_min = max(int(cfg.grace_min_dur_pos), int(cfg.min_seg_dur_pos), int(self._dur_min_pos))
        # steal-time mode requires main >= dur_min_pos
        grace_max = int(round(float(dur_pos) * float(cfg.grace_max_ratio)))
        grace_max = min(grace_max, dur_pos - int(self._dur_min_pos))
        if grace_max < grace_min:
            return None

        g = int(rng.integers(grace_min, grace_max + 1))

        # overlap: dt between grace -> main is negative (slight overlap), reduce how much main is shortened
        use_overlap = (float(rng.random()) < float(cfg.grace_overlap_prob)) and (g >= 2)
        if use_overlap:
            k_max = min(int(cfg.grace_overlap_max_pos), g - 1)
            k = int(rng.integers(1, k_max + 1))
            dt_grace_pos = -k
            main_dur_pos = dur_pos - g + k
        else:
            dt_grace_pos = 0
            main_dur_pos = dur_pos - g

        if main_dur_pos < int(self._dur_min_pos):
            return None

        grace = tok.copy()
        grace[0] = self._jitter_pitch_local(int(tok[0]), rng)
        grace[1] = self._encode_dur_local(g)
        grace[2] = self._encode_dt_local(dt_grace_pos)

        main = tok.copy()
        main[1] = self._encode_dur_local(main_dur_pos)
        # main[2] 保持原 tok dt（到下一个 token 的 dt）

        return [grace, main], [PI_INSERTED, orig_idx]

    def _op_post_grace(self, tok: np.ndarray, dur_pos: int, rng: np.random.Generator, orig_idx: int) -> Optional[Tuple[List[np.ndarray], List[int]]]:
        cfg = self.cfg
        if dur_pos < int(cfg.min_note_dur_pos_for_split):
            return None

        grace_min = max(int(cfg.grace_min_dur_pos), int(cfg.min_seg_dur_pos), int(self._dur_min_pos))
        grace_max = int(round(float(dur_pos) * float(cfg.grace_max_ratio)))
        grace_max = min(grace_max, dur_pos - int(self._dur_min_pos))
        if grace_max < grace_min:
            return None

        g = int(rng.integers(grace_min, grace_max + 1))

        use_overlap = (float(rng.random()) < float(cfg.grace_overlap_prob)) and (g >= 2)
        if use_overlap:
            k_max = min(int(cfg.grace_overlap_max_pos), g - 1)
            k = int(rng.integers(1, k_max + 1))
            dt_main_pos = -k
            main_dur_pos = dur_pos - g + k
        else:
            dt_main_pos = 0
            main_dur_pos = dur_pos - g

        if main_dur_pos < int(self._dur_min_pos):
            return None

        main = tok.copy()
        main[1] = self._encode_dur_local(main_dur_pos)
        main[2] = self._encode_dt_local(dt_main_pos)  # main -> grace

        grace = tok.copy()
        grace[0] = self._jitter_pitch_local(int(tok[0]), rng)
        grace[1] = self._encode_dur_local(g)
        grace[2] = tok[2]  # grace -> next 继承原 dt

        return [main, grace], [orig_idx, PI_INSERTED]

    def _op_trill(self, tok: np.ndarray, dur_pos: int, rng: np.random.Generator, orig_idx: int, budget: int) -> Optional[Tuple[List[np.ndarray], List[int]]]:
        cfg = self.cfg
        if dur_pos < int(cfg.trill_min_dur_pos):
            return None

        max_n = min(int(cfg.trill_max_segments), int(budget) + 1)
        if max_n < 3:
            return None

        # prefer odd n so alternating ends nicely; pick n in [3..max_n]
        n = int(rng.integers(3, max_n + 1))
        if n % 2 == 0 and n > 3:
            n -= 1
        if n < 3:
            return None

        min_each = max(int(cfg.min_seg_dur_pos), int(self._dur_min_pos))
        if dur_pos < n * min_each:
            return None

        parts = self._random_partition(dur_pos, n, min_each, rng)

        # make first segment longer (more "main-like")
        j_max = int(np.argmax(np.asarray(parts)))
        parts[0], parts[j_max] = parts[j_max], parts[0]

        main_pitch = int(tok[0])
        neigh_pitch = self._jitter_pitch_local(main_pitch, rng)

        out_toks: List[np.ndarray] = []
        out_pi: List[int] = []
        for i in range(n):
            t = tok.copy()
            t[0] = main_pitch if (i % 2 == 0) else neigh_pitch
            t[1] = self._encode_dur_local(parts[i])
            if i < n - 1:
                t[2] = self._encode_dt_local(0)
            else:
                t[2] = tok[2]  # last -> next keeps original dt
            out_toks.append(t)
            out_pi.append(orig_idx if i == 0 else PI_INSERTED)

        return out_toks, out_pi

    def _op_turn(self, tok: np.ndarray, dur_pos: int, rng: np.random.Generator, orig_idx: int, budget: int) -> Optional[Tuple[List[np.ndarray], List[int]]]:
        cfg = self.cfg
        if dur_pos < int(cfg.turn_min_dur_pos):
            return None

        n = int(cfg.turn_segments)
        n = 3 if n <= 3 else 4
        if budget < (n - 1):
            return None

        min_each = max(int(cfg.min_seg_dur_pos), int(self._dur_min_pos))
        if dur_pos < n * min_each:
            return None

        parts = self._random_partition(dur_pos, n, min_each, rng)

        # first longer
        j_max = int(np.argmax(np.asarray(parts)))
        parts[0], parts[j_max] = parts[j_max], parts[0]

        main_pitch = int(tok[0])
        up = self._jitter_pitch_local(main_pitch, rng)
        lo = self._jitter_pitch_local(main_pitch, rng)

        # crude: ensure up/lo are different directions if possible
        # (not guaranteed, but ok)
        pattern = [main_pitch]
        if n == 3:
            pattern += [up, main_pitch]
        else:
            pattern += [up, main_pitch, lo]

        out_toks: List[np.ndarray] = []
        out_pi: List[int] = []
        for i in range(n):
            t = tok.copy()
            t[0] = int(pattern[i])
            t[1] = self._encode_dur_local(parts[i])
            if i < n - 1:
                t[2] = self._encode_dt_local(0)
            else:
                t[2] = tok[2]
            out_toks.append(t)
            out_pi.append(orig_idx if i == 0 else PI_INSERTED)

        return out_toks, out_pi

    def _op_reartic(self, tok: np.ndarray, dur_pos: int, rng: np.random.Generator, orig_idx: int, budget: int) -> Optional[Tuple[List[np.ndarray], List[int]]]:
        cfg = self.cfg
        if dur_pos < int(cfg.reartic_min_dur_pos):
            return None

        max_n = min(int(cfg.reartic_max_repeats), int(budget) + 1)
        if max_n < 2:
            return None
        n = int(rng.integers(2, max_n + 1))

        min_each = max(int(cfg.min_seg_dur_pos), int(self._dur_min_pos))
        if dur_pos < n * min_each:
            return None

        # sample small internal gaps (dt between repeats); must be feasible
        gap_max = max(0, int(cfg.reartic_gap_max_pos))
        gaps = np.zeros(n - 1, dtype=np.int32)
        if gap_max > 0:
            budget_total = dur_pos - n * min_each
            for _ in range(16):
                cand = rng.integers(0, gap_max + 1, size=(n - 1,), dtype=np.int32)
                if int(cand.sum()) <= int(budget_total):
                    gaps = cand
                    break

        dur_for_notes = dur_pos - int(gaps.sum())
        if dur_for_notes < n * min_each:
            return None

        parts = self._random_partition(dur_for_notes, n, min_each, rng)

        # first longer
        j_max = int(np.argmax(np.asarray(parts)))
        parts[0], parts[j_max] = parts[j_max], parts[0]

        out_toks: List[np.ndarray] = []
        out_pi: List[int] = []
        for i in range(n):
            t = tok.copy()
            t[1] = self._encode_dur_local(parts[i])
            if i < n - 1:
                t[2] = self._encode_dt_local(int(gaps[i]))
            else:
                t[2] = tok[2]
            out_toks.append(t)
            out_pi.append(orig_idx if i == 0 else PI_INSERTED)

        return out_toks, out_pi

    def _op_pair_repeat(
        self,
        tok_a: np.ndarray,
        tok_b: np.ndarray,
        dur_a_pos: int,
        dur_b_pos: int,
        rng: np.random.Generator,
        orig_idx_a: int,
        orig_idx_b: int,
        budget: int,
    ) -> Optional[Tuple[List[np.ndarray], List[int]]]:
        """
        Two-note tremolo / pair repeat:
          original: A --(dt_ab)--> B --(dt_b)--> C
          augmented: A0 B0 A1 B1 ... A(n-1) B(n-1) --(dt_b)--> C
        Only A0/B0 are backbone; the rest are PI_INSERTED.
        Timing is preserved (to C onset) by keeping:
          sum(dur(Ai))=durA, keep dt_ab only on A0;
          sum(dur(Bi))=durB, keep dt_b only on last B.
        """
        cfg = self.cfg
        if int(dur_a_pos) < int(cfg.pair_repeat_min_note_dur_pos):
            return None
        if int(dur_b_pos) < int(cfg.pair_repeat_min_note_dur_pos):
            return None

        dt_ab_pos = self._decode_dt_pos(int(tok_a[2]))
        if (not bool(cfg.pair_repeat_allow_overlap_dt)) and dt_ab_pos < 0:
            return None

        # 至少需要 2 pairs => extra >= 2
        if int(budget) < 2:
            return None

        min_each = max(int(cfg.min_seg_dur_pos), int(self._dur_min_pos))

        max_pairs_budget = int(budget) // 2 + 1
        max_pairs_dur_a = int(dur_a_pos) // int(min_each)
        max_pairs_dur_b = int(dur_b_pos) // int(min_each)
        n_max = min(int(cfg.pair_repeat_max_pairs), max_pairs_budget, max_pairs_dur_a, max_pairs_dur_b)
        if n_max < 2:
            return None

        n_pairs = int(rng.integers(2, n_max + 1))

        parts_a = self._random_partition(int(dur_a_pos), n_pairs, min_each, rng)
        parts_b = self._random_partition(int(dur_b_pos), n_pairs, min_each, rng)

        # 让第一次出现更“主音”一点：把最长片段换到 index 0
        j = int(np.argmax(np.asarray(parts_a)))
        parts_a[0], parts_a[j] = parts_a[j], parts_a[0]
        j = int(np.argmax(np.asarray(parts_b)))
        parts_b[0], parts_b[j] = parts_b[j], parts_b[0]

        dt0 = self._encode_dt_local(int(dt_ab_pos))
        dt_zero = self._encode_dt_local(0)

        out_toks: List[np.ndarray] = []
        out_pi: List[int] = []
        for i in range(n_pairs):
            a = tok_a.copy()
            a[1] = self._encode_dur_local(parts_a[i])
            a[2] = dt0 if i == 0 else dt_zero
            out_toks.append(a)
            out_pi.append(int(orig_idx_a) if i == 0 else PI_INSERTED)

            b = tok_b.copy()
            b[1] = self._encode_dur_local(parts_b[i])
            b[2] = dt_zero if i < (n_pairs - 1) else int(tok_b[2])
            out_toks.append(b)
            out_pi.append(int(orig_idx_b) if i == 0 else PI_INSERTED)

        return out_toks, out_pi

    # -------------------------
    # between-note insert
    # -------------------------
    def _sample_between_pitch_local(self, cur_pitch_local: int, next_pitch_local: int, rng: np.random.Generator) -> int:
        cfg = self.cfg
        if (not self._is_note_pitch(cur_pitch_local)) or (not self._is_note_pitch(next_pitch_local)):
            return int(cur_pitch_local)

        cur = int(cur_pitch_local) - self.special_n
        nxt = int(next_pitch_local) - self.special_n

        use_next = float(rng.random()) < float(cfg.between_use_next_pitch_prob)
        base = nxt if use_next else cur

        diff = nxt - cur
        if abs(diff) >= 3:
            step = int(rng.choice(np.asarray(cfg.between_pitch_step_choices, dtype=np.int32)))
            sgn = 1 if diff > 0 else -1
            midi = cur + sgn * step
        else:
            midi = base + self._sample_nonzero_offset(rng)

        midi = int(np.clip(midi, int(cfg.pitch_min), int(cfg.pitch_max)))
        return int(self.special_n + midi)

    def _sample_between_run_pitches_local(
        self,
        cur_pitch_local: int,
        next_pitch_local: int,
        n_notes: int,
        rng: np.random.Generator,
    ) -> List[int]:
        """
        生成 run 的 pitch 序列（local-id），长度为 n_notes。
        策略：在 cur -> next 之间做线性插值（更像 passing notes），再以一定概率对每个音做微小 pitch jitter。
        """
        cfg = self.cfg
        n_notes = int(n_notes)
        if n_notes <= 0:
            return []

        if (not self._is_note_pitch(int(cur_pitch_local))) or (not self._is_note_pitch(int(next_pitch_local))):
            return [int(cur_pitch_local)] * n_notes

        cur_m = int(cur_pitch_local) - self.special_n
        nxt_m = int(next_pitch_local) - self.special_n

        if cur_m == nxt_m:
            mids = np.full((n_notes,), cur_m, dtype=np.int32)
        else:
            vals = np.linspace(cur_m, nxt_m, num=n_notes + 2, dtype=np.float64)[1:-1]
            mids = np.rint(vals).astype(np.int32)

        pj = float(getattr(cfg, "between_run_pitch_jitter_prob", 0.0))
        if pj > 0.0:
            for i in range(n_notes):
                if float(rng.random()) < pj:
                    mids[i] = int(mids[i]) + int(self._sample_nonzero_offset(rng))

        mids = np.clip(mids, int(cfg.pitch_min), int(cfg.pitch_max)).astype(np.int32)
        return [int(self.special_n + int(m)) for m in mids.tolist()]

    def _maybe_insert_between(
        self,
        out_tokens: List[np.ndarray],
        out_pi: List[int],
        next_tok: np.ndarray,
        rng: np.random.Generator,
        allow_time_extend: bool,
    ) -> int:
        """
        在 out_tokens[-1] 与 next_tok 之间插一个装饰音。
        返回消耗的 extra token 数（0 或 1）。
        """
        cfg = self.cfg
        if len(out_tokens) == 0:
            return 0

        cur = out_tokens[-1]
        if (not self._is_note_pitch(int(cur[0]))) or (not self._is_note_pitch(int(next_tok[0]))):
            return 0

        # 不在 ... -> EOS 之间插（避免把 EOS 当音乐内容）
        if cfg.eos_id is not None and int(next_tok[0]) == int(cfg.eos_id):
            return 0

        dt_gap = self._decode_dt_pos(int(cur[2]))
        if dt_gap < 0:
            # 负 gap（重叠）场景先跳过，避免把近单声部变太复杂
            return 0

        ins_dur = int(rng.integers(int(cfg.between_ins_min_dur_pos), int(cfg.between_ins_max_dur_pos) + 1))

        if not allow_time_extend:
            if dt_gap < int(cfg.between_min_gap_pos):
                return 0
            if dt_gap < ins_dur:
                return 0
            dt_before = int(rng.integers(0, dt_gap - ins_dur + 1))
            dt_after = dt_gap - dt_before - ins_dur
        else:
            # 允许 “dt_gap 不够塞下 ins_dur”，则总时值增长
            dt_before = int(rng.integers(0, dt_gap + 1)) if dt_gap > 0 else 0
            dt_after = dt_gap - dt_before

        # 修改 cur 的 dt（cur -> inserted）
        cur2 = cur.copy()
        cur2[2] = self._encode_dt_local(dt_before)
        out_tokens[-1] = cur2

        # 插入 token（inserted -> next）
        ins_pitch = self._sample_between_pitch_local(int(cur2[0]), int(next_tok[0]), rng)
        ins = np.array(
            [ins_pitch, self._encode_dur_local(ins_dur), self._encode_dt_local(dt_after)],
            dtype=cur2.dtype
        )
        out_tokens.append(ins)
        out_pi.append(PI_INSERTED)
        return 1

    def _maybe_insert_between_run(
        self,
        out_tokens: List[np.ndarray],
        out_pi: List[int],
        next_tok: np.ndarray,
        rng: np.random.Generator,
        allow_time_extend: bool,
        *,
        budget: int,
    ) -> int:
        """
        在 out_tokens[-1] 与 next_tok 之间插入一个 run（2~K 个装饰音）。

        返回消耗的 extra token 数（0 或 n_notes）。
        - 非 time-extend：尽量放进 dt_gap 里（保持 next onset 不变）
        - time-extend：行为对齐 _maybe_insert_between：无论 dt_gap 多大，总时值都增长（增长量=run 总dur）
        """
        cfg = self.cfg
        budget = int(budget)
        if budget <= 0:
            return 0
        if len(out_tokens) == 0:
            return 0

        cur = out_tokens[-1]
        if (not self._is_note_pitch(int(cur[0]))) or (not self._is_note_pitch(int(next_tok[0]))):
            return 0

        # 不在 ... -> EOS 之间插
        if cfg.eos_id is not None and int(next_tok[0]) == int(cfg.eos_id):
            return 0

        dt_gap = self._decode_dt_pos(int(cur[2]))
        if dt_gap < 0:
            return 0

        n_min = max(1, int(cfg.between_run_min_notes))
        n_max = max(n_min, int(cfg.between_run_max_notes))
        n_max = min(n_max, budget)  # 不能超过预算
        if n_max < n_min:
            return 0
        n = int(rng.integers(n_min, n_max + 1))

        # 采样每个 run note 的 duration（pos）
        dur_min = max(
            int(cfg.between_run_ins_min_dur_pos),
            int(cfg.min_seg_dur_pos),
            int(self._dur_min_pos),
        )
        dur_max = max(dur_min, int(cfg.between_run_ins_max_dur_pos))

        if not allow_time_extend:
            if dt_gap < int(cfg.between_min_gap_pos):
                return 0
            if dt_gap < n * dur_min:
                return 0

            # 采样一个能塞进 dt_gap 的 duration 序列
            durs = None
            total_dur = 0
            for _ in range(32):
                cand = rng.integers(dur_min, dur_max + 1, size=(n,), dtype=np.int32)
                s = int(cand.sum())
                if s <= dt_gap:
                    durs = cand
                    total_dur = s
                    break
            if durs is None:
                # 保底：直接把总dur设为 dt_gap，并保证每个 >= dur_min
                rem = dt_gap - n * dur_min  # >=0 (因为上面检查过)
                extra = rng.multinomial(rem, np.ones(n, dtype=np.float64) / float(n)).astype(np.int32)
                durs = extra + dur_min
                total_dur = int(durs.sum())

            dt_before = int(rng.integers(0, dt_gap - total_dur + 1))
            dt_after = int(dt_gap - dt_before - total_dur)
        else:
            # time-extend：dt 只在 gap 内分配，dur 不从 gap 里扣 -> 总时值必增长（对齐 _maybe_insert_between）
            durs = rng.integers(dur_min, dur_max + 1, size=(n,), dtype=np.int32)
            dt_before = int(rng.integers(0, dt_gap + 1)) if dt_gap > 0 else 0
            dt_after = int(dt_gap - dt_before)

        # 修改 cur 的 dt（cur -> first inserted）
        cur2 = cur.copy()
        cur2[2] = self._encode_dt_local(dt_before)
        out_tokens[-1] = cur2

        pitches = self._sample_between_run_pitches_local(int(cur2[0]), int(next_tok[0]), n, rng)
        if len(pitches) != n:
            return 0

        dt_zero = self._encode_dt_local(0)
        dt_last = self._encode_dt_local(dt_after)

        # 追加 n 个 inserted notes：内部 dt=0，最后一个 dt=dt_after
        for i in range(n):
            ins = np.array(
                [int(pitches[i]), self._encode_dur_local(int(durs[i])), dt_zero if i < (n - 1) else dt_last],
                dtype=cur2.dtype,
            )
            out_tokens.append(ins)
            out_pi.append(PI_INSERTED)

        return int(n)

    # -------------------------
    # public api
    # -------------------------
    def augment(self, x_local: np.ndarray, rng: np.random.Generator, *, max_extra_tokens: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        x_local: [L,3] local-id
        max_extra_tokens: 覆写 cfg.max_extra_tokens（用于 benchmark 控制 rho 下界等）
        return: (x_aug_local[L,3], pi[L])
        """
        cfg = self.cfg
        if (not cfg.enable) or (float(rng.random()) > float(cfg.p_apply)):
            return self._identity(x_local)

        L = int(x_local.shape[0])
        if L <= 0:
            return self._identity(x_local)

        valid_len = self._find_valid_len(x_local)
        if valid_len <= 0:
            return self._identity(x_local)

        eos_pos = self._find_eos_pos(x_local, valid_len)
        if eos_pos is not None:
            valid_len = eos_pos + 1  # 只处理到 EOS（含）

        # 预算：最多只能用 pad 区插 token；另外加 max_extra_tokens 限制强度
        pad_budget = max(0, L - valid_len)
        max_extra = int(cfg.max_extra_tokens if max_extra_tokens is None else max_extra_tokens)
        max_extra = max(0, max_extra)
        budget = min(max_extra, int(pad_budget))

        # budget 不足时，仍可做“无结构变化”的增强，但你需求主要是结构性加花 -> 直接返回 identity
        if budget <= 0:
            return self._identity(x_local)

        allow_time_extend = (float(rng.random()) < float(cfg.p_enable_time_extend_seq))

        out_tokens: List[np.ndarray] = []
        out_pi: List[int] = []

        ops = 0

        l = 0
        while l < valid_len:
            tok = x_local[l].copy()
            pitch = int(tok[0])

            # EOS / special 不做加花
            if cfg.eos_id is not None and pitch == int(cfg.eos_id):
                out_tokens.append(tok)
                out_pi.append(int(l))
                l += 1
                continue
            # ---- OOD op: pair repeat (uses current + next) ----
            if (
                budget >= 2
                and (ops < int(cfg.max_ops_per_seq))
                and (l < valid_len - 1)
                and (float(rng.random()) < float(cfg.p_pair_repeat))
                and self._is_note_pitch(pitch)
            ):
                tok2 = x_local[l + 1].copy()
                pitch2 = int(tok2[0])
                if (cfg.eos_id is None) or (pitch2 != int(cfg.eos_id)):
                    if self._is_note_pitch(pitch2):
                        dur1 = self._decode_dur_pos(int(tok[1]))
                        dur2 = self._decode_dur_pos(int(tok2[1]))
                        ret = self._op_pair_repeat(
                            tok, tok2, dur1, dur2, rng,
                            orig_idx_a=int(l), orig_idx_b=int(l + 1),
                            budget=int(budget),
                        )
                        if ret is not None:
                            toks, pis = ret
                            extra = len(toks) - 2
                            if extra <= budget:
                                out_tokens.extend(toks)
                                out_pi.extend(pis)
                                budget -= extra
                                ops += 1
                                l += 2
                                continue

            if not self._is_note_pitch(pitch):
                out_tokens.append(tok)
                out_pi.append(int(l))
                l += 1
                continue

            dur_pos = self._decode_dur_pos(int(tok[1]))

            applied = False

            # complex ops first
            if (not applied) and budget > 0 and (ops < int(cfg.max_ops_per_seq)):
                if float(rng.random()) < float(cfg.p_trill):
                    ret = self._op_trill(tok, dur_pos, rng, orig_idx=int(l), budget=budget)
                    if ret is not None:
                        toks, pis = ret
                        extra = len(toks) - 1
                        if extra <= budget:
                            out_tokens.extend(toks)
                            out_pi.extend(pis)
                            budget -= extra
                            ops += 1
                            applied = True

            if (not applied) and budget > 0 and (ops < int(cfg.max_ops_per_seq)):
                if float(rng.random()) < float(cfg.p_turn):
                    ret = self._op_turn(tok, dur_pos, rng, orig_idx=int(l), budget=budget)
                    if ret is not None:
                        toks, pis = ret
                        extra = len(toks) - 1
                        if extra <= budget:
                            out_tokens.extend(toks)
                            out_pi.extend(pis)
                            budget -= extra
                            ops += 1
                            applied = True

            if (not applied) and budget > 0 and (ops < int(cfg.max_ops_per_seq)):
                if float(rng.random()) < float(cfg.p_rearticulation):
                    ret = self._op_reartic(tok, dur_pos, rng, orig_idx=int(l), budget=budget)
                    if ret is not None:
                        toks, pis = ret
                        extra = len(toks) - 1
                        if extra <= budget:
                            out_tokens.extend(toks)
                            out_pi.extend(pis)
                            budget -= extra
                            ops += 1
                            applied = True

            # requested split ops
            if (not applied) and budget > 0 and (ops < int(cfg.max_ops_per_seq)):
                r = float(rng.random())
                if r < float(cfg.p_pre_grace):
                    ret = self._op_pre_grace(tok, dur_pos, rng, orig_idx=int(l))
                    if ret is not None:
                        toks, pis = ret
                        extra = len(toks) - 1
                        if extra <= budget:
                            out_tokens.extend(toks)
                            out_pi.extend(pis)
                            budget -= extra
                            ops += 1
                            applied = True

                elif r < float(cfg.p_pre_grace) + float(cfg.p_post_grace):
                    ret = self._op_post_grace(tok, dur_pos, rng, orig_idx=int(l))
                    if ret is not None:
                        toks, pis = ret
                        extra = len(toks) - 1
                        if extra <= budget:
                            out_tokens.extend(toks)
                            out_pi.extend(pis)
                            budget -= extra
                            ops += 1
                            applied = True

            if not applied:
                out_tokens.append(tok)
                out_pi.append(int(l))

            # between-note insertion: insert between current original token l and next original token l+1
            if budget > 0 and (ops < int(cfg.max_ops_per_seq)) and (l < valid_len - 1):
                next_tok = x_local[l + 1]  # next original token (not yet processed)
                r = float(rng.random())
                used = 0

                # 1) run insert (NEW)
                if r < float(cfg.p_between_run):
                    used = self._maybe_insert_between_run(
                        out_tokens, out_pi, next_tok, rng, allow_time_extend, budget=budget
                    )
                    # fallback: 若 run 条件不满足（gap太短等），尽量退化为单音插入
                    if used <= 0 and budget > 0:
                        used = self._maybe_insert_between(out_tokens, out_pi, next_tok, rng, allow_time_extend)

                # 2) single insert (existing)
                elif r < float(cfg.p_between_run) + float(cfg.p_between_insert):
                    used = self._maybe_insert_between(out_tokens, out_pi, next_tok, rng, allow_time_extend)

                if used > 0:
                    budget -= int(used)
                    ops += 1
            
            l += 1

        # pack to fixed length L
        x_aug = np.full((L, 3), fill_value=int(cfg.pad_id), dtype=x_local.dtype)
        pi = np.full((L,), fill_value=PI_PAD, dtype=np.int32)

        new_len = min(L, len(out_tokens))
        if new_len > 0:
            x_aug[:new_len] = np.stack(out_tokens[:new_len], axis=0)
            pi[:new_len] = np.asarray(out_pi[:new_len], dtype=np.int32)

        return x_aug, pi

    def _identity(self, x_local: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cfg = self.cfg
        x = np.array(x_local, copy=True)
        L = int(x.shape[0])
        pi = np.full((L,), fill_value=PI_PAD, dtype=np.int32)
        valid_len = self._find_valid_len(x)
        eos_pos = self._find_eos_pos(x, valid_len)
        if eos_pos is not None:
            valid_len = eos_pos + 1
        if valid_len > 0:
            pi[:valid_len] = np.arange(valid_len, dtype=np.int32)
        return x, pi
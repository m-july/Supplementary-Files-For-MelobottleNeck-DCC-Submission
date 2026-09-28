# main/retrieval/simplemono_rel.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from preproc.simplemono_preproc.decoder_midi import SimpleMonoDecodingConfig


@dataclass(frozen=True)
class RelTokenConfig:
    """
    检索用 note-transition 表示。

    mode:
      - dp_dur_ratio : 旧行为。tok = (dp+128)<<16 | num<<8 | den
      - dp_only      : 仅保留 pitch interval (dp)
      - dp_coarse_dur: tok = (dp+128)<<16 | bin_prev<<8 | bin_next

    其中：
      - dp: 相邻 note pitch difference
      - bin_prev / bin_next: 相邻两个音的 coarse duration bin
    """
    mode: str = "dp_dur_ratio"

    # common
    dp_offset: int = 128
    dur_min_pos: int = 1

    # exact dur-ratio mode
    num_max: int = 255
    den_max: int = 255

    # coarse duration mode: upper bounds (in pos units)
    # bin 0: dur<=1
    # bin 1: dur<=2
    # bin 2: dur<=3
    # ...
    # bin len(edges): dur > last_edge
    coarse_dur_bin_edges: Tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def load_decoding_cfg(simplemono_pkl: str | Path) -> SimpleMonoDecodingConfig:
    return SimpleMonoDecodingConfig.from_simplemono_pkl(Path(simplemono_pkl))


def infer_used_len_events(
    events_global: np.ndarray, cfg: SimpleMonoDecodingConfig
) -> int:
    """
    尽量稳健地推断 used_len_events（即 [0:used_len] 为有效，含 EOS）。
    优先找 EOS；否则找最后一个非 PAD。
    """
    ev = np.asarray(events_global)
    if ev.ndim != 2 or ev.shape[1] != 3:
        raise ValueError(f"events must be [L,3], got {ev.shape}")

    pitch = ev[:, 0].astype(np.int64, copy=False)
    pad = int(cfg.pad_id)
    eos = int(cfg.eos_id)

    # 有效区：pitch != pad
    valid = pitch != pad
    if not np.any(valid):
        return 0

    last_valid = int(np.nonzero(valid)[0][-1])
    used = last_valid + 1

    # EOS（只在有效区内找）
    eos_pos = np.where(pitch[:used] == eos)[0]
    if eos_pos.size > 0:
        return int(eos_pos[0] + 1)
    return int(used)


def extract_note_pitch_dur(
    events_global: np.ndarray,
    cfg: SimpleMonoDecodingConfig,
    *,
    used_len_events: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 events (global IDs) 中提取 note-level 的 pitch & duration(pos units).

    返回：
      pitch: int16 [N]
      dur_pos: int16 [N]  (>=1)
    """
    ev = np.asarray(events_global)
    if used_len_events is None:
        used_len_events = infer_used_len_events(ev, cfg)

    used_len_events = int(used_len_events)
    if used_len_events <= 0:
        return np.zeros((0,), np.int16), np.zeros((0,), np.int16)

    # note rows 通常是 [1 : used_len-1)，去掉 BOS 和 EOS
    st = 1
    ed = max(st, used_len_events - 1)
    if ed <= st:
        return np.zeros((0,), np.int16), np.zeros((0,), np.int16)

    rows = ev[st:ed]  # [N,3]
    pids = rows[:, 0].astype(np.int64, copy=False)

    # 过滤 special（保守一点）
    specials = {int(cfg.pad_id), int(cfg.bos_id), int(cfg.eos_id), int(cfg.unk_id), int(cfg.mask_id)}
    m = np.ones((pids.shape[0],), dtype=bool)
    for s in specials:
        m &= (pids != s)

    if not np.any(m):
        return np.zeros((0,), np.int16), np.zeros((0,), np.int16)

    pids = pids[m]
    dids = rows[:, 1].astype(np.int64, copy=False)[m]

    pitch = (pids - int(cfg.pitch_offset)).astype(np.int16, copy=False)

    dur_code = (dids - int(cfg.dur_offset)).astype(np.int32, copy=False)
    dur_code = np.clip(dur_code, 0, int(cfg.max_dur_code)).astype(np.int16, copy=False)
    dur_pos = np.maximum(dur_code, 1).astype(np.int16, copy=False)

    return pitch, dur_pos


def _dur_pos_to_coarse_bins(
    dur_pos: np.ndarray,
    *,
    edges: Tuple[int, ...],
) -> np.ndarray:
    """
    dur_pos -> coarse duration bin ids (uint32)

    edges 是升序的 upper bounds（单位：pos）。
    返回 bin 范围：
      0 .. len(edges)
    """
    if len(edges) == 0:
        raise ValueError("coarse_dur_bin_edges must not be empty.")

    e = np.asarray(edges, dtype=np.int32)
    if np.any(e <= 0):
        raise ValueError(f"coarse_dur_bin_edges must be positive, got {edges}")
    if np.any(e[1:] < e[:-1]):
        raise ValueError(f"coarse_dur_bin_edges must be non-decreasing, got {edges}")

    d = np.asarray(dur_pos, dtype=np.int32)
    b = np.searchsorted(e, d, side="left").astype(np.int32, copy=False)

    if b.size > 0 and int(b.max()) > 255:
        raise ValueError(
            f"Too many coarse duration bins for uint8 packing: max_bin={int(b.max())}. "
            f"Please use <=255 bins."
        )

    return b.astype(np.uint32, copy=False)


def pitch_dur_to_rel_tokens(
    pitch: np.ndarray,
    dur_pos: np.ndarray,
    *,
    rel_cfg: RelTokenConfig = RelTokenConfig(),
) -> np.ndarray:
    """
    pitch,dur -> rel transition tokens (uint32), length = N-1

    modes:
      - dp_dur_ratio : old exact relative-rhythm encoding
      - dp_only
      - dp_coarse_dur
    """
    p = np.asarray(pitch, dtype=np.int16)
    d = np.asarray(dur_pos, dtype=np.int16)
    if p.size < 2:
        return np.zeros((0,), dtype=np.uint32)

    # dp in [-127,127] -> uint8 space [0,255]
    dp = (p[1:].astype(np.int16) - p[:-1].astype(np.int16)).astype(np.int16)
    dp_u = (dp.astype(np.int32) + int(rel_cfg.dp_offset)).astype(np.int32)
    dp_u = np.clip(dp_u, 0, 255).astype(np.uint32)

    mode = str(rel_cfg.mode).lower().strip()

    if mode == "dp_only":
        return dp_u.astype(np.uint32, copy=False)

    d0 = np.maximum(d[:-1].astype(np.int32), int(rel_cfg.dur_min_pos))
    d1 = np.maximum(d[1:].astype(np.int32), int(rel_cfg.dur_min_pos))

    if mode == "dp_dur_ratio":
        g = np.gcd(d0, d1)
        g = np.maximum(g, 1)

        num = (d1 // g).astype(np.int32)
        den = (d0 // g).astype(np.int32)

        num = np.clip(num, 0, int(rel_cfg.num_max)).astype(np.uint32)
        den = np.clip(den, 0, int(rel_cfg.den_max)).astype(np.uint32)

        tok = (dp_u << np.uint32(16)) | (num << np.uint32(8)) | den
        return tok.astype(np.uint32, copy=False)

    if mode == "dp_coarse_dur":
        b0 = _dur_pos_to_coarse_bins(d0, edges=rel_cfg.coarse_dur_bin_edges)
        b1 = _dur_pos_to_coarse_bins(d1, edges=rel_cfg.coarse_dur_bin_edges)

        tok = (dp_u << np.uint32(16)) | (b0 << np.uint32(8)) | b1
        return tok.astype(np.uint32, copy=False)

    raise ValueError(f"Unknown RelTokenConfig.mode: {rel_cfg.mode}")


def events_to_rel_tokens(
    events_global: np.ndarray,
    cfg: SimpleMonoDecodingConfig,
    *,
    used_len_events: Optional[int] = None,
    rel_cfg: RelTokenConfig = RelTokenConfig(),
) -> np.ndarray:
    pitch, dur = extract_note_pitch_dur(events_global, cfg, used_len_events=used_len_events)
    return pitch_dur_to_rel_tokens(pitch, dur, rel_cfg=rel_cfg)


def decode_rel_token(
    tok: int,
    *,
    rel_cfg: RelTokenConfig = RelTokenConfig(),
) -> tuple[int, Optional[int], Optional[int]]:
    """
    debug helper

    returns:
      - dp_only      -> (dp, None, None)
      - dp_dur_ratio -> (dp, num, den)
      - dp_coarse_dur-> (dp, bin_prev, bin_next)
    """
    tok = int(tok) & 0xFFFFFFFF
    mode = str(rel_cfg.mode).lower().strip()

    if mode == "dp_only":
        dp_u = tok & 0xFF
        dp = dp_u - int(rel_cfg.dp_offset)
        return int(dp), None, None

    dp_u = (tok >> 16) & 0xFF
    a = (tok >> 8) & 0xFF
    b = tok & 0xFF
    dp = dp_u - int(rel_cfg.dp_offset)
    return int(dp), int(a), int(b)
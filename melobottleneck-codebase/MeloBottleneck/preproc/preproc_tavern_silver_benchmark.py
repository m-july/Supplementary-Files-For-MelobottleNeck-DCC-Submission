# preproc_tavern_silver_benchmark.py
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from tqdm import tqdm

from main.vocab_utils import load_vocab_info
from main.ornament import PI_INSERTED, PI_PAD

from .simplemono_preproc.score_readers import extract_voices
from .simplemono_preproc.transforms import trim_empty_measures
from .simplemono_preproc.types import NoteEv, VoiceData
from .simplemono_preproc.quantization import encode_dur_pos, encode_dt_pos
from .simplemono_preproc.encoder import encode_triples_to_events, pad_or_truncate_events
from .simplemono_preproc.vocab import SimpleMonoVocab
from .simplemono_preproc.npy_stream import NpyAppendWriter
from .simplemono_preproc.pipelines.pretrain import default_max_len_tokens

from preproc.dataset_stats import (
    DEFAULT_STATS_BPM,
    SeqExportStatsAccumulator,
    compute_span_pos_from_local_events,
    load_pos_resolution_from_vocab_pkl,
    write_json,
    aggregate_split_dirs_export_stats,
)


# -------------------------
# helpers copied from preproc_ornament_benchmark.py (same semantics)
# -------------------------
def _seq_len_including_eos(x_local: np.ndarray, *, pad_id: int, eos_id: Optional[int]) -> int:
    pitch = x_local[:, 0]
    L = int(pitch.shape[0])

    pad_idx = np.where(pitch == int(pad_id))[0]
    valid_len = int(pad_idx[0]) if pad_idx.size > 0 else L

    if eos_id is not None:
        eos_idx = np.where(pitch[:valid_len] == int(eos_id))[0]
        if eos_idx.size > 0:
            valid_len = int(eos_idx[0]) + 1
    return int(valid_len)


def _global_to_local(x_global: np.ndarray, vocab) -> np.ndarray:
    x_global = np.asarray(x_global, dtype=np.int64)
    L, A = x_global.shape
    if A != 3:
        raise ValueError(f"Expected x shape [L,3], got {x_global.shape}")

    pitch_g = x_global[:, 0]
    dur_g = x_global[:, 1]
    dt_g = x_global[:, 2]

    pitch_l = vocab.global2local_pitch[pitch_g]
    dur_l = vocab.global2local_duration[dur_g]
    dt_l = vocab.global2local_dt[dt_g]

    pitch = np.where(pitch_l >= 0, pitch_l, pitch_g)
    dur = np.where(dur_l >= 0, dur_l, dur_g)
    dt = np.where(dt_l >= 0, dt_l, dt_g)

    x_local = np.stack([pitch, dur, dt], axis=-1)
    return x_local.astype(np.int32)


# -------------------------
# Krn/*_score.krn pairing
# -------------------------
_RX_SCORE = re.compile(r"^([BM][A-Z0-9]+)_V?(\d{2})_([^_]+)_score\.krn$")


def parse_score_filename(fname: str) -> Optional[Dict[str, str]]:
    """
    Support both:
      - B063_01_02_score.krn
      - M613_V08_022_score.krn
    """
    m = _RX_SCORE.match(fname)
    if not m:
        return None
    opus, var_id, phrase_key = m.groups()
    return {
        "opus": opus,               # e.g. M613, BO76, B063
        "var_id": var_id,           # "00".."NN"
        "phrase_key": phrase_key,   # e.g. "01", "022", "e"
        "is_theme": (var_id == "00"),
    }


def phrase_base_from_key(phrase_key: str) -> Optional[str]:
    """
    Robust base phrase id for matching to theme:
      - "01"  -> "01"
      - "022" -> "02"   (duplicate occurrence of theme phrase 02)
      - "2"   -> "02"   (rare; be defensive)
      - "e"   -> None   (cannot match to theme by index)
    """
    m = re.match(r"^(\d+)", phrase_key)
    if not m:
        return None
    digits = m.group(1)
    if len(digits) == 1:
        return digits.zfill(2)
    return digits[:2]


def list_variation_theme_pairs_from_score(tavern_root: Path) -> List[Dict[str, Any]]:
    """
    Pairing rules (score-only):
      - theme files: var_id == "00"
      - variation files: var_id != "00"
      - match by (opus, phrase_base) where phrase_base is derived from phrase_key:
          phrase_key "022" => base "02" (pairs to theme phrase "02")
      - phrase_key like "e/f/g/h" => base None => skipped (no matching theme phrase)
    """
    tavern_root = Path(tavern_root)

    # themes[(opus, base)] = {"path": Path, "phrase_key": str}
    themes: Dict[Tuple[str, str], Dict[str, Any]] = {}
    variations: List[Dict[str, Any]] = []

    for krn_file in tavern_root.rglob("Krn/*_score.krn"):
        info = parse_score_filename(krn_file.name)
        if info is None:
            continue

        base = phrase_base_from_key(info["phrase_key"])
        info2 = dict(info)
        info2["phrase_base"] = base
        info2["path"] = krn_file

        if info["is_theme"]:
            if base is None:
                # theme without numeric phrase id is unexpected; skip
                continue
            k = (info["opus"], base)
            # prefer the shorter phrase_key for theme (e.g., "02" over "022" if both exist)
            if k not in themes or len(str(info["phrase_key"])) < len(str(themes[k]["phrase_key"])):
                themes[k] = {"path": krn_file, "phrase_key": info["phrase_key"], "phrase_base": base}
        else:
            variations.append(info2)

    pairs: List[Dict[str, Any]] = []
    for v in variations:
        base = v["phrase_base"]
        if base is None:
            continue
        k = (v["opus"], base)
        th = themes.get(k)
        if th is None:
            continue
        pairs.append({
            "opus": v["opus"],
            "variation": v["var_id"],
            "theme_phrase_base": base,
            "var_phrase_key": v["phrase_key"],
            "theme_file": th["path"],
            "variation_file": v["path"],
        })

    pairs.sort(key=lambda p: (p["opus"], p["variation"], p["theme_phrase_base"], p["var_phrase_key"], p["variation_file"].name))
    return pairs


# -------------------------
# treble + melody extraction
# -------------------------
def _notes_shift_to_zero(notes: List[NoteEv]) -> List[NoteEv]:
    if not notes:
        return []
    s0 = min(int(n.start) for n in notes)
    if s0 == 0:
        return notes
    out: List[NoteEv] = []
    for n in notes:
        st = int(n.start) - s0
        en = int(n.end) - s0
        if en <= st:
            en = st + 1
        out.append(NoteEv(int(n.pitch), st, en))
    return out


def _choose_treble_voice(voices: List[VoiceData]) -> Optional[VoiceData]:
    best = None
    best_med = None
    best_n = -1

    for v in voices:
        notes = trim_empty_measures(v.notes, v.measures)
        if not notes:
            continue
        pitches = np.asarray([int(n.pitch) for n in notes], dtype=np.int32)
        if pitches.size == 0:
            continue
        med = float(np.median(pitches))
        n = int(pitches.size)
        if best is None or med > float(best_med) + 1e-9 or (abs(med - float(best_med)) <= 1e-9 and n > best_n):
            best = v
            best_med = med
            best_n = n
    return best


def _melody_monophonize(notes: List[NoteEv], melody_mode: str) -> List[NoteEv]:
    if not notes:
        return []
    notes = sorted(notes, key=lambda x: (int(x.start), int(x.end), int(x.pitch)))
    notes = _notes_shift_to_zero(notes)

    if melody_mode == "all":
        return notes

    out: List[NoteEv] = []
    i = 0
    while i < len(notes):
        st = int(notes[i].start)
        j = i + 1
        while j < len(notes) and int(notes[j].start) == st:
            j += 1
        grp = notes[i:j]
        best = grp[0]
        for n in grp[1:]:
            if int(n.pitch) > int(best.pitch):
                best = n
            elif int(n.pitch) == int(best.pitch):
                if (int(n.end) - int(n.start)) > (int(best.end) - int(best.start)):
                    best = n
        out.append(best)
        i = j

    return out


def load_treble_melody_notes(
    krn_path: Path,
    melody_mode: str,
    cache: Dict[str, List[NoteEv]],
) -> List[NoteEv]:
    k = str(Path(krn_path).resolve())
    if k in cache:
        return cache[k]

    try:
        voices = extract_voices(Path(krn_path), xml_group="staff_voice")
    except Exception:
        cache[k] = []
        return []

    if not voices:
        cache[k] = []
        return []

    treble = _choose_treble_voice(voices)
    if treble is None:
        cache[k] = []
        return []

    notes = trim_empty_measures(treble.notes, treble.measures)
    notes_mono = _melody_monophonize(notes, melody_mode=melody_mode)
    cache[k] = notes_mono
    return notes_mono


# -------------------------
# Needleman–Wunsch alignment
# -------------------------
@dataclass
class AlignConfig:
    gap_ins: float = -0.60
    gap_del: float = -0.90

    w_pitch: float = 2.5
    w_time: float = 1.0
    w_dur: float = 1.0
    w_stability: float = 0.1

    pitch_pc_sigma: float = 2.0
    pitch_oct_sigma: float = 2.0
    time_sigma: float = 0.15
    dur_sigma: float = 0.25

    match_bias: float = 1.60


@dataclass
class FilterConfig:
    min_theme_notes: int = 6
    min_var_notes: int = 10
    min_skel_notes: int = 6

    min_match_score: float = 0.50
    min_matches: int = 6
    min_coverage: float = 0.70


def _compute_features(notes: List[NoteEv]) -> Dict[str, np.ndarray]:
    if not notes:
        return {
            "pitch": np.zeros((0,), dtype=np.int32),
            "on": np.zeros((0,), dtype=np.float32),
            "dur": np.zeros((0,), dtype=np.float32),
        }

    start = np.asarray([int(n.start) for n in notes], dtype=np.int32)
    end = np.asarray([int(n.end) for n in notes], dtype=np.int32)
    dur = np.maximum(1, end - start).astype(np.int32)
    p = np.asarray([int(n.pitch) for n in notes], dtype=np.int32)

    s0 = int(start.min())
    e1 = int(end.max())
    span = max(1, e1 - s0)

    on = (start - s0).astype(np.float32) / float(span)
    du = dur.astype(np.float32) / float(span)
    return {"pitch": p, "on": on, "dur": du}


def _match_score_matrix(theme: List[NoteEv], var: List[NoteEv], cfg: AlignConfig) -> np.ndarray:
    fy = _compute_features(theme)
    fx = _compute_features(var)
    py, ty, dy = fy["pitch"], fy["on"], fy["dur"]
    px, tx, dx = fx["pitch"], fx["on"], fx["dur"]

    if py.size == 0 or px.size == 0:
        return np.zeros((py.size, px.size), dtype=np.float32)

    diff = np.abs(py[:, None] - px[None, :]).astype(np.int32)
    dpc = (diff % 12).astype(np.float32)
    dpc = np.minimum(dpc, 12.0 - dpc)
    doct = (diff.astype(np.float32) / 12.0)

    pc_sigma = float(cfg.pitch_pc_sigma)
    oct_sigma = float(cfg.pitch_oct_sigma)
    pitch_sim = np.exp(- (dpc / pc_sigma) ** 2) * np.exp(- (doct / oct_sigma) ** 2)

    t_sigma = float(cfg.time_sigma)
    time_sim = np.exp(- (np.abs(ty[:, None] - tx[None, :]) / t_sigma) ** 2)

    d_sigma = float(cfg.dur_sigma)
    dur_sim = np.exp(- (np.abs(dy[:, None] - dx[None, :]) / d_sigma) ** 2)

    stability = dx[None, :]

    S = (
        float(cfg.w_pitch) * pitch_sim
        + float(cfg.w_time) * time_sim
        + float(cfg.w_dur) * dur_sim
        + float(cfg.w_stability) * stability
        - float(cfg.match_bias)
    )
    return S.astype(np.float32)


def needleman_wunsch(theme: List[NoteEv], var: List[NoteEv], cfg: AlignConfig) -> Tuple[List[Tuple[int, int]], np.ndarray, float]:
    N = len(theme)
    M = len(var)
    S = _match_score_matrix(theme, var, cfg)

    NEG = np.float32(-1e9)
    dp = np.full((N + 1, M + 1), NEG, dtype=np.float32)
    ptr = np.zeros((N + 1, M + 1), dtype=np.int8)  # 0 diag, 1 up, 2 left

    dp[0, 0] = np.float32(0.0)
    for j in range(1, M + 1):
        dp[0, j] = dp[0, j - 1] + np.float32(cfg.gap_ins)
        ptr[0, j] = 2
    for i in range(1, N + 1):
        dp[i, 0] = dp[i - 1, 0] + np.float32(cfg.gap_del)
        ptr[i, 0] = 1

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            s_diag = dp[i - 1, j - 1] + S[i - 1, j - 1]
            s_up = dp[i - 1, j] + np.float32(cfg.gap_del)
            s_left = dp[i, j - 1] + np.float32(cfg.gap_ins)

            if s_diag >= s_up and s_diag >= s_left:
                dp[i, j] = s_diag
                ptr[i, j] = 0
            elif s_up >= s_left:
                dp[i, j] = s_up
                ptr[i, j] = 1
            else:
                dp[i, j] = s_left
                ptr[i, j] = 2

    i, j = N, M
    matches: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        m = int(ptr[i, j])
        if m == 0:
            i -= 1
            j -= 1
            matches.append((i, j))
        elif m == 1:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    return matches, S, float(dp[N, M])


# -------------------------
# SimpleMono encoding from NoteEv
# -------------------------
def notes_to_triples(notes: List[NoteEv]) -> Tuple[List[Tuple[int, int, int]], int]:
    if not notes:
        return [], 0
    notes = sorted(notes, key=lambda x: (int(x.start), int(x.end), int(x.pitch)))
    bos_start_pos = int(notes[0].start)

    triples: List[Tuple[int, int, int]] = []
    for i, n in enumerate(notes):
        pitch = int(n.pitch)

        dur_pos = max(1, int(n.end - n.start))
        dur_code = encode_dur_pos(dur_pos)

        if i + 1 < len(notes):
            raw_dt = int(notes[i + 1].start - n.end)
        else:
            raw_dt = 0
        dt_code = encode_dt_pos(raw_dt)

        triples.append((pitch, int(dur_code), int(dt_code)))
    return triples, bos_start_pos


def encode_notes_to_padded_events_global(
    notes: List[NoteEv],
    vocab_global: SimpleMonoVocab,
    max_events: int,
) -> Tuple[np.ndarray, int]:
    triples, bos_start_pos = notes_to_triples(notes)
    events = encode_triples_to_events(triples, vocab_global, bos_start_pos=bos_start_pos)
    padded, used_len = pad_or_truncate_events(events, max_events=max_events, vocab=vocab_global)
    return padded, int(used_len)


# -------------------------
# build pi
# -------------------------
def build_pi_array(
    max_events: int,
    len_orn: int,
    len_skel: int,
    skel_note_indices_in_orn: List[int],
) -> np.ndarray:
    pi = np.full((max_events,), fill_value=int(PI_PAD), dtype=np.int32)
    if len_orn <= 0:
        return pi

    skel_note_indices_in_orn = sorted(skel_note_indices_in_orn)
    note2skel = {int(j): int(1 + k) for k, j in enumerate(skel_note_indices_in_orn)}

    pi[0] = 0  # BOS

    n_notes_orn = max(0, len_orn - 2)
    for j in range(n_notes_orn):
        t = 1 + j
        pi[t] = note2skel.get(j, int(PI_INSERTED))

    eos_t = len_orn - 1
    pi[eos_t] = int(len_skel - 1)  # EOS -> EOS
    return pi


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tavern_root", type=str, required=True, help="TAVERN-master root")
    ap.add_argument("--vocab_pkl", type=str, required=True, help="SimpleMono.pkl (for global->local mapping)")
    ap.add_argument("--output_dir", type=str, required=True, help="output root dir")
    ap.add_argument("--split_name", type=str, default="test", help="subfolder name (default test)")

    ap.add_argument("--max_len_tokens", type=int, default=0,
                    help="0 => default max_len_tokens. Must be divisible by 3.")

    ap.add_argument("--melody_mode", type=str, default="top", choices=["top", "all"])

    ap.add_argument("--min_rho", type=float, default=(1.0 / 3.0))
    ap.add_argument("--max_rho", type=float, default=1.0)

    ap.add_argument("--min_theme_notes", type=int, default=6)
    ap.add_argument("--min_var_notes", type=int, default=10)
    ap.add_argument("--min_skel_notes", type=int, default=6)
    ap.add_argument("--min_match_score", type=float, default=0.50)
    ap.add_argument("--min_matches", type=int, default=6)
    ap.add_argument("--min_coverage", type=float, default=0.70)

    ap.add_argument("--max_pairs", type=int, default=0, help="0 => all pairs")
    ap.add_argument("--write_meta_jsonl", action="store_true", help="write metadata.jsonl")
    args = ap.parse_args()

    tavern_root = Path(args.tavern_root)
    out_root = Path(args.output_dir) / str(args.split_name)

    if out_root.exists():
        raise FileExistsError(f"Output dir already exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)

    max_len_tokens = int(args.max_len_tokens)
    if max_len_tokens <= 0:
        max_len_tokens = int(default_max_len_tokens())
    if max_len_tokens % 3 != 0:
        raise ValueError(f"max_len_tokens must be divisible by 3, got {max_len_tokens}")
    max_events = max_len_tokens // 3
    if max_events < 2:
        raise ValueError("max_events must be >= 2")

    min_rho = float(args.min_rho)
    max_rho = float(args.max_rho)
    if not (0.0 < min_rho <= 1.0 and 0.0 < max_rho <= 1.0 and min_rho <= max_rho):
        raise ValueError(f"Bad rho range: [{min_rho}, {max_rho}]")

    align_cfg = AlignConfig()
    filt_cfg = FilterConfig(
        min_theme_notes=int(args.min_theme_notes),
        min_var_notes=int(args.min_var_notes),
        min_skel_notes=int(args.min_skel_notes),
        min_match_score=float(args.min_match_score),
        min_matches=int(args.min_matches),
        min_coverage=float(args.min_coverage),
    )

    print(f"[TAVERN] {tavern_root}")
    print(f"[OUT]   {out_root}")
    print(f"[LEN]   max_len_tokens={max_len_tokens} => max_events={max_events}")
    print(f"[MODE]  melody_mode={args.melody_mode}")
    print(f"[RHO]   range=[{min_rho:.4f}, {max_rho:.4f}]")
    print(f"[ALIGN] {asdict(align_cfg)}")
    print(f"[FILT]  {asdict(filt_cfg)}")

    vocab_info = load_vocab_info(str(args.vocab_pkl))
    vocab_global = SimpleMonoVocab.build()

    pos_resolution = load_pos_resolution_from_vocab_pkl(args.vocab_pkl, default=12)
    bpm = DEFAULT_STATS_BPM

    acc_x = SeqExportStatsAccumulator()
    acc_xorn = SeqExportStatsAccumulator()

    uniq_theme = set()
    uniq_var = set()

    pairs = list_variation_theme_pairs_from_score(tavern_root)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    print(f"[PAIR]  found {len(pairs)} score-based variation-theme phrase pairs.")

    w_x = NpyAppendWriter(out_root / "x.npy", dtype=np.int16, row_shape=(max_events, 3))
    w_xorn = NpyAppendWriter(out_root / "x_orn.npy", dtype=np.int16, row_shape=(max_events, 3))
    w_pi = NpyAppendWriter(out_root / "pi.npy", dtype=np.int32, row_shape=(max_events,))
    w_len_x = NpyAppendWriter(out_root / "len_x.npy", dtype=np.int32, row_shape=())
    w_len_xorn = NpyAppendWriter(out_root / "len_x_orn.npy", dtype=np.int32, row_shape=())
    w_rho = NpyAppendWriter(out_root / "rho.npy", dtype=np.float32, row_shape=())

    meta_f = None
    if args.write_meta_jsonl:
        meta_f = (out_root / "metadata.jsonl").open("w", encoding="utf-8")

    note_cache: Dict[str, List[NoteEv]] = {}

    skip = {
        "parse_or_empty": 0,
        "too_short": 0,
        "low_alignment": 0,
        "empty_after_trunc": 0,
        "rho_out_of_range": 0,
        "exception": 0,
    }
    rho_list: List[float] = []
    cov_list: List[float] = []
    match_list: List[int] = []

    max_notes = max_events - 2

    for p in tqdm(pairs, desc="[Build TAVERN O2B (score)]", dynamic_ncols=True, smoothing=0.0):
        try:
            theme_path: Path = p["theme_file"]
            var_path: Path = p["variation_file"]

            theme_notes = load_treble_melody_notes(theme_path, melody_mode=args.melody_mode, cache=note_cache)
            var_notes = load_treble_melody_notes(var_path, melody_mode=args.melody_mode, cache=note_cache)

            if (not theme_notes) or (not var_notes):
                skip["parse_or_empty"] += 1
                continue

            if len(theme_notes) < filt_cfg.min_theme_notes or len(var_notes) < filt_cfg.min_var_notes:
                skip["too_short"] += 1
                continue

            # prefix truncate (no windowing)
            theme_notes_t = theme_notes[:max_notes]
            var_notes_t = var_notes[:max_notes]

            matches, S, best_score = needleman_wunsch(theme_notes_t, var_notes_t, cfg=align_cfg)

            kept: List[Tuple[int, int, float]] = []
            for (i, j) in matches:
                sc = float(S[i, j])
                if sc >= float(filt_cfg.min_match_score):
                    kept.append((i, j, sc))

            kept_n = int(len(kept))
            N = len(theme_notes_t)
            M = len(var_notes_t)
            coverage = float(kept_n) / float(max(1, min(N, M)))

            if kept_n < int(filt_cfg.min_matches) or coverage < float(filt_cfg.min_coverage):
                skip["low_alignment"] += 1
                continue

            skel_note_indices = sorted({int(j) for (_, j, _) in kept})
            if len(skel_note_indices) < int(filt_cfg.min_skel_notes):
                skip["low_alignment"] += 1
                continue

            skel_note_indices = [j for j in skel_note_indices if 0 <= j < len(var_notes_t)]
            skel_notes_t = [var_notes_t[j] for j in skel_note_indices]

            if len(skel_notes_t) < int(filt_cfg.min_skel_notes):
                skip["empty_after_trunc"] += 1
                continue

            n_orn = len(var_notes_t)
            if n_orn >= 2 and len(skel_note_indices) >= 2:
                span_ratio = (max(skel_note_indices) - min(skel_note_indices)) / float(n_orn - 1)
                if span_ratio < 0.8:
                    skip["low_alignment"] += 1
                    continue

            xorn_g, _ = encode_notes_to_padded_events_global(var_notes_t, vocab_global, max_events=max_events)
            x_g, _ = encode_notes_to_padded_events_global(skel_notes_t, vocab_global, max_events=max_events)

            xorn_l = _global_to_local(xorn_g, vocab_info).astype(np.int16)
            x_l = _global_to_local(x_g, vocab_info).astype(np.int16)

            len_orn = _seq_len_including_eos(xorn_l, pad_id=vocab_info.pad_id, eos_id=vocab_info.eos_id)
            len_x = _seq_len_including_eos(x_l, pad_id=vocab_info.pad_id, eos_id=vocab_info.eos_id)

            rho = float(len_x) / float(max(1, len_orn))
            if rho < min_rho - 1e-9 or rho > max_rho + 1e-9:
                print(f"[WARN][skip rho] rho={rho:.4f} out of [{min_rho:.4f},{max_rho:.4f}] :: {var_path.name}")
                skip["rho_out_of_range"] += 1
                continue

            pi = build_pi_array(
                max_events=max_events,
                len_orn=len_orn,
                len_skel=len_x,
                skel_note_indices_in_orn=skel_note_indices,
            )

            w_x.append(x_l)
            w_xorn.append(xorn_l)
            w_pi.append(pi)
            w_len_x.append(np.int32(len_x))
            w_len_xorn.append(np.int32(len_orn))
            w_rho.append(np.float32(rho))

            rho_list.append(rho)
            cov_list.append(coverage)
            match_list.append(kept_n)

            n_notes_x = int(len_x) - 2
            n_notes_xorn = int(len_orn) - 2

            span_x = compute_span_pos_from_local_events(x_l, len_events=int(len_x), vocab=vocab_info)
            span_xorn = compute_span_pos_from_local_events(xorn_l, len_events=int(len_orn), vocab=vocab_info)

            acc_x.add(len_events=int(len_x), n_notes=int(n_notes_x), span_pos=int(span_x))
            acc_xorn.add(len_events=int(len_orn), n_notes=int(n_notes_xorn), span_pos=int(span_xorn))

            uniq_theme.add(str(theme_path.relative_to(tavern_root).as_posix()))
            uniq_var.add(str(var_path.relative_to(tavern_root).as_posix()))

            if meta_f is not None:
                meta_f.write(json.dumps({
                    "row_idx": int(w_x.count - 1),
                    "opus": p["opus"],
                    "variation": p["variation"],
                    "theme_phrase_base": p["theme_phrase_base"],
                    "var_phrase_key": p["var_phrase_key"],
                    "theme_file": str(theme_path.relative_to(tavern_root).as_posix()),
                    "variation_file": str(var_path.relative_to(tavern_root).as_posix()),
                    "theme_notes": int(len(theme_notes)),
                    "var_notes": int(len(var_notes)),
                    "theme_notes_trunc": int(len(theme_notes_t)),
                    "var_notes_trunc": int(len(var_notes_t)),
                    "kept_matches": int(kept_n),
                    "coverage": float(coverage),
                    "nw_best_score": float(best_score),
                    "len_x": int(len_x),
                    "len_x_orn": int(len_orn),
                    "rho": float(rho),
                }, ensure_ascii=False) + "\n")

        except Exception as e:
            skip["exception"] += 1
            print(f"[WARN][exception] {p.get('variation_file', '')}: {repr(e)}")
            continue

    w_x.close()
    w_xorn.close()
    w_pi.close()
    w_len_x.close()
    w_len_xorn.close()
    w_rho.close()
    if meta_f is not None:
        meta_f.close()

    N_out = int(len(rho_list))
    if N_out > 0:
        rho_min_v = float(min(rho_list))
        rho_max_v = float(max(rho_list))
        rho_mean_v = float(sum(rho_list) / float(N_out))
    else:
        rho_min_v, rho_max_v, rho_mean_v = math.nan, math.nan, math.nan

    meta = {
        "version": "tavern_silver_o2b_v1_score_only",
        "tavern_root": str(tavern_root),
        "split_name": str(args.split_name),
        "melody_mode": str(args.melody_mode),
        "max_len_tokens": int(max_len_tokens),
        "max_events": int(max_events),
        "vocab_pkl": str(args.vocab_pkl),
        "rho_range": [float(min_rho), float(max_rho)],
        "align_config": asdict(align_cfg),
        "filter_config": asdict(filt_cfg),
        "counts": {
            "pairs_found": int(len(pairs)),
            "pairs_kept": int(N_out),
            "skip": {k: int(v) for k, v in skip.items()},
        },
        "stats": {
            "rho_min": float(rho_min_v),
            "rho_max": float(rho_max_v),
            "rho_mean": float(rho_mean_v),
            "coverage_mean": float(np.mean(cov_list)) if cov_list else math.nan,
            "kept_matches_mean": float(np.mean(match_list)) if match_list else math.nan,
        },
    }

    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    export_stats = {
        "schema": "simplemono_export_stats_v1",
        "dataset_kind": "tavern_silver_o2b",
        "split_name": str(args.split_name),
        "tavern_root": str(tavern_root),
        "bpm": float(bpm),
        "pos_resolution": int(pos_resolution),
        "counts": {
            "pairs_kept": int(N_out),
            "sequences": int(N_out),
            "unique_theme_files_kept": int(len(uniq_theme)),
            "unique_variation_files_kept": int(len(uniq_var)),
            "unique_score_files_kept": int(len(set(uniq_theme) | set(uniq_var))),
        },
        "tracks": {
            "x": acc_x.to_dict(pos_resolution=pos_resolution, bpm=bpm),
            "x_orn": acc_xorn.to_dict(pos_resolution=pos_resolution, bpm=bpm),
        },
        "existing_meta_stats": meta.get("stats", {}),
    }

    write_json(out_root / "export_stats.json", export_stats)
    aggregate_split_dirs_export_stats(Path(args.output_dir))

    print("[DONE] TAVERN silver benchmark built (score-only).")
    print(json.dumps(meta["counts"], indent=2))
    print(json.dumps(meta["stats"], indent=2))


if __name__ == "__main__":
    main()

# usage:
# python -m preproc.preproc_tavern_silver_benchmark --tavern_root "J:\DATASETS\MIDIs\TAVERN-master\TAVERN-master" --vocab_pkl ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\tavern_silver_otb" --split_name test --melody_mode top --min_rho 0.3333333333 --max_rho 1.0 --write_meta_jsonl
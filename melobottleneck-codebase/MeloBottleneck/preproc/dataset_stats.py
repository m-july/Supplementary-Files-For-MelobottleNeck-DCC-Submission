# preproc/dataset_stats.py
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import math
import numpy as np


DEFAULT_STATS_BPM = 80.0


def load_pos_resolution_from_vocab_pkl(vocab_pkl: str | Path, default: int = 12) -> int:
    p = Path(vocab_pkl)
    with p.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        return int(default)
    qc = obj.get("quantization_config", {})
    if isinstance(qc, dict) and "pos_resolution" in qc:
        return int(qc["pos_resolution"])
    return int(default)


def pos_to_seconds(pos: int | float, *, pos_resolution: int, bpm: float = DEFAULT_STATS_BPM) -> float:
    return float(pos) / float(pos_resolution) * (60.0 / float(bpm))


def format_seconds(seconds: float) -> str:
    if seconds is None or not math.isfinite(float(seconds)):
        return "NaN"
    s = int(round(float(seconds)))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}d{h:02d}h{m:02d}m{s:02d}s"
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class RunningIntStats:
    count: int = 0
    sum: int = 0
    min: Optional[int] = None
    max: Optional[int] = None

    def update(self, x: int) -> None:
        x = int(x)
        self.count += 1
        self.sum += x
        self.min = x if self.min is None else min(self.min, x)
        self.max = x if self.max is None else max(self.max, x)

    def merge(self, other: "RunningIntStats") -> None:
        if other.count <= 0:
            return
        self.count += int(other.count)
        self.sum += int(other.sum)
        self.min = other.min if self.min is None else (other.min if other.min is not None and other.min < self.min else self.min)
        self.max = other.max if self.max is None else (other.max if other.max is not None and other.max > self.max else self.max)

    @property
    def mean(self) -> Optional[float]:
        if self.count <= 0:
            return None
        return float(self.sum) / float(self.count)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": int(self.count),
            "sum": int(self.sum),
            "min": None if self.min is None else int(self.min),
            "max": None if self.max is None else int(self.max),
            "mean": None if self.mean is None else float(self.mean),
        }


@dataclass
class SeqExportStatsAccumulator:
    n_seq: int = 0
    len_events: RunningIntStats = field(default_factory=RunningIntStats)
    n_notes: RunningIntStats = field(default_factory=RunningIntStats)
    span_pos: RunningIntStats = field(default_factory=RunningIntStats)

    def add(self, *, len_events: int, n_notes: int, span_pos: int) -> None:
        self.n_seq += 1
        self.len_events.update(int(len_events))
        self.n_notes.update(int(n_notes))
        self.span_pos.update(int(span_pos))

    def merge(self, other: "SeqExportStatsAccumulator") -> None:
        self.n_seq += int(other.n_seq)
        self.len_events.merge(other.len_events)
        self.n_notes.merge(other.n_notes)
        self.span_pos.merge(other.span_pos)

    def to_dict(self, *, pos_resolution: int, bpm: float = DEFAULT_STATS_BPM) -> Dict[str, Any]:
        total_seconds = pos_to_seconds(self.span_pos.sum, pos_resolution=pos_resolution, bpm=bpm)
        mean_seconds = None if self.span_pos.mean is None else pos_to_seconds(self.span_pos.mean, pos_resolution=pos_resolution, bpm=bpm)
        return {
            "sequences": int(self.n_seq),
            "len_events": self.len_events.to_dict(),
            "notes": self.n_notes.to_dict(),
            "span_pos": self.span_pos.to_dict(),
            "duration_80bpm": {
                "bpm": float(bpm),
                "pos_resolution": int(pos_resolution),
                "total_seconds": float(total_seconds),
                "total_human": format_seconds(total_seconds),
                "mean_seconds": None if mean_seconds is None else float(mean_seconds),
            },
        }


# -------------------------
# span computation
# -------------------------
def compute_span_pos_from_triples(triples, *, note_limit: int) -> int:
    """
    triples: List[(pitch, dur_code, dt_code_signed)]
    dur_code/dt_code are already pos-ish codes in your pipeline (uniform quantization).
    """
    n = min(int(note_limit), int(len(triples)))
    if n <= 0:
        return 0

    start = 0
    max_end = 0

    for i in range(n):
        dur_pos = int(triples[i][1])
        if dur_pos <= 0:
            dur_pos = 1

        end = start + dur_pos
        if end > max_end:
            max_end = end

        if i < n - 1:
            dt_pos = int(triples[i][2])
            # defensive: prevent start decreasing due to asymmetric clipping
            if dt_pos < -dur_pos:
                dt_pos = -dur_pos
            start = end + dt_pos

    return int(max_end)


def _decode_local_dur_pos(tok: int, vocab) -> int:
    # tok = special_n + dur_code
    code = int(tok) - int(vocab.special_n)
    if code < 0:
        return 1
    code = min(code, int(len(vocab.duration_code_to_pos) - 1))
    pos = int(vocab.duration_code_to_pos[code])
    return max(1, pos)


def _decode_local_dt_pos(tok: int, vocab) -> int:
    # tok = special_n + dt_code   (dt_code>=0)
    # tok = special_n + dt_samples + (mag-1)  (dt_code=-mag)
    special_n = int(vocab.special_n)
    dt_samples = int(vocab.deltatime_code_offset)
    idx = int(tok) - special_n

    if idx < 0:
        return 0

    if idx < dt_samples:
        code = idx
    else:
        mag = (idx - dt_samples) + 1
        code = -mag

    # clamp to representable signed range
    code = max(-dt_samples, min(code, dt_samples - 1))

    # map signed code -> pos using vocab table
    pos = int(vocab.deltatime_code_to_pos[code + dt_samples])
    return int(pos)


def compute_span_pos_from_local_events(x_local: np.ndarray, *, len_events: int, vocab) -> int:
    """
    x_local: [L,3] local ids (pitch, dur, dt)
    len_events: valid length including EOS (and BOS), excluding pads.
    """
    len_events = int(len_events)
    n_notes = max(0, len_events - 2)
    if n_notes <= 0:
        return 0

    start = 0
    max_end = 0

    for j in range(n_notes):
        row = 1 + j
        dur_tok = int(x_local[row, 1])
        dt_tok = int(x_local[row, 2])

        dur_pos = _decode_local_dur_pos(dur_tok, vocab)
        end = start + dur_pos
        if end > max_end:
            max_end = end

        if j < n_notes - 1:
            dt_pos = _decode_local_dt_pos(dt_tok, vocab)
            if dt_pos < -dur_pos:
                dt_pos = -dur_pos
            start = end + dt_pos

    return int(max_end)


def aggregate_split_dirs_export_stats(
    root_dir: str | Path,
    *,
    per_split_filename: str = "export_stats.json",
    out_filename: str = "export_stats_all_splits.json",
) -> Dict[str, Any]:
    root_dir = Path(root_dir)
    splits: Dict[str, Any] = {}

    for d in sorted(root_dir.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        p = d / per_split_filename
        if p.exists():
            try:
                splits[d.name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue

    # Best-effort overall sum (tracks -> totals)
    overall: Dict[str, Any] = {"splits_found": int(len(splits))}
    if splits:
        # assume consistent bpm/pos_resolution across splits; pick the first
        first = next(iter(splits.values()))
        overall["bpm"] = first.get("bpm", DEFAULT_STATS_BPM)
        overall["pos_resolution"] = first.get("pos_resolution", None)

        total_sequences = 0
        tracks_totals: Dict[str, Dict[str, float]] = {}  # {track: {notes, span_pos, seconds}}
        for sp, obj in splits.items():
            c = obj.get("counts", {})
            total_sequences += int(c.get("sequences", 0))

            tracks = obj.get("tracks", {})
            for tname, tstats in tracks.items():
                notes_sum = (tstats.get("notes", {}) or {}).get("sum", 0)
                span_pos_sum = (tstats.get("span_pos", {}) or {}).get("sum", 0)
                sec_sum = ((tstats.get("duration_80bpm", {}) or {}).get("total_seconds", 0.0))

                cur = tracks_totals.setdefault(tname, {"notes_sum": 0, "span_pos_sum": 0, "seconds_sum": 0.0})
                cur["notes_sum"] += int(notes_sum)
                cur["span_pos_sum"] += int(span_pos_sum)
                cur["seconds_sum"] += float(sec_sum)

        overall["counts"] = {"sequences": int(total_sequences)}
        overall["tracks"] = {k: v for k, v in tracks_totals.items()}

    out = {
        "schema": "export_stats_all_splits_v1",
        "root_dir": str(root_dir),
        "overall": overall,
        "splits": splits,
    }
    write_json(root_dir / out_filename, out)
    return out
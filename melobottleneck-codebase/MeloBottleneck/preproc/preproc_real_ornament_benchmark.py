from __future__ import annotations

import argparse
import bisect
import json
import os
import pickle
import shutil
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mido
import miditoolkit
import numpy as np
from tqdm import tqdm

from main.vocab_utils import load_vocab_info

from main.ornament import PI_INSERTED, PI_PAD

from .detect_ornamented_skeleton import (
    CandidateChannel,
    ChannelResolutionError,
    resolve_ornamented_skeleton,
)

from .simplemono_preproc.npy_stream import NpyAppendWriter
from .simplemono_preproc.pipelines.pretrain import default_max_len_tokens

from preproc.dataset_stats import (
    DEFAULT_STATS_BPM,
    SeqExportStatsAccumulator,
    compute_span_pos_from_local_events,
    write_json,
    aggregate_split_dirs_export_stats,
)


MAX_PITCH = 127
TRUNC_POS = 2 ** 16
MIDI_EXTS = (".mid", ".midi")


class SkipFileError(RuntimeError):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail)
        self.reason = str(reason)
        self.detail = str(detail)


@dataclass(frozen=True)
class TickNote:
    pitch: int
    start_tick: int
    end_tick: int
    src_order: int


@dataclass(frozen=True)
class PosNote:
    pitch: int
    start: int
    end: int
    src_order: int


def relpath_posix(p: Path, start: Path) -> str:
    return p.relative_to(start).as_posix()


def ensure_dir_empty_or_create(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"Output path exists and is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"Output dir already exists and is not empty: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)


def iter_midi_files(root: Path) -> List[Path]:
    out: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(MIDI_EXTS):
                out.append(Path(dirpath) / name)
    out.sort(key=os.fspath)
    return out


def write_keywords_txt(files: List[Path], out_path: Path) -> int:
    keywords = set()
    for p in files:
        for kw in p.stem.split():
            kw = kw.strip()
            if kw:
                keywords.add(kw)

    sorted_keywords = sorted(keywords)
    with out_path.open("w", encoding="utf-8") as f:
        for kw in sorted_keywords:
            f.write(kw + "\n")
    return len(sorted_keywords)


def load_pos_resolution_from_vocab_pkl(pkl_path: Path) -> int:
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    qc = obj.get("quantization_config", {})
    return int(qc.get("pos_resolution", 12))


def time_signature_reduce(
    numerator: int,
    denominator: int,
    max_ts_denominator: int = 6,
    max_notes_per_bar: int = 2,
) -> Tuple[int, int]:
    while (
        denominator > 2 ** max_ts_denominator
        and denominator % 2 == 0
        and numerator % 2 == 0
    ):
        denominator //= 2
        numerator //= 2
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def midi_tick_to_pos_with_resolution(tick: int, ticks_per_beat: int, pos_resolution: int) -> int:
    return int(round(int(tick) * int(pos_resolution) / int(ticks_per_beat)))


def build_midi_measures_with_resolution(
    midi_obj: miditoolkit.MidiFile,
    *,
    max_pos: int,
    pos_resolution: int,
) -> List[Tuple[int, int]]:
    tpq = int(midi_obj.ticks_per_beat)

    tsc = list(midi_obj.time_signature_changes)
    if not tsc:
        tsc = [miditoolkit.containers.TimeSignature(4, 4, 0)]
    tsc.sort(key=lambda x: x.time)

    ts_points: List[Tuple[int, int, int]] = []
    for ts in tsc:
        p = midi_tick_to_pos_with_resolution(int(ts.time), tpq, pos_resolution)
        num, den = time_signature_reduce(int(ts.numerator), int(ts.denominator))
        ts_points.append((p, num, den))

    if not ts_points or ts_points[0][0] != 0:
        ts_points.insert(0, (0, 4, 4))

    measures: List[Tuple[int, int]] = []
    cur_pos = 0
    ts_i = 0

    while cur_pos < max_pos:
        while ts_i + 1 < len(ts_points) and ts_points[ts_i + 1][0] <= cur_pos:
            ts_i += 1

        _ts_start, num, den = ts_points[ts_i]
        measure_len = int(round(num * 4 * pos_resolution / den))
        if measure_len <= 0:
            measure_len = 1

        next_ts_pos = ts_points[ts_i + 1][0] if ts_i + 1 < len(ts_points) else None
        me = cur_pos + measure_len
        if next_ts_pos is not None and next_ts_pos < me:
            me = next_ts_pos

        measures.append((cur_pos, me))
        if me <= cur_pos:
            break
        cur_pos = me

    return measures


def candidate_key(c: CandidateChannel) -> Tuple[int, int, int]:
    return int(c.track_index), int(c.channel), int(c.program)


def extract_candidate_note_ticks_multi(
    midi_path: Path,
    candidates: List[CandidateChannel],
) -> Dict[Tuple[int, int, int], List[TickNote]]:
    """
    Mirror the candidate granularity used in detect_ornamented_skeleton.py:
      (track_index, channel, program_at_note_on)
    """
    mf = mido.MidiFile(str(midi_path))

    wanted_by_track: Dict[int, set[Tuple[int, int]]] = defaultdict(set)
    out: Dict[Tuple[int, int, int], List[TickNote]] = {}

    for c in candidates:
        tk, ch, prog = candidate_key(c)
        wanted_by_track[tk].add((ch, prog))
        out[(tk, ch, prog)] = []

    for track_index, track in enumerate(mf.tracks):
        wanted_pairs = wanted_by_track.get(track_index)
        if not wanted_pairs:
            continue

        current_program = [0] * 16
        active: Dict[Tuple[int, int], deque[Tuple[int, int, int]]] = defaultdict(deque)
        abs_tick = 0
        note_on_order = 0

        for msg in track:
            abs_tick += int(msg.time)

            if msg.is_meta:
                continue

            if msg.type == "program_change":
                current_program[int(msg.channel)] = int(msg.program)
                continue

            if not hasattr(msg, "channel"):
                continue

            ch = int(msg.channel)

            if msg.type == "note_on" and int(msg.velocity) > 0:
                pitch = int(msg.note)
                active[(ch, pitch)].append(
                    (abs_tick, note_on_order, int(current_program[ch]))
                )
                note_on_order += 1
                continue

            if msg.type == "note_off" or (msg.type == "note_on" and int(msg.velocity) == 0):
                pitch = int(msg.note)
                dq = active.get((ch, pitch), None)
                if dq and len(dq) > 0:
                    st, order, prog_at_note_on = dq.popleft()
                    if (ch, prog_at_note_on) in wanted_pairs:
                        key = (track_index, ch, prog_at_note_on)
                        ed = max(abs_tick, st + 1)
                        out[key].append(
                            TickNote(
                                pitch=pitch,
                                start_tick=int(st),
                                end_tick=int(ed),
                                src_order=int(order),
                            )
                        )

        # close leftover active notes at track end
        for (ch, pitch), dq in active.items():
            while dq:
                st, order, prog_at_note_on = dq.popleft()
                if (ch, prog_at_note_on) in wanted_pairs:
                    key = (track_index, ch, prog_at_note_on)
                    ed = max(abs_tick, st + 1)
                    out[key].append(
                        TickNote(
                            pitch=int(pitch),
                            start_tick=int(st),
                            end_tick=int(ed),
                            src_order=int(order),
                        )
                    )

    for k in out:
        out[k].sort(key=lambda n: (n.start_tick, n.end_tick, n.pitch, n.src_order))
    return out


def quantize_tick_notes(
    tick_notes: List[TickNote],
    *,
    ticks_per_beat: int,
    pos_resolution: int,
) -> List[PosNote]:
    out: List[PosNote] = []
    for n in tick_notes:
        if n.pitch < 0 or n.pitch > MAX_PITCH:
            continue

        s = midi_tick_to_pos_with_resolution(n.start_tick, ticks_per_beat, pos_resolution)
        e = midi_tick_to_pos_with_resolution(n.end_tick, ticks_per_beat, pos_resolution)
        if e <= s:
            e = s + 1

        if s >= TRUNC_POS:
            continue
        e = min(e, TRUNC_POS)

        out.append(
            PosNote(
                pitch=int(n.pitch),
                start=int(s),
                end=int(e),
                src_order=int(n.src_order),
            )
        )

    out.sort(key=lambda x: (x.start, x.end, x.pitch, x.src_order))
    return out


def trim_empty_measures_joint(
    note_lists: List[List[PosNote]],
    measures: List[Tuple[int, int]] | None,
) -> Tuple[List[List[PosNote]], int, int]:
    """
    Remove only those measures that are empty in ALL channels jointly.
    This preserves cross-channel alignment.
    Returns:
      shifted_note_lists, removed_measure_count, removed_total_pos
    """
    if not measures:
        return note_lists, 0, 0

    all_notes = [n for notes in note_lists for n in notes]
    if not all_notes:
        return note_lists, 0, 0

    all_notes.sort(key=lambda x: (x.start, x.end))
    measures = sorted(measures, key=lambda x: x[0])

    empty: List[Tuple[int, int]] = []
    ni = 0
    for ms, me in measures:
        while ni < len(all_notes) and all_notes[ni].end <= ms:
            ni += 1
        if ni < len(all_notes) and all_notes[ni].start < me and all_notes[ni].end > ms:
            continue
        empty.append((ms, me))

    if not empty:
        return note_lists, 0, 0

    raw_empty_measure_count = len(empty)

    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = empty[0]
    for s, e in empty[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    cut_ends: List[int] = []
    prefix_removed: List[int] = []
    acc = 0
    for a, b in merged:
        acc += (b - a)
        cut_ends.append(b)
        prefix_removed.append(acc)

    def removed_before(t: int) -> int:
        k = bisect.bisect_right(cut_ends, t)
        return prefix_removed[k - 1] if k > 0 else 0

    shifted_lists: List[List[PosNote]] = []
    for notes in note_lists:
        shifted: List[PosNote] = []
        for n in notes:
            sh = removed_before(int(n.start))
            ns = int(n.start - sh)
            ne = int(n.end - sh)
            if ne <= ns:
                ne = ns + 1
            shifted.append(
                PosNote(
                    pitch=int(n.pitch),
                    start=int(ns),
                    end=int(ne),
                    src_order=int(n.src_order),
                )
            )
        shifted.sort(key=lambda x: (x.start, x.end, x.pitch, x.src_order))
        shifted_lists.append(shifted)

    removed_total_pos = sum(b - a for a, b in merged)
    return shifted_lists, raw_empty_measure_count, int(removed_total_pos)


def build_note_pi_from_onsets(
    skeleton_notes: List[PosNote],
    ornamented_notes: List[PosNote],
) -> Tuple[List[int], int, int, int, int]:
    """
    Build note-level pi for ornamented notes -> skeleton note indices.

    Mapping policy for shared onset groups:
      - require len(orn_group) >= len(sk_group)
      - map the LAST len(sk_group) ornamented notes at that onset
        to skeleton notes in order
      - earlier ornamented notes at same onset are treated as inserted
    """
    sk_groups: Dict[int, List[int]] = defaultdict(list)
    orn_groups: Dict[int, List[int]] = defaultdict(list)

    for i, n in enumerate(skeleton_notes):
        sk_groups[int(n.start)].append(int(i))
    for i, n in enumerate(ornamented_notes):
        orn_groups[int(n.start)].append(int(i))

    sk_onsets = set(sk_groups.keys())
    orn_onsets = set(orn_groups.keys())

    if not sk_onsets.issubset(orn_onsets):
        missing = sorted(sk_onsets - orn_onsets)
        show = missing[:16]
        suffix = "" if len(missing) <= 16 else f" ... (+{len(missing) - 16} more)"
        raise SkipFileError(
            "onset_not_subset",
            f"Skeleton onset set is not a subset of Ornamented onset set. "
            f"Missing onsets (quantized pos): {show}{suffix}",
        )

    pi_note = [int(PI_INSERTED)] * len(ornamented_notes)

    max_group_sk = 0
    max_group_orn = 0

    for onset in sorted(orn_groups.keys()):
        orn_idxs = orn_groups[onset]
        sk_idxs = sk_groups.get(onset, None)

        max_group_orn = max(max_group_orn, len(orn_idxs))
        if sk_idxs is not None:
            max_group_sk = max(max_group_sk, len(sk_idxs))

        if not sk_idxs:
            continue

        if len(orn_idxs) < len(sk_idxs):
            raise SkipFileError(
                "shared_onset_group_too_small",
                f"At onset={onset}, ornamented group size={len(orn_idxs)} "
                f"< skeleton group size={len(sk_idxs)}.",
            )

        mapped_orn_idxs = orn_idxs[len(orn_idxs) - len(sk_idxs):]
        for oi, si in zip(mapped_orn_idxs, sk_idxs):
            pi_note[int(oi)] = int(si)

    last = -1
    for v in pi_note:
        if v >= 0:
            if v < last:
                raise SkipFileError(
                    "pi_non_monotonic",
                    "Constructed note-level pi is not monotonic.",
                )
            last = v

    return (
        pi_note,
        int(len(sk_onsets)),
        int(len(orn_onsets)),
        int(max_group_sk),
        int(max_group_orn),
    )


def choose_common_prefix_lengths(pi_note: List[int], ornamented_note_budget: int) -> Tuple[int, int]:
    """
    Keep the first K_m ornamented notes; derive the required skeleton prefix
    from the mapped note indices appearing in that prefix.
    """
    k_m = min(len(pi_note), int(ornamented_note_budget))
    mapped = [int(v) for v in pi_note[:k_m] if int(v) >= 0]
    k_s = (max(mapped) + 1) if mapped else 0
    return int(k_s), int(k_m)


def encode_pitch_local(pitch: int, vocab) -> int:
    return int(vocab.special_n + int(pitch))


def encode_dur_local_from_pos(dur_pos: int, vocab) -> int:
    dur_pos = int(dur_pos)
    dur_pos = max(0, min(dur_pos, int(len(vocab.duration_pos_to_code) - 1)))
    code = int(vocab.duration_pos_to_code[dur_pos])
    return int(vocab.special_n + code)


def encode_dt_local_from_pos(dt_pos: int, vocab) -> int:
    dt_pos = int(dt_pos)
    dt_samples = int(vocab.deltatime_code_offset)
    dt_pos = max(-dt_samples, min(dt_pos, dt_samples - 1))
    idx = dt_pos + dt_samples
    code = int(vocab.deltatime_pos_to_code[idx])

    if code >= 0:
        return int(vocab.special_n + code)

    mag = -int(code)
    return int(vocab.special_n + dt_samples + (mag - 1))


def notes_to_local_events(notes: List[PosNote], vocab) -> np.ndarray:
    """
    Output local-id events with BOS/EOS included.
    Shape: [n_notes + 2, 3]
    """
    n = len(notes)
    ev = np.full((n + 2, 3), fill_value=int(vocab.pad_id), dtype=np.int32)

    bos_start_pos = int(notes[0].start) if n > 0 else 0
    ev[0, 0] = int(vocab.bos_id)
    ev[0, 1] = int(vocab.bos_id)
    ev[0, 2] = int(encode_dt_local_from_pos(bos_start_pos, vocab))

    for i, note in enumerate(notes):
        dur_pos = max(1, int(note.end) - int(note.start))
        if i + 1 < n:
            raw_dt = int(notes[i + 1].start) - int(note.end)
        else:
            raw_dt = 0

        ev[i + 1, 0] = int(encode_pitch_local(int(note.pitch), vocab))
        ev[i + 1, 1] = int(encode_dur_local_from_pos(dur_pos, vocab))
        ev[i + 1, 2] = int(encode_dt_local_from_pos(raw_dt, vocab))

    ev[n + 1, :] = int(vocab.eos_id)
    return ev


def pack_events_fixed(events: np.ndarray, *, max_events: int, pad_id: int) -> np.ndarray:
    if events.ndim != 2 or events.shape[1] != 3:
        raise ValueError(f"events must be [T,3], got {events.shape}")
    if events.shape[0] > max_events:
        raise ValueError(f"events length {events.shape[0]} > max_events {max_events}")

    out = np.full((max_events, 3), fill_value=int(pad_id), dtype=np.int16)
    out[: events.shape[0]] = events.astype(np.int16, copy=False)
    return out


def build_pi_row(note_pi_prefix: List[int], *, skeleton_note_count: int, max_events: int) -> np.ndarray:
    """
    Event-level pi row:
      BOS -> 0
      ornament note j -> skeleton note idx + 1, or PI_INSERTED
      EOS -> len_x - 1 = skeleton_note_count + 1
      PAD -> PI_PAD
    """
    k_m = len(note_pi_prefix)
    len_x_orn = k_m + 2

    pi = np.full((max_events,), fill_value=int(PI_PAD), dtype=np.int32)
    pi[0] = 0  # BOS -> BOS

    for j, v in enumerate(note_pi_prefix):
        pi[j + 1] = int(v + 1) if int(v) >= 0 else int(PI_INSERTED)

    pi[len_x_orn - 1] = int(skeleton_note_count + 1)  # EOS -> EOS
    return pi


def sanity_check_item(
    *,
    x_row: np.ndarray,
    x_orn_row: np.ndarray,
    pi_row: np.ndarray,
    len_x: int,
    len_x_orn: int,
    pad_id: int,
    bos_id: int,
    eos_id: int,
) -> None:
    if not (2 <= len_x <= x_row.shape[0]):
        raise ValueError(f"Bad len_x={len_x}")
    if not (2 <= len_x_orn <= x_orn_row.shape[0]):
        raise ValueError(f"Bad len_x_orn={len_x_orn}")

    if int(x_row[0, 0]) != int(bos_id) or int(x_orn_row[0, 0]) != int(bos_id):
        raise ValueError("BOS mismatch in packed rows.")
    if not np.all(x_row[len_x - 1] == int(eos_id)):
        raise ValueError("EOS mismatch in x row.")
    if not np.all(x_orn_row[len_x_orn - 1] == int(eos_id)):
        raise ValueError("EOS mismatch in x_orn row.")

    if not np.all(x_row[len_x:, 0] == int(pad_id)):
        raise ValueError("x row has non-pad after len_x.")
    if not np.all(x_orn_row[len_x_orn:, 0] == int(pad_id)):
        raise ValueError("x_orn row has non-pad after len_x_orn.")
    if not np.all(pi_row[len_x_orn:] == int(PI_PAD)):
        raise ValueError("pi row has non-PI_PAD after len_x_orn.")

    if int(pi_row[0]) != 0:
        raise ValueError("pi BOS entry must be 0.")
    if int(pi_row[len_x_orn - 1]) != int(len_x - 1):
        raise ValueError("pi EOS entry must point to x EOS.")

    valid_pi = pi_row[:len_x_orn]
    ok_mask = ((valid_pi >= 0) & (valid_pi < len_x)) | (valid_pi == int(PI_INSERTED))
    if not np.all(ok_mask):
        raise ValueError("pi contains out-of-range indices.")


def build_item_from_midi(
    *,
    midi_path: Path,
    input_dir: Path,
    vocab,
    pos_resolution: int,
    max_events: int,
    min_rho: float,
    max_rho: float,
    trim_joint_empty_measures: bool,
) -> Dict[str, object]:
    source_file = relpath_posix(midi_path, input_dir)

    try:
        resolution = resolve_ornamented_skeleton(midi_path)
    except ChannelResolutionError as e:
        raise SkipFileError("resolve_failed", str(e))

    cands = [resolution.ornamented, resolution.skeleton]
    tick_map = extract_candidate_note_ticks_multi(midi_path, cands)

    orn_tick = tick_map.get(candidate_key(resolution.ornamented), [])
    sk_tick = tick_map.get(candidate_key(resolution.skeleton), [])

    if len(orn_tick) == 0:
        raise SkipFileError("empty_ornamented_notes", "No extracted ornamented notes.")
    if len(sk_tick) == 0:
        raise SkipFileError("empty_skeleton_notes", "No extracted skeleton notes.")

    midi_obj = miditoolkit.MidiFile(str(midi_path))
    tpq = int(midi_obj.ticks_per_beat)

    orn_notes = quantize_tick_notes(orn_tick, ticks_per_beat=tpq, pos_resolution=pos_resolution)
    sk_notes = quantize_tick_notes(sk_tick, ticks_per_beat=tpq, pos_resolution=pos_resolution)

    if len(orn_notes) == 0:
        raise SkipFileError("empty_ornamented_notes_after_quant", "No ornamented notes after quantization.")
    if len(sk_notes) == 0:
        raise SkipFileError("empty_skeleton_notes_after_quant", "No skeleton notes after quantization.")

    removed_measure_count = 0
    removed_total_pos = 0
    if trim_joint_empty_measures:
        max_pos = max(
            max(n.end for n in orn_notes),
            max(n.end for n in sk_notes),
        ) + 1
        measures = build_midi_measures_with_resolution(
            midi_obj,
            max_pos=max_pos,
            pos_resolution=pos_resolution,
        )
        (orn_notes, sk_notes), removed_measure_count, removed_total_pos = trim_empty_measures_joint(
            [orn_notes, sk_notes],
            measures,
        )

    orn_notes.sort(key=lambda x: (x.start, x.end, x.pitch, x.src_order))
    sk_notes.sort(key=lambda x: (x.start, x.end, x.pitch, x.src_order))

    if len(orn_notes) == 0:
        raise SkipFileError("empty_ornamented_notes_after_trim", "No ornamented notes after joint trim.")
    if len(sk_notes) == 0:
        raise SkipFileError("empty_skeleton_notes_after_trim", "No skeleton notes after joint trim.")

    note_pi_full, n_onsets_sk, n_onsets_orn, max_group_sk, max_group_orn = build_note_pi_from_onsets(
        sk_notes,
        orn_notes,
    )

    note_budget = int(max_events - 2)
    if note_budget <= 0:
        raise ValueError(f"max_events must be >= 3, got {max_events}")

    k_s, k_m = choose_common_prefix_lengths(note_pi_full, ornamented_note_budget=note_budget)

    if k_m <= 0:
        raise SkipFileError(
            "empty_after_prefix_truncation",
            "No ornamented note survives prefix truncation.",
        )
    if k_s <= 0:
        raise SkipFileError(
            "no_skeleton_note_after_prefix_truncation",
            "The kept ornamented prefix does not yet contain any skeleton note.",
        )

    note_pi_use = note_pi_full[:k_m]
    if any((v >= k_s) for v in note_pi_use if v >= 0):
        raise SkipFileError(
            "prefix_alignment_inconsistent",
            "Kept ornamented prefix refers to a skeleton note outside kept skeleton prefix.",
        )

    orn_use = orn_notes[:k_m]
    sk_use = sk_notes[:k_s]

    x_events = notes_to_local_events(sk_use, vocab)
    x_orn_events = notes_to_local_events(orn_use, vocab)

    len_x = int(x_events.shape[0])
    len_x_orn = int(x_orn_events.shape[0])

    rho = float(len_x) / float(len_x_orn)
    rho_note = float(k_s) / float(k_m) if k_m > 0 else 0.0

    if rho < float(min_rho) - 1e-9 or rho > float(max_rho) + 1e-9:
        raise SkipFileError(
            "rho_out_of_range",
            f"rho={rho:.6f} not in [{min_rho:.6f}, {max_rho:.6f}] "
            f"(len_x={len_x}, len_x_orn={len_x_orn}, "
            f"note_k_s={k_s}, note_k_m={k_m}).",
        )

    x_row = pack_events_fixed(x_events, max_events=max_events, pad_id=int(vocab.pad_id))
    x_orn_row = pack_events_fixed(x_orn_events, max_events=max_events, pad_id=int(vocab.pad_id))
    pi_row = build_pi_row(note_pi_use, skeleton_note_count=k_s, max_events=max_events)

    sanity_check_item(
        x_row=x_row,
        x_orn_row=x_orn_row,
        pi_row=pi_row,
        len_x=len_x,
        len_x_orn=len_x_orn,
        pad_id=int(vocab.pad_id),
        bos_id=int(vocab.bos_id),
        eos_id=int(vocab.eos_id),
    )

    inserted_note_count_used = int(sum(1 for v in note_pi_use if int(v) < 0))

    meta_record = {
        "source_file": source_file,
        "resolution_strategy": str(resolution.strategy),
        "ornamented_candidate": resolution.ornamented.to_dict(),
        "skeleton_candidate": resolution.skeleton.to_dict(),
        "resolved_note_count_ornamented": int(resolution.ornamented.note_count),
        "resolved_note_count_skeleton": int(resolution.skeleton.note_count),
        "extracted_note_count_ornamented": int(len(orn_tick)),
        "extracted_note_count_skeleton": int(len(sk_tick)),
        "full_note_count_ornamented_after_quant_trim": int(len(orn_notes)),
        "full_note_count_skeleton_after_quant_trim": int(len(sk_notes)),
        "used_note_count_ornamented": int(k_m),
        "used_note_count_skeleton": int(k_s),
        "full_len_x": int(len(sk_notes) + 2),
        "full_len_x_orn": int(len(orn_notes) + 2),
        "len_x": int(len_x),
        "len_x_orn": int(len_x_orn),
        "rho": float(rho),
        "rho_note": float(rho_note),
        "onset_count_skeleton": int(n_onsets_sk),
        "onset_count_ornamented": int(n_onsets_orn),
        "max_same_onset_group_size_skeleton": int(max_group_sk),
        "max_same_onset_group_size_ornamented": int(max_group_orn),
        "used_inserted_note_count": int(inserted_note_count_used),
        "was_truncated": bool(len(orn_notes) > k_m),
        "joint_trim_removed_measure_count": int(removed_measure_count),
        "joint_trim_removed_total_pos": int(removed_total_pos),
        "pos_resolution": int(pos_resolution),
        "note_budget": int(note_budget),
        "pi_policy": "shared_onset_suffix_mapping",
        "prefix_policy": "ornament_prefix_then_backsolve_skeleton_prefix",
    }

    return {
        "x": x_row,
        "x_orn": x_orn_row,
        "pi": pi_row,
        "len_x": np.asarray(int(len_x), dtype=np.int32),
        "len_x_orn": np.asarray(int(len_x_orn), dtype=np.int32),
        "rho": np.asarray(float(rho), dtype=np.float32),
        "rho_note": float(rho_note),
        "meta": meta_record,
    }


def log_warning(
    warn_f,
    *,
    source_file: str,
    reason: str,
    detail: str,
    counter: Counter,
) -> None:
    counter[str(reason)] += 1
    obj = {
        "source_file": str(source_file),
        "reason": str(reason),
        "detail": str(detail),
    }
    warn_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    msg = str(detail).replace("\n", " | ")
    if len(msg) > 260:
        msg = msg[:257] + "..."
    print(f"[Warning][{reason}] {source_file}: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="Root dir of real benchmark MIDI files")
    ap.add_argument("--vocab_pkl", type=str, required=True, help="SimpleMono.pkl")
    ap.add_argument("--output_dir", type=str, required=True, help="Output root dir")
    ap.add_argument("--split_name", type=str, default="test", help="Default: test")

    ap.add_argument(
        "--max_len_tokens",
        type=int,
        default=0,
        help="0 => use default_max_len_tokens() from current repo; must be divisible by 3",
    )
    ap.add_argument("--min_rho", type=float, default=(1.0 / 3.0))
    ap.add_argument("--max_rho", type=float, default=1.0)
    ap.add_argument("--max_files", type=int, default=0, help="0 => all discovered MIDI files")

    ap.add_argument(
        "--disable_joint_empty_measure_trim",
        action="store_true",
        help="Disable joint empty-measure trim. Not recommended for aligned benchmark.",
    )
    ap.add_argument(
        "--write_keywords_txt",
        action="store_true",
        help="Write deduplicated stem-split keywords txt.",
    )
    ap.add_argument(
        "--keywords_txt_name",
        type=str,
        default="keywords.txt",
        help="Used only when --write_keywords_txt is set.",
    )
    ap.add_argument(
        "--copy_vocab_pkl",
        action="store_true",
        help="Optional: copy vocab pkl into output split dir.",
    )
    ap.add_argument(
        "--show_hist",
        action="store_true",
        help="Print rho histogram after build.",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    vocab_pkl = Path(args.vocab_pkl)
    out_root = Path(args.output_dir) / str(args.split_name)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")
    if not vocab_pkl.exists():
        raise FileNotFoundError(f"vocab_pkl not found: {vocab_pkl}")

    ensure_dir_empty_or_create(out_root)

    max_len_tokens = int(args.max_len_tokens)
    if max_len_tokens <= 0:
        max_len_tokens = int(default_max_len_tokens())
    if max_len_tokens % 3 != 0:
        raise ValueError(f"max_len_tokens must be divisible by 3, got {max_len_tokens}")
    max_events = max_len_tokens // 3
    if max_events < 3:
        raise ValueError(f"max_events must be >= 3, got {max_events}")

    min_rho = float(args.min_rho)
    max_rho = float(args.max_rho)
    if not (0.0 < min_rho <= max_rho <= 1.0):
        raise ValueError(f"Require 0 < min_rho <= max_rho <= 1, got [{min_rho}, {max_rho}]")

    vocab = load_vocab_info(str(vocab_pkl))
    pos_resolution = load_pos_resolution_from_vocab_pkl(vocab_pkl)
    bpm = DEFAULT_STATS_BPM
    acc_x = SeqExportStatsAccumulator()
    acc_xorn = SeqExportStatsAccumulator()

    files = iter_midi_files(input_dir)
    if int(args.max_files) > 0:
        files = files[: int(args.max_files)]

    print(f"[Input]  {input_dir}")
    print(f"[Vocab]  {vocab_pkl}")
    print(f"[Output] {out_root}")
    print(f"[Files]  {len(files)}")
    print(f"[Shape]  [N, {max_events}, 3]  (max_len_tokens={max_len_tokens})")
    print(f"[Rho]    [{min_rho}, {max_rho}]")
    print(f"[PosRes] {pos_resolution}")
    print(f"[Joint Trim Empty Measures] {not bool(args.disable_joint_empty_measure_trim)}")

    if args.write_keywords_txt:
        n_kw = write_keywords_txt(files, out_root / args.keywords_txt_name)
        print(f"[Keywords] wrote {n_kw} unique keywords -> {out_root / args.keywords_txt_name}")

    x_w = NpyAppendWriter(out_root / "x.npy", dtype=np.int16, row_shape=(max_events, 3))
    x_orn_w = NpyAppendWriter(out_root / "x_orn.npy", dtype=np.int16, row_shape=(max_events, 3))
    pi_w = NpyAppendWriter(out_root / "pi.npy", dtype=np.int32, row_shape=(max_events,))
    len_x_w = NpyAppendWriter(out_root / "len_x.npy", dtype=np.int32, row_shape=())
    len_x_orn_w = NpyAppendWriter(out_root / "len_x_orn.npy", dtype=np.int32, row_shape=())
    rho_w = NpyAppendWriter(out_root / "rho.npy", dtype=np.float32, row_shape=())

    meta_f = (out_root / "metadata.jsonl").open("w", encoding="utf-8")
    warn_f = (out_root / "warnings.jsonl").open("w", encoding="utf-8")

    written_n = 0
    skip_reason_counts: Counter = Counter()
    resolution_strategy_counts_written: Counter = Counter()
    rho_values: List[float] = []
    rho_note_values: List[float] = []

    try:
        for midi_path in tqdm(files, desc=f"[Build real OTB] {args.split_name}", dynamic_ncols=True, smoothing=0.0):
            source_file = relpath_posix(midi_path, input_dir)

            try:
                item = build_item_from_midi(
                    midi_path=midi_path,
                    input_dir=input_dir,
                    vocab=vocab,
                    pos_resolution=pos_resolution,
                    max_events=max_events,
                    min_rho=min_rho,
                    max_rho=max_rho,
                    trim_joint_empty_measures=(not bool(args.disable_joint_empty_measure_trim)),
                )

                x_w.append(item["x"])
                x_orn_w.append(item["x_orn"])
                pi_w.append(item["pi"])
                len_x_w.append(item["len_x"])
                len_x_orn_w.append(item["len_x_orn"])
                rho_w.append(item["rho"])

                len_x = int(item["len_x"])
                len_xorn = int(item["len_x_orn"])

                n_notes_x = len_x - 2
                n_notes_xorn = len_xorn - 2

                span_x = compute_span_pos_from_local_events(item["x"], len_events=len_x, vocab=vocab)
                span_xorn = compute_span_pos_from_local_events(item["x_orn"], len_events=len_xorn, vocab=vocab)

                acc_x.add(len_events=len_x, n_notes=n_notes_x, span_pos=span_x)
                acc_xorn.add(len_events=len_xorn, n_notes=n_notes_xorn, span_pos=span_xorn)

                record = dict(item["meta"])
                record["row_idx"] = int(written_n)
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                written_n += 1
                rho_values.append(float(item["rho"]))
                rho_note_values.append(float(item["rho_note"]))
                resolution_strategy_counts_written[str(record["resolution_strategy"])] += 1

            except SkipFileError as e:
                log_warning(
                    warn_f,
                    source_file=source_file,
                    reason=e.reason,
                    detail=e.detail,
                    counter=skip_reason_counts,
                )
                continue

            except Exception as e:
                tb = traceback.format_exc()
                log_warning(
                    warn_f,
                    source_file=source_file,
                    reason="unexpected_exception",
                    detail=f"{type(e).__name__}: {e}\n{tb}",
                    counter=skip_reason_counts,
                )
                continue

    finally:
        x_w.close()
        x_orn_w.close()
        pi_w.close()
        len_x_w.close()
        len_x_orn_w.close()
        rho_w.close()
        meta_f.close()
        warn_f.close()

    if args.copy_vocab_pkl:
        shutil.copy2(vocab_pkl, out_root / vocab_pkl.name)

    rho_mean = float(np.mean(rho_values)) if rho_values else None
    rho_min_out = float(np.min(rho_values)) if rho_values else None
    rho_max_out = float(np.max(rho_values)) if rho_values else None

    meta = {
        "version": "real_otb_benchmark_v1",
        "input_dir": str(input_dir),
        "vocab_pkl": str(vocab_pkl),
        "split_name": str(args.split_name),
        "N_discovered": int(len(files)),
        "N_written": int(written_n),
        "N_skipped": int(len(files) - written_n),
        "L": int(max_events),
        "max_len_tokens": int(max_len_tokens),
        "pos_resolution": int(pos_resolution),
        "rho_filter_min": float(min_rho),
        "rho_filter_max": float(max_rho),
        "joint_empty_measure_trim": bool(not args.disable_joint_empty_measure_trim),
        "same_onset_mapping_policy": "shared_onset_suffix_mapping",
        "prefix_truncation_policy": "ornament_prefix_then_backsolve_skeleton_prefix",
        "stats": {
            "rho_mean": rho_mean,
            "rho_min": rho_min_out,
            "rho_max": rho_max_out,
            "rho_note_mean": float(np.mean(rho_note_values)) if rho_note_values else None,
            "rho_note_min": float(np.min(rho_note_values)) if rho_note_values else None,
            "rho_note_max": float(np.max(rho_note_values)) if rho_note_values else None,
            "resolution_strategy_counts_written": {k: int(v) for k, v in resolution_strategy_counts_written.items()},
            "skip_reason_counts": {k: int(v) for k, v in skip_reason_counts.items()},
        },
    }

    with (out_root / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    export_stats = {
        "schema": "simplemono_export_stats_v1",
        "dataset_kind": "real_otb_benchmark",
        "split_name": str(args.split_name),
        "input_dir": str(input_dir),
        "bpm": float(bpm),
        "pos_resolution": int(pos_resolution),
        "counts": {
            "files_discovered": int(len(files)),
            "files_written": int(written_n),
            "files_skipped": int(len(files) - written_n),
            "sequences": int(written_n),
        },
        "tracks": {
            "x": acc_x.to_dict(pos_resolution=pos_resolution, bpm=bpm),
            "x_orn": acc_xorn.to_dict(pos_resolution=pos_resolution, bpm=bpm),
        },
        "skip_reason_counts": meta["stats"]["skip_reason_counts"],
    }

    write_json(out_root / "export_stats.json", export_stats)
    aggregate_split_dirs_export_stats(Path(args.output_dir))

    print("[Done] real benchmark built.")
    print(json.dumps(meta["stats"], ensure_ascii=False, indent=2))

    if args.show_hist and rho_values:
        bins = [1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 5.0 / 6.0, 1.0 + 1e-6]
        hist, edges = np.histogram(np.asarray(rho_values, dtype=np.float32), bins=bins)
        for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
            print(f"  rho in [{lo:.3f},{hi:.3f}): {int(c)}")


if __name__ == "__main__":
    main()

# python -m preproc.preproc_real_ornament_benchmark --input_dir ".\data_jiugong_o2g" --vocab_pkl ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\real_jiugongdacheng_otb_bench" --split_name test --max_len_tokens 0 --min_rho 0.3333333333 --max_rho 1.0 --show_hist
# simplemono_preproc/encoder.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from .constants import (
    WINDOW_MAX_NOTES,
    WINDOW_MIN_NOTES,
    WINDOW_STEP_NOTES,
)
from .quantization import encode_dur_pos, encode_dt_pos
from .score_readers import extract_voices
from .transforms import group_shuffle_by_onset, sliding_windows, trim_empty_measures
from .types import SequenceMeta, VoiceData
from .utils import relpath_posix, stable_int_hash
from .vocab import SimpleMonoVocab

def voice_to_simplemono_triples(
    voice: VoiceData,
    rng: random.Random
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    """
    VoiceData(notes in pos) -> (triples, note_starts_pos)
    """
    notes = trim_empty_measures(voice.notes, voice.measures)
    if not notes:
        return [], []

    notes = group_shuffle_by_onset(notes, rng)

    triples: List[Tuple[int, int, int]] = []
    note_starts_pos: List[int] = []

    for i, n in enumerate(notes):
        note_starts_pos.append(int(n.start))

        pitch = int(n.pitch)

        dur_pos = max(1, int(n.end - n.start))
        dur_code = encode_dur_pos(dur_pos)

        if i + 1 < len(notes):
            raw_dt = int(notes[i + 1].start - n.end)
        else:
            raw_dt = 0

        dt_code = encode_dt_pos(raw_dt)
        triples.append((pitch, dur_code, dt_code))

    return triples, note_starts_pos


def iter_piece_triple_windows(
    piece_path: Path,
    input_dir: Path,
    seed: int,
    xml_group: str,
    *,
    max_windows_per_voice: int = 0,  # 0 => unlimited
    max_windows_per_file: int = 0,   # 0 => unlimited
) -> Iterable[Tuple[List[Tuple[int, int, int]], SequenceMeta]]:
    voices = extract_voices(piece_path, xml_group=xml_group)
    if not voices:
        return

    source_rel = relpath_posix(piece_path, input_dir)

    max_windows_per_voice = int(max_windows_per_voice)
    max_windows_per_file = int(max_windows_per_file)
    if max_windows_per_voice < 0 or max_windows_per_file < 0:
        raise ValueError("max_windows_per_voice/max_windows_per_file must be >= 0")

    yielded_file = 0

    for voice_idx, v in enumerate(voices):
        v_seed = seed + stable_int_hash(str(piece_path) + "::" + v.voice_id)
        rng = random.Random(v_seed)

        triples, note_starts_pos = voice_to_simplemono_triples(v, rng)
        if len(triples) < WINDOW_MIN_NOTES:
            continue

        yielded_voice = 0

        for window_idx, (st, seg) in enumerate(
            sliding_windows(
                triples,
                max_notes=WINDOW_MAX_NOTES,
                step_notes=WINDOW_STEP_NOTES,
                min_notes=WINDOW_MIN_NOTES,
            )
        ):
            bos_start_pos = int(note_starts_pos[st]) if note_starts_pos else 0

            yield seg, SequenceMeta(
                source_file=source_rel,
                voice_id=v.voice_id,
                voice_idx=voice_idx,
                window_idx=window_idx,
                window_start_note_idx=int(st),
                bos_start_pos=bos_start_pos,
            )

            yielded_voice += 1
            yielded_file += 1

            # (A) 每个 voice 限制：例如只要首窗
            if max_windows_per_voice > 0 and yielded_voice >= max_windows_per_voice:
                break

            # (B) 每个 file 限制：例如整个 file 只要 1 个窗
            if max_windows_per_file > 0 and yielded_file >= max_windows_per_file:
                return


def encode_triples_to_events(
    triples: List[Tuple[int, int, int]],
    vocab: SimpleMonoVocab,
    *,
    bos_start_pos: int = 0,
) -> np.ndarray:
    """
    Event[0]   = (<s>, <s>, <2-bos_start_pos>)
    Event[-1]  = (</s>, </s>, </s>)
    Middle     = (<0-p>, <1-d>, <2-dt>)
    """
    n = len(triples)
    events = np.empty((n + 2, 3), dtype=np.int32)

    events[0, 0] = vocab.bos_id
    events[0, 1] = vocab.bos_id
    events[0, 2] = vocab.dt_id(encode_dt_pos(int(bos_start_pos)))  # NEW

    for i, (p, d, dt) in enumerate(triples):
        events[i + 1, 0] = vocab.pitch_id(int(p))
        events[i + 1, 1] = vocab.dur_id(int(d))
        events[i + 1, 2] = vocab.dt_id(int(dt))

    events[n + 1, :] = vocab.eos_id
    return events


def pad_or_truncate_events(events: np.ndarray, max_events: int, vocab: SimpleMonoVocab) -> Tuple[np.ndarray, int]:
    """
    Pad to (max_events,3). Always keeps BOS and EOS.
    Returns:
      padded_events, used_len_events (excluding pads)
    """
    if max_events < 2:
        raise ValueError("max_events must be >= 2 (need BOS/EOS).")

    cur = int(events.shape[0])

    out = np.full((max_events, 3), vocab.pad_id, dtype=np.int32)

    if cur <= max_events:
        out[:cur] = events
        used = cur
        return out, used

    # truncate: keep BOS + first mid + EOS
    out[0] = events[0]
    mid_len = max_events - 2
    out[1 : 1 + mid_len] = events[1 : 1 + mid_len]
    out[max_events - 1] = events[-1]
    used = max_events
    return out, used


def triples_to_token_line(triples: List[Tuple[int, int, int]], *, bos_start_pos: int = 0) -> str:
    def triple_to_tokens(pitch: int, dur: int, dtime_signed: int) -> List[str]:
        dt = f"!{abs(dtime_signed)}" if dtime_signed < 0 else f"{dtime_signed}"
        return [f"<0-{pitch}>", f"<1-{dur}>", f"<2-{dt}>"]

    bos_dt = encode_dt_pos(int(bos_start_pos))
    bos_dt_s = f"!{abs(bos_dt)}" if bos_dt < 0 else f"{bos_dt}"

    words = ["<s>", "<s>", f"<2-{bos_dt_s}>"]   # NEW: BOS 第3列写 dtime
    for p, d, dt in triples:
        words.extend(triple_to_tokens(int(p), int(d), int(dt)))
    words.extend(["</s>", "</s>", "</s>"])
    return " ".join(words)
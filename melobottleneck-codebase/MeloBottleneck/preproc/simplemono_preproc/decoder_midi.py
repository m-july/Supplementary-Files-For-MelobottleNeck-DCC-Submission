# simplemono_preproc/decoder_midi.py
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _lcm(a: int, b: int) -> int:
    return a // gcd(a, b) * b


def sanitize_filename(s: str, repl: str = "_") -> str:
    # Windows illegal: <>:"/\|?*
    illegal = '<>:"/\\|?*'
    out = []
    for ch in s:
        out.append(repl if ch in illegal else ch)
    # avoid trailing dot/space on Windows
    return "".join(out).rstrip(" .")


@dataclass(frozen=True)
class SimpleMonoDecodingConfig:
    # quantization
    pos_resolution: int
    dur_samples: int
    dt_samples: int
    max_pitch: int

    # ids
    pad_id: int
    bos_id: int
    eos_id: int
    unk_id: int
    mask_id: int

    # offsets in GLOBAL vocab
    pitch_offset: int
    dur_offset: int
    dt_pos_offset: int
    dt_neg_offset: int

    @property
    def max_dur_code(self) -> int:
        return self.dur_samples - 1

    @property
    def dt_code_min(self) -> int:
        return -self.dt_samples

    @property
    def dt_code_max(self) -> int:
        return self.dt_samples - 1

    @classmethod
    def from_simplemono_pkl(cls, pkl_path: Path) -> "SimpleMonoDecodingConfig":
        import pickle

        obj = pickle.load(Path(pkl_path).open("rb"))
        token2id: Dict[str, int] = obj["token2id"]
        qc = obj.get("quantization_config", {})

        pos_resolution = int(qc.get("pos_resolution", 12))
        dur_samples = int(qc.get("dur_samples", 96))
        dt_samples = int(qc.get("dt_samples", 96))

        # infer max_pitch from vocab (more robust than hardcode 127)
        max_pitch = 0
        for tok in token2id.keys():
            if tok.startswith("<0-") and tok.endswith(">"):
                try:
                    p = int(tok[3:-1])
                    max_pitch = max(max_pitch, p)
                except Exception:
                    pass

        # offsets: read from known tokens
        pitch_offset = token2id["<0-0>"]
        dur_offset = token2id["<1-0>"]
        dt_pos_offset = token2id["<2-0>"]
        dt_neg_offset = token2id["<2-!1>"]

        return cls(
            pos_resolution=pos_resolution,
            dur_samples=dur_samples,
            dt_samples=dt_samples,
            max_pitch=max_pitch,
            pad_id=token2id["<pad>"],
            unk_id=token2id["<unk>"],
            bos_id=token2id["<s>"],
            eos_id=token2id["</s>"],
            mask_id=token2id["<mask>"],
            pitch_offset=pitch_offset,
            dur_offset=dur_offset,
            dt_pos_offset=dt_pos_offset,
            dt_neg_offset=dt_neg_offset,
        )


def global_dt_id_to_signed_code(
    dtid: int,
    cfg: SimpleMonoDecodingConfig,
    *,
    strict: bool = False,
) -> int:
    dtid = int(dtid)

    # allow specials => treat as 0
    if dtid in (cfg.pad_id, cfg.bos_id, cfg.eos_id, cfg.unk_id, cfg.mask_id):
        return 0

    if cfg.dt_pos_offset <= dtid < cfg.dt_neg_offset:
        return int(dtid - cfg.dt_pos_offset)

    if cfg.dt_neg_offset <= dtid < (cfg.dt_neg_offset + cfg.dt_samples):
        mag = int(dtid - cfg.dt_neg_offset) + 1
        return -mag

    if strict:
        raise ValueError(f"Bad dt global id: {dtid}")
    return 0


def decode_bos_start_pos(
    events_global_ids: np.ndarray,
    cfg: SimpleMonoDecodingConfig,
    *,
    used_len_events: Optional[int] = None,
    strict: bool = False,
) -> int:
    ev = np.asarray(events_global_ids)
    if ev.ndim != 2 or ev.shape[1] != 3:
        return 0
    if used_len_events is not None:
        ev = ev[: int(used_len_events)]
    if ev.shape[0] == 0:
        return 0

    # default: assume row0 is BOS
    if int(ev[0, 0]) == cfg.bos_id:
        return global_dt_id_to_signed_code(int(ev[0, 2]), cfg, strict=strict)

    # fallback: search first BOS row
    for row in ev:
        if int(row[0]) == cfg.bos_id:
            return global_dt_id_to_signed_code(int(row[2]), cfg, strict=strict)

    return 0


def global_row_to_semantic_triple(
    row_global_ids: np.ndarray,
    cfg: SimpleMonoDecodingConfig,
    *,
    strict: bool = False,
) -> Optional[Tuple[int, int, int]]:
    """
    row: (pitch_global_id, dur_global_id, dt_global_id)
    returns: (pitch, dur_code, dt_code_signed) in semantic space
    """
    pid = int(row_global_ids[0])
    did = int(row_global_ids[1])
    dtid = int(row_global_ids[2])

    # skip specials
    if pid in (cfg.pad_id, cfg.bos_id, cfg.eos_id):
        return None

    # pitch
    if not (cfg.pitch_offset <= pid < cfg.dur_offset):
        if strict:
            raise ValueError(f"Bad pitch global id: {pid}")
        return None
    pitch = pid - cfg.pitch_offset
    if pitch < 0 or pitch > cfg.max_pitch:
        if strict:
            raise ValueError(f"Pitch out of range: {pitch}")
        return None

    # duration
    if not (cfg.dur_offset <= did < cfg.dt_pos_offset):
        if strict:
            raise ValueError(f"Bad duration global id: {did}")
        return None
    dur_code = did - cfg.dur_offset

    # dt
    if cfg.dt_pos_offset <= dtid < cfg.dt_neg_offset:
        dt_code = dtid - cfg.dt_pos_offset
    elif cfg.dt_neg_offset <= dtid < (cfg.dt_neg_offset + cfg.dt_samples):
        mag = (dtid - cfg.dt_neg_offset) + 1  # 1..dt_samples
        dt_code = -mag
    else:
        if strict:
            raise ValueError(f"Bad dt global id: {dtid}")
        return None

    return int(pitch), int(dur_code), int(dt_code)


def decode_events_to_triples(
    events_global_ids: np.ndarray,
    cfg: SimpleMonoDecodingConfig,
    *,
    used_len_events: Optional[int] = None,
    strict: bool = False,
) -> List[Tuple[int, int, int]]:
    """
    events_global_ids: (T, 3) int
    """
    ev = np.asarray(events_global_ids)
    if ev.ndim != 2 or ev.shape[1] != 3:
        raise ValueError(f"events must be (T,3), got {ev.shape}")

    if used_len_events is not None:
        ev = ev[: int(used_len_events)]

    triples: List[Tuple[int, int, int]] = []
    for row in ev:
        if int(row[0]) == cfg.eos_id:
            break
        t = global_row_to_semantic_triple(row, cfg, strict=strict)
        if t is None:
            continue
        triples.append(t)
    return triples


def triples_to_notes_pos(
    triples: List[Tuple[int, int, int]],
    cfg: SimpleMonoDecodingConfig,
    *,
    start_pos0: int = 0,
    shift_to_nonneg: bool = True,
    shift_unit_pos: Optional[int] = None,   # default: whole note
) -> List[Tuple[int, int, int]]:
    notes: List[Tuple[int, int, int]] = []
    cur_start = int(start_pos0)

    for pitch, dur_code, dt_code in triples:
        dur_code = int(max(0, min(dur_code, cfg.max_dur_code)))
        dur_pos = max(1, dur_code)

        start_pos = int(cur_start)
        end_pos = int(start_pos + dur_pos)
        notes.append((int(pitch), start_pos, end_pos))

        dt_code = int(max(cfg.dt_code_min, min(dt_code, cfg.dt_code_max)))
        cur_start = int(end_pos + dt_code)   # NEW: 不要逐步 clamp

    if shift_to_nonneg and notes:
        min_start = min(s for _, s, _ in notes)
        if min_start < 0:
            base = int(shift_unit_pos) if shift_unit_pos is not None else (4 * int(cfg.pos_resolution))
            if base <= 0:
                base = 1
            shift = ((-min_start + base - 1) // base) * base  # ceil
            notes = [(p, s + shift, e + shift) for (p, s, e) in notes]

    # safety (理论上 shift 后不会负)
    fixed: List[Tuple[int, int, int]] = []
    for p, s, e in notes:
        if s < 0:
            # 极端异常才会进来
            delta = -s
            s = 0
            e = e + delta
        if e <= s:
            e = s + 1
        fixed.append((p, s, e))
    return fixed


def notes_pos_to_midi(
    notes_pos: List[Tuple[int, int, int]],
    out_midi_path: Path,
    cfg: SimpleMonoDecodingConfig,
    *,
    program: int = 0,
    velocity: int = 80,
    tempo_bpm: float = 120.0,
    base_ticks_per_beat: int = 480,
) -> None:
    """
    Write a single-track MIDI (one Instrument) from (pitch, start_pos, end_pos).
    """
    import miditoolkit

    out_midi_path = Path(out_midi_path)
    out_midi_path.parent.mkdir(parents=True, exist_ok=True)

    tpq = _lcm(int(base_ticks_per_beat), int(cfg.pos_resolution))
    ticks_per_pos = tpq // int(cfg.pos_resolution)

    midi = miditoolkit.MidiFile(ticks_per_beat=tpq)

    # tempo / time signature (not encoded in your data, so we set defaults)
    midi.tempo_changes = [miditoolkit.TempoChange(float(tempo_bpm), 0)]
    midi.time_signature_changes = [miditoolkit.TimeSignature(4, 4, 0)]

    inst = miditoolkit.Instrument(program=int(program), is_drum=False, name="SimpleMono")
    for pitch, s_pos, e_pos in notes_pos:
        st = int(s_pos) * ticks_per_pos
        ed = int(e_pos) * ticks_per_pos
        if ed <= st:
            ed = st + 1
        inst.notes.append(miditoolkit.Note(int(velocity), int(pitch), int(st), int(ed)))

    midi.instruments = [inst]
    midi.dump(str(out_midi_path))


def decode_events_to_midi(
    events_global_ids: np.ndarray,
    out_midi_path: Path,
    cfg: SimpleMonoDecodingConfig,
    *,
    used_len_events: Optional[int] = None,
    strict: bool = False,
    program: int = 0,
    velocity: int = 80,
    tempo_bpm: float = 120.0,
) -> None:
    bos_start = decode_bos_start_pos(
        events_global_ids,
        cfg,
        used_len_events=used_len_events,
        strict=strict,
    )

    triples = decode_events_to_triples(
        events_global_ids,
        cfg,
        used_len_events=used_len_events,
        strict=strict,
    )

    notes_pos = triples_to_notes_pos(
        triples,
        cfg,
        start_pos0=bos_start,
        shift_to_nonneg=True,
        shift_unit_pos=4 * cfg.pos_resolution,  # whole note
    )

    notes_pos_to_midi(
        notes_pos,
        out_midi_path,
        cfg,
        program=program,
        velocity=velocity,
        tempo_bpm=tempo_bpm,
    )
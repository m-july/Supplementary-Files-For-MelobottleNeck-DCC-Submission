# simplemono_preproc/score_readers.py
from __future__ import annotations

import io
import os
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import (
    MAX_PITCH,
    MIN_NOTES_PER_VOICE,
    MIN_UNIQUE_PITCHES,
    POS_RESOLUTION,
    SUPPORTED_EXTS,
    TRUNC_POS,
)
from .types import NoteEv, VoiceData


def iter_score_files(root: Path) -> List[Path]:
    _EXTS_TUPLE = tuple(SUPPORTED_EXTS)             # 给 endswith 用（C 层实现，快）
    out: List[Path] = []
    # followlinks 默认 False：避免符号链接循环（也更快/更安全）
    for dirpath, dirnames, filenames in os.walk(root):
        # 如可行，剪枝巨大无关目录（见下文方案 D）
        for name in filenames:
            if name.lower().endswith(_EXTS_TUPLE):
                out.append(Path(dirpath, name))
    out.sort(key=os.fspath)  # sort Path 也行；key=os.fspath 往往更轻一点
    return out


# ----------------------------
# MusicXML (.mxl) helper
# ----------------------------
def read_mxl_as_musicxml_bytes(mxl_path: Path) -> bytes:
    """
    Read .mxl (zip) and extract the main MusicXML file bytes.
    Heuristics:
      1) if META-INF/container.xml exists, follow rootfile full-path
      2) else pick the first non-META-INF *.xml/*.musicxml entry with smallest name length
    """
    with zipfile.ZipFile(mxl_path, "r") as zf:
        names = zf.namelist()

        # Try container.xml
        if "META-INF/container.xml" in names:
            import xml.etree.ElementTree as ET

            container = zf.read("META-INF/container.xml")
            root = ET.fromstring(container)
            rootfiles = []
            for el in root.iter():
                if el.tag.lower().endswith("rootfile") and "full-path" in el.attrib:
                    rootfiles.append(el.attrib["full-path"])
            if rootfiles:
                target = rootfiles[0]
                if target in names:
                    return zf.read(target)

        # Fallback: choose a candidate xml file
        cand = []
        for n in names:
            nl = n.lower()
            if nl.startswith("meta-inf/"):
                continue
            if nl.endswith(".xml") or nl.endswith(".musicxml"):
                cand.append(n)
        if not cand:
            raise ValueError(f"No MusicXML found inside {mxl_path}")
        cand.sort(key=lambda x: (len(x), x))
        return zf.read(cand[0])


# ----------------------------
# MIDI parsing (miditoolkit)
# ----------------------------
def time_signature_reduce(
    numerator: int,
    denominator: int,
    max_ts_denominator: int = 6,
    max_notes_per_bar: int = 2,
):
    # same idea as MusicBERT script
    while (
        denominator > 2**max_ts_denominator
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


def midi_tick_to_pos(tick: int, ticks_per_beat: int) -> int:
    """
    Convert MIDI ticks to pos units.

    Args:
        tick: MIDI tick value (depends on file's ticks_per_beat)
        ticks_per_beat: MIDI file's time resolution (PPQ/TPQN)

    Returns:
        Position in pos units (based on POS_RESOLUTION per quarter note)
    """
    return int(round(tick * POS_RESOLUTION / ticks_per_beat))


def build_midi_measures(midi_obj, max_pos: int) -> List[Tuple[int, int]]:
    """
    Create measure boundaries in pos units.
    If a TS change happens mid-measure, cut the measure at the change point.

    Returns:
        List of (measure_start, measure_end) tuples in pos units
    """
    import miditoolkit

    tpq = midi_obj.ticks_per_beat

    tsc = list(midi_obj.time_signature_changes)
    if not tsc:
        tsc = [miditoolkit.containers.TimeSignature(4, 4, 0)]
    tsc.sort(key=lambda x: x.time)

    ts_points = []
    for ts in tsc:
        p = midi_tick_to_pos(ts.time, tpq)
        num, den = time_signature_reduce(int(ts.numerator), int(ts.denominator))
        ts_points.append((p, num, den))
    if ts_points[0][0] != 0:
        ts_points.insert(0, (0, 4, 4))

    measures: List[Tuple[int, int]] = []
    cur_pos = 0
    ts_i = 0

    while cur_pos < max_pos:
        while ts_i + 1 < len(ts_points) and ts_points[ts_i + 1][0] <= cur_pos:
            ts_i += 1

        _ts_start, num, den = ts_points[ts_i]
        # Measure length in pos units: (num/den bars) * (4 quarter notes/bar) * POS_RESOLUTION
        measure_len = int(round(num * 4 * POS_RESOLUTION / den))
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


def extract_midi_voices(midi_path: Path) -> List[VoiceData]:
    import miditoolkit

    midi_obj = miditoolkit.MidiFile(str(midi_path))
    tpq = midi_obj.ticks_per_beat

    all_notes_pos = []
    for inst in midi_obj.instruments:
        for n in inst.notes:
            s = midi_tick_to_pos(int(n.start), tpq)
            e = midi_tick_to_pos(int(n.end), tpq)
            if e <= s:
                e = s + 1
            all_notes_pos.append(e)
    if not all_notes_pos:
        return []

    max_pos = min(max(all_notes_pos) + 1, TRUNC_POS)
    measures = build_midi_measures(midi_obj, max_pos=max_pos)

    voices: List[VoiceData] = []
    for idx, inst in enumerate(midi_obj.instruments):
        if inst.is_drum:
            continue

        notes: List[NoteEv] = []
        for n in inst.notes:
            s = midi_tick_to_pos(int(n.start), tpq)
            e = midi_tick_to_pos(int(n.end), tpq)
            if s >= TRUNC_POS:
                continue
            if e <= s:
                e = s + 1
            e = min(e, TRUNC_POS)
            pitch = int(n.pitch)
            if pitch < 0 or pitch > MAX_PITCH:
                continue
            notes.append(NoteEv(pitch, s, e))

        if len(notes) < MIN_NOTES_PER_VOICE:
            continue
        uniq = len(set(nn.pitch for nn in notes))
        if uniq < MIN_UNIQUE_PITCHES:
            continue

        voice_id = f"midi:inst={idx}:program={inst.program}"
        voices.append(VoiceData(voice_id=voice_id, notes=notes, measures=measures))

    return voices


# ----------------------------
# MusicXML parsing (partitura)
# ----------------------------
def _require_partitura():
    try:
        import partitura as pt
        from partitura.score import Measure
    except ImportError as e:
        raise ImportError(
            "Missing dependency: partitura. Install via `pip install partitura`."
        ) from e
    return pt, Measure


def part_quarter(part, t_div: int) -> float:
    return float(np.asarray(part.quarter_map(t_div)).reshape(-1)[0])


def extract_musicxml_voices(xml_path: Path, xml_group: str = "staff_voice") -> List[VoiceData]:
    """
    xml_group:
      - "part": one voice per part (merge staves/voices)
      - "staff": one voice per (part, staff)
      - "staff_voice": one voice per (part, staff, voice)
    """
    pt, Measure = _require_partitura()

    suffix = xml_path.suffix.lower()
    if suffix == ".mxl":
        xml_bytes = read_mxl_as_musicxml_bytes(xml_path)
        score = pt.load_musicxml(io.BytesIO(xml_bytes), force_note_ids="keep")
    else:
        score = pt.load_musicxml(str(xml_path), force_note_ids="keep")

    voices: List[VoiceData] = []

    for p_idx, part in enumerate(score.parts):
        measures_div = list(part.iter_all(Measure))
        measures_pos: Optional[List[Tuple[int, int]]] = None
        if measures_div:
            ms = []
            for m in measures_div:
                if m.start is None or m.end is None:
                    continue
                qs = part_quarter(part, int(m.start.t))
                qe = part_quarter(part, int(m.end.t))
                # Convert measure boundaries from quarter notes to pos units
                ps = int(round(qs * POS_RESOLUTION))
                pe = int(round(qe * POS_RESOLUTION))
                if pe > ps:
                    ms.append((ps, pe))
            if ms:
                measures_pos = sorted(ms, key=lambda x: x[0])

        na = part.note_array(
            include_staff=True,
            include_grace_notes=True,
        )
        if na is None or len(na) == 0:
            continue

        groups: Dict[Tuple, List[NoteEv]] = {}
        names = set(na.dtype.names)
        has_staff = "staff" in names
        has_voice = "voice" in names
        has_grace = "is_grace" in names

        for row in na:
            if has_grace and int(row["is_grace"]) == 1:
                continue

            pitch = int(row["pitch"])
            if pitch < 0 or pitch > MAX_PITCH:
                continue

            onset_q = float(row["onset_quarter"])
            dur_q = float(row["duration_quarter"])
            if dur_q <= 0:
                continue

            # Convert from quarter notes to pos units
            s = int(round(onset_q * POS_RESOLUTION))
            d = int(round(dur_q * POS_RESOLUTION))
            if d <= 0:
                d = 1
            e = s + d
            if s >= TRUNC_POS:
                continue
            e = min(e, TRUNC_POS)

            staff = int(row["staff"]) if has_staff else 0
            voice = int(row["voice"]) if has_voice else 0

            if xml_group == "part":
                key = (p_idx,)
            elif xml_group == "staff":
                key = (p_idx, staff)
            else:
                key = (p_idx, staff, voice)

            groups.setdefault(key, []).append(NoteEv(pitch, s, e))

        for key, notes in groups.items():
            if len(notes) < MIN_NOTES_PER_VOICE:
                continue
            uniq = len(set(nn.pitch for nn in notes))
            if uniq < MIN_UNIQUE_PITCHES:
                continue

            if xml_group == "part":
                voice_id = f"xml:part={p_idx}"
            elif xml_group == "staff":
                voice_id = f"xml:part={p_idx}:staff={key[1]}"
            else:
                voice_id = f"xml:part={p_idx}:staff={key[1]}:voice={key[2]}"

            voices.append(VoiceData(voice_id=voice_id, notes=notes, measures=measures_pos))

    return voices


# ----------------------------
# Humdrum/Kern (.krn) parsing (music21)
# ----------------------------
def _ql_to_pos(qL) -> int:
    """
    Convert music21 quarterLength (quarter notes) to pos units.

    Args:
        qL: Quarter length value (Fraction or float)

    Returns:
        Position in pos units (based on POS_RESOLUTION per quarter note)
    """
    if isinstance(qL, Fraction):
        val = qL
    else:
        val = Fraction(qL).limit_denominator(4096)
    return int(round(float(val * POS_RESOLUTION)))


def extract_krn_voices(krn_path: Path) -> List[VoiceData]:
    """
    Parse Humdrum **kern (.krn) with music21 and return VoiceData list.

    Strategy:
      - Each music21 Part -> one VoiceData
      - Notes/Chords -> NoteEv
      - Rests are ignored (time gaps captured by offsets)
      - Ties merged into sustained note
      - Measure boundaries from music21 measures if available
    """
    try:
        from music21 import converter, note as m21note, chord as m21chord, stream as m21stream
    except ImportError as e:
        raise ImportError(
            "Missing dependency: music21. Install via `pip install music21` to enable .krn parsing."
        ) from e

    score = converter.parse(str(krn_path))
    voices: List[VoiceData] = []

    for p_idx, part in enumerate(score.parts):
        measures_pos: Optional[List[Tuple[int, int]]] = None
        ms: List[Tuple[int, int]] = []
        for m in part.getElementsByClass(m21stream.Measure):
            ms_q = m.offset
            if m.barDuration is not None:
                me_q = ms_q + m.barDuration.quarterLength
            else:
                me_q = ms_q + m.duration.quarterLength

            ps = _ql_to_pos(ms_q)
            pe = _ql_to_pos(me_q)
            if pe > ps:
                ms.append((ps, pe))
        if ms:
            measures_pos = ms

        notes_out: List[NoteEv] = []
        active_ties: Dict[int, Tuple[int, int]] = {}

        elems = list(part.recurse().notesAndRests)

        def _sort_key(el):
            off = float(el.getOffsetInHierarchy(part))
            return (off, 0)

        elems.sort(key=_sort_key)

        for el in elems:
            if getattr(el, "duration", None) is not None and getattr(el.duration, "isGrace", False):
                continue

            off_q = float(el.getOffsetInHierarchy(part))
            ql = getattr(el.duration, "quarterLength", None)
            if ql is None or float(ql) <= 0:
                continue

            s_pos = _ql_to_pos(off_q)
            e_pos = _ql_to_pos(off_q + ql)
            if e_pos <= s_pos:
                e_pos = s_pos + 1

            if s_pos >= TRUNC_POS:
                continue
            e_pos = min(e_pos, TRUNC_POS)

            if isinstance(el, m21note.Rest):
                continue

            pitch_items: List[Tuple[int, Optional[str]]] = []
            if isinstance(el, m21chord.Chord):
                for p in el.pitches:
                    midi = int(p.midi)
                    pitch_items.append((midi, None))
            elif isinstance(el, m21note.Note):
                midi = int(el.pitch.midi)
                tie_type = el.tie.type if el.tie is not None else None
                pitch_items.append((midi, tie_type))
            else:
                continue

            for midi, tie_type in pitch_items:
                if midi < 0 or midi > MAX_PITCH:
                    continue

                if tie_type in ("start", "continue"):
                    if midi in active_ties:
                        st0, en0 = active_ties[midi]
                        active_ties[midi] = (st0, max(en0, e_pos))
                    else:
                        active_ties[midi] = (s_pos, e_pos)
                    continue

                if tie_type == "stop":
                    if midi in active_ties:
                        st0, en0 = active_ties[midi]
                        notes_out.append(NoteEv(midi, st0, max(en0, e_pos)))
                        del active_ties[midi]
                    else:
                        notes_out.append(NoteEv(midi, s_pos, e_pos))
                    continue

                if midi in active_ties:
                    st0, en0 = active_ties[midi]
                    notes_out.append(NoteEv(midi, st0, en0))
                    del active_ties[midi]

                notes_out.append(NoteEv(midi, s_pos, e_pos))

        for midi, (st0, en0) in active_ties.items():
            if en0 > st0:
                notes_out.append(NoteEv(midi, st0, en0))

        if not notes_out:
            continue

        voice_id = f"krn:part={p_idx}"
        voices.append(VoiceData(voice_id=voice_id, notes=notes_out, measures=measures_pos))

    return voices


def extract_voices(score_path: Path, xml_group: str = "staff_voice") -> List[VoiceData]:
    suf = score_path.suffix.lower()
    if suf in (".mid", ".midi"):
        return extract_midi_voices(score_path)
    if suf == ".krn":
        return extract_krn_voices(score_path)
    # xml/musicxml/mxl
    return extract_musicxml_voices(score_path, xml_group=xml_group)

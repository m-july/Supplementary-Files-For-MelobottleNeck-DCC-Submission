# simplemono_preproc/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class NoteEv:
    """
    A single note event with quantized time positions.

    Time units (pos):
        - Based on POS_RESOLUTION ticks per quarter note (see constants.py)
        - Independent of tempo; purely positional/symbolic time
        - Example: if POS_RESOLUTION=16, then 1 pos = 1/16 quarter note
    """
    pitch: int
    start: int  # start position in pos units
    end: int    # end position in pos units (exclusive)


@dataclass
class VoiceData:
    voice_id: str
    notes: List[NoteEv]
    measures: Optional[List[Tuple[int, int]]] = None  # (m_start_pos, m_end_pos) in pos units


@dataclass
class SequenceMeta:
    source_file: str
    voice_id: str
    voice_idx: int
    window_idx: int
    window_start_note_idx: int
    bos_start_pos: int   # pos units, can be negative in future

# simplemono_preproc/transforms.py
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import random

from .types import NoteEv


def group_shuffle_by_onset(notes: List[NoteEv], rng: random.Random) -> List[NoteEv]:
    notes = sorted(notes, key=lambda x: (x.start, x.end, x.pitch))
    out: List[NoteEv] = []
    i = 0
    while i < len(notes):
        j = i + 1
        while j < len(notes) and notes[j].start == notes[i].start:
            j += 1
        grp = notes[i:j]
        rng.shuffle(grp)
        out.extend(grp)
        i = j
    return out


def sliding_windows(
    triples: List[Tuple[int, int, int]],
    max_notes: int,
    step_notes: int,
    min_notes: int,
) -> Iterable[Tuple[int, List[Tuple[int, int, int]]]]:
    n = len(triples)
    if n < min_notes:
        return
    for st in range(0, n, step_notes):
        ed = min(st + max_notes, n)
        if ed - st >= min_notes:
            yield st, triples[st:ed]
        if ed == n:
            break


def build_cut_prefix(cuts: List[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    cut_ends = []
    prefix = []
    acc = 0
    for a, b in cuts:
        acc += (b - a)
        cut_ends.append(b)
        prefix.append(acc)
    return cut_ends, prefix


def trim_empty_measures(notes: List[NoteEv], measures: Optional[List[Tuple[int, int]]]) -> List[NoteEv]:
    """
    Remove measures that contain only rests for THIS voice.
    After removing, shift subsequent notes left (time compression).
    """
    if not measures or not notes:
        return notes

    import bisect

    notes = sorted(notes, key=lambda x: (x.start, x.end))
    measures = sorted(measures, key=lambda x: x[0])

    empty: List[Tuple[int, int]] = []
    ni = 0
    for ms, me in measures:
        while ni < len(notes) and notes[ni].end <= ms:
            ni += 1
        if ni < len(notes) and notes[ni].start < me and notes[ni].end > ms:
            continue
        empty.append((ms, me))

    if not empty:
        return notes

    merged = []
    cur_s, cur_e = empty[0]
    for s, e in empty[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    empty = merged

    cut_ends, prefix = build_cut_prefix(empty)

    def removed_before(t: int) -> int:
        k = bisect.bisect_right(cut_ends, t)
        return prefix[k - 1] if k > 0 else 0

    shifted = []
    for n in notes:
        sh = removed_before(n.start)
        ns = n.start - sh
        ne = n.end - sh
        if ne <= ns:
            ne = ns + 1
        shifted.append(NoteEv(n.pitch, ns, ne))
    return shifted

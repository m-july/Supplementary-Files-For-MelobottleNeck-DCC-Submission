# preproc/gttm_preproc/mudep_compat.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union, Optional

import numpy as np
import xml.etree.ElementTree as ET


PathLike = Union[str, Path]


def load_score_musicxml(score_file: PathLike):
    """
    Load MusicXML using partitura, matching MuDeP behavior.
    Important: force_note_ids=True for stable IDs.
    """
    try:
        import partitura as pt
    except ImportError as e:
        raise ImportError("Missing dependency: partitura. Install via `pip install partitura`.") from e
    return pt.load_musicxml(str(score_file), force_note_ids=True)


def get_nra(score):
    """
    Build tied-note + rest array (nra) exactly as MuDeP:
      - tied note array from ensure_notearray(score)
      - rest array from ensure_rest_array(score.parts[0])
      - hstack + sort by onset_div

    Returns
    -------
    nra : np.ndarray structured
        fields at least: onset_div, duration_div, pitch, id
    """
    import partitura as pt

    na = pt.utils.music.ensure_notearray(score)[["onset_div", "duration_div", "pitch", "id"]]
    ra = pt.utils.music.ensure_rest_array(score.parts[0])[["onset_div", "duration_div", "pitch", "id"]]
    nra = np.hstack([na, ra])
    nra.sort(order="onset_div")
    return nra


def ts_xml_to_dependency_tree(ts_xml_file: PathLike) -> Tuple[List[Tuple[str, str]], str]:
    """
    Convert GTTM TS.xml (time-span tree) to dependency arcs in GTTM-style ids.

    Returns
    -------
    dep_arcs : list[(head_id, dep_id)] with GTTM-style ids and 'ROOT'
    root_id  : root note id in GTTM-style
    """
    tree = ET.parse(str(ts_xml_file))
    xml_root = tree.getroot()
    dep_arcs, root = _iterative_parse(xml_root)

    # add artificial root arcs (MuDeP style)
    dep_arcs.append(("ROOT", root))
    dep_arcs.append(("ROOT", "ROOT"))  # self-loop at ROOT
    return dep_arcs, root


def _iterative_parse(xml_elem) -> Tuple[List[Tuple[str, str]], str]:
    """
    MuDeP logic:
      - primary child is head
      - secondary child depends on primary
      - return (deps_in_subtree, head_id_of_subtree)
    """
    ts = xml_elem.find("ts")
    if ts is None:
        raise ValueError("Bad TS.xml: missing <ts> node.")

    primary_children = ts.find("primary")
    secondary_children = ts.find("secondary")

    if primary_children is None:
        # leaf: <head><chord><note id=...>
        if secondary_children is not None:
            raise ValueError("Bad TS.xml: leaf has secondary child.")
        head = ts.find("head")
        if head is None:
            raise ValueError("Bad TS.xml: leaf missing <head>.")
        chord = head.find("chord")
        if chord is None:
            raise ValueError("Bad TS.xml: leaf missing <chord>.")
        note = chord.find("note")
        if note is None:
            raise ValueError("Bad TS.xml: leaf missing <note>.")
        nid = note.attrib.get("id", None)
        if nid is None:
            raise ValueError("Bad TS.xml: <note> missing id attribute.")
        return [], nid

    if secondary_children is None:
        raise ValueError("Bad TS.xml: non-leaf missing <secondary>.")

    deps: List[Tuple[str, str]] = []
    deps_p, head_p = _iterative_parse(primary_children)
    deps_s, head_s = _iterative_parse(secondary_children)
    deps.extend(deps_p)
    deps.extend(deps_s)

    # primary -> secondary
    deps.append((head_p, head_s))
    return deps, head_p


def gttm_style_to_id_dependency_ts(
    gttm_ts_dependency: List[Tuple[str, str]],
    measure_mapping: np.ndarray,
    nra_untied: np.ndarray,
    nra_tied: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Convert GTTM-style ids (like P1-3-1) to indices in tied nra.

    Returns
    -------
    dep_list : list[(head_idx, dep_idx)] in tied nra indexing, ROOT -> -1
    """
    dep_list: List[Tuple[int, int]] = []
    for head_gttm, dep_gttm in gttm_ts_dependency:
        head_pt_id = gttm_id_to_pt_id(head_gttm, measure_mapping, nra_untied)
        dep_pt_id = gttm_id_to_pt_id(dep_gttm, measure_mapping, nra_untied)
        head_idx = note_id_to_note_array_index(head_pt_id, nra_tied)
        dep_idx = note_id_to_note_array_index(dep_pt_id, nra_tied)
        dep_list.append((head_idx, dep_idx))

    # check single head per dependent
    ends = np.asarray([e[1] for e in dep_list], dtype=np.int64)
    _, end_count = np.unique(ends, return_counts=True)
    if not np.all(end_count == 1):
        raise ValueError("Some nodes have multiple heads (not a tree).")

    return dep_list


def gttm_id_to_pt_id(gttm_id: str, measure_mapping: np.ndarray, nra_untied: np.ndarray):
    """
    Translate GTTM-style 'P1-3-1' to partitura note id in nra_untied.
    ROOT stays ROOT.
    """
    if gttm_id == "ROOT":
        return "ROOT"
    parts = gttm_id.split("-")
    if len(parts) < 3:
        raise ValueError(f"Bad GTTM id format: {gttm_id}")
    measure_number = int(parts[1])
    note_number = int(parts[2])

    idxs = np.where(measure_mapping == measure_number)[0]
    if idxs.size == 0:
        raise ValueError(f"No notes in measure {measure_number} for id={gttm_id}")

    pick = int(note_number) - 1
    if pick < 0 or pick >= idxs.size:
        raise ValueError(f"Note number out of range: id={gttm_id}, idxs={idxs.size}")

    nra_index = idxs[pick]
    return nra_untied[nra_index]["id"]


def note_id_to_note_array_index(note_id, nra_tied: np.ndarray) -> int:
    """
    Translate partitura note id to index in tied nra.
    ROOT -> -1
    Rests are not allowed (MuDeP throws).
    """
    if note_id == "ROOT":
        return -1

    # rest id starts with 'r' in MuDeP assumption
    # be robust for bytes ids
    s = note_id.decode("utf-8", errors="ignore") if isinstance(note_id, (bytes, bytearray)) else str(note_id)
    if s.startswith("r"):
        raise ValueError("Trying to build an arc from a rest")

    idxs = np.where(nra_tied["id"] == note_id)[0]
    if idxs.size != 1:
        raise ValueError(f"Problem with finding note id in tied nra: id={note_id}, matches={idxs.size}")
    return int(idxs[0])


def get_dependency_arcs(ts_xml_file: PathLike, score, nra_tied: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[str, str]]]:
    """
    Compute gold dependency arcs in tied nra indexing.
    Includes:
      - (-1, root_note_idx)
      - (-1, -1) root self-loop
      - (head_idx, dep_idx) for all other deps
    """
    import partitura as pt

    gttm_ts, _ = ts_xml_to_dependency_tree(ts_xml_file)

    # Build untied note+rest array for GTTM counting
    na_untied = pt.utils.music.note_array_from_note_list(score.parts[0].notes)
    ra_untied = pt.utils.music.rest_array_from_rest_list(score.parts[0].rests)
    ra_fields = list(ra_untied.dtype.names)
    nra_untied = np.hstack([na_untied[ra_fields], ra_untied])
    nra_untied.sort(order="onset_div")

    # remove grace notes
    nra_untied = nra_untied[nra_untied["duration_div"] != 0]

    m_map = score.parts[0].measure_number_map(nra_untied["onset_div"])

    try:
        dep_list = gttm_style_to_id_dependency_ts(gttm_ts, m_map, nra_untied, nra_tied)
        return dep_list, gttm_ts
    except Exception:
        # common pickup-measure mismatch: first rest not counted
        # MuDeP tries dropping first 1 or 2 elements.
        try:
            dep_list = gttm_style_to_id_dependency_ts(gttm_ts, m_map[1:], nra_untied[1:], nra_tied)
            return dep_list, gttm_ts
        except Exception:
            dep_list = gttm_style_to_id_dependency_ts(gttm_ts, m_map[2:], nra_untied[2:], nra_tied)
            return dep_list, gttm_ts


def shift_dep_arcs_add1(dep_list: List[Tuple[int, int]]) -> np.ndarray:
    """
    MuDeP shifts indices by +1 to reserve 0 as ROOT.
      -1 -> 0
       i -> i+1

    Returns
    -------
    dep_arcs : np.ndarray shape [E,2], int32
    """
    arr = np.asarray(dep_list, dtype=np.int32)
    return arr + 1
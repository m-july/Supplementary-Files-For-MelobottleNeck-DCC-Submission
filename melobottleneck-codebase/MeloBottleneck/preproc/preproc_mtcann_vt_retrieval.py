# preproc/preproc_mtcann_vt_retrieval.py
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .simplemono_preproc.npy_stream import NpyAppendWriter
from .simplemono_preproc.vocab import SimpleMonoVocab
from .simplemono_preproc.pipelines.pretrain import default_max_len_tokens
from .simplemono_preproc.score_readers import extract_voices
from .simplemono_preproc.transforms import trim_empty_measures
from .simplemono_preproc.types import NoteEv, VoiceData
from .simplemono_preproc.quantization import encode_dur_pos, encode_dt_pos
from .simplemono_preproc.encoder import encode_triples_to_events, pad_or_truncate_events


def _relpath_posix(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return p.as_posix()


def _read_two_col_csv(path: Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rd = csv.reader(f)
        for row in rd:
            if not row or len(row) < 2:
                continue
            a = str(row[0]).strip()
            b = str(row[1]).strip()
            if not a or not b:
                continue
            rows.append((a, b))
    return rows


def _validate_simplemono_pkl(simplemono_pkl: Path, vocab: SimpleMonoVocab) -> None:
    """
    你原来的 Tavern retrieval 脚本里 --simplemono_pkl 其实没有真正参与编码。
    这里我至少做一个一致性校验，防止你现在代码里的 constants/vocab
    和模型训练时的 SimpleMono.pkl 已经不一致，却静默继续跑。
    """
    with simplemono_pkl.open("rb") as f:
        obj = pickle.load(f)

    cur = vocab.build_simplemono_pkl_object()

    if obj.get("token2id") != cur["token2id"]:
        raise ValueError(
            "Provided SimpleMono.pkl token2id does not match current SimpleMonoVocab.build(). "
            "This would make retrieval preprocessing inconsistent with your trained model."
        )

    if obj.get("quantization_config") != cur["quantization_config"]:
        raise ValueError(
            "Provided SimpleMono.pkl quantization_config does not match current code. "
            "This would change symbolic-time encoding semantics."
        )


def list_mtcann_member_reference_pairs(mtc_root: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Build non-self member->reference pairs from:
      - MTC-ANN-tune-family-labels.csv       songid -> tunefamily
      - MTC-ANN-referencemelodies.csv        tunefamily -> reference_songid

    We use krn/*.krn by default, because:
      - MTC-ANN is symbolic and clean in krn
      - your current MIDI reader has MIN_NOTES_PER_VOICE filtering, which may drop short folk tunes
    """
    meta_dir = mtc_root / "metadata"
    labels_csv = meta_dir / "MTC-ANN-tune-family-labels.csv"
    refs_csv = meta_dir / "MTC-ANN-referencemelodies.csv"

    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)
    if not refs_csv.exists():
        raise FileNotFoundError(refs_csv)

    label_rows = _read_two_col_csv(labels_csv)   # (songid, tunefamily)
    ref_rows = _read_two_col_csv(refs_csv)       # (tunefamily, reference_songid)

    family_to_ref: Dict[str, str] = {}
    family_order: Dict[str, int] = {}
    for idx, (fam, ref_songid) in enumerate(ref_rows):
        if fam not in family_to_ref:
            family_to_ref[fam] = ref_songid
            family_order[fam] = idx

    pairs: List[Dict[str, Any]] = []
    self_pairs = 0
    missing_reference = 0
    missing_files = 0

    for songid, tunefamily in label_rows:
        ref_songid = family_to_ref.get(tunefamily, None)
        if ref_songid is None:
            missing_reference += 1
            continue

        if songid == ref_songid:
            self_pairs += 1
            continue

        query_file = mtc_root / "krn" / f"{songid}.krn"
        doc_file = mtc_root / "krn" / f"{ref_songid}.krn"

        if (not query_file.exists()) or (not doc_file.exists()):
            missing_files += 1
            continue

        pairs.append({
            "tunefamily": tunefamily,
            "query_songid": songid,
            "reference_songid": ref_songid,
            "query_file": query_file,
            "doc_file": doc_file,
        })

    pairs.sort(
        key=lambda x: (
            int(family_order.get(x["tunefamily"], 10**9)),
            str(x["tunefamily"]),
            str(x["query_songid"]),
        )
    )

    stats = {
        "songs_total": int(len(label_rows)),
        "families_total": int(len(set(f for _, f in label_rows))),
        "references_total": int(len(ref_rows)),
        "self_pairs_excluded": int(self_pairs),
        "missing_reference": int(missing_reference),
        "missing_files": int(missing_files),
        "pairs_after_file_check": int(len(pairs)),
    }
    return pairs, stats


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


def _melody_monophonize(notes: List[NoteEv], melody_mode: str) -> List[NoteEv]:
    """
    Same spirit as your Tavern helper:
      - sort notes
      - shift to zero
      - if melody_mode == "top", keep highest pitch per onset group
      - if melody_mode == "all", keep all notes as-is
    """
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
                dur_n = int(n.end) - int(n.start)
                dur_b = int(best.end) - int(best.start)
                if dur_n > dur_b:
                    best = n
        out.append(best)
        i = j

    return out


def _choose_primary_voice(voices: List[VoiceData]) -> Optional[VoiceData]:
    """
    MTC-ANN should usually be essentially monophonic.
    But to be robust against odd parsing results, choose:
      1) most notes
      2) higher median pitch as tiebreak
      3) higher max pitch as second tiebreak
    """
    best: Optional[VoiceData] = None
    best_key: Optional[Tuple[int, float, int]] = None

    for v in voices:
        notes = trim_empty_measures(v.notes, v.measures)
        if not notes:
            continue

        pitches = np.asarray([int(n.pitch) for n in notes], dtype=np.int32)
        if pitches.size == 0:
            continue

        key = (
            int(len(notes)),
            float(np.median(pitches)),
            int(np.max(pitches)),
        )

        if best is None or key > best_key:
            best = v
            best_key = key

    return best


def load_mtc_melody_notes(
    score_path: Path,
    melody_mode: str,
    cache: Dict[str, List[NoteEv]],
) -> List[NoteEv]:
    k = str(Path(score_path).resolve())
    if k in cache:
        return cache[k]

    try:
        voices = extract_voices(Path(score_path), xml_group="staff_voice")
    except Exception:
        cache[k] = []
        return []

    if not voices:
        cache[k] = []
        return []

    primary = _choose_primary_voice(voices)
    if primary is None:
        cache[k] = []
        return []

    notes = trim_empty_measures(primary.notes, primary.measures)
    notes_mono = _melody_monophonize(notes, melody_mode=melody_mode)
    cache[k] = notes_mono
    return notes_mono


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtc_root", type=str, required=True, help="MTC-ANN-2.0.1 root")
    ap.add_argument(
        "--simplemono_pkl",
        type=str,
        required=True,
        help="SimpleMono.pkl; used here for vocab/quantization consistency check",
    )
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--split_name", type=str, default="test")

    ap.add_argument(
        "--max_len_tokens",
        type=int,
        default=0,
        help="0 => default_max_len_tokens(); must be divisible by 3",
    )
    ap.add_argument("--melody_mode", type=str, default="top", choices=["top", "all"])

    ap.add_argument("--min_doc_notes", type=int, default=8)
    ap.add_argument("--min_query_notes", type=int, default=8)
    ap.add_argument("--max_pairs", type=int, default=0, help="0 => all")

    ap.add_argument("--docs_name", type=str, default="docs.npy")
    ap.add_argument("--queries_name", type=str, default="queries.npy")
    ap.add_argument(
        "--targets_name",
        type=str,
        default="target_doc_ids.npy",
        help='"" => do not write target_doc_ids.npy',
    )

    ap.add_argument(
        "--write_meta_jsonl",
        action="store_true",
        help="write docs_meta.jsonl + queries_meta.jsonl (names follow docs_name/queries_name stems)",
    )

    args = ap.parse_args()

    mtc_root = Path(args.mtc_root)
    out_root = Path(args.output_dir) / str(args.split_name)

    if out_root.exists():
        raise FileExistsError(f"Output dir already exists: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)

    if not str(args.docs_name).strip():
        raise ValueError("--docs_name must be non-empty")
    if not str(args.queries_name).strip():
        raise ValueError("--queries_name must be non-empty")

    max_len_tokens = int(args.max_len_tokens)
    if max_len_tokens <= 0:
        max_len_tokens = int(default_max_len_tokens())
    if max_len_tokens % 3 != 0:
        raise ValueError(f"max_len_tokens must be divisible by 3, got {max_len_tokens}")

    max_events = max_len_tokens // 3
    max_notes = max_events - 2

    print(f"[MTC-ANN] {mtc_root}")
    print(f"[OUT]     {out_root}")
    print(f"[LEN]     max_len_tokens={max_len_tokens} => max_events={max_events}")
    print(f"[MODE]    melody_mode={args.melody_mode}")

    vocab_global = SimpleMonoVocab.build()
    _validate_simplemono_pkl(Path(args.simplemono_pkl), vocab_global)

    pairs, pair_scan = list_mtcann_member_reference_pairs(mtc_root)
    if int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]

    print(f"[PAIR]    loaded {len(pairs)} non-self member->prototype pairs.")
    print(f"[SCAN]    {json.dumps(pair_scan, ensure_ascii=False)}")

    # -----------------------------
    # pass 1: filter pairs by parsing + min length
    # -----------------------------
    note_cache: Dict[str, List[NoteEv]] = {}

    kept_pairs: List[Dict[str, Any]] = []
    skip = {
        "parse_or_empty_doc": 0,
        "parse_or_empty_query": 0,
        "too_short_doc": 0,
        "too_short_query": 0,
        "exception": 0,
    }

    for p in tqdm(pairs, desc="[Scan] parse+filter", dynamic_ncols=True, smoothing=0.0):
        try:
            doc_notes = load_mtc_melody_notes(Path(p["doc_file"]), melody_mode=args.melody_mode, cache=note_cache)
            if not doc_notes:
                skip["parse_or_empty_doc"] += 1
                continue
            if len(doc_notes) < int(args.min_doc_notes):
                skip["too_short_doc"] += 1
                continue

            query_notes = load_mtc_melody_notes(Path(p["query_file"]), melody_mode=args.melody_mode, cache=note_cache)
            if not query_notes:
                skip["parse_or_empty_query"] += 1
                continue
            if len(query_notes) < int(args.min_query_notes):
                skip["too_short_query"] += 1
                continue

            kept_pairs.append(p)

        except Exception:
            skip["exception"] += 1
            continue

    print(f"[KEEP]    kept_pairs={len(kept_pairs)} / {len(pairs)}")
    print(f"[SKIP]    {skip}")

    # -----------------------------
    # build doc list (unique references among kept pairs)
    # -----------------------------
    doc_entries: Dict[str, Dict[str, Any]] = {}
    for p in kept_pairs:
        ref_songid = str(p["reference_songid"])
        if ref_songid not in doc_entries:
            doc_entries[ref_songid] = {
                "reference_songid": ref_songid,
                "tunefamily": p["tunefamily"],
                "path": Path(p["doc_file"]),
            }

    # preserve reference-family order as much as possible
    ref_rows = _read_two_col_csv(mtc_root / "metadata" / "MTC-ANN-referencemelodies.csv")
    family_order = {fam: idx for idx, (fam, _) in enumerate(ref_rows)}

    docs_sorted = sorted(
        doc_entries.values(),
        key=lambda d: (
            int(family_order.get(d["tunefamily"], 10**9)),
            str(d["tunefamily"]),
            str(d["reference_songid"]),
        ),
    )

    # -----------------------------
    # writers
    # -----------------------------
    docs_name = str(args.docs_name)
    queries_name = str(args.queries_name)
    docs_stem = Path(docs_name).stem
    queries_stem = Path(queries_name).stem

    w_docs = NpyAppendWriter(out_root / docs_name, dtype=np.int32, row_shape=(max_events, 3))
    w_q = NpyAppendWriter(out_root / queries_name, dtype=np.int32, row_shape=(max_events, 3))

    w_tgt = None
    if str(args.targets_name).strip():
        w_tgt = NpyAppendWriter(out_root / str(args.targets_name), dtype=np.int32, row_shape=())

    f_docs_meta = None
    f_q_meta = None
    if bool(args.write_meta_jsonl):
        f_docs_meta = (out_root / f"{docs_stem}_meta.jsonl").open("w", encoding="utf-8")
        f_q_meta = (out_root / f"{queries_stem}_meta.jsonl").open("w", encoding="utf-8")

    ref_songid_to_docid: Dict[str, int] = {}

    # -----------------------------
    # encode docs (reference melodies)
    # -----------------------------
    for d in tqdm(docs_sorted, desc="[Encode] docs", dynamic_ncols=True, smoothing=0.0):
        doc_notes = load_mtc_melody_notes(d["path"], melody_mode=args.melody_mode, cache=note_cache)
        doc_notes_t = doc_notes[:max_notes]
        ev, used_len = encode_notes_to_padded_events_global(doc_notes_t, vocab_global, max_events=max_events)

        w_docs.append(ev.astype(np.int32, copy=False))
        doc_id = int(w_docs.count - 1)
        ref_songid_to_docid[str(d["reference_songid"])] = doc_id

        if f_docs_meta is not None:
            f_docs_meta.write(json.dumps({
                "doc_id": doc_id,
                "tunefamily": d["tunefamily"],
                "reference_songid": d["reference_songid"],
                "doc_file": _relpath_posix(Path(d["path"]), mtc_root),
                "doc_notes": int(len(doc_notes)),
                "doc_notes_trunc": int(len(doc_notes_t)),
                "used_len_events": int(used_len),
            }, ensure_ascii=False) + "\n")

    # -----------------------------
    # encode queries (family members)
    # -----------------------------
    for p in tqdm(kept_pairs, desc="[Encode] queries", dynamic_ncols=True, smoothing=0.0):
        tgt = ref_songid_to_docid.get(str(p["reference_songid"]), None)
        if tgt is None:
            continue

        query_path = Path(p["query_file"])
        query_notes = load_mtc_melody_notes(query_path, melody_mode=args.melody_mode, cache=note_cache)
        query_notes_t = query_notes[:max_notes]
        ev, used_len = encode_notes_to_padded_events_global(query_notes_t, vocab_global, max_events=max_events)

        w_q.append(ev.astype(np.int32, copy=False))
        qid = int(w_q.count - 1)

        if w_tgt is not None:
            w_tgt.append(np.int32(tgt))

        if f_q_meta is not None:
            f_q_meta.write(json.dumps({
                "query_id": qid,
                "target_doc_id": int(tgt),
                "tunefamily": p["tunefamily"],
                "query_songid": p["query_songid"],
                "reference_songid": p["reference_songid"],
                "query_file": _relpath_posix(query_path, mtc_root),
                "reference_file": _relpath_posix(Path(p["doc_file"]), mtc_root),
                "query_notes": int(len(query_notes)),
                "query_notes_trunc": int(len(query_notes_t)),
                "used_len_events": int(used_len),
            }, ensure_ascii=False) + "\n")

    # close
    w_docs.close()
    w_q.close()
    if w_tgt is not None:
        w_tgt.close()
    if f_docs_meta is not None:
        f_docs_meta.close()
    if f_q_meta is not None:
        f_q_meta.close()

    meta = {
        "version": "mtcann_member_reference_retrieval_v1",
        "mtc_root": str(mtc_root),
        "split_name": str(args.split_name),
        "simplemono_pkl": str(args.simplemono_pkl),
        "score_format": "krn",
        "melody_mode": str(args.melody_mode),
        "max_len_tokens": int(max_len_tokens),
        "max_events": int(max_events),
        "max_notes": int(max_notes),
        "filters": {
            "min_doc_notes": int(args.min_doc_notes),
            "min_query_notes": int(args.min_query_notes),
        },
        "outputs": {
            "docs_name": str(args.docs_name),
            "queries_name": str(args.queries_name),
            "targets_name": str(args.targets_name) if str(args.targets_name).strip() else None,
            "docs_meta_name": f"{docs_stem}_meta.jsonl" if bool(args.write_meta_jsonl) else None,
            "queries_meta_name": f"{queries_stem}_meta.jsonl" if bool(args.write_meta_jsonl) else None,
        },
        "pair_scan": {k: int(v) for k, v in pair_scan.items()},
        "counts": {
            "pairs_loaded": int(len(pairs)),
            "pairs_kept": int(len(kept_pairs)),
            "docs": int(w_docs.count),
            "queries": int(w_q.count),
            "skip": {k: int(v) for k, v in skip.items()},
        },
    }

    (out_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[DONE] MTC-ANN member->prototype retrieval pack built.")
    print(json.dumps(meta["counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
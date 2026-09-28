# preproc/preproc_gttm_benchmark.py
from __future__ import annotations

import argparse
import json
import math
import shutil
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm

from main.vocab_utils import load_vocab_info

from .gttm_preproc.mudep_compat import (
    load_score_musicxml,
    get_nra,
    get_dependency_arcs,
    shift_dep_arcs_add1,
)

# keep consistent with SimpleMono pos grid
try:
    from .simplemono_preproc.constants import POS_RESOLUTION
except Exception:
    POS_RESOLUTION = 12


# -------------------------
# Output config
# -------------------------
@dataclass(frozen=True)
class CutConfig:
    rho_min: float = 1.0 / 3.0
    rho_max: float = 1.0
    n_cuts: int = 21              # number of rho points for precomputed gold_cut_masks
    write_cut_masks: bool = True  # can turn off if you don't want it

    def cut_rhos(self) -> np.ndarray:
        return np.linspace(self.rho_min, self.rho_max, int(self.n_cuts), dtype=np.float32)


# -------------------------
# SimpleMono local encoder (pos -> local ids)
# -------------------------
@dataclass
class SimpleMonoLocalEncoder:
    pad_id: int
    bos_id: int
    eos_id: int
    special_n: int

    # duration
    duration_pos_to_code: np.ndarray   # [pos] -> code
    duration_min_code: int = 1         # IMPORTANT: match main.quantization.DurationQuantizer default

    # dt (signed)
    deltatime_code_offset: int = 96
    deltatime_pos_to_code: np.ndarray = None  # [pos + offset] -> signed code in [-dt_samples .. dt_samples-1]

    def encode_pitch_local(self, midi_pitch: int) -> int:
        midi_pitch = int(midi_pitch)
        midi_pitch = max(0, min(127, midi_pitch))
        return int(self.special_n + midi_pitch)

    def encode_duration_local_from_pos(self, dur_pos: int) -> int:
        dur_pos = int(dur_pos)
        if dur_pos < 0:
            dur_pos = 0

        pos_max = int(self.duration_pos_to_code.shape[0] - 1)
        dur_pos = max(0, min(dur_pos, pos_max))

        code = int(self.duration_pos_to_code[dur_pos])
        # clamp to [min_code, max_code]
        max_code = int(np.max(self.duration_pos_to_code))
        code = max(int(self.duration_min_code), min(code, max_code))
        return int(self.special_n + code)

    def encode_dt_local_from_pos(self, dt_pos_signed: int) -> int:
        dt_pos_signed = int(dt_pos_signed)

        dt_samples = int(self.deltatime_code_offset)
        # supported signed pos range: [-dt_samples .. dt_samples-1]
        dt_pos_signed = max(-dt_samples, min(dt_pos_signed, dt_samples - 1))
        idx = dt_pos_signed + dt_samples
        idx = max(0, min(idx, 2 * dt_samples - 1))

        code = int(self.deltatime_pos_to_code[idx])  # signed code
        code = max(-dt_samples, min(code, dt_samples - 1))

        if code >= 0:
            return int(self.special_n + code)

        mag = -code
        mag = max(1, min(mag, dt_samples))
        # negative codes are stored after positive codes in local-id space:
        # [special_n + dt_samples ... special_n + 2*dt_samples - 1] -> -1 .. -dt_samples
        return int((self.special_n + dt_samples) + (mag - 1))


# -------------------------
# IO utils
# -------------------------
def _ensure_empty_dir(p: Path, overwrite: bool):
    if p.exists():
        if not overwrite:
            # allow empty existing dir
            if any(p.iterdir()):
                raise FileExistsError(f"Output dir not empty: {p} (use --overwrite)")
        else:
            shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _find_unique_file(folder: Path, prefix: str, suffix: str = "") -> Path:
    cands = []
    for f in folder.iterdir():
        if not f.is_file():
            continue
        name = f.name
        if not name.startswith(prefix):
            continue
        if suffix and (not name.lower().endswith(suffix.lower())):
            continue
        cands.append(f)
    if len(cands) != 1:
        raise FileNotFoundError(f"Expect exactly 1 file in {folder} starting with '{prefix}' and ending with '{suffix}', got {len(cands)}")
    return cands[0]


def _list_piece_dirs(input_dir: Path) -> List[Path]:
    dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    dirs.sort(key=lambda x: x.name)
    return dirs


def _ids_as_unicode(arr: np.ndarray) -> np.ndarray:
    # robust for bytes ('S'), unicode ('U'), object
    try:
        return arr.astype("U")
    except Exception:
        return np.asarray([str(x) for x in arr], dtype="U")


# -------------------------
# Core: depth from dep arcs
# -------------------------
def compute_note_depth_from_dep_arcs(
    dep_arcs_shifted: np.ndarray,
    note_node_ids: np.ndarray,
) -> Dict[int, int]:
    """
    dep_arcs_shifted: [E,2], ROOT is node 0, other nodes are nra_idx+1
    note_node_ids: [N_note] subset of nodes that are notes (no rests)

    Returns
    -------
    depth_map: dict[node_id] = depth (root-note depth=0)
    """
    dep_arcs_shifted = np.asarray(dep_arcs_shifted, dtype=np.int32)
    note_set = set(int(x) for x in np.asarray(note_node_ids, dtype=np.int32).tolist())

    children: Dict[int, List[int]] = {}
    for h, d in dep_arcs_shifted.tolist():
        h = int(h)
        d = int(d)
        if h == 0 and d == 0:
            # ROOT self-loop
            continue
        if d == 0:
            continue
        # dep must be a note node
        if d not in note_set:
            # should not happen; ignore for safety
            continue
        if h != 0 and h not in note_set:
            # should not happen
            continue
        children.setdefault(h, []).append(d)

    # BFS from ROOT (depth[0]=-1 => root-note depth=0)
    depth: Dict[int, int] = {0: -1}
    q: List[int] = [0]
    while q:
        u = q.pop(0)
        for v in children.get(u, []):
            if v in depth:
                continue
            depth[v] = depth[u] + 1
            q.append(v)

    # sanity: all notes reached?
    missing = [n for n in note_set if n not in depth]
    if missing:
        raise ValueError(f"Dependency tree depth missing nodes: {missing[:10]} (total_missing={len(missing)})")

    # drop ROOT
    depth.pop(0, None)
    return depth


# -------------------------
# Process one piece folder
# -------------------------
@dataclass
class PieceResult:
    piece_id: str
    score_file: str
    ts_file: str
    ts_num: int
    ts_den: int

    x: np.ndarray              # [L,3] int16 local
    len_x: int
    note_oldidx: np.ndarray    # [L] int32
    gold_depth: np.ndarray     # [L] int16
    gold_score: np.ndarray     # [L] float32
    gold_cut_masks: Optional[np.ndarray]  # [C,L] uint8 or None

    n_notes_total: int
    n_notes_kept: int
    truncated: bool


def _extract_time_signature_mode(score, nra: np.ndarray) -> Tuple[int, int]:
    """
    Lightweight TS extraction for stratified split.
    We choose the most frequent (numerator, denominator) pair across nra onsets.
    """
    part = score.parts[0]
    ts_map = np.asarray(part.time_signature_map(nra["onset_div"]))
    if ts_map.ndim < 2 or ts_map.shape[1] < 2:
        # fallback
        return 4, 4

    num = np.asarray(ts_map[:, 0]).astype(int)
    den = np.asarray(ts_map[:, 1]).astype(int)

    pairs = np.stack([num, den], axis=1)
    # count mode
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    best = uniq[int(np.argmax(counts))]
    return int(best[0]), int(best[1])


def process_piece(
    piece_dir: Path,
    *,
    encoder: SimpleMonoLocalEncoder,
    L: int,
    cut_cfg: CutConfig,
    score_prefix: str = "score",
    ts_prefix: str = "TS",
) -> PieceResult:
    score_file = _find_unique_file(piece_dir, prefix=score_prefix, suffix=".xml")
    ts_file = _find_unique_file(piece_dir, prefix=ts_prefix, suffix=".xml")

    score = load_score_musicxml(score_file)
    nra = get_nra(score)
    # remove grace notes (MuDeP)
    nra = nra[nra["duration_div"] != 0]

    ids_u = _ids_as_unicode(nra["id"])
    is_rest = np.char.startswith(ids_u, "r")
    note_indices_all = np.where(~is_rest)[0].astype(np.int32)
    if note_indices_all.size == 0:
        raise ValueError("No notes after filtering (all rests?)")

    # dep arcs in tied nra indexing (with ROOT=-1)
    dep_list, _gttm_style = get_dependency_arcs(ts_file, score, nra_tied=nra)
    dep_arcs = shift_dep_arcs_add1(dep_list)  # ROOT=0, node = nra_idx+1

    # compute depth on note nodes only
    note_node_ids_all = note_indices_all + 1
    depth_map = compute_note_depth_from_dep_arcs(dep_arcs, note_node_ids_all)

    # time signature (for stratified split)
    ts_num, ts_den = _extract_time_signature_mode(score, nra)

    # convert to note-only SimpleMono sequence
    part = score.parts[0]
    onset_q = np.asarray(part.quarter_map(nra["onset_div"]), dtype=np.float64).reshape(-1)
    off_div = nra["onset_div"] + nra["duration_div"]
    offset_q = np.asarray(part.quarter_map(off_div), dtype=np.float64).reshape(-1)
    dur_q = offset_q - onset_q

    onset_pos = np.rint(onset_q * float(POS_RESOLUTION)).astype(np.int64)
    dur_pos = np.rint(dur_q * float(POS_RESOLUTION)).astype(np.int64)
    dur_pos = np.maximum(dur_pos, 1)  # avoid 0

    # note-only arrays in MuDeP order
    note_on = onset_pos[note_indices_all]
    note_dur = dur_pos[note_indices_all]
    note_pitch = np.asarray(nra["pitch"][note_indices_all], dtype=np.int64)

    # dt between notes (note-only)
    n_notes_total = int(note_indices_all.size)
    dt_pos = np.zeros((n_notes_total,), dtype=np.int64)
    if n_notes_total >= 2:
        end_pos = note_on + note_dur
        dt_pos[:-1] = note_on[1:] - end_pos[:-1]
        dt_pos[-1] = 0

    # BOS dt = first note onset
    bos_start_pos = int(note_on[0])

    # pack to fixed length
    max_notes_kept = max(0, int(L) - 2)
    n_notes_kept = min(n_notes_total, max_notes_kept)
    truncated = (n_notes_total > n_notes_kept)

    x = np.full((L, 3), fill_value=int(encoder.pad_id), dtype=np.int16)
    note_oldidx = np.full((L,), fill_value=-1, dtype=np.int32)
    gold_depth = np.full((L,), fill_value=-1, dtype=np.int16)
    gold_score = np.zeros((L,), dtype=np.float32)

    # BOS
    x[0, 0] = int(encoder.bos_id)
    x[0, 1] = int(encoder.bos_id)
    x[0, 2] = int(encoder.encode_dt_local_from_pos(bos_start_pos))

    # notes
    for j in range(n_notes_kept):
        tok_pos = j + 1
        nra_idx = int(note_indices_all[j])   # MuDeP indexing (in filtered nra)
        node_id = nra_idx + 1                # shifted node id

        pitch_l = encoder.encode_pitch_local(int(note_pitch[j]))
        dur_l = encoder.encode_duration_local_from_pos(int(note_dur[j]))
        dt_l = encoder.encode_dt_local_from_pos(int(dt_pos[j] if j < n_notes_total else 0))

        x[tok_pos, 0] = int(pitch_l)
        x[tok_pos, 1] = int(dur_l)
        x[tok_pos, 2] = int(dt_l)

        note_oldidx[tok_pos] = int(nra_idx)

        d = int(depth_map[int(node_id)])
        gold_depth[tok_pos] = int(d)
        gold_score[tok_pos] = float(math.exp(-float(d)))

    # EOS
    eos_pos = n_notes_kept + 1
    if eos_pos >= L:
        raise ValueError(f"Not enough space for EOS: L={L}, n_notes_kept={n_notes_kept}")
    x[eos_pos, :] = int(encoder.eos_id)
    len_x = eos_pos + 1

    # precompute cut masks (optional)
    gold_cut_masks = None
    if cut_cfg.write_cut_masks:
        cut_rhos = cut_cfg.cut_rhos()
        C = int(cut_rhos.shape[0])
        gold_cut_masks = np.zeros((C, L), dtype=np.uint8)

        # note positions in token space
        note_positions = np.arange(1, 1 + n_notes_kept, dtype=np.int32)
        if note_positions.size > 0:
            depths = gold_depth[note_positions].astype(np.int32)
            # stable sort by (depth, position)
            order = np.lexsort((note_positions, depths))
            pos_sorted = note_positions[order]

            Nn = int(note_positions.size)
            for ci, rho in enumerate(cut_rhos.tolist()):
                k = int(math.ceil(float(rho) * float(Nn)))
                k = max(1, min(k, Nn))
                sel = pos_sorted[:k]
                gold_cut_masks[ci, sel] = 1

    return PieceResult(
        piece_id=piece_dir.name,
        score_file=str(score_file),
        ts_file=str(ts_file),
        ts_num=int(ts_num),
        ts_den=int(ts_den),
        x=x,
        len_x=int(len_x),
        note_oldidx=note_oldidx,
        gold_depth=gold_depth,
        gold_score=gold_score,
        gold_cut_masks=gold_cut_masks,
        n_notes_total=int(n_notes_total),
        n_notes_kept=int(n_notes_kept),
        truncated=bool(truncated),
    )


# -------------------------
# Split util (no sklearn dependency)
# -------------------------
def stratified_train_test_split(
    keys: List[str],
    labels: List[int],
    *,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if len(keys) != len(labels):
        raise ValueError("keys and labels length mismatch")

    rng = np.random.default_rng(int(seed))

    by_lab: Dict[int, List[str]] = {}
    for k, y in zip(keys, labels):
        by_lab.setdefault(int(y), []).append(k)

    train: List[str] = []
    test: List[str] = []

    for y, ks in sorted(by_lab.items(), key=lambda kv: kv[0]):
        ks = list(ks)
        rng.shuffle(ks)
        nt = int(round(len(ks) * float(test_ratio)))
        nt = max(1, min(nt, len(ks) - 1)) if len(ks) >= 2 else 0
        test.extend(ks[:nt])
        train.extend(ks[nt:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True, help="GTTM root dir containing 300 piece folders")
    ap.add_argument("--vocab_pkl", type=str, required=True, help="SimpleMono.pkl (for quant tables + special ids)")
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--max_events", type=int, default=514, help="Fixed sequence length L (must match model max_seq_len)")
    ap.add_argument("--seed", type=int, default=1234)

    # split
    ap.add_argument("--test_ratio", type=float, default=0.10)
    ap.add_argument("--stratify_by_ts_num", action="store_true", help="Stratify split by time signature numerator")

    # cut config
    ap.add_argument("--rho_min", type=float, default=1.0 / 3.0)
    ap.add_argument("--n_cuts", type=int, default=21)
    ap.add_argument("--write_cut_masks", action="store_true", help="If set, write gold_cut_masks.npy")

    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found: {input_dir}")

    L = int(args.max_events)
    if L < 4:
        raise ValueError("max_events too small (need BOS + >=1 note + EOS + maybe PAD).")

    _ensure_empty_dir(out_root, overwrite=bool(args.overwrite))

    vocab = load_vocab_info(str(args.vocab_pkl))

    enc = SimpleMonoLocalEncoder(
        pad_id=int(vocab.pad_id),
        bos_id=int(vocab.bos_id),
        eos_id=int(vocab.eos_id),
        special_n=int(vocab.special_n),
        duration_pos_to_code=np.asarray(vocab.duration_pos_to_code, dtype=np.int32),
        duration_min_code=1,  # IMPORTANT: match main.quantization default
        deltatime_code_offset=int(vocab.deltatime_code_offset),
        deltatime_pos_to_code=np.asarray(vocab.deltatime_pos_to_code, dtype=np.int32),
    )

    cut_cfg = CutConfig(
        rho_min=float(args.rho_min),
        rho_max=1.0,
        n_cuts=int(args.n_cuts),
        write_cut_masks=bool(args.write_cut_masks),
    )

    if not (0.0 < cut_cfg.rho_min <= 1.0):
        raise ValueError(f"rho_min must be in (0,1], got {cut_cfg.rho_min}")

    piece_dirs = _list_piece_dirs(input_dir)
    print(f"[GTTM] found piece dirs: {len(piece_dirs)}")

    errors_path = out_root / "errors.jsonl"
    valid_path = out_root / "valid_pieces.json"

    results: Dict[str, PieceResult] = {}
    errors_f = errors_path.open("w", encoding="utf-8")

    for pd in tqdm(piece_dirs, desc="[Parse GTTM]", dynamic_ncols=True, smoothing=0.0):
        try:
            r = process_piece(pd, encoder=enc, L=L, cut_cfg=cut_cfg)
            results[r.piece_id] = r
        except Exception as e:
            err = {
                "piece_id": pd.name,
                "piece_dir": str(pd),
                "error": repr(e),
                "traceback": traceback.format_exc(limit=50),
            }
            errors_f.write(json.dumps(err, ensure_ascii=False) + "\n")

    errors_f.close()

    valid_piece_ids = sorted(results.keys())
    valid_path.write_text(json.dumps(valid_piece_ids, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[GTTM] valid pieces: {len(valid_piece_ids)} / {len(piece_dirs)}")
    print(f"[GTTM] errors saved to: {errors_path}")

    # -------------------------
    # split (single split, solidified)
    # -------------------------
    test_ratio = float(args.test_ratio)
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")

    if len(valid_piece_ids) < 4:
        raise ValueError("Too few valid pieces to split.")

    if bool(args.stratify_by_ts_num):
        labels = [int(results[k].ts_num) for k in valid_piece_ids]
        train_ids, test_ids = stratified_train_test_split(valid_piece_ids, labels, test_ratio=test_ratio, seed=int(args.seed))
    else:
        rng = np.random.default_rng(int(args.seed))
        ids = valid_piece_ids[:]
        rng.shuffle(ids)
        n_test = int(round(len(ids) * test_ratio))
        n_test = max(1, min(n_test, len(ids) - 1))
        test_ids = sorted(ids[:n_test])
        train_ids = sorted(ids[n_test:])

    split = {"train": train_ids, "test": test_ids}
    (out_root / "split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Split] train={len(train_ids)} test={len(test_ids)} (seed={args.seed}, stratify={bool(args.stratify_by_ts_num)})")

    # -------------------------
    # write split arrays
    # -------------------------
    def _write_split(split_name: str, ids: List[str]):
        out_dir = out_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        N = len(ids)
        x = np.zeros((N, L, 3), dtype=np.int16)
        len_x = np.zeros((N,), dtype=np.int32)
        note_oldidx = np.zeros((N, L), dtype=np.int32)
        gold_depth = np.zeros((N, L), dtype=np.int16)
        gold_score = np.zeros((N, L), dtype=np.float32)

        cut_rhos = cut_cfg.cut_rhos()
        C = int(cut_rhos.shape[0])
        gold_cut_masks = None
        if cut_cfg.write_cut_masks:
            gold_cut_masks = np.zeros((N, C, L), dtype=np.uint8)

        pieces_f = (out_dir / "pieces.jsonl").open("w", encoding="utf-8")

        trunc_n = 0
        notes_total = 0
        notes_kept = 0

        for i, pid in enumerate(ids):
            r = results[pid]
            x[i] = r.x
            len_x[i] = int(r.len_x)
            note_oldidx[i] = r.note_oldidx
            gold_depth[i] = r.gold_depth
            gold_score[i] = r.gold_score
            if gold_cut_masks is not None:
                assert r.gold_cut_masks is not None
                gold_cut_masks[i] = r.gold_cut_masks

            trunc_n += int(r.truncated)
            notes_total += int(r.n_notes_total)
            notes_kept += int(r.n_notes_kept)

            pieces_f.write(json.dumps({
                "row_idx": i,
                "piece_id": r.piece_id,
                "score_file": r.score_file,
                "ts_file": r.ts_file,
                "time_signature": [int(r.ts_num), int(r.ts_den)],
                "n_notes_total": int(r.n_notes_total),
                "n_notes_kept": int(r.n_notes_kept),
                "len_x": int(r.len_x),
                "truncated": bool(r.truncated),
            }, ensure_ascii=False) + "\n")

        pieces_f.close()

        np.save(out_dir / "x.npy", x)
        np.save(out_dir / "len_x.npy", len_x)
        np.save(out_dir / "note_oldidx.npy", note_oldidx)
        np.save(out_dir / "gold_depth.npy", gold_depth)
        np.save(out_dir / "gold_score.npy", gold_score)
        if gold_cut_masks is not None:
            np.save(out_dir / "gold_cut_masks.npy", gold_cut_masks)

        meta = {
            "version": "gttm_benchmark_v1",
            "split": split_name,
            "N": int(N),
            "L": int(L),
            "seed": int(args.seed),
            "pos_resolution": int(POS_RESOLUTION),
            "rho_min": float(cut_cfg.rho_min),
            "cut_rhos": [float(x) for x in cut_rhos.tolist()],
            "write_cut_masks": bool(cut_cfg.write_cut_masks),
            "stats": {
                "truncated_pieces": int(trunc_n),
                "avg_notes_total": float(notes_total / max(1, N)),
                "avg_notes_kept": float(notes_kept / max(1, N)),
            }
        }
        (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Write] {split_name}: N={N} truncated={trunc_n} avg_notes={meta['stats']['avg_notes_kept']:.2f}")

    _write_split("train", train_ids)
    _write_split("test", test_ids)

    # root meta
    root_meta = {
        "version": "gttm_benchmark_v1",
        "input_dir": str(input_dir),
        "vocab_pkl": str(args.vocab_pkl),
        "output_dir": str(out_root),
        "valid_pieces": len(valid_piece_ids),
        "total_piece_dirs": len(piece_dirs),
        "split_sizes": {"train": len(train_ids), "test": len(test_ids)},
        "pos_resolution": int(POS_RESOLUTION),
        "max_events": int(L),
        "seed": int(args.seed),
        "test_ratio": float(test_ratio),
        "stratify_by_ts_num": bool(args.stratify_by_ts_num),
        "cut_config": asdict(cut_cfg),
    }
    (out_root / "meta.json").write_text(json.dumps(root_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Done] GTTM benchmark built.")
    print(f"  output: {out_root}")


if __name__ == "__main__":
    main()

# python -m preproc.preproc_gttm_benchmark --input_dir ".\preproc\input_data_external\gttm_from_mudep_repo" --vocab_pkl ".\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\SimpleMono.pkl" --output_dir "./preproc/output/gttm_bench_v1.2" --max_events 514 --seed 1234 --test_ratio 0.10 --stratify_by_ts_num --rho_min 0.3333333 --n_cuts 21 --write_cut_masks
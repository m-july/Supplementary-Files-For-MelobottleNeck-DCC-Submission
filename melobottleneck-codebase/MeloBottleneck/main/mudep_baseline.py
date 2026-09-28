# main/mudep_baseline.py
from __future__ import annotations

import json
import math
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Piece meta IO
# -------------------------
@dataclass(frozen=True)
class GTTMPieceMeta:
    row_idx: int
    piece_id: str
    score_file: str = ""
    ts_file: str = ""
    time_signature: Optional[Tuple[int, int]] = None


def load_pieces_jsonl(path: str | Path) -> List[GTTMPieceMeta]:
    path = Path(path)
    metas: List[GTTMPieceMeta] = []
    with path.open("r", encoding="utf-8") as f:
        for li, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            row_idx = int(d.get("row_idx", li))
            piece_id = str(d["piece_id"])
            score_file = str(d.get("score_file", ""))
            ts_file = str(d.get("ts_file", ""))
            ts = d.get("time_signature", None)
            ts_tup = (int(ts[0]), int(ts[1])) if isinstance(ts, (list, tuple)) and len(ts) >= 2 else None
            metas.append(GTTMPieceMeta(
                row_idx=row_idx,
                piece_id=piece_id,
                score_file=score_file,
                ts_file=ts_file,
                time_signature=ts_tup,
            ))
    metas.sort(key=lambda m: m.row_idx)
    return metas


def _find_unique_file(folder: Path, prefix: str, suffix: str = ".xml") -> Path:
    cands = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if not p.name.startswith(prefix):
            continue
        if suffix and (not p.name.lower().endswith(suffix.lower())):
            continue
        cands.append(p)
    if len(cands) != 1:
        raise FileNotFoundError(f"Expect exactly 1 file under {folder} starting with '{prefix}', got {len(cands)}")
    return cands[0]


def _normalize_path_like(s: str) -> Path:
    """
    Robust path normalizer across Windows/Linux.
    If the path doesn't exist, try replacing backslashes with slashes.
    """
    p = Path(s)
    if p.exists():
        return p
    if "\\" in s:
        p2 = Path(s.replace("\\", "/"))
        if p2.exists():
            return p2
    if "/" in s and os.sep == "\\":
        p3 = Path(s.replace("/", "\\"))
        if p3.exists():
            return p3
    return p


def resolve_score_ts_paths(
    meta: GTTMPieceMeta,
    *,
    gttm_raw_dir: Optional[str | Path] = None,
    score_prefix: str = "score",
    ts_prefix: str = "TS",
) -> Tuple[Path, Path]:
    """
    Prefer resolving via gttm_raw_dir/piece_id for cross-platform stability.
    Fallback to meta.score_file/meta.ts_file if gttm_raw_dir is None.
    """
    if gttm_raw_dir is not None:
        piece_dir = Path(gttm_raw_dir) / meta.piece_id
        if not piece_dir.is_dir():
            raise FileNotFoundError(f"Piece dir not found: {piece_dir}")
        score_p = _find_unique_file(piece_dir, prefix=score_prefix, suffix=".xml")
        ts_p = _find_unique_file(piece_dir, prefix=ts_prefix, suffix=".xml")
        return score_p, ts_p

    if not meta.score_file or not meta.ts_file:
        raise ValueError("gttm_raw_dir is None but meta.score_file/meta.ts_file is empty.")
    return _normalize_path_like(meta.score_file), _normalize_path_like(meta.ts_file)


# -------------------------
# MuDeP feature / arc building
# -------------------------
def _import_mudep_modules():
    """
    Lazy imports, so that this file can still be imported even if MuDeP deps are missing.
    """
    from preproc.gttm_preproc.mudep_compat import (
        load_score_musicxml,
        get_nra,
        get_dependency_arcs,
        shift_dep_arcs_add1,
    )
    import module_external.musicparser.data_loading as mudep_dl

    return load_score_musicxml, get_nra, get_dependency_arcs, shift_dep_arcs_add1, mudep_dl


def build_mudep_note_features(score_file: Path) -> Tuple[torch.Tensor, Any, Any]:
    """
    Returns:
      note_features: torch.LongTensor [N,4] (pitch, is_rest, duration_idx, metrical_strength)
      score: partitura score
      nra: tied note+rest array (after grace removal), length N
    """
    load_score_musicxml, get_nra, _, _, mudep_dl = _import_mudep_modules()

    score = load_score_musicxml(score_file)
    nra = get_nra(score)
    # remove grace notes (MuDeP convention)
    nra = nra[nra["duration_div"] != 0]

    feats_np, _ts = mudep_dl.get_note_features(score, nra)
    feats = torch.as_tensor(feats_np, dtype=torch.long)
    if feats.ndim != 2 or feats.size(1) < 4:
        raise ValueError(f"Bad note_features shape: {tuple(feats.shape)}")
    return feats, score, nra


def build_potential_arcs_from_features(note_features: torch.Tensor) -> Tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Build pot_arcs in MuDeP shifted indexing:
      node 0 = ROOT
      node i (1..N) = nra_idx = i-1

    Rules (clean & robust):
      - include ROOT self-loop (0,0)
      - no other self-loops
      - dependent cannot be ROOT
      - exclude arcs whose head or dependent is a REST (ROOT is not REST)
    """
    if note_features.ndim != 2:
        raise ValueError("note_features must be [N,F]")

    N = int(note_features.size(0))

    # rest flag: column 1 in MuDeP features
    is_rest_event = (note_features[:, 1] > 0)  # [N] bool-ish
    is_rest_node = torch.zeros((N + 1,), dtype=torch.bool)
    is_rest_node[1:] = is_rest_event.to(torch.bool)

    idx = torch.arange(N + 1, dtype=torch.long)
    pot = torch.cartesian_prod(idx, idx)  # [M,2]
    # remove self loops
    pot = pot[pot[:, 0] != pot[:, 1]]
    # dep != ROOT
    pot = pot[pot[:, 1] != 0]

    # filter rest endpoints
    head_rest = is_rest_node[pot[:, 0]]
    dep_rest = is_rest_node[pot[:, 1]]
    pot = pot[~(head_rest | dep_rest)]

    # add ROOT self-loop at the beginning
    pot = torch.cat([torch.tensor([[0, 0]], dtype=torch.long), pot], dim=0)
    return pot, is_rest_node


def build_gold_dep_arcs_shifted(ts_file: Path, score: Any, nra: Any) -> torch.LongTensor:
    """
    Returns dep_arcs_shifted: torch.LongTensor [E,2], with ROOT=0, node=(nra_idx+1).
    Includes (0,0) root self-loop.
    """
    _, _, get_dependency_arcs, shift_dep_arcs_add1, _ = _import_mudep_modules()

    dep_list, _gttm_style = get_dependency_arcs(ts_file, score, nra_tied=nra)
    dep_arcs = shift_dep_arcs_add1(dep_list)  # np.ndarray int32, ROOT->0
    dep_arcs_t = torch.as_tensor(dep_arcs, dtype=torch.long)
    if dep_arcs_t.ndim != 2 or dep_arcs_t.size(1) != 2:
        raise ValueError(f"Bad dep_arcs shape: {tuple(dep_arcs_t.shape)}")
    return dep_arcs_t


def build_truth_mask_and_head_seq(
    *,
    dep_arcs_shifted: torch.LongTensor,  # [E,2]
    pot_arcs_shifted: torch.LongTensor,  # [M,2]
    num_events: int,                     # N
) -> Tuple[torch.BoolTensor, torch.Tensor]:
    """
    truth_mask: [M] bool
    head_seq:   [N+1] float tensor (MuDeP style; rests are -1)
    """
    _, _, _, _, mudep_dl = _import_mudep_modules()

    truth_mask = mudep_dl.get_edges_mask(dep_arcs_shifted, pot_arcs_shifted, transpose=False, check_strict_subset=True)
    head_seq = mudep_dl.get_head_seq(dep_arcs_shifted, num_notes=int(num_events))
    return truth_mask.bool(), head_seq


# -------------------------
# Dataset for training MuDeP on *your fixed split*
# -------------------------
@dataclass
class MuDePSupervisedItem:
    note_features: torch.LongTensor   # [N,4]
    truth_mask: torch.BoolTensor      # [M]
    pot_arcs: torch.LongTensor        # [M,2]
    head_seq: torch.Tensor            # [N+1] (float with -1 for rests)
    piece_id: str


class MuDePGTTMSupervisedDataset(Dataset):
    """
    Cache-all-in-memory dataset (like MuDeP original TSDataset).
    Works with batch_size=1 (recommended).
    """

    def __init__(
        self,
        metas: List[GTTMPieceMeta],
        *,
        gttm_raw_dir: Optional[str | Path],
        strict: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.items: List[MuDePSupervisedItem] = []
        self.errors: List[Dict[str, Any]] = []

        for meta in metas:
            try:
                score_p, ts_p = resolve_score_ts_paths(meta, gttm_raw_dir=gttm_raw_dir)
                feats, score, nra = build_mudep_note_features(score_p)
                pot_arcs, _is_rest_node = build_potential_arcs_from_features(feats)
                dep_arcs = build_gold_dep_arcs_shifted(ts_p, score, nra)

                truth_mask, head_seq = build_truth_mask_and_head_seq(
                    dep_arcs_shifted=dep_arcs,
                    pot_arcs_shifted=pot_arcs,
                    num_events=int(feats.size(0)),
                )

                self.items.append(MuDePSupervisedItem(
                    note_features=feats,
                    truth_mask=truth_mask,
                    pot_arcs=pot_arcs,
                    head_seq=head_seq,
                    piece_id=meta.piece_id,
                ))
            except Exception as e:
                err = {
                    "piece_id": meta.piece_id,
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=50),
                }
                self.errors.append(err)
                if verbose:
                    print(f"[MuDeP][Dataset] error piece={meta.piece_id}: {e}")

                if strict:
                    raise

        if verbose:
            print(f"[MuDeP][Dataset] built items={len(self.items)} errors={len(self.errors)}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        # match MuDeP's (note_seq, truth_mask, pot_arcs, head_seq)
        return it.note_features, it.truth_mask, it.pot_arcs, it.head_seq

    def get_positive_weight(self) -> float:
        """
        Same definition as MuDeP:
          mean_i ( len(truth_mask_i) / sum(truth_mask_i) )
        """
        vals: List[float] = []
        for it in self.items:
            pos = float(it.truth_mask.sum().item())
            tot = float(it.truth_mask.numel())
            if pos <= 0:
                continue
            vals.append(tot / pos)
        return float(np.mean(vals)) if vals else 1.0


# -------------------------
# Inference: head_seq -> depth -> token-aligned score
# -------------------------
def build_adj_logits_root_safe(
    pot_arcs: torch.LongTensor,       # [M,2]
    arc_logits: torch.Tensor,         # [M]
    num_events: int,                  # N (no root)
) -> torch.Tensor:
    """
    Safer than MuDeP's compute_adj_logits_root (which treats 0 as absent edge).
    Returns dense [N+1, N+1] with -inf for absent edges.
    """
    N = int(num_events)
    device = arc_logits.device
    adj = torch.full((N + 1, N + 1), float("-inf"), device=device, dtype=arc_logits.dtype)
    adj[pot_arcs[:, 0], pot_arcs[:, 1]] = arc_logits
    return adj


def head_seq_to_depth(
    head_seq: np.ndarray,            # [N+1]
    is_rest_node: np.ndarray,        # [N+1] bool
) -> np.ndarray:
    """
    Depth convention:
      depth(ROOT=0) = -1
      depth(root_note) = 0
    Rests are ignored and keep depth=-1.
    """
    head_seq = np.asarray(head_seq, dtype=np.int64)
    is_rest_node = np.asarray(is_rest_node, dtype=bool)
    if head_seq.ndim != 1:
        raise ValueError("head_seq must be 1d")
    if head_seq.shape != is_rest_node.shape:
        raise ValueError("head_seq/is_rest shape mismatch")

    N = int(head_seq.shape[0] - 1)
    depth = np.full((N + 1,), -1, dtype=np.int16)
    depth[0] = -1

    children: List[List[int]] = [[] for _ in range(N + 1)]
    for dep in range(1, N + 1):
        if is_rest_node[dep]:
            continue
        h = int(head_seq[dep])
        if h < 0:
            # disconnected (shouldn't happen for non-rest after postprocess), keep -1
            continue
        if is_rest_node[h]:
            raise ValueError(f"Invalid head: dep={dep} head={h} is REST")
        if h == dep:
            continue
        children[h].append(dep)

    # BFS from ROOT
    from collections import deque
    q = deque([0])
    while q:
        h = q.popleft()
        for dep in children[h]:
            if depth[dep] != -1:
                continue
            depth[dep] = depth[h] + 1
            q.append(dep)

    missing = [i for i in range(1, N + 1) if (not is_rest_node[i]) and depth[i] < 0]
    if missing:
        raise ValueError(f"Depth missing nodes (first10={missing[:10]}, total={len(missing)})")

    return depth


def depth_to_score(depth: np.ndarray, *, temp: float = 1.0) -> np.ndarray:
    depth = np.asarray(depth)
    out = np.zeros_like(depth, dtype=np.float32)
    m = depth >= 0
    out[m] = np.exp(-depth[m].astype(np.float32) / max(float(temp), 1e-6))
    return out


def align_node_depth_to_token_positions(
    note_oldidx_row: np.ndarray,     # [L], token_pos -> nra_idx (note-only positions >=0)
    node_depth: np.ndarray,          # [N+1], node_id = nra_idx+1
) -> np.ndarray:
    note_oldidx_row = np.asarray(note_oldidx_row, dtype=np.int64)
    node_depth = np.asarray(node_depth, dtype=np.int64)

    idx = note_oldidx_row + 1  # -1 -> 0 (ROOT)
    if idx.max() >= node_depth.shape[0]:
        raise ValueError(f"note_oldidx has out-of-range index: max(nra_idx+1)={idx.max()} but node_depth len={len(node_depth)}")

    tok_depth = node_depth[idx].astype(np.int16)
    # make non-note positions -1 explicitly
    tok_depth[note_oldidx_row < 0] = -1
    return tok_depth


def predict_mudep_scores_for_split(
    *,
    model,                              # ArcPredictionLightModel (loaded)
    bench_split_dir: str | Path,         # .../gttm_bench_v1/test
    metas: List[GTTMPieceMeta],
    gttm_raw_dir: Optional[str | Path],
    device: torch.device,
    out_dir: str | Path,
    postprocess_alg: str = "eisner",
    score_temp: float = 1.0,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Writes:
      out_dir/pred_depth.npy  [N,L] int16
      out_dir/pred_score.npy  [N,L] float32
      out_dir/errors.jsonl
      out_dir/meta.json
    """
    bench_split_dir = Path(bench_split_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    note_oldidx = np.load(bench_split_dir / "note_oldidx.npy", mmap_mode="r")  # [N,L]
    N, L = note_oldidx.shape

    if len(metas) != N:
        raise ValueError(f"metas length mismatch: metas={len(metas)} note_oldidx N={N}")

    pred_depth = np.full((N, L), -1, dtype=np.int16)
    pred_score = np.zeros((N, L), dtype=np.float32)

    errors_path = out_dir / "errors.jsonl"
    ef = errors_path.open("w", encoding="utf-8")

    model = model.to(device)
    model.eval()

    from preproc.gttm_preproc.mudep_compat import load_score_musicxml, get_nra
    import module_external.musicparser.data_loading as mudep_dl

    with torch.inference_mode():
        for i, meta in enumerate(metas):
            try:
                score_p, _ts_p = resolve_score_ts_paths(meta, gttm_raw_dir=gttm_raw_dir)
                # build features
                score = load_score_musicxml(score_p)
                nra = get_nra(score)
                nra = nra[nra["duration_div"] != 0]
                feats_np, _ = mudep_dl.get_note_features(score, nra)
                feats = torch.as_tensor(feats_np, dtype=torch.long)

                pot_arcs, is_rest_node = build_potential_arcs_from_features(feats)
                num_events = int(feats.size(0))  # N

                # move to device
                feats_d = feats.to(device)
                pot_d = pot_arcs.to(device)

                # forward
                arc_logits = model.module(feats_d, pot_d)  # [M]
                adj_logits_root = build_adj_logits_root_safe(pot_d, arc_logits, num_events=num_events)

                # postprocess -> head_seq_postp (len N+1)
                is_rest_node_t = is_rest_node.to(device)
                _adj_postp, _pred_arcs_postp, head_seq_postp = model.postprocess(
                    adj_logits_root,
                    num_events,
                    is_rest_node_t,
                    alg=postprocess_alg,
                )
                head_seq_postp = head_seq_postp.detach().cpu().numpy()

                depth_node = head_seq_to_depth(
                    head_seq=head_seq_postp,
                    is_rest_node=is_rest_node.cpu().numpy(),
                )
                score_node = depth_to_score(depth_node, temp=float(score_temp))

                # align to token positions using note_oldidx
                tok_depth = align_node_depth_to_token_positions(note_oldidx[i], depth_node)  # [L]
                tok_score = np.zeros((L,), dtype=np.float32)
                m = tok_depth >= 0
                tok_score[m] = np.exp(-tok_depth[m].astype(np.float32) / max(float(score_temp), 1e-6))

                pred_depth[i] = tok_depth
                pred_score[i] = tok_score

            except Exception as e:
                ef.write(json.dumps({
                    "row_idx": int(i),
                    "piece_id": meta.piece_id,
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=80),
                }, ensure_ascii=False) + "\n")
                if strict:
                    ef.close()
                    raise

    ef.close()

    np.save(out_dir / "pred_depth.npy", pred_depth)
    np.save(out_dir / "pred_score.npy", pred_score)

    meta_out = {
        "split_dir": str(bench_split_dir),
        "out_dir": str(out_dir),
        "N": int(N),
        "L": int(L),
        "postprocess_alg": str(postprocess_alg),
        "score_temp": float(score_temp),
        "gttm_raw_dir": (str(gttm_raw_dir) if gttm_raw_dir is not None else None),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_out
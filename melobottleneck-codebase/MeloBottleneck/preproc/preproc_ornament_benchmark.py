# preproc_ornament_benchmark.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
from tqdm import tqdm

from main.vocab_utils import load_vocab_info
from main.quantization import MusicQuantizationTables
from main.ornament import MusicOrnamenter, MusicOrnamentConfig, PI_INSERTED, PI_PAD

from preproc.dataset_stats import (
    DEFAULT_STATS_BPM,
    SeqExportStatsAccumulator,
    compute_span_pos_from_local_events,
    load_pos_resolution_from_vocab_pkl,
    write_json,
    aggregate_split_dirs_export_stats,
)


def _seq_len_including_eos(x_local: np.ndarray, *, pad_id: int, eos_id: Optional[int]) -> int:
    """
    x_local: [L,3]
    returns valid length (<=L), ending at EOS (inclusive) if EOS exists, else first PAD.
    """
    pitch = x_local[:, 0]
    L = int(pitch.shape[0])

    pad_idx = np.where(pitch == int(pad_id))[0]
    valid_len = int(pad_idx[0]) if pad_idx.size > 0 else L

    if eos_id is not None:
        eos_idx = np.where(pitch[:valid_len] == int(eos_id))[0]
        if eos_idx.size > 0:
            valid_len = int(eos_idx[0]) + 1
    return int(valid_len)


def _global_to_local(x_global: np.ndarray, vocab) -> np.ndarray:
    """
    x_global: [L,3] global ids (or already-local).
    Returns x_local: [L,3] local ids.

    Safety: if mapping gives -1 (unknown), fallback to original id.
    """
    x_global = np.asarray(x_global, dtype=np.int64)
    L, A = x_global.shape
    if A != 3:
        raise ValueError(f"Expected x shape [L,3], got {x_global.shape}")

    pitch_g = x_global[:, 0]
    dur_g = x_global[:, 1]
    dt_g = x_global[:, 2]

    pitch_l = vocab.global2local_pitch[pitch_g]
    dur_l = vocab.global2local_duration[dur_g]
    dt_l = vocab.global2local_dt[dt_g]

    pitch = np.where(pitch_l >= 0, pitch_l, pitch_g)
    dur = np.where(dur_l >= 0, dur_l, dur_g)
    dt = np.where(dt_l >= 0, dt_l, dt_g)

    x_local = np.stack([pitch, dur, dt], axis=-1)
    return x_local.astype(np.int32)


def _load_json_overrides(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise ValueError("ornament_json must be a JSON dict.")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npy", type=str, required=True, help="valid.npy or test.npy (base split)")
    ap.add_argument("--vocab_pkl", type=str, required=True, help="SimpleMono.pkl")
    ap.add_argument("--output_dir", type=str, required=True, help="output root dir")
    ap.add_argument("--split_name", type=str, required=True, help="valid|test|... used as subfolder name")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--rho_min", type=float, default=(1.0 / 3.0))
    ap.add_argument("--min_extra_tokens", type=int, default=1)

    ap.add_argument("--max_tries", type=int, default=8)
    ap.add_argument("--max_samples", type=int, default=0, help="0 => all samples")

    ap.add_argument("--ornament_json", type=str, default="", help="Optional: JSON overrides for MusicOrnamentConfig")
    ap.add_argument("--show_hist", action="store_true", help="Print rho histogram after build")

    args = ap.parse_args()

    in_path = Path(args.input_npy)
    out_root = Path(args.output_dir) / str(args.split_name)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[Input]  {in_path}")
    print(f"[Output] {out_root}")

    vocab = load_vocab_info(str(args.vocab_pkl))

    pos_resolution = load_pos_resolution_from_vocab_pkl(args.vocab_pkl, default=12)
    bpm = DEFAULT_STATS_BPM

    acc_x = SeqExportStatsAccumulator()
    acc_xorn = SeqExportStatsAccumulator()

    quant_tables = MusicQuantizationTables(
        special_n=vocab.special_n,
        duration_code_to_pos=vocab.duration_code_to_pos,
        duration_pos_to_code=vocab.duration_pos_to_code,
        deltatime_code_offset=vocab.deltatime_code_offset,
        deltatime_code_to_pos=vocab.deltatime_code_to_pos,
        deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    )

    # ---- benchmark ornament config (default strong + always apply) ----
    # NOTE: max_extra_tokens 可在 JSON 覆写；实际每条样本还会再用 augment(..., max_extra_tokens=extra_cap) 进一步限制
    orn_cfg = MusicOrnamentConfig(
        enable=True,
        p_apply=1.0,
        pad_id=vocab.pad_id,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        max_extra_tokens=256,
        quantization_tables=quant_tables,
    )

    # Optional JSON overrides (OOD benchmark etc.)
    if str(args.ornament_json).strip():
        overrides = _load_json_overrides(args.ornament_json)
        legal = set(MusicOrnamentConfig.__dataclass_fields__.keys())
        overrides = {k: v for k, v in overrides.items() if k in legal}
        overrides.pop("pad_id", None)
        overrides.pop("bos_id", None)
        overrides.pop("eos_id", None)
        overrides.pop("quantization_tables", None)
        orn_cfg = replace(orn_cfg, **overrides)

    ornamenter = MusicOrnamenter(orn_cfg)
    max_extra_tokens_cfg = int(orn_cfg.max_extra_tokens)

    base = np.load(in_path, mmap_mode="r", allow_pickle=True)
    if not isinstance(base, np.ndarray):
        raise ValueError("input_npy must be a numpy array.")
    if base.ndim != 3 or base.shape[-1] != 3:
        raise ValueError(f"input_npy must have shape [N,L,3], got {base.shape}")

    N_total, L, _ = base.shape
    N = int(N_total)
    if int(args.max_samples) > 0:
        N = min(N, int(args.max_samples))

    # ---- output arrays (memmap .npy) ----
    x_out = np.lib.format.open_memmap(out_root / "x.npy", mode="w+", dtype=np.int16, shape=(N, L, 3))
    x_orn_out = np.lib.format.open_memmap(out_root / "x_orn.npy", mode="w+", dtype=np.int16, shape=(N, L, 3))
    pi_out = np.lib.format.open_memmap(out_root / "pi.npy", mode="w+", dtype=np.int32, shape=(N, L))
    len_x_out = np.lib.format.open_memmap(out_root / "len_x.npy", mode="w+", dtype=np.int32, shape=(N,))
    len_x_orn_out = np.lib.format.open_memmap(out_root / "len_x_orn.npy", mode="w+", dtype=np.int32, shape=(N,))
    rho_out = np.lib.format.open_memmap(out_root / "rho.npy", mode="w+", dtype=np.float32, shape=(N,))

    rho_min = float(args.rho_min)
    if not (0.0 < rho_min <= 1.0):
        raise ValueError(f"rho_min must be in (0,1], got {rho_min}")
    min_extra = max(0, int(args.min_extra_tokens))
    max_tries = max(1, int(args.max_tries))

    fail_n = 0
    inserted_n_total = 0
    extra_total = 0

    for i in tqdm(range(N), desc=f"[Build OTB] {args.split_name}", dynamic_ncols=True, smoothing=0.0):
        # deterministic per-sample RNG
        rng = np.random.default_rng(int(args.seed) + i * 10007)

        x_global = np.asarray(base[i], dtype=np.int64)  # [L,3]
        x_local = _global_to_local(x_global, vocab)     # [L,3] local ids

        # original length
        len_x = _seq_len_including_eos(x_local, pad_id=vocab.pad_id, eos_id=vocab.eos_id)
        pad_budget = max(0, int(L) - int(len_x))

        # enforce rho_min by capping max extra tokens:
        #   rho = len_x / (len_x + extra) >= rho_min
        # => extra <= len_x*(1/rho_min - 1)
        extra_cap_rho = int(math.floor(float(len_x) * (1.0 / rho_min - 1.0) + 1e-9))
        extra_cap_rho = max(0, extra_cap_rho)

        extra_cap = min(max_extra_tokens_cfg, pad_budget, extra_cap_rho)

        best = None
        last = None
        for _ in range(max_tries):
            x_aug, pi = ornamenter.augment(x_local, rng=rng, max_extra_tokens=extra_cap)

            valid = (pi != int(PI_PAD))
            len_orn = int(valid.sum())
            extra = int(len_orn - len_x)
            rho = float(len_x) / float(max(1, len_orn))

            inserted_n = int((pi == int(PI_INSERTED)).sum())
            last = (x_aug, pi, len_orn, rho, inserted_n, extra)

            if (rho + 1e-9) >= rho_min and extra >= min_extra:
                best = last
                break

        if best is None:
            best = last
            fail_n += 1

        x_aug, pi, len_orn, rho, inserted_n, extra = best

        x_out[i] = x_local
        x_orn_out[i] = x_aug
        pi_out[i] = pi
        len_x_out[i] = int(len_x)
        len_x_orn_out[i] = int(len_orn)
        rho_out[i] = float(rho)

        inserted_n_total += inserted_n
        extra_total += extra

        n_notes_x = int(len_x) - 2
        n_notes_xorn = int(len_orn) - 2

        span_x = compute_span_pos_from_local_events(x_local, len_events=int(len_x), vocab=vocab)
        span_xorn = compute_span_pos_from_local_events(x_aug, len_events=int(len_orn), vocab=vocab)

        acc_x.add(len_events=int(len_x), n_notes=int(n_notes_x), span_pos=int(span_x))
        acc_xorn.add(len_events=int(len_orn), n_notes=int(n_notes_xorn), span_pos=int(span_xorn))

    # flush
    x_out.flush()
    x_orn_out.flush()
    pi_out.flush()
    len_x_out.flush()
    len_x_orn_out.flush()
    rho_out.flush()

    rho_arr = np.asarray(rho_out)
    meta = {
        "version": "otb_benchmark_v1",
        "input_npy": str(in_path),
        "split_name": str(args.split_name),
        "N": int(N),
        "L": int(L),
        "seed": int(args.seed),
        "rho_min": float(rho_min),
        "min_extra_tokens": int(min_extra),
        "max_tries": int(max_tries),
        "ornament_config": {k: v for k, v in asdict(orn_cfg).items() if k != "quantization_tables"},
        "stats": {
            "fail_n": int(fail_n),
            "rho_mean": float(np.mean(rho_arr)),
            "rho_min": float(np.min(rho_arr)),
            "rho_max": float(np.max(rho_arr)),
            "avg_extra_tokens": float(extra_total / max(1, N)),
            "avg_inserted_tokens": float(inserted_n_total / max(1, N)),
        },
    }

    with open(out_root / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    export_stats = {
        "schema": "simplemono_export_stats_v1",
        "dataset_kind": "synthetic_otb_benchmark",
        "split_name": str(args.split_name),
        "input_npy": str(in_path),
        "bpm": float(bpm),
        "pos_resolution": int(pos_resolution),
        "counts": {
            "sequences": int(N),
            # optional: try infer file count from metadata.jsonl if exists
        },
        "tracks": {
            "x": acc_x.to_dict(pos_resolution=pos_resolution, bpm=bpm),
            "x_orn": acc_xorn.to_dict(pos_resolution=pos_resolution, bpm=bpm),
        },
        "existing_meta_stats": meta.get("stats", {}),
    }

    write_json(out_root / "export_stats.json", export_stats)

    # root aggregate (output_dir contains multiple split subfolders)
    aggregate_split_dirs_export_stats(Path(args.output_dir))

    print("[Done] benchmark built.")
    print(json.dumps(meta["stats"], indent=2))

    if args.show_hist:
        bins = [1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0, 5.0 / 6.0, 1.0 + 1e-6]
        hist, edges = np.histogram(rho_arr, bins=bins)
        for c, lo, hi in zip(hist, edges[:-1], edges[1:]):
            print(f"  rho in [{lo:.3f},{hi:.3f}): {int(c)}")


if __name__ == "__main__":
    main()

# STEP 1 use preproc.preproc_pretrain

# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\skeletion_unsup_corpus_v260411" --output_dir ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file --num_workers 8 --seed 10101

# STEP 2 valid set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\valid.npy" --vocab_pkl ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name valid_ood --rho_min 0.3333333333 --seed 10101 --show_hist --ornament_json ".\preproc\ornament_ood.json"

# STEP 3 test set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\test.npy" --vocab_pkl ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed10101\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name test_ood --rho_min 0.3333333333 --seed 10101 --show_hist --ornament_json ".\preproc\ornament_ood.json"

# 20202

# STEP 1 use preproc.preproc_pretrain

# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\skeletion_unsup_corpus_v260411" --output_dir ".\preproc\output\seed20202\skeletion_unsup_corpus_v260411_with_ornamented_split" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file --num_workers 8 --seed 20202

# STEP 2 valid set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed20202\skeletion_unsup_corpus_v260411_with_ornamented_split\valid.npy" --vocab_pkl ".\preproc\output\seed20202\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed20202\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name valid_ood --rho_min 0.3333333333 --seed 20202 --show_hist --ornament_json ".\preproc\ornament_ood.json"

# STEP 3 test set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed20202\skeletion_unsup_corpus_v260411_with_ornamented_split\test.npy" --vocab_pkl ".\preproc\output\seed20202\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed20202\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name test_ood --rho_min 0.3333333333 --seed 20202 --show_hist --ornament_json ".\preproc\ornament_ood.json"

# 30303

# STEP 1 use preproc.preproc_pretrain

# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\skeletion_unsup_corpus_v260411" --output_dir ".\preproc\output\seed30303\skeletion_unsup_corpus_v260411_with_ornamented_split" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file --num_workers 8 --seed 30303

# STEP 2 valid set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed30303\skeletion_unsup_corpus_v260411_with_ornamented_split\valid.npy" --vocab_pkl ".\preproc\output\seed30303\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed30303\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name valid_ood --rho_min 0.3333333333 --seed 30303 --show_hist --ornament_json ".\preproc\ornament_ood.json"

# STEP 3 test set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed30303\skeletion_unsup_corpus_v260411_with_ornamented_split\test.npy" --vocab_pkl ".\preproc\output\seed30303\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed30303\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name test_ood --rho_min 0.3333333333 --seed 30303 --show_hist --ornament_json ".\preproc\ornament_ood.json"


# 40404

# STEP 1 use preproc.preproc_pretrain

# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\skeletion_unsup_corpus_v260411" --output_dir ".\preproc\output\seed40404\skeletion_unsup_corpus_v260411_with_ornamented_split" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file --num_workers 8 --seed 40404

# STEP 2 valid set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed40404\skeletion_unsup_corpus_v260411_with_ornamented_split\valid.npy" --vocab_pkl ".\preproc\output\seed40404\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed40404\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name valid_ood --rho_min 0.3333333333 --seed 40404 --show_hist --ornament_json ".\preproc\ornament_ood.json"

# STEP 3 test set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed40404\skeletion_unsup_corpus_v260411_with_ornamented_split\test.npy" --vocab_pkl ".\preproc\output\seed40404\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed40404\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name test_ood --rho_min 0.3333333333 --seed 40404 --show_hist --ornament_json ".\preproc\ornament_ood.json"


# 50505

# STEP 1 use preproc.preproc_pretrain

# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\skeletion_unsup_corpus_v260411" --output_dir ".\preproc\output\seed50505\skeletion_unsup_corpus_v260411_with_ornamented_split" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file --num_workers 8 --seed 50505

# STEP 2 valid set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed50505\skeletion_unsup_corpus_v260411_with_ornamented_split\valid.npy" --vocab_pkl ".\preproc\output\seed50505\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed50505\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name valid_ood --rho_min 0.3333333333 --seed 50505 --show_hist --ornament_json ".\preproc\ornament_ood.json"

# STEP 3 test set

# python -m preproc.preproc_ornament_benchmark --input_npy ".\preproc\output\seed50505\skeletion_unsup_corpus_v260411_with_ornamented_split\test.npy" --vocab_pkl ".\preproc\output\seed50505\skeletion_unsup_corpus_v260411_with_ornamented_split\SimpleMono.pkl" --output_dir ".\preproc\output\seed50505\skeletion_unsup_corpus_v260411_with_ornamented_split\otb_bench" --split_name test_ood --rho_min 0.3333333333 --seed 50505 --show_hist --ornament_json ".\preproc\ornament_ood.json"
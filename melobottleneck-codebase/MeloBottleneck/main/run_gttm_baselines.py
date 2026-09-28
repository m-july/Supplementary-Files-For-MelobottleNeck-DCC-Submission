from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

from .vocab_utils import load_vocab_info
from .evaluation import (
    make_gttm_benchmark_dataloader,
    GTTMBackboneEvalConfig,
    evaluate_gttm_backbone,
    build_pointer_adapter_from_model,
)

from .models.skeleton.baselines import (
    OTBBaselineModel,
    OTBRandomDownsampleCompressor,
    OTBUniformTimeDownsampleCompressor,
    OTBTopKDurationCompressor,
)


def _pick_device(dev: str) -> torch.device:
    dev = str(dev).strip().lower()
    if dev == "cuda":
        if not torch.cuda.is_available():
            print("[Warn] cuda requested but not available; fallback to cpu.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def _summarize(prefix: str, m: Dict[str, float]) -> str:
    def g(k: str) -> float:
        return float(m.get(prefix + k, float("nan")))

    return (
        f"f1_topk={g('f1_hard_topk'):.4f} | "
        f"ap={g('ap_soft'):.4f} | "
        f"ndcg@ref={g('ndcg_soft'):.4f} | "
        f"ndcg@full={g('ndcg_full'):.4f} | "
        f"cut_auc={g('cut_f1_auc_norm'):.4f} | "
        f"mean_ndcg_cut={g('mean_ndcg_cut'):.4f} | "
        f"spearman={g('spearman'):.4f} | "
        f"rho={g('rho'):.4f} | n={int(g('n'))}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_pkl", type=str, required=True)
    ap.add_argument("--gttm_split_dir", type=str, required=True, help=".../gttm_bench/test or train")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_batches", type=int, default=0)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--baselines", type=str, default="random,uniform_time,topk_duration")

    ap.add_argument("--eval_rho", type=float, default=2.0 / 3.0)
    ap.add_argument("--out_json", type=str, default="")
    ap.add_argument("--no_cut_curve", action="store_true")

    args = ap.parse_args()
    device = _pick_device(args.device)

    vocab = load_vocab_info(args.vocab_pkl)

    req = [s.strip() for s in str(args.baselines).split(",") if s.strip()]
    legal = {"random", "uniform_time", "topk_duration"}
    for r in req:
        if r not in legal:
            raise ValueError(f"Unknown baseline: {r}. Legal={sorted(list(legal))}")

    loader = make_gttm_benchmark_dataloader(
        args.gttm_split_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    eval_cfg = GTTMBackboneEvalConfig(
        eval_rho=float(args.eval_rho),
        force_z_len_rho=float(args.eval_rho),   # IMPORTANT: baselines need explicit length
        compute_cut_curve=(not args.no_cut_curve),
        max_batches=(None if int(args.max_batches) <= 0 else int(args.max_batches)),
        amp=False,
        show_progress=True,
    )

    compressors = {}
    if "random" in req:
        compressors["random"] = OTBRandomDownsampleCompressor(
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            special_n=vocab.special_n,
            duration_code_to_pos=vocab.duration_code_to_pos,
            deltatime_code_to_pos=vocab.deltatime_code_to_pos,
            deltatime_code_offset=vocab.deltatime_code_offset,
            seed=int(args.seed),
        )
    if "uniform_time" in req:
        compressors["uniform_time"] = OTBUniformTimeDownsampleCompressor(
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            special_n=vocab.special_n,
            duration_code_to_pos=vocab.duration_code_to_pos,
            deltatime_code_to_pos=vocab.deltatime_code_to_pos,
            deltatime_code_offset=vocab.deltatime_code_offset,
            seed=int(args.seed),
        )
    if "topk_duration" in req:
        compressors["topk_duration"] = OTBTopKDurationCompressor(
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            special_n=vocab.special_n,
            duration_code_to_pos=vocab.duration_code_to_pos,
            deltatime_code_to_pos=vocab.deltatime_code_to_pos,
            deltatime_code_offset=vocab.deltatime_code_offset,
            seed=int(args.seed),
        )

    all_metrics: Dict[str, float] = {}

    print(f"\n========== GTTM Split: {args.gttm_split_dir} ==========")
    for name, comp in compressors.items():
        model = OTBBaselineModel(
            compressor=comp,
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            special_n=vocab.special_n,
            tau=1.0,
        ).to(device)

        adapter = build_pointer_adapter_from_model(model)
        prefix = f"gttm/baseline/{name}/"
        m = evaluate_gttm_backbone(adapter, loader, device=device, cfg=eval_cfg, prefix=prefix)
        all_metrics.update(m)

        print(f"[{name}] " + _summarize(prefix, m))

    if str(args.out_json).strip():
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[Saved] {outp}")


if __name__ == "__main__":
    main()
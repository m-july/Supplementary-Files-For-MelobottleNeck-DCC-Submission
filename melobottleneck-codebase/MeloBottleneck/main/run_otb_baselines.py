# run_otb_baselines.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch

from .vocab_utils import load_vocab_info
from .evaluation import (
    make_ornament_benchmark_dataloader,
    OrnamentToBackboneEvalConfig,
    evaluate_ornament_to_backbone,
    build_pointer_adapter_from_model,   # NEW
)

from .models.skeleton.baselines import (
    OTBBaselineModel,
    OTBRandomDownsampleCompressor,
    OTBUniformTimeDownsampleCompressor,
    OTBTopKDurationCompressor,
    OTBAMRNoHarmonyCompressor,
)

from .quantization import MusicQuantizationTables, build_quantizers
from .evaluation.music_prior_proxy_eval import MusicPriorProxyEvalConfig, evaluate_music_prior_proxy


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
        f"f1_hard={g('f1_hard'):.4f} | "
        f"f1_topk={g('f1_hard_topk'):.4f} | "
        f"gap_f1={g('gap_f1'):.4f} | "
        f"ap_soft={g('ap_soft'):.4f} | "
        f"ndcg_soft={g('ndcg_soft'):.4f} | "
        f"ins_mass_soft={g('ins_mass_soft'):.4f} | "
        f"cut_f1_auc_norm={g('cut_f1_auc_norm'):.4f} | "
        f"mean_ndcg_cut={g('mean_ndcg_cut'):.4f} | "
        f"rho_mean={g('rho'):.4f} | n={int(g('n'))}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_pkl", type=str, required=True)

    ap.add_argument("--valid_bench_dir", type=str, default="")
    ap.add_argument("--test_bench_dir", type=str, default="")

    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max_batches", type=int, default=0, help="0 => all batches")

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--baselines", type=str, default="random,uniform_time,topk_duration,amr_no_harmony",
                    help="comma-separated: random,uniform_time,topk_duration,amr_no_harmony")

    ap.add_argument("--out_json", type=str, default="", help="optional path to save metrics json")
    ap.add_argument("--no_cut_curve", action="store_true")

    ap.add_argument("--proxy", action="store_true", help="compute music-prior proxy metrics")
    ap.add_argument("--pos_per_beat", type=int, default=12)
    ap.add_argument("--strong_period_beats", type=int, default=2)

    args = ap.parse_args()
    device = _pick_device(args.device)

    vocab = load_vocab_info(args.vocab_pkl)

    quant_tables = MusicQuantizationTables(
        special_n=vocab.special_n,
        duration_code_to_pos=vocab.duration_code_to_pos,
        duration_pos_to_code=vocab.duration_pos_to_code,
        deltatime_code_offset=vocab.deltatime_code_offset,
        deltatime_code_to_pos=vocab.deltatime_code_to_pos,
        deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    )
    dur_q, dt_q = build_quantizers(quant_tables)
    dur_q = dur_q.to(device)
    dt_q = dt_q.to(device)

    # baseline names
    req = [s.strip() for s in str(args.baselines).split(",") if s.strip()]
    legal = {"random", "uniform_time", "topk_duration", "amr_no_harmony"}
    for r in req:
        if r not in legal:
            raise ValueError(f"Unknown baseline: {r}. Legal={sorted(list(legal))}")

    # loaders
    pin = (device.type == "cuda")
    loaders: List[Tuple[str, Any]] = []
    if str(args.valid_bench_dir).strip():
        loaders.append(("valid", make_ornament_benchmark_dataloader(
            args.valid_bench_dir, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin
        )))
    if str(args.test_bench_dir).strip():
        loaders.append(("test", make_ornament_benchmark_dataloader(
            args.test_bench_dir, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin
        )))

    if not loaders:
        raise ValueError("Please provide at least one of --valid_bench_dir / --test_bench_dir")

    # eval cfg
    eval_cfg = OrnamentToBackboneEvalConfig(
        exclude_last_step=True,
        ignore_special_tokens=True,
        compute_cut_curve=(not args.no_cut_curve),
        max_batches=(None if int(args.max_batches) <= 0 else int(args.max_batches)),
        amp=False,               # baselines don't need AMP
        show_progress=True,
    )

    # build compressors
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
    if "amr_no_harmony" in req:
        compressors["amr_no_harmony"] = OTBAMRNoHarmonyCompressor(
            pad_id=vocab.pad_id,
            bos_id=vocab.bos_id,
            eos_id=vocab.eos_id,
            special_n=vocab.special_n,
            duration_code_to_pos=vocab.duration_code_to_pos,
            deltatime_code_to_pos=vocab.deltatime_code_to_pos,
            deltatime_code_offset=vocab.deltatime_code_offset,
            pos_per_beat=12,     # POS_RESOLUTION
            nbpm=4,              # 若数据是 4/4 拍子，则 nbpm=4
            lambda_time=1.0,
            dist_eta=1.6,
            time_subdiv=4,       # 16th units
            start_bias=1.0,
            seed=int(args.seed),
            rhy_param=0.0,
        )

    all_metrics: Dict[str, float] = {}

    for split, loader in loaders:
        print(f"\n========== OTB Split: {split} ==========")
        for name, comp in compressors.items():
            model = OTBBaselineModel(
                compressor=comp,
                pad_id=vocab.pad_id,
                bos_id=vocab.bos_id,
                eos_id=vocab.eos_id,
                special_n=vocab.special_n,
                tau=1.0,
            ).to(device)

            prefix = f"{split}/baseline/{name}/"
            adapter = build_pointer_adapter_from_model(model)
            m = evaluate_ornament_to_backbone(adapter, loader, device=device, cfg=eval_cfg, prefix=prefix)
            all_metrics.update(m)

            print(f"[{split}][{name}] " + _summarize(prefix, m))

            if args.proxy:
                proxy_cfg = MusicPriorProxyEvalConfig(
                    pos_per_beat=int(args.pos_per_beat),
                    strong_period_beats=int(args.strong_period_beats),
                    use_cummax_onset=True,
                    max_batches=(None if int(args.max_batches) <= 0 else int(args.max_batches)),
                    amp=False,  # baselines no need AMP
                    show_progress=True,
                    token_key="x_orn",
                    z_len_key="len_x",
                )
                proxy_prefix = f"{split}/proxy/{name}/"
                mp = evaluate_music_prior_proxy(
                    adapter,
                    loader,
                    device=device,
                    duration_q=dur_q,
                    dt_q=dt_q,
                    cfg=proxy_cfg,
                    prefix=proxy_prefix,
                )
                all_metrics.update(mp)

                def g(k: str) -> float:
                    return float(mp.get(proxy_prefix + k, float("nan")))

                print(
                    f"[{split}][{name}][proxy] "
                    f"JS(cnt)={g('pchist_js_cnt'):.4f} "
                    f"JS(dur)={g('pchist_js_dur'):.4f} "
                    f"Lift(strong)={g('lift_strong'):.3f} "
                    f"Lift(dur)={g('lift_duration'):.3f} "
                    f"Lift(ext)={g('lift_extrema'):.3f}"
                )

    if str(args.out_json).strip():
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] {outp}")


if __name__ == "__main__":
    main()

# for valid (first 50 batches)

# python -m main.run_otb_baselines --vocab_pkl .\preproc\output\bart_pretrain_corpus_v260205_with_ornamented_split\SimpleMono.pkl --valid_bench_dir .\preproc\output\bart_pretrain_corpus_v260205_with_ornamented_split\otb_bench\valid_ood --batch_size 32 --max_batches 50 --device cpu

# for test

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/bart_pretrain_corpus_v260205_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/bart_pretrain_corpus_v260205_with_ornamented_split/otb_bench/test_ood --batch_size 64 --device cpu --out_json ./baseline_metrics.json

# O2B TEST Split 10101

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/seed10101/skeletion_unsup_corpus_v260411_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/seed10101/skeletion_unsup_corpus_v260411_with_ornamented_split/otb_bench/test_ood --batch_size 64 --device cpu --out_json ./baseline_metrics.json --proxy --seed 10101

# O2B TEST Split 20202

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/seed20202/skeletion_unsup_corpus_v260411_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/seed20202/skeletion_unsup_corpus_v260411_with_ornamented_split/otb_bench/test_ood --batch_size 64 --device cpu --out_json ./baseline_metrics.json --proxy --seed 20202

# O2B TEST Split 30303

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/seed30303/skeletion_unsup_corpus_v260411_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/seed30303/skeletion_unsup_corpus_v260411_with_ornamented_split/otb_bench/test_ood --batch_size 64 --device cpu --out_json ./baseline_metrics.json --proxy --seed 30303

# O2B TEST Split 40404

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/seed40404/skeletion_unsup_corpus_v260411_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/seed40404/skeletion_unsup_corpus_v260411_with_ornamented_split/otb_bench/test_ood --batch_size 64 --device cpu --out_json ./baseline_metrics.json --proxy --seed 40404

# O2B TEST Split 50505

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/seed50505/skeletion_unsup_corpus_v260411_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/seed50505/skeletion_unsup_corpus_v260411_with_ornamented_split/otb_bench/test_ood --batch_size 64 --device cpu --out_json ./baseline_metrics.json --proxy --seed 50505

# TAVERN test

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/seed10101/skeletion_unsup_corpus_v260411_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/tavern_silver_otb/test --batch_size 64 --device cpu --out_json ./baseline_metrics.json

# Jiugongdacheng test

# python -m main.run_otb_baselines --vocab_pkl ./preproc/output/seed10101/skeletion_unsup_corpus_v260411_with_ornamented_split/SimpleMono.pkl --test_bench_dir ./preproc/output/real_jiugongdacheng_otb_bench/test --batch_size 64 --device cpu --out_json ./baseline_metrics.json
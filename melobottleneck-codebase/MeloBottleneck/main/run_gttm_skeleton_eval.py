from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .vocab_utils import load_vocab_info
from .quantization import MusicQuantizationTables
from .models.bart import MusicBartConfig, MusicBartBackboneConfig
from .nn_modules import MusicBartBackbone
from .models.skeleton.model import MusicSkeletonModelIII, MusicSkeletonIIIConfig

from .evaluation import (
    make_gttm_benchmark_dataloader,
    GTTMBackboneEvalConfig,
    evaluate_gttm_backbone,
    build_pointer_adapter_from_model,
)


def _pick_device(dev: str) -> torch.device:
    dev = str(dev).strip().lower()
    if dev == "cuda":
        if not torch.cuda.is_available():
            print("[Warn] cuda requested but not available; fallback to cpu.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def build_music_bart_cfg_from_dict(model_cfg: Dict[str, Any], vocab_cfg, dropout: float = 0.0) -> MusicBartConfig:
    backbone_cfg = MusicBartBackboneConfig(
        max_seq_len=int(model_cfg["max_seq_len"]),
        d_embed=int(model_cfg["d_embed"]),
        d_model=int(model_cfg["d_model"]),
        n_encoder_layers=int(model_cfg["n_encoder_layers"]),
        n_decoder_layers=int(model_cfg["n_decoder_layers"]),
        n_heads=int(model_cfg["n_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        dropout=float(dropout),
    )
    return MusicBartConfig(vocab=vocab_cfg, backbone=backbone_cfg)


def load_skeleton_model_from_ckpt(
    ckpt_path: Path,
    *,
    vocab_pkl: str,
    device: torch.device,
) -> MusicSkeletonModelIII:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    cfg = ckpt.get("config", None)
    if cfg is None:
        raise ValueError("Checkpoint missing 'config' (saved by train_skeleton_end2end.py).")

    vocab = load_vocab_info(vocab_pkl)
    vocab_cfg = vocab.to_music_bart_vocab_config()

    music_bart_cfg = build_music_bart_cfg_from_dict(cfg["model"], vocab_cfg, dropout=0.0)
    backbone = MusicBartBackbone(cfg=music_bart_cfg).to(device)

    quant_tables = MusicQuantizationTables(
        special_n=vocab.special_n,
        duration_code_to_pos=vocab.duration_code_to_pos,
        duration_pos_to_code=vocab.duration_pos_to_code,
        deltatime_code_offset=vocab.deltatime_code_offset,
        deltatime_code_to_pos=vocab.deltatime_code_to_pos,
        deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    )

    sk_cfg = MusicSkeletonIIIConfig(**cfg["skeleton_model"])
    model = MusicSkeletonModelIII(
        backbone=backbone,
        quant=quant_tables,
        cfg=sk_cfg,
        lm_prior=None,
        use_bias_in_lm_head=True,
    ).to(device)

    state = ckpt.get("model_state_dict", None)
    if state is None:
        raise ValueError("Checkpoint missing 'model_state_dict'.")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Warn] missing keys: {missing[:10]} (total={len(missing)})")
    if unexpected:
        print(f"[Warn] unexpected keys: {unexpected[:10]} (total={len(unexpected)})")

    model.eval()
    return model


def _summarize(prefix: str, m: Dict[str, float]) -> str:
    def g(k: str) -> float:
        return float(m.get(prefix + k, float("nan")))
    return (
        f"f1_path={g('f1_hard'):.4f} | "
        f"f1_topk={g('f1_hard_topk'):.4f} | "
        f"gap_f1={g('gap_f1'):.4f} | "
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
    ap.add_argument("--ckpt_path", type=str, required=True, help="stageC skeleton_last.pt or skeleton_epoch*.pt")
    ap.add_argument("--vocab_pkl", type=str, required=True)
    ap.add_argument("--gttm_split_dir", type=str, required=True, help=".../gttm_bench/test or train")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_batches", type=int, default=0)

    ap.add_argument("--eval_rho", type=float, default=2.0 / 3.0)
    ap.add_argument("--no_cut_curve", action="store_true")
    ap.add_argument("--out_json", type=str, default="")

    args = ap.parse_args()
    device = _pick_device(args.device)

    model = load_skeleton_model_from_ckpt(
        Path(args.ckpt_path),
        vocab_pkl=str(args.vocab_pkl),
        device=device,
    )
    adapter = build_pointer_adapter_from_model(model)

    loader = make_gttm_benchmark_dataloader(
        args.gttm_split_dir,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    eval_cfg = GTTMBackboneEvalConfig(
        eval_rho=float(args.eval_rho),
        force_z_len_rho=float(args.eval_rho),  # fix evaluation length for fair comparison
        compute_cut_curve=(not bool(args.no_cut_curve)),
        max_batches=(None if int(args.max_batches) <= 0 else int(args.max_batches)),
        amp=(device.type == "cuda"),
        show_progress=True,
    )

    prefix = "gttm/ours/"
    metrics = evaluate_gttm_backbone(adapter, loader, device=device, cfg=eval_cfg, prefix=prefix)
    print("[Ours] " + _summarize(prefix, metrics))

    if str(args.out_json).strip():
        outp = Path(args.out_json)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[Saved] {outp}")


if __name__ == "__main__":
    main()

# example usage:
# python -m main.run_gttm_skeleton_eval --ckpt_path .\ckpt\skel-260401\pt_logit_s=8\stageC_skeleton\skeleton_last.pt --vocab_pkl .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\SimpleMono.pkl --gttm_split_dir .\preproc\output\gttm_bench_v1.2\test --device cuda --eval_rho 0.6666667
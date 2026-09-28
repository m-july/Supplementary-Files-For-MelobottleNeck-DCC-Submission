from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .mudep_baseline import (
    load_pieces_jsonl,
    MuDePGTTMSupervisedDataset,
    predict_mudep_scores_for_split,
)

from .vocab_utils import load_vocab_info
from .evaluation import (
    make_gttm_benchmark_dataloader,
    GTTMBackboneEvalConfig,
    evaluate_gttm_backbone,
    ScoreArrayAdapter,
)


def _pick_device(dev: str) -> torch.device:
    dev = str(dev).strip().lower()
    if dev == "cuda":
        if not torch.cuda.is_available():
            print("[Warn] cuda requested but not available; fallback to cpu.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def train_mudep(
    *,
    gttm_bench_dir: Path,
    gttm_raw_dir: Optional[Path],
    out_dir: Path,
    device: torch.device,
    seed: int,
    # MuDeP hyperparams
    n_layers: int,
    n_hidden: int,
    dropout: float,
    lr: float,
    weight_decay: float,
    activation: str,
    embeddings: int,
    n_heads: int,
    pos_enc: str,
    biaffine: bool,
    pos_weight: bool,
    encoder_type: str,
    loss: str,
    optimizer: str,
    warmup_steps: int,
    max_epochs: int,
    data_augmentation: str,
    no_validation: bool,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from pytorch_lightning import Trainer, seed_everything
    except Exception as e:
        raise ImportError("pytorch_lightning is required for MuDeP training. Install it first.") from e

    seed_everything(int(seed), workers=True)

    train_metas = load_pieces_jsonl(gttm_bench_dir / "train" / "pieces.jsonl")
    ds_train = MuDePGTTMSupervisedDataset(train_metas, gttm_raw_dir=gttm_raw_dir, strict=True, verbose=True)

    if data_augmentation == "preprocess":
        import module_external.musicparser.data_loading as mudep_dl
        print("[MuDeP] Preprocess augmentation: transpose -12..12")
        ds_train_aug = mudep_dl.TSDatasetAugmented([ds_train[i] for i in range(len(ds_train))])
        train_loader = torch.utils.data.DataLoader(ds_train_aug, batch_size=1, shuffle=True, num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(ds_train, batch_size=1, shuffle=True, num_workers=0)

    if pos_weight:
        pw = float(ds_train.get_positive_weight())
        print(f"[MuDeP] Using pos_weight={pw:.4f}")
    else:
        pw = 1.0

    from module_external.musicparser.models import ArcPredictionLightModel

    use_embeddings = embeddings > 0
    embedding_dim = {"sum": int(embeddings)} if use_embeddings else {}
    input_dim = int(embeddings) if use_embeddings else 25
    rpr = (str(pos_enc).strip().lower() == "relative")

    model = ArcPredictionLightModel(
        input_dim,
        int(n_hidden),
        pos_weight=int(pw) if pos_weight else 1,
        dropout=float(dropout),
        lr=float(lr),
        weight_decay=float(weight_decay),
        n_layers=int(n_layers),
        activation=str(activation),
        use_embeddings=bool(use_embeddings),
        embedding_dim=embedding_dim,
        biaffine=bool(biaffine),
        encoder_type=str(encoder_type),
        n_heads=int(n_heads),
        data_type="notes",
        rpr=bool(rpr),
        pretrain_mode=False,
        loss_type=str(loss),
        optimizer=str(optimizer),
        warmup_steps=int(warmup_steps),
        max_epochs=int(max_epochs),
        len_train_dataloader=len(train_loader),
    )

    if device.type == "cuda":
        trainer = Trainer(
            max_epochs=int(max_epochs),
            accelerator="gpu",
            devices=1,
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
        )
    else:
        trainer = Trainer(
            max_epochs=int(max_epochs),
            accelerator="cpu",
            devices=1,
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
        )

    print("[MuDeP] Start training...")
    trainer.fit(model, train_dataloaders=train_loader)

    ckpt_path = out_dir / "mudep_last.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    print(f"[MuDeP] Saved checkpoint: {ckpt_path}")

    (out_dir / "train_meta.json").write_text(json.dumps({
        "seed": int(seed),
        "n_layers": int(n_layers),
        "n_hidden": int(n_hidden),
        "dropout": float(dropout),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "activation": str(activation),
        "embeddings": int(embeddings),
        "n_heads": int(n_heads),
        "pos_enc": str(pos_enc),
        "rpr": bool(rpr),
        "biaffine": bool(biaffine),
        "pos_weight": bool(pos_weight),
        "encoder_type": str(encoder_type),
        "loss": str(loss),
        "optimizer": str(optimizer),
        "warmup_steps": int(warmup_steps),
        "max_epochs": int(max_epochs),
        "data_augmentation": str(data_augmentation),
        "gttm_bench_dir": str(gttm_bench_dir),
        "gttm_raw_dir": (str(gttm_raw_dir) if gttm_raw_dir is not None else None),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return ckpt_path


def _summarize(prefix: str, m: dict) -> str:
    def g(k: str) -> float:
        return float(m.get(prefix + k, float("nan")))
    return (
        f"f1_hard_topk={g('f1_hard_topk'):.4f} | "
        f"ap_soft={g('ap_soft'):.4f} | "
        f"ndcg_soft={g('ndcg_soft'):.4f} | "
        f"ndcg_full={g('ndcg_full'):.4f} | "
        f"cut_f1_auc_norm={g('cut_f1_auc_norm'):.4f} | "
        f"mean_ndcg_cut={g('mean_ndcg_cut'):.4f} | "
        f"spearman={g('spearman'):.4f} | "
        f"rho={g('rho'):.4f} | n={int(g('n'))}"
    )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p: argparse.ArgumentParser):
        p.add_argument("--gttm_bench_dir", type=str, required=True)
        p.add_argument("--gttm_raw_dir", type=str, default="")
        p.add_argument("--out_dir", type=str, required=True)
        p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
        p.add_argument("--seed", type=int, default=0)

        # NEW: vocab for evaluation
        p.add_argument("--vocab_pkl", type=str, default="", help="Required for eval (pad_id/special_n)")

    # train
    p_train = sub.add_parser("train")
    add_common(p_train)
    p_train.add_argument("--n_layers", type=int, default=2)
    p_train.add_argument("--n_hidden", type=int, default=64)
    p_train.add_argument("--dropout", type=float, default=0.21)
    p_train.add_argument("--lr", type=float, default=4e-4)
    p_train.add_argument("--weight_decay", type=float, default=0.05)
    p_train.add_argument("--activation", type=str, default="gelu")
    p_train.add_argument("--embeddings", type=int, default=96)
    p_train.add_argument("--n_heads", type=int, default=8)
    p_train.add_argument("--pos_enc", type=str, default="relative", choices=["absolute", "relative"])
    p_train.add_argument("--biaffine", action="store_true")
    p_train.add_argument("--pos_weight", action="store_true")
    p_train.add_argument("--encoder_type", type=str, default="transformer", choices=["rnn", "transformer"])
    p_train.add_argument("--loss", type=str, default="both", choices=["bce", "ce", "both"])
    p_train.add_argument("--optimizer", type=str, default="warmadamw", choices=["adamw", "radam", "warmadamw", "warmadam"])
    p_train.add_argument("--warmup_steps", type=int, default=50)
    p_train.add_argument("--max_epochs", type=int, default=15)
    p_train.add_argument("--data_augmentation", type=str, default="preprocess", choices=["no", "preprocess"])
    p_train.add_argument("--no_validation", action="store_true")

    # predict
    p_pred = sub.add_parser("predict")
    add_common(p_pred)
    p_pred.add_argument("--ckpt_path", type=str, required=True)
    p_pred.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p_pred.add_argument("--postprocess_alg", type=str, default="eisner", choices=["eisner", "chuliu_edmonds"])
    p_pred.add_argument("--score_temp", type=float, default=1.0)
    p_pred.add_argument("--strict", action="store_true")
    # NEW: eval options
    p_pred.add_argument("--no_eval", action="store_true")
    p_pred.add_argument("--eval_batch_size", type=int, default=32)
    p_pred.add_argument("--eval_rho", type=float, default=2.0 / 3.0)
    p_pred.add_argument("--no_cut_curve", action="store_true")
    p_pred.add_argument("--max_eval_batches", type=int, default=0)

    # train + predict
    p_tp = sub.add_parser("train_predict")
    add_common(p_tp)
    for a in p_train._actions:
        if a.dest in {"help", "cmd"}:
            continue
        if a.dest in {"gttm_bench_dir", "gttm_raw_dir", "out_dir", "device", "seed", "vocab_pkl"}:
            continue
        p_tp._add_action(a)
    p_tp.add_argument("--postprocess_alg", type=str, default="eisner", choices=["eisner", "chuliu_edmonds"])
    p_tp.add_argument("--score_temp", type=float, default=1.0)
    p_tp.add_argument("--strict", action="store_true")
    # NEW: eval options
    p_tp.add_argument("--no_eval", action="store_true")
    p_tp.add_argument("--eval_batch_size", type=int, default=32)
    p_tp.add_argument("--eval_rho", type=float, default=2.0 / 3.0)
    p_tp.add_argument("--no_cut_curve", action="store_true")
    p_tp.add_argument("--max_eval_batches", type=int, default=0)

    args = ap.parse_args()

    gttm_bench_dir = Path(args.gttm_bench_dir)
    gttm_raw_dir = Path(args.gttm_raw_dir) if str(args.gttm_raw_dir).strip() else None
    out_dir = Path(args.out_dir)
    device = _pick_device(args.device)

    if args.cmd == "train":
        _ = train_mudep(
            gttm_bench_dir=gttm_bench_dir,
            gttm_raw_dir=gttm_raw_dir,
            out_dir=out_dir,
            device=device,
            seed=args.seed,
            n_layers=args.n_layers,
            n_hidden=args.n_hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            activation=args.activation,
            embeddings=args.embeddings,
            n_heads=args.n_heads,
            pos_enc=args.pos_enc,
            biaffine=args.biaffine,
            pos_weight=args.pos_weight,
            encoder_type=args.encoder_type,
            loss=args.loss,
            optimizer=args.optimizer,
            warmup_steps=args.warmup_steps,
            max_epochs=args.max_epochs,
            data_augmentation=args.data_augmentation,
            no_validation=args.no_validation,
        )
        return

    if args.cmd in {"predict", "train_predict"}:
        if args.cmd == "train_predict":
            ckpt_path = train_mudep(
                gttm_bench_dir=gttm_bench_dir,
                gttm_raw_dir=gttm_raw_dir,
                out_dir=out_dir,
                device=device,
                seed=args.seed,
                n_layers=args.n_layers,
                n_hidden=args.n_hidden,
                dropout=args.dropout,
                lr=args.lr,
                weight_decay=args.weight_decay,
                activation=args.activation,
                embeddings=args.embeddings,
                n_heads=args.n_heads,
                pos_enc=args.pos_enc,
                biaffine=args.biaffine,
                pos_weight=args.pos_weight,
                encoder_type=args.encoder_type,
                loss=args.loss,
                optimizer=args.optimizer,
                warmup_steps=args.warmup_steps,
                max_epochs=args.max_epochs,
                data_augmentation=args.data_augmentation,
                no_validation=True,
            )
            split = "test"
        else:
            ckpt_path = Path(args.ckpt_path)
            split = str(args.split)

        from module_external.musicparser.models import ArcPredictionLightModel
        model = ArcPredictionLightModel.load_from_checkpoint(str(ckpt_path), map_location="cpu", weights_only=False)

        split_dir = gttm_bench_dir / split
        metas = load_pieces_jsonl(split_dir / "pieces.jsonl")

        pred_out = out_dir / f"pred_{split}"
        predict_mudep_scores_for_split(
            model=model,
            bench_split_dir=split_dir,
            metas=metas,
            gttm_raw_dir=gttm_raw_dir,
            device=device,
            out_dir=pred_out,
            postprocess_alg=str(args.postprocess_alg),
            score_temp=float(args.score_temp),
            strict=bool(args.strict),
        )
        print(f"[Done] predictions saved to: {pred_out}")

        # ===========================
        # NEW: eval right after predict
        # ===========================
        if not bool(getattr(args, "no_eval", False)):
            if not str(args.vocab_pkl).strip():
                raise ValueError("--vocab_pkl is required for evaluation (pad_id/special_n).")

            vocab = load_vocab_info(args.vocab_pkl)

            # load pred_score
            pred_score = np.load(pred_out / "pred_score.npy", mmap_mode="r")  # [N,L]
            adapter = ScoreArrayAdapter(pred_score, pad_id=vocab.pad_id, special_n=vocab.special_n)

            loader = make_gttm_benchmark_dataloader(
                split_dir,
                batch_size=int(args.eval_batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )

            eval_cfg = GTTMBackboneEvalConfig(
                eval_rho=float(args.eval_rho),
                force_z_len_rho=None,   # score-array model doesn't need z_len; metrics still use eval_rho
                compute_cut_curve=(not bool(args.no_cut_curve)),
                max_batches=(None if int(args.max_eval_batches) <= 0 else int(args.max_eval_batches)),
                amp=False,
                show_progress=True,
            )

            # evaluation itself can run on CPU safely (no NN forward)
            eval_device = torch.device("cpu")
            metrics = evaluate_gttm_backbone(
                adapter,
                loader,
                device=eval_device,
                cfg=eval_cfg,
                prefix=f"{split}/mudep/",
            )

            print(f"[GTTM-{split}][MuDeP] " + _summarize(f"{split}/mudep/", metrics))

            (pred_out / "metrics.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[Saved] {pred_out / 'metrics.json'}")

        return


if __name__ == "__main__":
    main()
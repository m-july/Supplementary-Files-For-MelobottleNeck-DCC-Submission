# main/export_aug_ornament_npy.py
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .vocab_utils import load_vocab_info
from .data import make_dataloader
from .augment import MusicAugmentConfig
from .quantization import MusicQuantizationTables
from .ornament import MusicOrnamenter, MusicOrnamentConfig, PI_INSERTED, PI_PAD


def build_quant_tables_from_vocab(vocab) -> MusicQuantizationTables:
    return MusicQuantizationTables(
        special_n=vocab.special_n,
        duration_code_to_pos=vocab.duration_code_to_pos,
        duration_pos_to_code=vocab.duration_pos_to_code,
        deltatime_code_offset=vocab.deltatime_code_offset,
        deltatime_code_to_pos=vocab.deltatime_code_to_pos,
        deltatime_pos_to_code=vocab.deltatime_pos_to_code,
    )


def invert_global2local(global2local: np.ndarray, n_local: int) -> np.ndarray:
    """
    从 global_id -> local_id 的映射数组，构建 local_id -> global_id 的反向查找表。
    """
    g2l = np.asarray(global2local, dtype=np.int64)
    local2global = np.full((int(n_local),), -1, dtype=np.int64)

    valid = (g2l >= 0) & (g2l < int(n_local))
    # np.nonzero(valid)[0] 给出 global_id
    local2global[g2l[valid]] = np.nonzero(valid)[0]
    return local2global

def debug_orn(x_local, ornamenter, rng=np.random.default_rng(0)):
    cfg = ornamenter.cfg
    L = x_local.shape[0]
    pad = int(cfg.pad_id)
    eos = None if cfg.eos_id is None else int(cfg.eos_id)

    pitch = x_local[:, 0].astype(int)

    valid_len0 = ornamenter._find_valid_len(x_local)
    eos_pos = ornamenter._find_eos_pos(x_local, valid_len0)
    valid_len = (eos_pos + 1) if eos_pos is not None else valid_len0
    pad_budget = L - valid_len

    print("pad_id=", pad, "eos_id=", eos, "special_n=", ornamenter.special_n)
    print("valid_len0(last!=pad)=", valid_len0, "eos_pos=", eos_pos, "valid_len_used=", valid_len)
    print("pad_budget=", pad_budget, "tail_pitch_unique=", np.unique(pitch[valid_len:])[:10])

    # 关键断言：valid_len 后面应该全是 PAD（至少 pitch 列）
    bad_tail = np.where(pitch[valid_len:] != pad)[0]
    print("non-pad in tail after valid_len:", bad_tail[:20], "count=", bad_tail.size)

    xa, pi = ornamenter.augment(x_local, rng=rng)
    ins = int((pi == PI_INSERTED).sum())
    print("inserted=", ins, "new_nonpad=", int((xa[:,0] != pad).sum()))

    # 更关键：检查原序列 [0..valid_len-1] 的 index 是否都还在 pi 里出现过
    present = set(int(i) for i in pi[pi >= 0])
    missing = [i for i in range(valid_len) if i not in present]
    print("missing original indices (should be empty):", missing[:50], "count=", len(missing))

    # EOS 新位置
    if eos is not None:
        eos_new = np.where(xa[:,0].astype(int) == eos)[0]
        print("eos_new_positions=", eos_new[:10], "count=", eos_new.size)

    return xa, pi


def identity_pi_for_seq(
    x_local: np.ndarray,  # [L,3]
    *,
    pad_id: int,
    eos_id: Optional[int],
) -> np.ndarray:
    """
    ornament 关闭时，为了仍可输出 pi，这里构造 identity pi：
      - 有效 token（直到 EOS（含）/ 或直到 PAD）: pi[k]=k
      - pad 区：PI_PAD
    """
    L = int(x_local.shape[0])
    pi = np.full((L,), PI_PAD, dtype=np.int32)

    pitch = x_local[:, 0]
    idx_pad = np.where(pitch == int(pad_id))[0]
    valid_len = int(idx_pad[0]) if idx_pad.size > 0 else L

    if eos_id is not None:
        idx_eos = np.where(pitch[:valid_len] == int(eos_id))[0]
        if idx_eos.size > 0:
            valid_len = int(idx_eos[0]) + 1

    if valid_len > 0:
        pi[:valid_len] = np.arange(valid_len, dtype=np.int32)
    return pi


def local_tokens_to_global(
    x_local: np.ndarray,  # [...,3] int64
    *,
    local2global_pitch: np.ndarray,
    local2global_dur: np.ndarray,
    local2global_dt: np.ndarray,
) -> np.ndarray:
    """
    向量化 local->global 映射。
    """
    if x_local.dtype != np.int64:
        x_local = x_local.astype(np.int64, copy=False)

    if x_local.min() < 0:
        raise ValueError(f"Found negative local ids in tokens (min={int(x_local.min())}). "
                         "This usually indicates vocab mismatch or bad mapping.")

    out = np.empty_like(x_local, dtype=np.int64)
    out[..., 0] = local2global_pitch[x_local[..., 0]]
    out[..., 1] = local2global_dur[x_local[..., 1]]
    out[..., 2] = local2global_dt[x_local[..., 2]]

    if (out < 0).any():
        bad = int((out < 0).sum())
        raise ValueError(
            f"local->global produced {bad} negative ids. "
            "This usually indicates incomplete mapping tables or vocab mismatch."
        )
    return out


def main():
    ap = argparse.ArgumentParser("Export dataset after (optional) augment + (optional) train ornamenter")
    ap.add_argument("--input_npy", type=str, required=True, help="Input npy: [N,L,3] GLOBAL ids")
    ap.add_argument("--output_npy", type=str, required=True, help="Output npy: [N,L,3] GLOBAL ids")

    ap.add_argument("--vocab_pkl", type=str, required=True, help="SimpleMono.pkl (must match input npy vocab)")

    # dataloader
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--max_samples", type=int, default=None)

    # reproducibility
    ap.add_argument("--seed", type=int, default=1234)

    # toggle modules
    ap.add_argument("--no_augment", action="store_true", help="Disable MusicAugmenter")
    ap.add_argument("--no_ornament", action="store_true", help="Disable MusicOrnamenter")

    # augment params (optional overrides)
    ap.add_argument("--augment_seed", type=int, default=None)
    ap.add_argument("--transpose_sigma", type=float, default=None)
    ap.add_argument("--transpose_min", type=int, default=None)
    ap.add_argument("--transpose_max", type=int, default=None)
    ap.add_argument("--p_time_scale_2x", type=float, default=None)
    ap.add_argument("--p_time_scale_half", type=float, default=None)

    # ornament params (optional overrides)
    ap.add_argument("--ornament_seed", type=int, default=None)
    ap.add_argument("--ornament_p_apply", type=float, default=1.0, help="Train-time strong view often uses 1.0")
    ap.add_argument("--ornament_max_extra_tokens", type=int, default=None)

    # extra outputs
    ap.add_argument("--output_pi_npy", type=str, default=None, help="Optional: save pi [N,L] (int32)")
    ap.add_argument("--output_aug_only_npy", type=str, default=None, help="Optional: save post-augment/pre-ornament tokens")

    args = ap.parse_args()

    # -------------------------
    # Seed
    # -------------------------
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # -------------------------
    # Load vocab + quant tables
    # -------------------------
    vocab = load_vocab_info(args.vocab_pkl)
    quant_tables = build_quant_tables_from_vocab(vocab)

    # -------------------------
    # Build augment config (StageC-like by default)
    # -------------------------
    aug_cfg = MusicAugmentConfig(quantization_tables=quant_tables)
    # optional overrides
    if args.transpose_sigma is not None:
        aug_cfg = replace(aug_cfg, transpose_sigma=float(args.transpose_sigma))
    if args.transpose_min is not None:
        aug_cfg = replace(aug_cfg, transpose_min=int(args.transpose_min))
    if args.transpose_max is not None:
        aug_cfg = replace(aug_cfg, transpose_max=int(args.transpose_max))
    if args.p_time_scale_2x is not None:
        aug_cfg = replace(aug_cfg, p_time_scale_2x=float(args.p_time_scale_2x))
    if args.p_time_scale_half is not None:
        aug_cfg = replace(aug_cfg, p_time_scale_half=float(args.p_time_scale_half))

    augment_enabled = (not bool(args.no_augment))
    ornament_enabled = (not bool(args.no_ornament))

    augment_seed = int(args.augment_seed) if args.augment_seed is not None else int(args.seed)

    # -------------------------
    # Ornamenter (StageC-like)
    # -------------------------
    ornamenter: Optional[MusicOrnamenter] = None
    orn_rng: Optional[np.random.Generator] = None
    if ornament_enabled:
        pad_id_local = int(vocab.global2local_pitch[int(vocab.pad_id)])
        bos_id_local = int(vocab.global2local_pitch[int(vocab.bos_id)])
        eos_id_local = int(vocab.global2local_pitch[int(vocab.eos_id)])
        orn_cfg = MusicOrnamentConfig(
            enable=True,
            p_apply=float(args.ornament_p_apply),
            pad_id=pad_id_local,
            bos_id=bos_id_local,
            eos_id=eos_id_local,
            quantization_tables=quant_tables,
        )
        if args.ornament_max_extra_tokens is not None:
            orn_cfg = replace(orn_cfg, max_extra_tokens=int(args.ornament_max_extra_tokens))

        ornamenter = MusicOrnamenter(orn_cfg)

        orn_seed = int(args.ornament_seed) if args.ornament_seed is not None else (int(args.seed) + 260312)
        orn_rng = np.random.default_rng(orn_seed)

    # -------------------------
    # Input shape (N,L,3)
    # -------------------------
    inp = np.load(args.input_npy, mmap_mode="r")
    if inp.ndim != 3 or inp.shape[-1] != 3:
        raise ValueError(f"Expected input [N,L,3], got {inp.shape}")
    N_in, L, _ = inp.shape
    del inp

    # -------------------------
    # Dataloader (GLOBAL -> LOCAL, optional augment)
    # -------------------------
    loader = make_dataloader(
        args.input_npy,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        drop_last=False,
        global2local_pitch=vocab.global2local_pitch,
        global2local_duration=vocab.global2local_duration,
        global2local_dt=vocab.global2local_dt,
        augment=augment_enabled,
        augment_seed=augment_seed,
        augment_config=aug_cfg,          # augment=False 时也无妨
        max_samples=int(args.max_samples) if args.max_samples is not None else None,
    )

    N_out = len(loader.dataset)

    # -------------------------
    # local->global reverse tables
    # -------------------------
    local2global_pitch = invert_global2local(vocab.global2local_pitch, vocab.n_pitch)
    local2global_dur = invert_global2local(vocab.global2local_duration, vocab.n_duration)
    local2global_dt = invert_global2local(vocab.global2local_dt, vocab.n_dt)

    # -------------------------
    # Prepare outputs (memmap .npy)
    # -------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_npy)), exist_ok=True)
    out_mm = np.lib.format.open_memmap(
        args.output_npy, mode="w+", dtype=np.int64, shape=(N_out, int(L), 3)
    )

    aug_mm = None
    if args.output_aug_only_npy is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_aug_only_npy)), exist_ok=True)
        aug_mm = np.lib.format.open_memmap(
            args.output_aug_only_npy, mode="w+", dtype=np.int64, shape=(N_out, int(L), 3)
        )

    pi_mm = None
    if args.output_pi_npy is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_pi_npy)), exist_ok=True)
        pi_mm = np.lib.format.open_memmap(
            args.output_pi_npy, mode="w+", dtype=np.int32, shape=(N_out, int(L))
        )

    # -------------------------
    # Main loop
    # -------------------------
    write_pos = 0
    pbar = tqdm(loader, desc="[Export] augment+ornament", dynamic_ncols=True)
    for x_local_t in pbar:
        # x_local_t: [B,L,3] LOCAL ids, already augmented if augment_enabled=True
        x_local = x_local_t.contiguous().numpy().astype(np.int64, copy=False)  # view/copy safe

        B = int(x_local.shape[0])

        # (optional) save post-augment/pre-ornament
        if aug_mm is not None:
            x_aug_global = local_tokens_to_global(
                x_local,
                local2global_pitch=local2global_pitch,
                local2global_dur=local2global_dur,
                local2global_dt=local2global_dt,
            )
            aug_mm[write_pos:write_pos + B] = x_aug_global

        # ornament
        if ornamenter is not None:
            assert orn_rng is not None
            x2 = np.empty_like(x_local)
            pi_b = np.empty((B, int(L)), dtype=np.int32)

            for b in range(B):
                xa, pib = ornamenter.augment(x_local[b], rng=orn_rng)
                x2[b] = xa
                pi_b[b] = pib

            x_local_out = x2
            pi_out = pi_b
        else:
            x_local_out = x_local
            pi_out = None

        # if need pi but ornament disabled -> identity pi
        if pi_mm is not None:
            if pi_out is None:
                pi_out = np.stack(
                    [
                        identity_pi_for_seq(
                            x_local_out[b],
                            pad_id=int(vocab.pad_id),
                            eos_id=int(vocab.eos_id),
                        )
                        for b in range(B)
                    ],
                    axis=0,
                )
            pi_mm[write_pos:write_pos + B] = pi_out

        # write final tokens
        x_global_out = local_tokens_to_global(
            x_local_out,
            local2global_pitch=local2global_pitch,
            local2global_dur=local2global_dur,
            local2global_dt=local2global_dt,
        )
        out_mm[write_pos:write_pos + B] = x_global_out

        write_pos += B

    # flush
    out_mm.flush()
    if aug_mm is not None:
        aug_mm.flush()
    if pi_mm is not None:
        pi_mm.flush()

    print(f"[Saved] output tokens -> {args.output_npy} | shape={(N_out, int(L), 3)}")
    if args.output_aug_only_npy is not None:
        print(f"[Saved] aug-only tokens -> {args.output_aug_only_npy} | shape={(N_out, int(L), 3)}")
    if args.output_pi_npy is not None:
        print(f"[Saved] pi -> {args.output_pi_npy} | shape={(N_out, int(L))}")
        print(f"        pi values: inserted={PI_INSERTED}, pad={PI_PAD}, >=0 means 'points to original index'")

    print("---------------------")

    debug_orn(x_local, ornamenter, rng=orn_rng)


if __name__ == "__main__":
    main()

# usage:

# python -m main.run_export_aug_ornament_npy --input_npy .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\train.npy --output_npy .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\train_aug_orn.npy --vocab_pkl  .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\SimpleMono.pkl --batch_size 128 --seed 1234 --output_pi_npy .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\train_aug_orn_pi.npy --output_aug_only_npy .\preproc\output\skeletion_unsup_corpus_v260331_with_ornamented_split\train_aug_only.npy
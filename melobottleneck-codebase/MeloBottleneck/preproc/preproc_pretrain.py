# simplemono_encode.py
from __future__ import annotations

import argparse
from pathlib import Path

from .simplemono_preproc.pipelines.pretrain import build_pretrain_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--xml_group", type=str, default="staff_voice", choices=["part", "staff", "staff_voice"])

    # split / anti-leak
    ap.add_argument("--train_ratio", type=float, default=0.90)
    ap.add_argument("--valid_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.05)
    ap.add_argument("--group_mode", type=str, default="file",
                    help="file|stem|parent_stem|regex:<pattern>  (avoid leakage by grouping)")

    # output controls
    ap.add_argument("--max_len_tokens", type=int, default=0,
                    help="0 => use default fixed max_len for windowed pretrain (6150). Must be divisible by 3.")
    ap.add_argument("--write_txt_debug", action="store_true",
                    help="Optional: dump train/valid/test.txt for inspection (not needed for training).")
    ap.add_argument("--write_dict_txt", action="store_true",
                    help="Optional: dump dict.txt with token counts (not needed if you only use SimpleMono.pkl).")
    ap.add_argument(
        "--max_windows_per_file", type=int, default=0,
        help="0 => keep all windows (old behavior). 1 => keep only the first window per source file."
    )
    ap.add_argument(
        "--max_windows_per_voice", type=int, default=0,
        help="0 => keep all windows per voice. 1 => keep only the first window per extracted voice."
    )

    # speed
    ap.add_argument("--num_workers", type=int, default=0,
                    help="0 => auto(cpu_count-1); 1 => single process (debug).")
    ap.add_argument("--chunksize", type=int, default=8,
                    help="multiprocessing imap(_unordered) chunksize.")
    ap.add_argument("--maxtasksperchild", type=int, default=1000,
                    help="restart workers every N tasks to avoid memory leaks. 0 => disabled.")
    ap.add_argument("--ordered", action="store_true",
                    help="keep deterministic output order (slower); default unordered for speed.")

    args = ap.parse_args()

    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios} sum={sum(ratios)}")

    build_pretrain_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        xml_group=args.xml_group,
        ratios=ratios,
        group_mode=args.group_mode,
        max_len_tokens=args.max_len_tokens,
        write_txt_debug=args.write_txt_debug,
        write_dict_txt=args.write_dict_txt,
        max_windows_per_file=args.max_windows_per_file,
        max_windows_per_voice=args.max_windows_per_voice,

        num_workers=args.num_workers,
        chunksize=args.chunksize,
        maxtasksperchild=args.maxtasksperchild,
        ordered=args.ordered,
    )


if __name__ == "__main__":
    main()

# Example:
# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\essen" --output_dir ".\preproc\output\tmp_essen" --train_ratio 0.8 --valid_ratio 0.1 --test_ratio 0.1 --max_len_tokens 0 --group_mode file

# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\bart_pretrain_corpus_v260205" --output_dir ".\preproc\output\bart_pretrain_corpus_v260205" --train_ratio 1.0 --valid_ratio 0 --test_ratio 0 --max_len_tokens 0

# python -m preproc.preproc_pretrain --input_dir "J:\DATASETS\MIDIs\bart_pretrain_corpus_v260205" --output_dir ".\preproc\output\bart_pretrain_corpus_v260205_with_ornamented_split" --train_ratio 0.9 --valid_ratio 0.05 --test_ratio 0.05 --max_len_tokens 0 --group_mode file
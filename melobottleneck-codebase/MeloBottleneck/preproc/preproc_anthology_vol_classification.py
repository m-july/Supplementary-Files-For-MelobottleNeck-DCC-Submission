# simplemono_encode.py
from __future__ import annotations

import re

import argparse
from pathlib import Path
from typing import List

from .simplemono_preproc.pipelines.manifest_task import build_from_manifest_items, ManifestItem

def get_anthology_vol_from_folder_name(folder_name: str) -> str:
    pattern = r'[0-9_\- ]'  # 匹配数字、下划线、横杠和空格
    result = re.sub(pattern, '', folder_name)
    return result

def get_anthology_vol(file_path: Path) -> str:
    return get_anthology_vol_from_folder_name(file_path.parent.name)

def get_anthology_group_name(file_path: Path) -> str:
    anthology_name = re.sub(r'（.*）$', '', re.sub(r'^\d+_', '', re.sub(r'[\s\u3000]', '', file_path.stem)))
    return f"{get_anthology_vol(file_path)}/{anthology_name}"

def build_manifest_items(input_dir: Path, file_ext_list: List[str] = ["mid", "midi", "musicxml", "xml"]) -> List[ManifestItem]:
    
    # list all subfolders in input_dir
    folders = [f for f in input_dir.iterdir() if f.is_dir()]

    # list all volumes
    volumes = set()
    for folder in folders:
        volumes.add(get_anthology_vol_from_folder_name(folder.name))
    
    # build vol to label mapping
    vol_to_label = {vol: i for i, vol in enumerate(sorted(volumes))}

    # build manifest items
    manifest_items = []
    for i, folder in enumerate(folders):
        for file in folder.iterdir():
            if file.suffix.lower()[1:] in file_ext_list:
                manifest_items.append(ManifestItem(path=file, label=vol_to_label[get_anthology_vol_from_folder_name(folder.name)], group=get_anthology_group_name(file)))
    return manifest_items

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

    # file extension
    ap.add_argument("--file_ext_list", type=str, nargs="+", default=["mid", "midi", "musicxml", "xml"])

    # output controls
    ap.add_argument("--max_len_tokens", type=int, default=0,
                    help="0 => use default fixed max_len for windowed pretrain (6150). Must be divisible by 3.")
    ap.add_argument("--write_txt_debug", action="store_true",
                    help="Optional: dump train/valid/test.txt for inspection (not needed for training).")
    ap.add_argument("--write_dict_txt", action="store_true",
                    help="Optional: dump dict.txt with token counts (not needed if you only use SimpleMono.pkl).")

    args = ap.parse_args()

    ratios = (args.train_ratio, args.valid_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratios} sum={sum(ratios)}")
    
    manifest_items = build_manifest_items(Path(args.input_dir), file_ext_list=args.file_ext_list)

    build_from_manifest_items(
        manifest_items=manifest_items,
        output_dir=Path(args.output_dir),
        seed=args.seed,
        xml_group=args.xml_group,
        ratios=ratios,
        max_len_tokens=args.max_len_tokens,
    )


if __name__ == "__main__":
    main()

# Example:
# python .\preproc_anthology_vol_classification.py --input_dir "J:\ACADEMIC\JianpuDigitization2025\DATASET-release\v251218\lyrics-included" --output_dir ".\output\anthology_v251218_lyrics_included" --train_ratio 0.7 --valid_ratio 0.15 --test_ratio 0.15 --max_len_tokens 0 --file_ext_list "musicxml" "xml"


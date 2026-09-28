# simplemono_preproc/splitting.py
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from .utils import relpath_posix

from tqdm import tqdm


GroupKeyFn = Callable[[Path], str]


def make_group_key_fn(mode: str, input_dir: Path) -> GroupKeyFn:
    """
    mode:
      - file: each file is its own group (default pretrain behavior)
      - stem: group by filename without extension (avoid same-name leakage)
      - parent_stem: group by "parent_dir/stem" (often better than stem)
      - regex:<pattern>: use regex on the relative path string; group=first match; fallback to stem
    """
    mode = mode.strip()

    if mode == "file":
        return lambda p: relpath_posix(p, input_dir)

    if mode == "stem":
        return lambda p: p.stem

    if mode == "parent_stem":
        return lambda p: f"{p.parent.name}/{p.stem}"
    
    if mode == "anthology_style":
        # 去除括号部分，头部数字和下划线，删除空白符。
        get_anthology_name = lambda path: re.sub(r'（.*）$', '', re.sub(r'^\d+_', '', re.sub(r'[\s\u3000]', '', path.stem)))
        return lambda p: f"{p.parent.name}/{get_anthology_name(p)}"

    if mode.startswith("regex:"):
        pat = mode[len("regex:") :]
        rx = re.compile(pat)

        def _fn(p: Path) -> str:
            s = relpath_posix(p, input_dir)
            m = rx.search(s)
            if m:
                return m.group(1) if m.groups() else m.group(0)
            return p.stem

        return _fn

    raise ValueError(f"Unknown group mode: {mode}")


def split_by_groups(
    items: List[Path],
    group_key: GroupKeyFn,
    seed: int,
    ratios: Tuple[float, float, float],
) -> Dict[str, List[Path]]:
    """
    Split with leakage prevention: items with the same group_key never go to different splits.

    We aim to match ratios by *item count*, assigning groups sequentially after shuffle.
    """
    r_train, r_valid, r_test = ratios
    if abs((r_train + r_valid + r_test) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.0")

    groups: Dict[str, List[Path]] = {}
    for p in tqdm(items, desc="Splitting by groups", unit="files", total=len(items)):
        k = group_key(p)
        groups.setdefault(k, []).append(p)

    group_list = list(groups.items())

    rng = random.Random(seed)
    rng.shuffle(group_list)

    total = sum(len(v) for _, v in group_list)
    t_train = int(round(total * r_train))
    t_valid = int(round(total * r_valid))
    # rest -> test

    out = {"train": [], "valid": [], "test": []}
    cur = 0

    # fill train
    for k, gitems in group_list:
        if cur < t_train:
            out["train"].extend(gitems)
            cur += len(gitems)
        else:
            break

    # remaining groups
    remain = group_list[len({id(x) for x in out["train"]}) :]  # not reliable, ignore
    # simpler: re-walk group_list but track assigned groups
    assigned_groups = set()
    for p in out["train"]:
        assigned_groups.add(group_key(p))
    remaining_groups = [(k, v) for k, v in group_list if k not in assigned_groups]

    cur_valid = 0
    for k, gitems in remaining_groups:
        if cur_valid < t_valid:
            out["valid"].extend(gitems)
            cur_valid += len(gitems)
        else:
            out["test"].extend(gitems)

    # stable sort each split for reproducibility (optional)
    for sp in out:
        out[sp] = sorted(out[sp], key=lambda x: str(x))

    return out


def split_files(
    files: List[Path],
    input_dir: Path,
    seed: int,
    ratios: Tuple[float, float, float],
    group_mode: str = "file",
) -> Dict[str, List[Path]]:
    group_key = make_group_key_fn(group_mode, input_dir=input_dir)
    return split_by_groups(files, group_key=group_key, seed=seed, ratios=ratios)

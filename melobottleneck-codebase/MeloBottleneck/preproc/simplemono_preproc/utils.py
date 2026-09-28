# simplemono_preproc/utils.py
from __future__ import annotations

import hashlib
from pathlib import Path


def stable_int_hash(s: str, mod: int = 2**31 - 1) -> int:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % mod


def ensure_dir_empty(path: Path):
    if path.exists():
        raise FileExistsError(f"Output dir already exists: {path}")
    path.mkdir(parents=True, exist_ok=False)


def relpath_posix(p: Path, start: Path) -> str:
    """
    For metadata: always store POSIX-style paths (stable across OS).
    """
    return p.relative_to(start).as_posix()

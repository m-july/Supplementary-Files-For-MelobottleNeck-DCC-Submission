# simplemono_preproc/quantization.py
from __future__ import annotations

from .constants import (
    DUR_SAMPLES,
    DT_SAMPLES,
    MAX_DUR_CODE,
    DT_CODE_MIN,
    DT_CODE_MAX,
)

# ----------------------------
# Uniform mapping tables
# ----------------------------
# Duration: code <-> pos are identity within representable range.
# Encoding rule: dur_code = clip(dur_pos, 0, MAX_DUR_CODE)
POS_TO_DUR_CODE = list(range(DUR_SAMPLES))   # idx: dur_pos (0..MAX_DUR_CODE) -> dur_code
DUR_CODE_TO_POS = list(range(DUR_SAMPLES))   # idx: dur_code -> dur_pos

# DeltaTime: signed integer range [-DT_SAMPLES, DT_SAMPLES-1]
# We export mapping tables using an offset so we can index by (signed_value + offset).
DT_CODE_OFFSET = DT_SAMPLES  # so (dt_code_signed + offset) is in [0, 2*DT_SAMPLES-1]
DT_CODE_TO_POS = [DT_CODE_MIN + i for i in range(2 * DT_SAMPLES)]  # idx: dt_code+offset -> dt_pos
POS_TO_DT_CODE = DT_CODE_TO_POS.copy()  # idx: dt_pos+offset -> dt_code (identity)

def _clamp_int(x: int, lo: int, hi: int) -> int:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x

def encode_dur_pos(dur_pos: int) -> int:
    """
    Uniform duration encoding.

    Args:
        dur_pos: duration in pos units (non-negative int)

    Returns:
        dur_code in [0, DUR_SAMPLES-1]
    """
    if dur_pos < 0:
        raise ValueError("encode_dur_pos expects non-negative")
    return _clamp_int(int(dur_pos), 0, MAX_DUR_CODE)

def decode_dur_code(dur_code: int) -> int:
    """
    Uniform duration decoding.

    Args:
        dur_code: duration code

    Returns:
        duration in pos units (clipped)
    """
    return _clamp_int(int(dur_code), 0, MAX_DUR_CODE)

def encode_dt_pos(dt_pos: int) -> int:
    """
    Uniform delta-time encoding (signed).

    Args:
        dt_pos: delta-time in pos units (signed int)

    Returns:
        dt_code_signed in [-DT_SAMPLES, DT_SAMPLES-1]
    """
    return _clamp_int(int(dt_pos), DT_CODE_MIN, DT_CODE_MAX)

def decode_dt_code(dt_code_signed: int) -> int:
    """
    Uniform delta-time decoding (signed).

    Args:
        dt_code_signed: signed code

    Returns:
        delta-time in pos units (clipped)
    """
    return _clamp_int(int(dt_code_signed), DT_CODE_MIN, DT_CODE_MAX)

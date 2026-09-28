# simplemono_preproc/constants.py
from __future__ import annotations

# ----------------------------
# Quantization (uniform duration / delta-time)
# ----------------------------
# Time grid resolution: defines how many 'pos' units fit in one quarter note.
# All note start/end times are quantized to this grid.
POS_RESOLUTION = 12  # RESOLUTION: positions per quarter note (default=12)

# Uniform coding ranges (in pos units):
#   - duration_code in [0, DUR_SAMPLES-1]
#   - deltatime_code_signed in [-DT_SAMPLES, DT_SAMPLES-1]
DUR_SAMPLES = 96
DT_SAMPLES = 96

# Derived ranges (handy for clipping & for downstream config export)
MAX_DUR_CODE = DUR_SAMPLES - 1          # 95
MAX_DT_POS_CODE = DT_SAMPLES - 1        # 95
MAX_DT_NEG_MAG = DT_SAMPLES             # 96 (so we can represent -96)

DT_CODE_MIN = -DT_SAMPLES               # -96
DT_CODE_MAX = DT_SAMPLES - 1            # 95

TRUNC_POS = 2 ** 16     # maximum position value in pos units

MAX_PITCH = 127

# Filters (for MIDI and MusicXML only)
MIN_NOTES_PER_VOICE = 40
MIN_UNIQUE_PITCHES = 5

# Windowing
WINDOW_MAX_NOTES = 512
WINDOW_STEP_NOTES = 256
WINDOW_MIN_NOTES = 16

# File types
SUPPORTED_EXTS = {".mid", ".midi", ".xml", ".musicxml", ".mxl", ".krn"}

# Vocab specials (order matters)
SPECIAL_TOKENS = ("<pad>", "<unk>", "<s>", "</s>", "<mask>")

DEFAULT_SPLIT = (0.90, 0.05, 0.05)

# simplemono_preproc/vocab.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pickle

from .constants import (
    DUR_SAMPLES,
    DT_SAMPLES,
    MAX_PITCH,
    POS_RESOLUTION,
    SPECIAL_TOKENS,
    MAX_DUR_CODE,
    MAX_DT_POS_CODE,
    MAX_DT_NEG_MAG,
    DT_CODE_MIN,
    DT_CODE_MAX,
)


def build_simplemono_token_list() -> List[str]:
    tokens: List[str] = []
    tokens.extend(list(SPECIAL_TOKENS))

    # pitch
    for p in range(0, MAX_PITCH + 1):
        tokens.append(f"<0-{p}>")

    # duration codes: 0..(DUR_SAMPLES-1)
    for d in range(0, DUR_SAMPLES):
        tokens.append(f"<1-{d}>")

    # delta-time codes:
    #   positive: 0..(DT_SAMPLES-1)
    #   negative: !1..!DT_SAMPLES   (so signed range is [-DT_SAMPLES, DT_SAMPLES-1])
    for dt in range(0, DT_SAMPLES):
        tokens.append(f"<2-{dt}>")
    for mag in range(1, DT_SAMPLES + 1):
        tokens.append(f"<2-!{mag}>")

    return tokens


def load_vocab_from_dict_txt(dict_path: Path) -> Tuple[Dict[str, int], List[str]]:
    token2id: Dict[str, int] = {}
    id2token: List[str] = []
    with dict_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tok = line.split()[0]
            if tok in token2id:
                continue
            token2id[tok] = len(id2token)
            id2token.append(tok)
    return token2id, id2token


@dataclass
class SimpleMonoVocab:
    """
    Deterministic vocab.

    IDs (with defaults DUR_SAMPLES=96, DT_SAMPLES=96):
      0..4: <pad>, <unk>, <s>, </s>, <mask>                         (5)
      5..132: <0-0>.. <0-127>                                       (128)
      133..228: <1-0>.. <1-95>                                      (96)
      229..324: <2-0>.. <2-95>                                      (96)
      325..420: <2-!1>.. <2-!96>                                    (96)
    """
    token2id: Dict[str, int]
    id2token: List[str]

    max_dur_code: int
    max_dt_pos_code: int
    max_dt_neg_mag: int

    pitch_offset: int
    dur_offset: int
    dt_pos_offset: int
    dt_neg_offset: int

    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int
    mask_id: int

    @classmethod
    def build(cls) -> "SimpleMonoVocab":
        id2token = build_simplemono_token_list()
        token2id = {t: i for i, t in enumerate(id2token)}

        pitch_offset = len(SPECIAL_TOKENS)
        dur_offset = pitch_offset + (MAX_PITCH + 1)
        dt_pos_offset = dur_offset + DUR_SAMPLES
        dt_neg_offset = dt_pos_offset + DT_SAMPLES

        return cls(
            token2id=token2id,
            id2token=id2token,
            max_dur_code=MAX_DUR_CODE,
            max_dt_pos_code=MAX_DT_POS_CODE,
            max_dt_neg_mag=MAX_DT_NEG_MAG,
            pitch_offset=pitch_offset,
            dur_offset=dur_offset,
            dt_pos_offset=dt_pos_offset,
            dt_neg_offset=dt_neg_offset,
            pad_id=token2id["<pad>"],
            unk_id=token2id["<unk>"],
            bos_id=token2id["<s>"],
            eos_id=token2id["</s>"],
            mask_id=token2id["<mask>"],
        )

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    def pitch_id(self, pitch: int) -> int:
        if pitch < 0 or pitch > MAX_PITCH:
            return self.unk_id
        return self.pitch_offset + pitch

    def dur_id(self, dur_code: int) -> int:
        if dur_code < 0:
            return self.unk_id
        if dur_code > self.max_dur_code:
            dur_code = self.max_dur_code
        return self.dur_offset + dur_code

    def dt_id(self, dt_code_signed: int) -> int:
        # positive (including 0): 0..max_dt_pos_code
        if dt_code_signed >= 0:
            dt = dt_code_signed
            if dt > self.max_dt_pos_code:
                dt = self.max_dt_pos_code
            return self.dt_pos_offset + dt

        # negative: -1..-max_dt_neg_mag  => token is <2-!mag>
        mag = abs(dt_code_signed)
        if mag <= 0:
            return self.dt_pos_offset + 0
        if mag > self.max_dt_neg_mag:
            mag = self.max_dt_neg_mag
        return self.dt_neg_offset + (mag - 1)

    def write_dict_txt(self, out_path: Path, counts_by_id: Optional[np.ndarray] = None):
        if counts_by_id is None:
            counts_by_id = np.zeros((self.vocab_size,), dtype=np.int64)
        if len(counts_by_id) != self.vocab_size:
            raise ValueError("counts_by_id length mismatch")

        with out_path.open("w", encoding="utf-8") as f:
            for i, tok in enumerate(self.id2token):
                f.write(f"{tok} {int(counts_by_id[i])}\n")

    def build_simplemono_pkl_object(self) -> Dict:
        SPECIAL = set(SPECIAL_TOKENS)

        token2id = dict(self.token2id)
        id2token = {i: t for i, t in enumerate(self.id2token)}

        event2word = {
            "Global": {},
            "Special": {},
            "Pitch": {},
            "Duration": {},
            "DeltaTime": {},
        }
        word2event = {k: {} for k in event2word.keys()}

        for idx, tok in enumerate(self.id2token):
            event2word["Global"][tok] = idx
            word2event["Global"][idx] = tok

            if tok in SPECIAL:
                for cls_name in ["Special", "Pitch", "Duration", "DeltaTime"]:
                    event2word[cls_name][tok] = idx
                    word2event[cls_name][idx] = tok
            elif tok.startswith("<0-"):
                event2word["Pitch"][tok] = idx
                word2event["Pitch"][idx] = tok
            elif tok.startswith("<1-"):
                event2word["Duration"][tok] = idx
                word2event["Duration"][idx] = tok
            elif tok.startswith("<2-"):
                event2word["DeltaTime"][tok] = idx
                word2event["DeltaTime"][idx] = tok
            else:
                raise ValueError(f"Unrecognized token format: {tok}")

        # Expose quantization mapping tables in the pkl object
        from .quantization import (
            POS_TO_DUR_CODE,
            DUR_CODE_TO_POS,
            DT_CODE_OFFSET,
            DT_CODE_TO_POS,
            POS_TO_DT_CODE,
        )

        return {
            "token2id": token2id,
            "id2token": id2token,
            "event2word": event2word,
            "word2event": word2event,

            # ----------------------------
            # NEW: uniform quantization tables
            # ----------------------------
            # Duration (pos >= 0):
            "duration_pos_to_code": POS_TO_DUR_CODE.copy(),   # idx: pos (0..MAX_DUR_CODE) -> code
            "duration_code_to_pos": DUR_CODE_TO_POS.copy(),   # idx: code -> pos

            # DeltaTime (signed):
            # index with (signed_value + deltatime_code_offset)
            "deltatime_code_offset": DT_CODE_OFFSET,
            "deltatime_code_to_pos": DT_CODE_TO_POS.copy(),   # idx: code+offset -> pos
            "deltatime_pos_to_code": POS_TO_DT_CODE.copy(),   # idx: pos+offset  -> code

            # Optional: config for downstream usage
            "quantization_config": {
                "pos_resolution": POS_RESOLUTION,
                "dur_samples": DUR_SAMPLES,
                "dt_samples": DT_SAMPLES,
                "duration_code_min": 0,
                "duration_code_max": MAX_DUR_CODE,
                "deltatime_code_min": DT_CODE_MIN,
                "deltatime_code_max": DT_CODE_MAX,
                "deltatime_code_offset": DT_CODE_OFFSET,
                "max_dt_pos_code": MAX_DT_POS_CODE,
                "max_dt_neg_mag": MAX_DT_NEG_MAG,
            },
        }

    def save_simplemono_pkl(self, out_path: Path):
        obj = self.build_simplemono_pkl_object()
        with out_path.open("wb") as f:
            pickle.dump(obj, f)

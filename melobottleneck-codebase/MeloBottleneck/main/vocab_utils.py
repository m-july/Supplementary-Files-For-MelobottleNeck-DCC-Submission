from __future__ import annotations
from dataclasses import dataclass
import pickle
import numpy as np

from .models.bart import MusicBartVocabConfig


@dataclass(frozen=True)
class VocabInfo:
    n_pitch: int
    n_duration: int
    n_dt: int

    pad_id: int
    bos_id: int          # <s>
    eos_id: int          # </s>
    mask_id: int         # <mask>

    # 全局 ID -> 局部 ID 的映射表
    # global2local_pitch[global_id] = local_pitch_id
    global2local_pitch: np.ndarray    # shape: [max_global_id + 1]
    global2local_duration: np.ndarray
    global2local_dt: np.ndarray
    # ===== NEW: quantization tables from SimpleMono.pkl =====
    special_n: int
    duration_code_to_pos: np.ndarray      # [dur_code] -> pos
    duration_pos_to_code: np.ndarray      # [pos] -> dur_code
    deltatime_code_offset: int            # offset for signed indexing
    deltatime_code_to_pos: np.ndarray     # [dt_code_signed + offset] -> pos
    deltatime_pos_to_code: np.ndarray     # [pos + offset] -> dt_code_signed

    def to_music_bart_vocab_config(self) -> MusicBartVocabConfig:
        return MusicBartVocabConfig(
            n_pitch=self.n_pitch,
            n_duration=self.n_duration,
            n_dt=self.n_dt,
            pad_id=self.pad_id,
            bos_id=self.bos_id,
            eos_id=self.eos_id,
            mask_id=self.mask_id,
            special_n=self.special_n
        )


def load_vocab_info(pkl_path: str) -> VocabInfo:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    event2word = obj["event2word"]
    special = event2word["Special"]
    special_n = len(event2word["Special"])
    # ===== NEW: load quantization tables =====
    duration_code_to_pos = np.asarray(obj["duration_code_to_pos"], dtype=np.int32)
    duration_pos_to_code = np.asarray(obj["duration_pos_to_code"], dtype=np.int32)
    deltatime_code_offset = int(obj["deltatime_code_offset"])
    deltatime_code_to_pos = np.asarray(obj["deltatime_code_to_pos"], dtype=np.int32)
    deltatime_pos_to_code = np.asarray(obj["deltatime_pos_to_code"], dtype=np.int32)
    # (optional sanity checks)
    if deltatime_code_to_pos.shape[0] != 2 * deltatime_code_offset:
        raise ValueError("Bad deltatime_code_to_pos length in pkl.")
    if deltatime_pos_to_code.shape[0] != 2 * deltatime_code_offset:
        raise ValueError("Bad deltatime_pos_to_code length in pkl.")

    # 这里默认 Special 里的 id 与各 attribute vocab 的 special id 兼容（通常 <pad>=0 等）
    pad_id = int(special["<pad>"])
    bos_id = int(special["<s>"])
    eos_id = int(special["</s>"])
    mask_id = int(special["<mask>"])

    n_pitch = len(event2word["Pitch"])
    n_duration = len(event2word["Duration"])
    n_dt = len(event2word["DeltaTime"])

    # 构建全局 ID -> 局部 ID 的映射表
    # 注意：event2word 中的值是全局 ID，不是局部 ID
    # 我们需要重新编号为 0, 1, 2, ... len-1
    token2id = obj["token2id"]

    # 找到最大的全局 ID
    max_global_id = max(token2id.values())

    # 初始化映射表，默认值设为 -1（表示无效映射）
    global2local_pitch = np.full(max_global_id + 1, -1, dtype=np.int32)
    global2local_duration = np.full(max_global_id + 1, -1, dtype=np.int32)
    global2local_dt = np.full(max_global_id + 1, -1, dtype=np.int32)

    # 填充映射表
    # 我们需要为每个属性类别创建从 0 开始的局部 ID
    # 方法：按全局 ID 排序，然后依次分配局部 ID 0, 1, 2, ...

    # Pitch: 收集所有 token 的全局 ID，排序后分配局部 ID
    pitch_global_ids = [token2id[token_str] for token_str in event2word["Pitch"].keys()]
    pitch_global_ids.sort()
    for local_id, global_id in enumerate(pitch_global_ids):
        global2local_pitch[global_id] = local_id

    # Duration: 同样处理
    duration_global_ids = [token2id[token_str] for token_str in event2word["Duration"].keys()]
    duration_global_ids.sort()
    for local_id, global_id in enumerate(duration_global_ids):
        global2local_duration[global_id] = local_id

    # DeltaTime: 同样处理
    dt_global_ids = [token2id[token_str] for token_str in event2word["DeltaTime"].keys()]
    dt_global_ids.sort()
    for local_id, global_id in enumerate(dt_global_ids):
        global2local_dt[global_id] = local_id

    return VocabInfo(
        n_pitch=n_pitch,
        n_duration=n_duration,
        n_dt=n_dt,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        mask_id=mask_id,
        global2local_pitch=global2local_pitch,
        global2local_duration=global2local_duration,
        global2local_dt=global2local_dt,
        special_n=special_n,
        duration_code_to_pos=duration_code_to_pos,
        duration_pos_to_code=duration_pos_to_code,
        deltatime_code_offset=deltatime_code_offset,
        deltatime_code_to_pos=deltatime_code_to_pos,
        deltatime_pos_to_code=deltatime_pos_to_code,
    )

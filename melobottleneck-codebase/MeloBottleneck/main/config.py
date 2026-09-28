from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import torch


@dataclass
class BartDenoiseConfig:
    """
    BART-style 音乐序列损毁配置。

    约定：
    - 输入 token: LongTensor, shape [B, L, 3]，最后一维是 (pitch, duration, dt)
    - attention_mask: Bool/LongTensor, shape [B, L]，1/True 表示有效 token，0/False 表示 padding

    粒度记法：
      - "token@attribute"       : Token@Attribute
      - "token@token"           : Token@Token
      - "n-token@attribute"     : n-Token@Attribute
      - "n-token@token"         : n-Token@Token
    """

    # === 序列基本信息 ===
    num_attributes: int = 3          # 你的复合 token 维度，固定为 3
    pad_token_id: int = 0           # [PAD] 的 id（pitch/duration/dt 都会被设为这个值）
    mask_token_id: int = 1          # [MASK] 的 id（pitch/duration/dt 都会被设为这个值）

    # === 1. Token Masking & Text Infilling ===
    enable_masking: bool = True
    # 有多少比例的有效 token 被「选中」，用于做 masking / span infilling
    masking_noise_density: float = 0.333
    # n-Token 时 span 长度的 Poisson(lambda) 期望，可看作「平均 span 长度」
    masking_span_lambda: float = 3.0

    # 四种粒度的选择概率（按 batch 内每个样本独立采样一种粒度）
    masking_granularity_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "token@attribute": 0.25,
            "token@token": 0.25,
            "n-token@attribute": 0.25,
            "n-token@token": 0.25,
        }
    )

    # 对于 n-Token@Token 粒度：
    # True  : 按 BART 文本填空方式，把一个 span 压缩为单个 [MASK] token
    # False : 只是在 span 内每个 token 全部改成 [MASK]，序列长度不变
    masking_compress_n_token_token_level: bool = True

    # === 2. Token Deletion ===
    enable_deletion: bool = True
    # 删除概率 = 有多少比例的有效 token 被删除（平均意义上）
    deletion_prob: float = 0.2
    deletion_span_lambda: float = 3.0

    # 删除任务只允许 Token@Token / n-Token@Token 两种粒度
    deletion_granularity_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "token@token": 0.5,
            "n-token@token": 0.5,
        }
    )

    # === 3. Document Rotation ===
    enable_rotation: bool = True
    # 每个样本被做一次 rotation 的概率
    rotation_prob: float = 0.5

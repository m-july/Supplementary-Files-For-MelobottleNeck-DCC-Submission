from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class MultiAttrLogits:
    pitch: torch.Tensor      # [B, T, n_pitch] 或 [B, n_pitch]
    duration: torch.Tensor   # [B, T, n_duration] 或 [B, n_duration]
    dt: torch.Tensor         # [B, T, n_dt] 或 [B, n_dt]


@dataclass
class MusicSeq2SeqLMOutput:
    loss: Optional[torch.Tensor]
    loss_dict: Optional[Dict[str, torch.Tensor]]
    logits: MultiAttrLogits
    decoder_last_hidden_state: Optional[torch.Tensor] = None
    encoder_last_hidden_state: Optional[torch.Tensor] = None
    past_key_values: Optional[Tuple] = None


@dataclass
class MusicTokenClassificationOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor                # [B, L, C]
    hidden_states: Optional[torch.Tensor] = None


@dataclass
class MusicSequenceClassificationOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor                # [B, C]
    pooled: Optional[torch.Tensor] = None

# main/models/skeleton/baseline_keep_encoder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ...nn_modules import MusicBartBackbone

from ...nn_modules.heads import SkeletonTokenScoreHead


@dataclass(frozen=True)
class MusicSkeletonKeepBaselineConfig:
    head_hidden: int = 256      # 0 => 直接线性
    head_dropout: float = 0.1


class MusicSkeletonKeepBaseline(nn.Module):
    """
    Encoder-only baseline:
      x_orn -> BartEncoder -> per-token keep logits (binary)
    keep=1: original (pi>=0)
    keep=0: inserted (pi==-1)
    """
    def __init__(self, *, backbone: MusicBartBackbone, cfg: MusicSkeletonKeepBaselineConfig):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        d = int(backbone.cfg.backbone.d_model)
        h = int(cfg.head_hidden)

        self.keep_head = SkeletonTokenScoreHead(
            d_model=d,
            hidden=h,
            dropout=float(cfg.head_dropout),
        )

    @property
    def pad_id(self) -> int:
        return int(self.backbone.pad_id)

    @property
    def special_n(self) -> int:
        return int(self.backbone.cfg.vocab.special_n)

    def forward(
        self,
        *,
        src_tokens: torch.LongTensor,                 # [B,L,3]
        src_attention_mask: Optional[torch.Tensor] = None,  # [B,L] bool/0-1
    ) -> torch.Tensor:
        if src_attention_mask is None:
            src_attention_mask = (src_tokens[..., 0] != self.pad_id)

        src_embeds = self.backbone.embed(src_tokens)  # [B,L,D]
        enc = self.backbone.encode(src_embeds=src_embeds, src_attention_mask=src_attention_mask)
        h = enc.last_hidden_state                     # [B,L,D]

        keep_logits = self.keep_head(h)               # [B,L]
        return keep_logits

    @torch.no_grad()
    def keep_prob(
        self,
        *,
        src_tokens: torch.LongTensor,
        src_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(src_tokens=src_tokens, src_attention_mask=src_attention_mask)
        return torch.sigmoid(logits.to(torch.float32))  # [B,L] float32
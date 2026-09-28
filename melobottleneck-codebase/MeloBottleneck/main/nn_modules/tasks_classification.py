from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .backbone import MusicBartBackbone
from .heads import TokenClassificationHead, SequenceClassificationHead
from ..outputs import MusicTokenClassificationOutput, MusicSequenceClassificationOutput

from ..utils import infer_attention_mask_from_tokens, right_shift_tokens


class MusicBartForTokenClassification(nn.Module):
    """
    token 级分类：
      - encoder_only=True：更轻更稳（你笔记里推荐）
      - encoder_only=False：full seq2seq，用 decoder hidden 做分类（对齐 PianoBART 方案）
    """

    def __init__(
        self,
        backbone: MusicBartBackbone,
        *,
        num_classes: int,
        encoder_only: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.encoder_only = encoder_only
        self.head = TokenClassificationHead(backbone.cfg.backbone.d_model, num_classes, dropout=dropout)

    def forward(
        self,
        *,
        tokens: torch.LongTensor,                      # [B,L,3]
        attention_mask: Optional[torch.Tensor] = None, # [B,L]
        labels: Optional[torch.LongTensor] = None,     # [B,L]
        ignore_index: int = -100,
        return_hidden: bool = False,
    ) -> MusicTokenClassificationOutput:
        if attention_mask is None:
            attention_mask = infer_attention_mask_from_tokens(tokens, pad_id=self.backbone.pad_id)

        if self.encoder_only:
            enc = self.backbone.encode(src_tokens=tokens, src_attention_mask=attention_mask)
            hidden = enc.last_hidden_state  # [B,L,D]
        else:
            # full seq2seq: src=tokens, decoder_input=shift_right(tokens)
            dec_in = right_shift_tokens(tokens, bos_id=self.backbone.bos_id, pad_id=self.backbone.pad_id)
            dec_mask = infer_attention_mask_from_tokens(dec_in, pad_id=self.backbone.pad_id)

            out = self.backbone.bart(
                inputs_embeds=self.backbone.embed(tokens),
                attention_mask=attention_mask.to(dtype=torch.long),
                decoder_inputs_embeds=self.backbone.embed(dec_in),
                decoder_attention_mask=dec_mask.to(dtype=torch.long),
                use_cache=False,
                return_dict=True,
            )
            hidden = out.last_hidden_state  # [B,L,D]

        logits = self.head(hidden)  # [B,L,C]

        loss = None
        if labels is not None:
            # padding 位置一般置 ignore_index；你也可以在外部先处理 labels
            labels2 = labels.masked_fill(attention_mask.to(dtype=torch.bool) == 0, ignore_index)
            loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels2.view(-1))

        return MusicTokenClassificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden if return_hidden else None,
        )


class MusicBartForSequenceClassification(nn.Module):
    """
    序列级分类：默认 encoder-only + pooling（AWAL / mean）。
    """

    def __init__(
        self,
        backbone: MusicBartBackbone,
        *,
        num_classes: int,
        dropout: float = 0.1,
        pooling: str = "awal",
    ):
        super().__init__()
        self.backbone = backbone
        self.head = SequenceClassificationHead(
            d_model=backbone.cfg.backbone.d_model,
            num_classes=num_classes,
            dropout=dropout,
            pooling=pooling,
        )

    def forward(
        self,
        *,
        tokens: torch.LongTensor,                       # [B,L,3]
        attention_mask: Optional[torch.Tensor] = None,  # [B,L]
        labels: Optional[torch.LongTensor] = None,      # [B]
        return_pooled: bool = False,
    ) -> MusicSequenceClassificationOutput:
        if attention_mask is None:
            attention_mask = infer_attention_mask_from_tokens(tokens, pad_id=self.backbone.pad_id)

        enc = self.backbone.encode(src_tokens=tokens, src_attention_mask=attention_mask)
        hidden = enc.last_hidden_state  # [B,L,D]

        logits, pooled = self.head(hidden, attention_mask)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return MusicSequenceClassificationOutput(
            loss=loss,
            logits=logits,
            pooled=pooled if return_pooled else None,
        )

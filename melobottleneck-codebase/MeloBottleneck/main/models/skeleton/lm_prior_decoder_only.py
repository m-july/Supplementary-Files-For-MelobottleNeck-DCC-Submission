# models/skeleton/lm_prior_decoder_only.py
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from ...nn_modules import MusicBartBackbone
from ...nn_modules.heads import MusicBartMultiAttrLMHead
from ...outputs import MultiAttrLogits
from ...utils import infer_attention_mask_from_tokens, right_shift_tokens


class MusicBartDecoderOnlyLM(nn.Module):
    """
    一个“decoder-only 的多属性 LM”：
      logits_t = p_LM(z_t | z_<t)

    这里我们直接调用 HF BartModel 的 decoder（encoder_hidden_states=None -> 不走 cross-attn）。
    """

    def __init__(self, backbone: MusicBartBackbone, *, freeze: bool = True, use_bias_in_lm_head: bool = True):
        super().__init__()
        self.backbone = backbone
        self.lm_head = backbone.lm_head

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    @property
    def pad_id(self) -> int:
        return self.backbone.pad_id

    @property
    def bos_id(self) -> int:
        return self.backbone.bos_id

    def logits(
        self,
        *,
        tokens: torch.LongTensor,                     # [B,T,3]
        attention_mask: Optional[torch.Tensor] = None,# [B,T]
        decoder_input_tokens: Optional[torch.LongTensor] = None,
    ) -> MultiAttrLogits:
        if attention_mask is None:
            attention_mask = infer_attention_mask_from_tokens(tokens, pad_id=self.pad_id)

        if decoder_input_tokens is None:
            decoder_input_tokens = right_shift_tokens(tokens, bos_id=self.bos_id, pad_id=self.pad_id)

        dec_embeds = self.backbone.embed(decoder_input_tokens)

        dec_out = self.backbone.bart.decoder(
            inputs_embeds=dec_embeds,
            attention_mask=attention_mask.to(dtype=torch.long),
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            return_dict=True,
        )
        h = dec_out.last_hidden_state  # [B,T,D]
        p, d, dt = self.lm_head(h)
        return MultiAttrLogits(pitch=p, duration=d, dt=dt)

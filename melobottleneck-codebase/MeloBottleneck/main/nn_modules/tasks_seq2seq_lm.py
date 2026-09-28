from __future__ import annotations
from typing import Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn

from ..nn_modules import MusicBartBackbone
from ..outputs import MultiAttrLogits, MusicSeq2SeqLMOutput
from ..utils import (
    infer_attention_mask_from_tokens,
    right_shift_tokens,
    mask_labels_with_ignore_index,
)
from .heads import MusicBartMultiAttrLMHead

from ..nn_funcs.token_ce_loss import multi_attribute_ce_loss


NextTokenSelector = Callable[[MultiAttrLogits], torch.LongTensor]
# 期望返回: next_tokens [B, A]，A=3 (pitch, duration, dt)


def greedy_selector(logits: MultiAttrLogits) -> torch.LongTensor:
    # logits.*: [B, V]
    pitch = logits.pitch.argmax(dim=-1)
    dur = logits.duration.argmax(dim=-1)
    dt = logits.dt.argmax(dim=-1)
    return torch.stack([pitch, dur, dt], dim=-1)  # [B,3]


class MusicBartForSeq2SeqLM(nn.Module):
    """
    通用 Seq2Seq LM wrapper：
      - 预训练 denoising：src=corrupted, tgt=clean
      - prompt->continuation：src=prompt, tgt=continuation
    都是同一个 forward。

    关键点：
      - decoder_input_tokens 可显式传入（支持 scheduled sampling / 特殊 decoder prompt 等）
      - mask 可不传，自动从 tokens 推断（用 tokens[...,0] != pad_id）
    """

    def __init__(self, backbone: MusicBartBackbone, *, use_bias_in_lm_head: bool = True):
        super().__init__()
        self.backbone = backbone
        self.lm_head = backbone.lm_head

    @property
    def pad_id(self) -> int:
        return self.backbone.pad_id

    @property
    def bos_id(self) -> int:
        return self.backbone.bos_id

    @property
    def eos_id(self) -> int:
        return self.backbone.eos_id

    def forward(
        self,
        *,
        src_tokens: torch.LongTensor,                      # [B,S,3]
        tgt_tokens: torch.LongTensor,                      # [B,T,3]
        src_attention_mask: Optional[torch.Tensor] = None, # [B,S]
        tgt_attention_mask: Optional[torch.Tensor] = None, # [B,T] (for loss masking on labels)
        decoder_input_tokens: Optional[torch.LongTensor] = None,   # [B,T,3]
        decoder_attention_mask: Optional[torch.Tensor] = None,     # [B,T]
        attr_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ignore_index: int = -100,
        return_hidden: bool = False,
    ) -> MusicSeq2SeqLMOutput:
        if src_attention_mask is None:
            src_attention_mask = infer_attention_mask_from_tokens(src_tokens, pad_id=self.pad_id)
        if tgt_attention_mask is None:
            tgt_attention_mask = infer_attention_mask_from_tokens(tgt_tokens, pad_id=self.pad_id)

        if decoder_input_tokens is None:
            decoder_input_tokens = right_shift_tokens(tgt_tokens, bos_id=self.bos_id, pad_id=self.pad_id)

        if decoder_attention_mask is None:
            decoder_attention_mask = infer_attention_mask_from_tokens(decoder_input_tokens, pad_id=self.pad_id)

        src_embeds = self.backbone.embed(src_tokens)
        dec_embeds = self.backbone.embed(decoder_input_tokens)

        out = self.backbone.bart(
            inputs_embeds=src_embeds,
            attention_mask=src_attention_mask.to(dtype=torch.long),
            decoder_inputs_embeds=dec_embeds,
            decoder_attention_mask=decoder_attention_mask.to(dtype=torch.long),
            use_cache=False,
            return_dict=True,
        )

        h_dec = out.last_hidden_state  # [B,T,D]
        pitch_logits, duration_logits, dt_logits = self.lm_head(h_dec)

        logits = MultiAttrLogits(
            pitch=pitch_logits,
            duration=duration_logits,
            dt=dt_logits,
        )

        labels = mask_labels_with_ignore_index(tgt_tokens.clone(), tgt_attention_mask, ignore_index=ignore_index)

        total_loss, loss_dict = multi_attribute_ce_loss(
            labels=labels,
            pitch_logits=pitch_logits,
            duration_logits=duration_logits,
            dt_logits=dt_logits,
            attr_weights=attr_weights,
            ignore_index=ignore_index,
            reduction="mean",
        )

        return MusicSeq2SeqLMOutput(
            loss=total_loss,
            loss_dict=loss_dict,
            logits=logits,
            decoder_last_hidden_state=h_dec if return_hidden else None,
            encoder_last_hidden_state=out.encoder_last_hidden_state if return_hidden else None,
            past_key_values=None,
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        src_tokens: torch.LongTensor,                          # [B,S,3]
        src_attention_mask: Optional[torch.Tensor] = None,     # [B,S]
        max_new_tokens: int = 256,
        selector: NextTokenSelector = greedy_selector,
        stop_on_eos: bool = True,
        eos_attr_idx: int = 0,  # 默认用 pitch 维判断 EOS（你也可以改成“3维全等”）
    ) -> torch.LongTensor:
        """
        一个“最小可用”的多属性自回归生成例程（支持 KV cache）。
        你之后可以替换 selector 实现 top-k/top-p/温度等策略，或加入约束采样。
        """
        device = src_tokens.device
        bsz = src_tokens.size(0)

        if src_attention_mask is None:
            src_attention_mask = infer_attention_mask_from_tokens(src_tokens, pad_id=self.pad_id)

        # encode once
        enc_out = self.backbone.encode(src_tokens=src_tokens, src_attention_mask=src_attention_mask)

        # init decoder with BOS
        cur = torch.full((bsz, 1, 3), fill_value=self.bos_id, dtype=torch.long, device=device)

        past = None
        finished = torch.zeros((bsz,), dtype=torch.bool, device=device)
        generated = []

        for _ in range(max_new_tokens):
            # only feed last token embedding (KV cache)
            step_tokens = cur[:, -1:, :]  # [B,1,3]
            step_embeds = self.backbone.embed(step_tokens)

            out = self.backbone.decode(
                decoder_embeds=step_embeds,
                encoder_outputs=enc_out,
                encoder_attention_mask=src_attention_mask,
                use_cache=True,
                past_key_values=past,
            )

            past = out.past_key_values
            h_t = out.last_hidden_state[:, -1, :]  # [B,D]

            p, d, dt = self.lm_head(h_t)  # each [B,V]
            next_tok = selector(MultiAttrLogits(pitch=p, duration=d, dt=dt))  # [B,3]
            generated.append(next_tok)

            if stop_on_eos:
                finished = finished | (next_tok[:, eos_attr_idx] == self.eos_id)
                if torch.all(finished):
                    break

            cur = torch.cat([cur, next_tok[:, None, :]], dim=1)

        if len(generated) == 0:
            return cur[:, 1:, :]  # empty
        return torch.stack(generated, dim=1)  # [B, T_new, 3]

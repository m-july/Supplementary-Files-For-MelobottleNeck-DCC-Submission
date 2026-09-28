# models/skeleton/reconstructor_from_embeds.py
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ...nn_modules import MusicBartBackbone
from ...nn_modules.heads import MusicBartMultiAttrLMHead
from ...outputs import MultiAttrLogits, MusicSeq2SeqLMOutput
from ...utils import (
    infer_attention_mask_from_tokens,
    right_shift_tokens,
    mask_labels_with_ignore_index,
)
from ...nn_funcs.token_ce_loss_weighted import multi_attribute_ce_loss_weighted


class MusicBartSeq2SeqReconstructorFromEmbeds(nn.Module):
    """
    Reconstructor R:
      src = z_embeds（阴阳embedding序列）
      tgt = x_tokens（teacher forcing）
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
    
    @property
    def mask_id(self) -> int:
        return int(self.backbone.cfg.vocab.mask_id)
    
    @property
    def special_n(self) -> int:
        return int(self.backbone.cfg.vocab.special_n)

    def _sample_ids(
        self,
        logits: torch.Tensor,          # [B,T,V]
        temperature: float,
        use_greedy: bool,
    ) -> torch.LongTensor:
        if use_greedy:
            return logits.argmax(dim=-1)

        temp = max(float(temperature), 1e-6)
        probs = torch.softmax(logits.float() / temp, dim=-1)   # float32 for stability
        B, T, V = probs.shape
        ids = torch.multinomial(probs.reshape(B * T, V), num_samples=1).reshape(B, T)
        return ids

    def _sample_multi_attr_tokens(
        self,
        *,
        pitch_logits: torch.Tensor,    # [B,T,Vp]
        dur_logits: torch.Tensor,      # [B,T,Vd]
        dt_logits: torch.Tensor,       # [B,T,Vt]
        temperature: float,
        use_greedy: bool,
        sync_special: bool,
    ) -> torch.LongTensor:
        pitch_ids = self._sample_ids(pitch_logits, temperature, use_greedy)  # [B,T]

        if (not sync_special) or (self.special_n <= 0):
            dur_ids = self._sample_ids(dur_logits, temperature, use_greedy)
            dt_ids = self._sample_ids(dt_logits, temperature, use_greedy)
            return torch.stack([pitch_ids, dur_ids, dt_ids], dim=-1)

        sN = self.special_n
        is_special = pitch_ids < sN      # [B,T]
        normal = ~is_special

        # 对 pitch 是正常音符的位置，dur/dt 不允许采样 special
        if normal.any():
            dur_logits = dur_logits.clone()
            dt_logits = dt_logits.clone()
            dur_logits[..., :sN] = dur_logits[..., :sN].masked_fill(normal.unsqueeze(-1), float("-inf"))
            dt_logits[..., :sN]  = dt_logits[..., :sN].masked_fill(normal.unsqueeze(-1), float("-inf"))

        dur_ids = self._sample_ids(dur_logits, temperature, use_greedy)
        dt_ids  = self._sample_ids(dt_logits, temperature, use_greedy)

        # pitch 是 special 的位置：三列同步成同一个 special id
        dur_ids = torch.where(is_special, pitch_ids, dur_ids)
        dt_ids  = torch.where(is_special, pitch_ids, dt_ids)

        return torch.stack([pitch_ids, dur_ids, dt_ids], dim=-1)

    def forward(
        self,
        *,
        src_embeds: torch.Tensor,                          # [B,S,D]
        src_attention_mask: torch.Tensor,                  # [B,S]
        tgt_tokens: torch.LongTensor,                      # [B,T,3]
        tgt_attention_mask: Optional[torch.Tensor] = None, # [B,T]
        decoder_input_tokens: Optional[torch.LongTensor] = None,  # [B,T,3]
        decoder_attention_mask: Optional[torch.Tensor] = None,     # [B,T]
        attr_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        token_weights: Optional[torch.Tensor] = None,       # [B,T]
        ignore_index: int = -100,
        return_hidden: bool = False,
        scheduled_sampling_prob: float = 0.0,
        scheduled_sampling_temperature: float = 1.0,
        scheduled_sampling_use_greedy: bool = False,
        scheduled_sampling_sync_special: bool = True,
        # ---- NEW ----
        decoder_input_mask_prob: float = 0.0,
        decoder_input_mask_keep_special: bool = True,
    ) -> MusicSeq2SeqLMOutput:
        if tgt_attention_mask is None:
            tgt_attention_mask = infer_attention_mask_from_tokens(tgt_tokens, pad_id=self.pad_id)

        if tgt_attention_mask is None:
            tgt_attention_mask = infer_attention_mask_from_tokens(tgt_tokens, pad_id=self.pad_id)

        # ---- build TF decoder inputs first ----
        user_provided_decoder_inputs = (decoder_input_tokens is not None)

        if decoder_input_tokens is None:
            decoder_input_tokens_tf = right_shift_tokens(tgt_tokens, bos_id=self.bos_id, pad_id=self.pad_id)
        else:
            decoder_input_tokens_tf = decoder_input_tokens

        if decoder_attention_mask is None:
            # 重要：mask 用 TF 的，不要用被替换后的 tokens 推
            decoder_attention_mask = infer_attention_mask_from_tokens(decoder_input_tokens_tf, pad_id=self.pad_id)

        decoder_input_tokens_final = decoder_input_tokens_tf

        # ---- scheduled sampling (only when using default TF inputs) ----
        p = float(scheduled_sampling_prob)
        if (not user_provided_decoder_inputs) and self.training and (p > 1e-6):
            p = max(0.0, min(1.0, p))
            device = tgt_tokens.device

            with torch.no_grad():
                # 1) TF forward to get logits for sampling
                dec_embeds_tf = self.backbone.embed(decoder_input_tokens_tf)

                out_tf = self.backbone.bart(
                    inputs_embeds=src_embeds,
                    attention_mask=src_attention_mask.to(dtype=torch.long),
                    decoder_inputs_embeds=dec_embeds_tf,
                    decoder_attention_mask=decoder_attention_mask.to(dtype=torch.long),
                    use_cache=False,
                    return_dict=True,
                )
                h_tf = out_tf.last_hidden_state  # [B,T,D]
                pitch_tf, dur_tf, dt_tf = self.lm_head(h_tf)

                # 2) sample predicted tokens (x_hat_t)
                pred_tokens = self._sample_multi_attr_tokens(
                    pitch_logits=pitch_tf,
                    dur_logits=dur_tf,
                    dt_logits=dt_tf,
                    temperature=scheduled_sampling_temperature,
                    use_greedy=scheduled_sampling_use_greedy,
                    sync_special=scheduled_sampling_sync_special,
                )  # [B,T,3]

            # 3) shift predictions to build "prev token" inputs (x_hat_{t-1})
            pred_in = right_shift_tokens(pred_tokens, bos_id=self.bos_id, pad_id=self.pad_id)  # [B,T,3]

            # 4) decide which positions to replace
            B, T, _ = decoder_input_tokens_tf.shape
            replace = (torch.rand((B, T), device=device) < p) & decoder_attention_mask.to(torch.bool)
            replace[:, 0] = False  # BOS 永远不替换

            decoder_input_tokens_final = torch.where(
                replace.unsqueeze(-1),
                pred_in,
                decoder_input_tokens_tf,
            )

        # ---- NEW: decoder-input mask dropout (word dropout) ----
        mp = float(decoder_input_mask_prob)
        if (not user_provided_decoder_inputs) and self.training and (mp > 1e-6):
            mp = max(0.0, min(1.0, mp))
            device = decoder_input_tokens_final.device
            B, T, A = decoder_input_tokens_final.shape

            valid = decoder_attention_mask.to(torch.bool)
            valid[:, 0] = False  # 不 mask decoder 起始位（BOS）

            if decoder_input_mask_keep_special and (self.special_n > 0):
                # 只 mask “正常音符”(pitch >= special_n)，保留 BOS/EOS/PAD/MASK 等 special
                valid = valid & (decoder_input_tokens_final[..., 0] >= self.special_n)

            drop = (torch.rand((B, T), device=device) < mp) & valid
            mask_tok = decoder_input_tokens_final.new_full((B, T, A), fill_value=self.mask_id)

            decoder_input_tokens_final = torch.where(
                drop.unsqueeze(-1),
                mask_tok,
                decoder_input_tokens_final,
            )

        # ---- now do the real forward with (possibly) mixed inputs ----
        dec_embeds = self.backbone.embed(decoder_input_tokens_final)

        # dec_embeds = self.backbone.embed(decoder_input_tokens)

        out = self.backbone.bart(
            inputs_embeds=src_embeds,
            attention_mask=src_attention_mask.to(dtype=torch.long),
            decoder_inputs_embeds=dec_embeds,
            decoder_attention_mask=decoder_attention_mask.to(dtype=torch.long),
            use_cache=False,
            return_dict=True,
        )

        h_dec = out.last_hidden_state  # [B,T,D]
        pitch_logits, dur_logits, dt_logits = self.lm_head(h_dec)

        logits = MultiAttrLogits(pitch=pitch_logits, duration=dur_logits, dt=dt_logits)

        labels = mask_labels_with_ignore_index(tgt_tokens.clone(), tgt_attention_mask, ignore_index=ignore_index)

        total_loss, loss_dict = multi_attribute_ce_loss_weighted(
            labels=labels,
            pitch_logits=pitch_logits,
            duration_logits=dur_logits,
            dt_logits=dt_logits,
            attr_weights=attr_weights,
            token_weights=token_weights,
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

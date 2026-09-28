from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from transformers import BartModel
from transformers.modeling_outputs import BaseModelOutput

from ..models.bart.config import MusicBartConfig

from ..utils import infer_attention_mask_from_tokens

# 复用你现有模块
from .token_embed import TokenEmbedding

from .heads import MusicBartMultiAttrLMHead


class MusicBartBackbone(nn.Module):
    """
    迁移友好的 backbone：
      Multi-attr TokenEmbedding -> HF BartModel(encoder-decoder)
    不包含任何 task-specific head。
    """

    def __init__(self, cfg: MusicBartConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab = cfg.vocab

        self.token_embed = TokenEmbedding(
            n_pitch=cfg.vocab.n_pitch,
            n_duration=cfg.vocab.n_duration,
            n_dt=cfg.vocab.n_dt,
            d_embed=cfg.backbone.d_embed,
            d_model=cfg.backbone.d_model,
            max_seq_len=cfg.backbone.max_seq_len,
        )

        te = self.token_embed
        self.lm_head = MusicBartMultiAttrLMHead(
            d_model=cfg.backbone.d_model,
            d_embed=cfg.backbone.d_embed,
            pitch_embed=te.pitch_embed,
            duration_embed=te.duration_embed,
            dt_embed=te.dt_embed,
            use_bias=True,
        )

        hf_cfg = cfg.backbone.to_hf_bart_config(cfg.vocab)

        # ---- NEW: force Transformers to use SDPA backend ----
        if hasattr(hf_cfg, "attn_implementation"):
            hf_cfg.attn_implementation = "sdpa"
        elif hasattr(hf_cfg, "_attn_implementation"):
            hf_cfg._attn_implementation = "sdpa"

        self.bart = BartModel(hf_cfg)

    @property
    def pad_id(self) -> int:
        return self.vocab.pad_id

    @property
    def bos_id(self) -> int:
        return self.vocab.bos_id

    @property
    def eos_id(self) -> int:
        return self.vocab.eos_id

    def embed(self, tokens: torch.LongTensor) -> torch.Tensor:
        """tokens: [B,L,A] -> [B,L,D]"""
        return self.token_embed(tokens)

    def encode(
        self,
        src_tokens: Optional[torch.LongTensor] = None,
        src_embeds: Optional[torch.Tensor] = None,
        src_attention_mask: Optional[torch.Tensor] = None,
    ) -> BaseModelOutput:
        if src_embeds is None:
            if src_tokens is None:
                raise ValueError("Either src_tokens or src_embeds must be provided.")
            src_embeds = self.embed(src_tokens)

        if src_attention_mask is None:
            if src_tokens is None:
                raise ValueError("src_attention_mask is None and src_tokens is None; can't infer mask.")
            src_attention_mask = infer_attention_mask_from_tokens(src_tokens, pad_id=self.pad_id)

        return self.bart.encoder(
            inputs_embeds=src_embeds,
            attention_mask=src_attention_mask.to(dtype=torch.long),
            return_dict=True,
        )

    def decode(
        self,
        decoder_tokens: Optional[torch.LongTensor] = None,
        decoder_embeds: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values=None,
    ):
        if decoder_embeds is None:
            if decoder_tokens is None:
                raise ValueError("Either decoder_tokens or decoder_embeds must be provided.")
            decoder_embeds = self.embed(decoder_tokens)

        if decoder_attention_mask is None and decoder_tokens is not None:
            decoder_attention_mask = infer_attention_mask_from_tokens(decoder_tokens, pad_id=self.pad_id)

        # NEW: if encoder_outputs is given, call decoder directly (avoid BartModel wrapper)
        if encoder_outputs is not None:
            enc_hid = encoder_outputs.last_hidden_state
            return self.bart.decoder(
                inputs_embeds=decoder_embeds,
                attention_mask=None if decoder_attention_mask is None else decoder_attention_mask.to(torch.long),
                encoder_hidden_states=enc_hid,
                encoder_attention_mask=None if encoder_attention_mask is None else encoder_attention_mask.to(torch.long),
                use_cache=use_cache,
                past_key_values=past_key_values,
                return_dict=True,
            )

        # fallback: decoder-only path (no cross-attn)
        return self.bart(
            encoder_outputs=None,
            attention_mask=None,
            decoder_inputs_embeds=decoder_embeds,
            decoder_attention_mask=None if decoder_attention_mask is None else decoder_attention_mask.to(torch.long),
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=True,
        )

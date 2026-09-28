from __future__ import annotations
from dataclasses import dataclass

from transformers import BartConfig


@dataclass(frozen=True)
class MusicBartVocabConfig:
    """多属性 token 的词表/特殊符号配置。"""
    n_pitch: int
    n_duration: int
    n_dt: int

    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    mask_id: int = 1

    special_n: int = 5

    @property
    def n_attr(self) -> int:
        return 3


@dataclass(frozen=True)
class MusicBartBackboneConfig:
    """BART backbone + embedding 的结构超参。"""
    max_seq_len: int
    d_embed: int
    d_model: int

    # HF BartModel 超参
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_function: str = "gelu"

    # 位置表冗余
    position_offset: int = 8

    # 由于我们使用 inputs_embeds，vocab_size 只是 dummy
    dummy_vocab_size: int = 16

    def to_hf_bart_config(self, vocab: MusicBartVocabConfig) -> BartConfig:
        return BartConfig(
            vocab_size=self.dummy_vocab_size,
            d_model=self.d_model,
            encoder_layers=self.n_encoder_layers,
            decoder_layers=self.n_decoder_layers,
            encoder_attention_heads=self.n_heads,
            decoder_attention_heads=self.n_heads,
            encoder_ffn_dim=self.d_ff,
            decoder_ffn_dim=self.d_ff,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_function=self.activation_function,
            max_position_embeddings=self.max_seq_len + self.position_offset,
            pad_token_id=vocab.pad_id,
            bos_token_id=vocab.bos_id,
            eos_token_id=vocab.eos_id,
            decoder_start_token_id=vocab.bos_id,
            scale_embedding=True,
        )


@dataclass(frozen=True)
class MusicBartConfig:
    vocab: MusicBartVocabConfig
    backbone: MusicBartBackboneConfig
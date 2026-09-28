# main/evaluation/score_extraction.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from ..pointer_utils import (
    compute_step_mask,
    hard_indices_to_mask,
    marginal_importance_from_pointer_soft,
    importance_from_score_logits,
)


@dataclass
class BackbonePrediction:
    """
    统一的评估输出格式。
    约定：
      - scores 越大表示 token 越重要
      - scores 最好是非负 importance，而不是任意 logits
    """
    scores: torch.Tensor                          # [B,L]
    hard_mask: Optional[torch.Tensor] = None      # [B,L] bool
    hard_indices: Optional[torch.LongTensor] = None
    step_mask: Optional[torch.Tensor] = None      # [B,T] bool
    z_mask: Optional[torch.Tensor] = None         # [B,T] bool
    extras: Dict[str, torch.Tensor] = field(default_factory=dict)


class PointerModelAdapter:
    """
    适配当前的 pointer-compressor 类模型：
      - 你的 MusicSkeletonModelIII
      - 以及当前的 OTBBaselineModel

    支持两种使用方式：
      1) 显式给 z_len / rho
      2) 不给 z_len/rho 时：
         - 如果 model 有 predict_compression_length()，就走 native rho/z_len
         - 否则尝试用 model.cfg.rho
    """

    def __init__(self, model, *, pad_id: int, eos_id: int, special_n: int):
        self.module = model
        self.model = model
        self.pad_id = int(pad_id)
        self.eos_id = int(eos_id)
        self.special_n = int(special_n)

    def _resolve_tau(self, tau: Optional[float]) -> float:
        if tau is not None:
            return float(tau)
        cfg = getattr(self.model, "cfg", None)
        if cfg is None:
            return 1.0
        return float(getattr(cfg, "tau", 1.0))

    def predict(
        self,
        *,
        src_tokens: torch.LongTensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        z_len: Optional[torch.LongTensor] = None,
        rho: Optional[float] = None,
        tau: Optional[float] = None,
        exclude_last_step: bool = True,
        **kwargs,
    ) -> BackbonePrediction:
        if src_attention_mask is None:
            src_attention_mask = (src_tokens[..., 0] != self.pad_id)

        len_out = None

        # native length mode
        if z_len is None and rho is None:
            if hasattr(self.model, "predict_compression_length"):
                len_out = self.model.predict_compression_length(
                    x_tokens=src_tokens,
                    x_attention_mask=src_attention_mask,
                )
                z_len = len_out.z_len_hard
            else:
                cfg = getattr(self.model, "cfg", None)
                rho_cfg = None if cfg is None else getattr(cfg, "rho", None)
                if rho_cfg is None:
                    raise ValueError(
                        "This adapter needs explicit z_len or rho for the current model."
                    )
                rho = float(rho_cfg)

        comp = self.model.compressor(
            src_tokens=src_tokens,
            src_attention_mask=src_attention_mask,
            z_len=z_len,
            rho=rho,
            tau=self._resolve_tau(tau),
        )

        step_mask = compute_step_mask(comp.z_mask, exclude_last_step=exclude_last_step)
        if getattr(comp, "score_logits", None) is not None and getattr(comp, "score_mask", None) is not None:
            # encoder_topk: 静态 score -> soft importance（ranking 与 hard top-k 一致）
            scores = importance_from_score_logits(
                comp.score_logits,
                comp.score_mask,
                temperature=self._resolve_tau(tau),
            )
        else:
            # pointer_decoder / baseline: 仍用旧的 marginal(pointer_soft)
            scores = marginal_importance_from_pointer_soft(
                comp.pointer_soft,
                step_mask=step_mask,
            )
        hard_mask = hard_indices_to_mask(
            comp.hard_indices,
            step_mask,
            L=src_tokens.size(1),
        )

        extras = {
            "pointer_soft": comp.pointer_soft,
            "eos_pos": comp.eos_pos,
            "z_len": comp.z_len,
        }
        if len_out is not None:
            extras["rho_pred"] = len_out.rho_pred
            extras["z_len_cont"] = len_out.z_len_cont
        if getattr(comp, "score_logits", None) is not None:
            extras["score_logits"] = comp.score_logits
            extras["score_mask"] = comp.score_mask

        return BackbonePrediction(
            scores=scores,
            hard_mask=hard_mask,
            hard_indices=comp.hard_indices,
            step_mask=step_mask,
            z_mask=comp.z_mask,
            extras=extras,
        )


def build_pointer_adapter_from_model(model) -> PointerModelAdapter:
    return PointerModelAdapter(
        model,
        pad_id=int(model.backbone.pad_id),
        eos_id=int(model.backbone.eos_id),
        special_n=int(model.vocab.special_n),
    )

class ScoreArrayAdapter:
    """
    Adapter for models that already produced per-token scores on disk,
    e.g. MuDeP baseline -> pred_score.npy [N,L].

    It ignores src_tokens and returns scores by idx.
    """

    def __init__(
        self,
        scores: np.ndarray,
        *,
        pad_id: int,
        special_n: int,
    ) -> None:
        self.scores = scores  # [N,L] float32/float64
        self.pad_id = int(pad_id)
        self.special_n = int(special_n)
        self.module = None

        if self.scores.ndim != 2:
            raise ValueError(f"ScoreArrayAdapter expects [N,L], got {self.scores.shape}")

    def predict(
        self,
        *,
        src_tokens: torch.LongTensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        idx: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BackbonePrediction:
        if idx is None:
            raise ValueError("ScoreArrayAdapter requires idx in batch.")

        # idx: [B] on any device
        idx_cpu = idx.detach().cpu().numpy().astype(np.int64)
        s = np.asarray(self.scores[idx_cpu])  # [B,L]
        scores_t = torch.from_numpy(np.asarray(s, dtype=np.float32)).to(src_tokens.device)

        return BackbonePrediction(scores=scores_t, hard_mask=None)

def _topk_mask_per_row_torch(
    scores: torch.Tensor,   # [B,L]
    mask: torch.Tensor,     # [B,L] bool
    k: torch.LongTensor,    # [B]
) -> torch.Tensor:
    """
    Select top-k[b] within mask[b].
    Returns bool [B,L].
    """
    scores = scores.to(torch.float32)
    mask = mask.to(torch.bool)
    B, L = scores.shape

    # invalid -> very negative so they go to the end
    neg = scores.new_full((), -1e4)
    s = torch.where(mask, scores, neg)

    order = s.argsort(dim=1, descending=True)  # [B,L]
    # ranks[pos] = rank in sorted order
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(L, device=scores.device)[None, :].expand(B, L))

    k = k.to(torch.long).clamp(min=0)
    return mask & (ranks < k[:, None])


class KeepProbModelAdapter:
    """
    Adapter for encoder-only keep-prob baseline.
    """
    def __init__(
        self,
        model,
        *,
        pad_id: int,
        special_n: int,
        hard_policy: str = "topk_zlen",   # "none" | "threshold" | "topk_zlen"
        threshold: float = 0.5,
        zero_special_scores: bool = True, # prevent dumping mass to BOS/EOS for ins_mass
    ):
        self.module = model
        self.model = model
        self.pad_id = int(pad_id)
        self.special_n = int(special_n)

        self.hard_policy = str(hard_policy)
        self.threshold = float(threshold)
        self.zero_special_scores = bool(zero_special_scores)

    def predict(
        self,
        *,
        src_tokens: torch.LongTensor,                 # [B,L,3]
        src_attention_mask: Optional[torch.Tensor] = None,
        z_len: Optional[torch.LongTensor] = None,
        rho: Optional[float] = None,
        tau: Optional[float] = None,
        exclude_last_step: bool = True,
        **kwargs,
    ) -> BackbonePrediction:

        if src_attention_mask is None:
            src_attention_mask = (src_tokens[..., 0] != self.pad_id)

        pitch = src_tokens[..., 0]
        note_mask = src_attention_mask.to(torch.bool) & (pitch >= self.special_n)

        keep_logits = self.model(src_tokens=src_tokens, src_attention_mask=src_attention_mask)
        keep_prob = torch.sigmoid(keep_logits.to(torch.float32))  # [B,L]

        scores = keep_prob
        if self.zero_special_scores:
            scores = scores * note_mask.to(scores.dtype)

        hard_mask = None
        if self.hard_policy == "none":
            hard_mask = None

        elif self.hard_policy == "threshold":
            hard_mask = (keep_prob >= self.threshold) & note_mask

        elif self.hard_policy == "topk_zlen":
            if z_len is None:
                raise ValueError("hard_policy='topk_zlen' requires z_len.")
            # estimate how many notes are in the backbone sequence:
            # k_note = len_x - (#special tokens in valid region)
            special_valid = src_attention_mask.to(torch.bool) & (pitch < self.special_n)
            n_special = special_valid.sum(dim=1).to(torch.long)          # [B]
            k_note = (z_len.to(torch.long) - n_special).clamp(min=0)     # [B]
            hard_mask = _topk_mask_per_row_torch(scores, note_mask, k_note)

        else:
            raise ValueError(f"Unknown hard_policy: {self.hard_policy}")

        return BackbonePrediction(scores=scores, hard_mask=hard_mask)


def build_keep_adapter_from_model(
    model,
    *,
    hard_policy: str = "topk_zlen",
    threshold: float = 0.5,
    zero_special_scores: bool = True,
) -> KeepProbModelAdapter:
    return KeepProbModelAdapter(
        model,
        pad_id=int(model.backbone.pad_id),
        special_n=int(model.backbone.cfg.vocab.special_n),
        hard_policy=hard_policy,
        threshold=threshold,
        zero_special_scores=zero_special_scores,
    )
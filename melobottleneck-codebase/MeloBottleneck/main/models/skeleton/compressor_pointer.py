# models/skeleton/compressor_pointer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...nn_modules import MusicBartBackbone
from ...utils import infer_attention_mask_from_tokens
from ...nn_funcs.gumbel import gumbel_softmax_st
from ...nn_modules.heads import SkeletonPointerHead, SkeletonTokenScoreHead

def _sample_gumbel_like(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    u = torch.rand_like(x)
    u = u.clamp(min=eps, max=1.0 - eps)
    return -torch.log(-torch.log(u))

@dataclass
class PointerCompressorOutput:
    pointer_logits: torch.Tensor      # [B, T, L] (masked logits, pre-softmax, after forcing/masking)
    pointer_soft: torch.Tensor                  # [B, T, L] train-soft (possibly leaky)
    pointer_soft_strict: torch.Tensor           # [B, T, L] strict-soft over allowed set
    hard_indices: torch.LongTensor    # [B, T]
    z_tokens_hard: torch.LongTensor   # [B, T, 3]
    z_embeds_soft: torch.Tensor       # [B, T, D]  (soft gather from E_ori)
    z_embeds_hard: torch.Tensor       # [B, T, D]  (hard gather from E_ori)
    z_embeds_st: torch.Tensor         # [B, T, D]  (ST on-the-fly)
    z_mask: torch.Tensor              # [B, T] bool
    eos_pos: torch.LongTensor         # [B]
    z_len: torch.LongTensor           # [B] 目标长度（包含EOS步）
    score_logits: Optional[torch.Tensor] = None   # [B,L] static logits (before top-k)
    score_mask: Optional[torch.Tensor] = None     # [B,L] bool, positions eligible for importance mass


def _find_eos_pos(
    src_tokens: torch.LongTensor,  # [B,L,3]
    src_mask: torch.Tensor,        # [B,L]
    eos_id: int,
) -> torch.LongTensor:
    pitch = src_tokens[..., 0]
    is_eos = (pitch == eos_id) & (src_mask.to(dtype=torch.bool))
    has = is_eos.any(dim=1)
    # argmax: 若全0会返回0，所以要用 has 修正
    first = is_eos.to(dtype=torch.long).argmax(dim=1)
    last_valid = src_mask.to(dtype=torch.long).sum(dim=1).clamp_min(1) - 1
    return torch.where(has, first, last_valid)


class MusicSkeletonPointerCompressor(nn.Module):
    """
    Compressor C:
      x --(encoder)--> memory
      decoder step-by-step -> pointer logits over x positions
      gumbel-softmax(ST) -> choose indices l_t and feed gathered embedding as next decoder input
    """

    def __init__(
        self,
        backbone: MusicBartBackbone,
        *,
        pointer_d_head: int = 256,
        tau: float = 1.0,
        max_z_len: int = 256,
        mask_eos_before_last: bool = True,
        use_gumbel_in_train: bool = True,
        soft_mask_penalty: float = 0.0,
        soft_mask_apply_k: int = 1,
        soft_mask_use_gumbel: bool = False,
        gumbel_scale: float = 1.0,
        pointer_normalize: bool = False,
        pointer_logit_scale: float = 1.0,
        compressor_mode: Literal["pointer_decoder", "encoder_topk", "encoder_topk_sampling"] = "encoder_topk",
        score_head_hidden: int = 256,
        score_head_dropout: float = 0.1,
        topk_select_note_only: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.keep_head = SkeletonTokenScoreHead(
            d_model=backbone.cfg.backbone.d_model,
            hidden=score_head_hidden,
            dropout=score_head_dropout,
        )
        self.topk_select_note_only = bool(topk_select_note_only)

        self.tau = float(tau)
        self.max_z_len = int(max_z_len)
        self.mask_eos_before_last = bool(mask_eos_before_last)
        self.use_gumbel_in_train = bool(use_gumbel_in_train)

        self.soft_mask_penalty = float(soft_mask_penalty)
        self.soft_mask_apply_k = int(soft_mask_apply_k)
        self.soft_mask_use_gumbel = bool(soft_mask_use_gumbel)
        self.gumbel_scale = float(gumbel_scale)

        self.pointer_logit_scale = float(pointer_logit_scale)
        self.compressor_mode = compressor_mode

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
    def special_n(self) -> int:
        return int(self.backbone.cfg.vocab.special_n)

    def _build_can_pick_mask(
        self,
        *,
        src_tokens: torch.LongTensor,   # [B,L,3]
        src_mask_bool: torch.Tensor,    # [B,L]
        eos_pos: torch.LongTensor,      # [B]
    ) -> torch.Tensor:
        """
        top-k 可选集合：
          - valid
          - strictly before EOS
          - optional: note-only (exclude BOS/EOS/PAD/MASK/...)
        """
        B, L, _ = src_tokens.shape
        l_ids = torch.arange(L, device=src_tokens.device)[None, :].expand(B, L)

        can_pick = src_mask_bool & (l_ids < eos_pos[:, None])

        if self.topk_select_note_only:
            can_pick = can_pick & (src_tokens[..., 0] >= self.special_n)

        return can_pick

    def _score_from_encoder_memory(self, memory: torch.Tensor) -> torch.Tensor:
        """
        tokenwise MLP scorer, analogous to O2B-Learner keep head.
        Returns raw logits [B,L] (do NOT sigmoid here).
        """
        return self.keep_head(memory) * self.pointer_logit_scale

    def _forward_static_topk(
        self,
        *,
        score_logits: torch.Tensor,             # [B,L]
        can_pick: torch.Tensor,                 # [B,L] bool
        valid: torch.Tensor,                    # [B,L] bool
        eos_pos: torch.LongTensor,              # [B]
        E_ori: torch.Tensor,                    # [B,L,D]
        z_len: torch.LongTensor,                # [B]
        tau: float,
        return_pointer_logits: bool,
        return_z_embeds_hard: bool,
        return_pointer_soft_strict: bool,
        sample_hard_in_train: bool,
    ):
        device = score_logits.device
        B, L = score_logits.shape
        D = E_ori.size(-1)

        k_sel = (z_len - 1).clamp(min=0)   # number of non-EOS picks
        K = int(k_sel.max().item())

        logits_list = []
        psoft_list = []
        psoft_strict_list = []
        idx_list = []
        z_soft_list = []
        z_hard_list = []
        z_st_list = []

        picked = torch.zeros((B, L), device=device, dtype=torch.bool)
        l_ids = torch.arange(L, device=device)[None, :].expand(B, L)
        neg_inf = torch.finfo(score_logits.dtype).min

        for t in range(K):
            after_end = (t >= k_sel)  # [B]

            allowed_pick = can_pick & (~picked)
            allowed_after = valid & (l_ids == eos_pos[:, None])  # dummy EOS-only step for shorter rows
            allowed = torch.where(after_end[:, None], allowed_after, allowed_pick)

            logits_raw = score_logits
            logits_hard = logits_raw.masked_fill(~allowed, neg_inf)

            logits_soft = logits_hard
            if self.soft_mask_penalty > 0.0:
                allowed_n = allowed.to(torch.int32).sum(dim=1)   # [B]
                leak = (allowed_n <= self.soft_mask_apply_k)     # [B]
                if leak.any():
                    leak_pool = torch.where(after_end[:, None], valid, can_pick)
                    logits_soft_all_valid = logits_raw.masked_fill(~leak_pool, neg_inf)
                    disallowed = (~allowed) & leak_pool & leak[:, None]
                    logits_soft = torch.where(leak[:, None], logits_soft_all_valid, logits_hard)
                    logits_soft = logits_soft - disallowed.to(logits_soft.dtype) * (self.soft_mask_penalty * tau)

            if sample_hard_in_train:
                g = _sample_gumbel_like(logits_raw) * self.gumbel_scale
                idx = (logits_hard + g).argmax(dim=-1)  # [B]

                if self.soft_mask_use_gumbel:
                    p_soft = torch.softmax(((logits_soft + g) / tau).float(), dim=-1).to(dtype=logits_raw.dtype)
                else:
                    p_soft = torch.softmax((logits_soft / tau).float(), dim=-1).to(dtype=logits_raw.dtype)
            else:
                idx = logits_hard.argmax(dim=-1)
                p_soft = torch.softmax((logits_soft / tau).float(), dim=-1).to(dtype=logits_raw.dtype)

            if return_pointer_soft_strict:
                p_soft_strict = torch.softmax((logits_hard / tau).float(), dim=-1).to(dtype=logits_raw.dtype)
            else:
                p_soft_strict = None

            soft_embed = torch.bmm(p_soft.unsqueeze(1), E_ori).squeeze(1)  # [B,D]
            hard_embed = E_ori.detach()[torch.arange(B, device=device), idx]  # [B,D]
            st_embed = (hard_embed - soft_embed).detach() + soft_embed

            if return_pointer_logits:
                logits_list.append(logits_soft)
            psoft_list.append(p_soft)
            if return_pointer_soft_strict:
                psoft_strict_list.append(p_soft_strict)
            idx_list.append(idx)
            z_soft_list.append(soft_embed)
            if return_z_embeds_hard:
                z_hard_list.append(hard_embed)
            z_st_list.append(st_embed)

            # update picked only on still-active rows
            upd = (~after_end)
            if upd.any():
                one = torch.zeros((B, L), device=device, dtype=torch.bool)
                one.scatter_(
                    dim=1,
                    index=idx[:, None],
                    src=torch.ones((B, 1), device=device, dtype=torch.bool),
                )
                picked = picked | (one & upd[:, None])

        # ---- sort selected indices into time order ----
        if K == 0:
            idx_sel_sorted = torch.empty((B, 0), device=device, dtype=torch.long)
            psoft_sel_sorted = torch.empty((B, 0, L), device=device, dtype=score_logits.dtype)
            z_embeds_soft_sel_sorted = torch.empty((B, 0, D), device=device, dtype=E_ori.dtype)
            z_embeds_st_sel_sorted = torch.empty((B, 0, D), device=device, dtype=E_ori.dtype)

            psoft_strict_sel_sorted = (
                torch.empty((B, 0, L), device=device, dtype=score_logits.dtype)
                if return_pointer_soft_strict else None
            )
            z_embeds_hard_sel_sorted = (
                torch.empty((B, 0, D), device=device, dtype=E_ori.dtype)
                if return_z_embeds_hard else None
            )
            pointer_logits_sel_sorted = (
                torch.empty((B, 0, L), device=device, dtype=score_logits.dtype)
                if return_pointer_logits else None
            )
        else:
            idx_sel = torch.stack(idx_list, dim=1)            # [B,K]
            psoft_sel = torch.stack(psoft_list, dim=1)        # [B,K,L]
            z_embeds_soft_sel = torch.stack(z_soft_list, dim=1)  # [B,K,D]
            z_embeds_st_sel = torch.stack(z_st_list, dim=1)      # [B,K,D]

            psoft_strict_sel = torch.stack(psoft_strict_list, dim=1) if return_pointer_soft_strict else None
            z_embeds_hard_sel = torch.stack(z_hard_list, dim=1) if return_z_embeds_hard else None
            pointer_logits_sel = torch.stack(logits_list, dim=1) if return_pointer_logits else None

            perm = idx_sel.argsort(dim=1)  # [B,K]
            perm_L = perm.unsqueeze(-1).expand(-1, -1, L)
            perm_D = perm.unsqueeze(-1).expand(-1, -1, D)

            idx_sel_sorted = idx_sel.gather(dim=1, index=perm)
            psoft_sel_sorted = psoft_sel.gather(dim=1, index=perm_L)
            z_embeds_soft_sel_sorted = z_embeds_soft_sel.gather(dim=1, index=perm_D)
            z_embeds_st_sel_sorted = z_embeds_st_sel.gather(dim=1, index=perm_D)

            psoft_strict_sel_sorted = (
                psoft_strict_sel.gather(dim=1, index=perm_L)
                if return_pointer_soft_strict else None
            )
            z_embeds_hard_sel_sorted = (
                z_embeds_hard_sel.gather(dim=1, index=perm_D)
                if return_z_embeds_hard else None
            )
            pointer_logits_sel_sorted = (
                pointer_logits_sel.gather(dim=1, index=perm_L)
                if return_pointer_logits else None
            )

        # ---- append EOS step ----
        p_eos = F.one_hot(
            eos_pos.clamp(min=0, max=L - 1),
            num_classes=L,
        ).to(dtype=E_ori.dtype)  # [B,L]

        eos_soft = torch.bmm(p_eos.unsqueeze(1), E_ori).squeeze(1)  # [B,D]
        eos_hard = E_ori.detach()[torch.arange(B, device=device), eos_pos]  # [B,D]
        eos_st = (eos_hard - eos_soft).detach() + eos_soft

        K_max = int(z_len.max().item()) - 1

        idx_list = [idx_sel_sorted[:, t] for t in range(K_max)] + [eos_pos]
        psoft_list = [psoft_sel_sorted[:, t, :] for t in range(K_max)] + [p_eos]
        z_soft_list = [z_embeds_soft_sel_sorted[:, t, :] for t in range(K_max)] + [eos_soft]
        z_st_list = [z_embeds_st_sel_sorted[:, t, :] for t in range(K_max)] + [eos_st]

        if return_pointer_soft_strict:
            psoft_strict_list = [psoft_strict_sel_sorted[:, t, :] for t in range(K_max)] + [p_eos]
        else:
            psoft_strict_list = []

        if return_z_embeds_hard:
            z_hard_list = [z_embeds_hard_sel_sorted[:, t, :] for t in range(K_max)] + [eos_hard]
        else:
            z_hard_list = []

        if return_pointer_logits:
            eos_logits = torch.full((B, L), fill_value=neg_inf, device=device, dtype=score_logits.dtype)
            eos_logits.scatter_(
                dim=1,
                index=eos_pos[:, None],
                src=torch.zeros((B, 1), device=device, dtype=score_logits.dtype),
            )
            logits_list = [pointer_logits_sel_sorted[:, t, :] for t in range(K_max)] + [eos_logits]
        else:
            logits_list = []

        return logits_list, psoft_list, psoft_strict_list, idx_list, z_soft_list, z_hard_list, z_st_list

    def forward(
        self,
        *,
        src_tokens: torch.LongTensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        rho: Optional[float] = None,
        z_len: Optional[torch.LongTensor] = None,
        tau: Optional[float] = None,
        # ---- NEW: reuse encoder work ----
        src_embeds: Optional[torch.Tensor] = None,       # [B,L,D]
        encoder_outputs: Optional[object] = None,        # HF BaseModelOutput
        # ---- NEW: avoid useless outputs when not needed ----
        return_pointer_logits: bool = False,
        return_z_embeds_hard: bool = False,
        return_pointer_soft_strict: bool = False,
    ) -> PointerCompressorOutput:

        device = src_tokens.device
        B, L, _ = src_tokens.shape
        tau = float(self.tau if tau is None else tau)

        if src_attention_mask is None:
            src_attention_mask = infer_attention_mask_from_tokens(src_tokens, pad_id=self.pad_id)
        src_mask_bool = src_attention_mask.to(dtype=torch.bool)

        eos_pos = _find_eos_pos(src_tokens, src_attention_mask, eos_id=self.eos_id)  # [B]
        L_x = eos_pos + 1  # 包含 EOS 的长度估计（更贴近你“指向Lx结束”设定）

        if z_len is None:
            if rho is None:
                raise ValueError("Either rho or z_len must be provided.")
            z_len = torch.ceil(L_x.to(torch.float32) * float(rho)).to(torch.long)

        z_len = z_len.clamp(min=1, max=self.max_z_len)  # [B]

        can_pick_pre = None
        if self.compressor_mode in ("encoder_topk", "encoder_topk_sampling"):
            can_pick_pre = self._build_can_pick_mask(
                src_tokens=src_tokens,
                src_mask_bool=src_mask_bool,
                eos_pos=eos_pos,
            )
            max_feasible_z_len = can_pick_pre.to(torch.long).sum(dim=1) + 1  # + EOS
            z_len = torch.minimum(z_len, max_feasible_z_len).clamp(min=1, max=self.max_z_len)

        T = int(z_len.max().item())
        z_mask = (torch.arange(T, device=device)[None, :] < z_len[:, None])  # bool

        # --- encoder: encode x ---
        E_ori = self.backbone.embed(src_tokens) if src_embeds is None else src_embeds

        if encoder_outputs is None:
            enc_out = self.backbone.encode(src_embeds=E_ori, src_attention_mask=src_attention_mask)
        else:
            enc_out = encoder_outputs

        score_logits_out = None
        score_mask_out = None

        if self.compressor_mode == "pointer_decoder":

            # ------------------------------------------------------------------------
            # Pointer network implementation
            # ------------------------------------------------------------------------

            memory = enc_out.last_hidden_state

            # --- decoder init with BOS embed ---
            bos_tok = torch.full((B, 1, 3), fill_value=self.bos_id, dtype=torch.long, device=device)
            cur_embed = self.backbone.embed(bos_tok).squeeze(1)  # [B,D]
            prev_idx = torch.full((B,), -1, dtype=torch.long, device=device)

            past = None

            logits_list = []
            psoft_list = []
            psoft_strict_list = []
            idx_list = []
            z_soft_list = []
            z_hard_list = []
            z_st_list = []

            l_ids = torch.arange(L, device=device)[None, :].expand(B, L)  # [B,L]
            valid = src_mask_bool  # [B,L]

            # ---- NEW: cache projected keys ----
            k = self.pointer.k_proj(memory)  # [B,L,H]
            if self.pointer.normalize:
                k = F.normalize(k, dim=-1)

            for t in range(T):
                out = self.backbone.decode(
                    decoder_embeds=cur_embed[:, None, :],      # [B,1,D]
                    encoder_outputs=enc_out,
                    encoder_attention_mask=src_attention_mask,
                    use_cache=True,
                    past_key_values=past,
                )
                past = out.past_key_values
                h_t = out.last_hidden_state[:, -1, :]  # [B,D]

                q = self.pointer.q_proj(h_t)        # [B,H]
                if self.pointer.normalize:
                    q = F.normalize(q, dim=-1)

                logits = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)  # [B,L]
                logits = logits * self.pointer_logit_scale

                # ---------------------------
                # masking / forcing
                # ---------------------------
                remaining = (z_len - 1 - t)           # [B]
                # 对 after_end 的样本 remaining 是负数，这里不用于正常步，后面单独处理
                max_allowed = eos_pos - remaining     # [B]

                allowed_normal = valid & (l_ids > prev_idx[:, None]) & (l_ids <= max_allowed[:, None])

                last_step = (t == (z_len - 1))
                after_end = (t >= z_len)

                allowed_last = valid & (l_ids == eos_pos[:, None]) & (l_ids > prev_idx[:, None])
                allowed_after = valid & (l_ids == eos_pos[:, None])   # after_end 允许重复 eos，避免空集合导致 NaN

                allowed = torch.where(
                    after_end[:, None],
                    allowed_after,
                    torch.where(last_step[:, None], allowed_last, allowed_normal)
                )

                logits_raw = logits  # [B,L]
                neg_inf = torch.finfo(logits_raw.dtype).min

                # ---- hard logits: strictly feasible (for idx / hard path) ----
                logits_hard = logits_raw.masked_fill(~allowed, neg_inf)

                # ---- soft logits: default = strict; only selected steps become leaky ----
                logits_soft = logits_hard
                if self.soft_mask_penalty > 0.0:
                    allowed_n = allowed.to(torch.int32).sum(dim=1)   # [B]
                    leak = (allowed_n <= self.soft_mask_apply_k)     # [B]

                    if leak.any():
                        logits_soft_all_valid = logits_raw.masked_fill(~valid, neg_inf)
                        disallowed = (~allowed) & valid & leak[:, None]
                        logits_soft = torch.where(leak[:, None], logits_soft_all_valid, logits_hard)
                        logits_soft = logits_soft - disallowed.to(logits_soft.dtype) * (self.soft_mask_penalty * tau)

                # ---- sample hard idx (strict) + build soft distribution (leaky) ----
                if self.training and self.use_gumbel_in_train:
                    g = _sample_gumbel_like(logits_raw)
                    g = g * self.gumbel_scale

                    idx = (logits_hard + g).argmax(dim=-1)  # [B]
                    # p_hard = torch.nn.functional.one_hot(idx, num_classes=L).to(dtype=logits_raw.dtype)

                    if self.soft_mask_use_gumbel:
                        p_soft = torch.softmax(((logits_soft + g) / tau).float(), dim=-1).to(dtype=logits_raw.dtype)
                    else:
                        p_soft = torch.softmax((logits_soft / tau).float(), dim=-1).to(dtype=logits_raw.dtype)
                else:
                    idx = logits_hard.argmax(dim=-1)
                    # p_hard = torch.nn.functional.one_hot(idx, num_classes=L).to(dtype=logits_raw.dtype)
                    p_soft = torch.softmax((logits_soft / tau).float(), dim=-1).to(dtype=logits_raw.dtype)

                if return_pointer_soft_strict:
                    p_soft_strict = torch.softmax((logits_hard / tau).float(), dim=-1).to(dtype=logits_raw.dtype)
                else:
                    p_soft_strict = None

                # gather embeddings from E_ori
                soft_embed = torch.bmm(p_soft.unsqueeze(1), E_ori).squeeze(1)  # [B,D]
                # hard_embed = torch.bmm(p_hard.unsqueeze(1), E_ori).squeeze(1)  # [B,D]

                # hard path does not need grad (ST detach), detach E_ori to save graph/memory
                hard_embed = E_ori.detach()[torch.arange(B, device=device), idx]  # [B,D]
                st_embed = (hard_embed - soft_embed).detach() + soft_embed

                if return_pointer_logits:
                    logits_list.append(logits_soft)
                psoft_list.append(p_soft)
                if return_pointer_soft_strict:
                    psoft_strict_list.append(p_soft_strict)
                idx_list.append(idx)
                z_soft_list.append(soft_embed)
                if return_z_embeds_hard:
                    z_hard_list.append(hard_embed)
                z_st_list.append(st_embed)

                # next step
                cur_embed = st_embed
                prev_idx = idx

        elif self.compressor_mode in ("encoder_topk", "encoder_topk_sampling"):

            memory = enc_out.last_hidden_state  # [B,L,D]
            valid = src_mask_bool               # [B,L]

            can_pick = can_pick_pre
            if can_pick is None:
                can_pick = self._build_can_pick_mask(
                    src_tokens=src_tokens,
                    src_mask_bool=valid,
                    eos_pos=eos_pos,
                )

            # NEW: tokenwise MLP scorer (O2B-Learner style)
            score_logits = self._score_from_encoder_memory(memory)  # [B,L]

            score_logits_out = score_logits
            score_mask_out = can_pick

            (
                logits_list,
                psoft_list,
                psoft_strict_list,
                idx_list,
                z_soft_list,
                z_hard_list,
                z_st_list,
            ) = self._forward_static_topk(
                score_logits=score_logits,
                can_pick=can_pick,
                valid=valid,
                eos_pos=eos_pos,
                E_ori=E_ori,
                z_len=z_len,
                tau=tau,
                return_pointer_logits=return_pointer_logits,
                return_z_embeds_hard=return_z_embeds_hard,
                return_pointer_soft_strict=return_pointer_soft_strict,
                sample_hard_in_train=(
                    self.compressor_mode == "encoder_topk_sampling"
                    and self.training
                    and self.use_gumbel_in_train
                ),
            )

        else:
            raise ValueError(f"Invalid compressor_mode: {self.compressor_mode}")

        if return_pointer_logits:
            pointer_logits = torch.stack(logits_list, dim=1)      # [B,T,L]
        else:
            pointer_logits = None
        pointer_soft = torch.stack(psoft_list, dim=1)         # [B,T,L]
        if return_pointer_soft_strict:
            pointer_soft_strict = torch.stack(psoft_strict_list, dim=1)  # [B,T,L]
        else:
            pointer_soft_strict = None
        hard_indices = torch.stack(idx_list, dim=1)           # [B,T]
        z_embeds_soft = torch.stack(z_soft_list, dim=1)       # [B,T,D]
        if return_z_embeds_hard:
            z_embeds_hard = torch.stack(z_hard_list, dim=1)       # [B,T,D]
        else:
            z_embeds_hard = None
        z_embeds_st = torch.stack(z_st_list, dim=1)           # [B,T,D]

        # gather hard tokens from src
        T_eff = hard_indices.size(1)
        gather_idx = hard_indices.unsqueeze(-1).expand(B, T_eff, 3)
        z_tokens_hard = src_tokens.gather(dim=1, index=gather_idx)  # [B,T,3]

        # mask out beyond z_mask
        pad_tok = torch.full((B, 1, 3), fill_value=self.pad_id, dtype=torch.long, device=device)
        z_tokens_hard = torch.where(z_mask.unsqueeze(-1), z_tokens_hard, pad_tok)

        # zero out embeds beyond mask（可选，但对稳定日志/后续模块更安全）
        z_embeds_soft = z_embeds_soft * z_mask.unsqueeze(-1).to(z_embeds_soft.dtype)
        if z_embeds_hard is not None:
            z_embeds_hard = z_embeds_hard * z_mask.unsqueeze(-1).to(z_embeds_hard.dtype)
        z_embeds_st = z_embeds_st * z_mask.unsqueeze(-1).to(z_embeds_st.dtype)

        return PointerCompressorOutput(
            pointer_logits=pointer_logits,
            pointer_soft=pointer_soft,
            pointer_soft_strict=pointer_soft_strict,
            hard_indices=hard_indices,
            z_tokens_hard=z_tokens_hard,
            z_embeds_soft=z_embeds_soft,
            z_embeds_hard=z_embeds_hard,
            z_embeds_st=z_embeds_st,
            z_mask=z_mask,
            eos_pos=eos_pos,
            z_len=z_len,
            score_logits=score_logits_out,
            score_mask=score_mask_out,
        )

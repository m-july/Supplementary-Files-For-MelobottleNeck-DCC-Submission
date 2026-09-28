from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from .config import BartDenoiseConfig


class BartStyleDenoiser:
    """
    Fast BART-style denoiser for prefix-aligned padded sequences.

    Important assumption
    --------------------
    attention_mask is prefix-aligned:
        [True, True, ..., True, False, ..., False]

    This matches your Stage-A pipeline and lets us avoid the old per-sample
    nonzero()/item()/while-loop heavy implementation that was causing lots of
    Python overhead and host-device synchronisation.
    """

    _MASKING_KEYS = ("token@attribute", "token@token", "n-token@attribute", "n-token@token")
    _DELETION_KEYS = ("token@token", "n-token@token")
    _ENABLE_EPS = 1e-3

    def __init__(self, config: Optional[BartDenoiseConfig]):
        self.config = config if config is not None else BartDenoiseConfig()
        self._arange_cache = {}

    # ------------------------------------------------------------------
    # Runtime control
    # ------------------------------------------------------------------
    def set_runtime_noise(
        self,
        *,
        masking_noise_density: Optional[float] = None,
        deletion_prob: Optional[float] = None,
        rotation_prob: Optional[float] = None,
    ) -> None:
        if masking_noise_density is not None:
            v = float(masking_noise_density)
            v = min(max(v, 0.0), 1.0)
            self.config.masking_noise_density = v
            self.config.enable_masking = v > self._ENABLE_EPS

        if deletion_prob is not None:
            v = float(deletion_prob)
            v = min(max(v, 0.0), 1.0)
            self.config.deletion_prob = v
            self.config.enable_deletion = v > self._ENABLE_EPS

        if rotation_prob is not None:
            v = float(rotation_prob)
            v = min(max(v, 0.0), 1.0)
            self.config.rotation_prob = v
            self.config.enable_rotation = v > self._ENABLE_EPS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def corrupt(
        self,
        input_tokens: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        apply_masking: bool = True,
        apply_deletion: bool = True,
        apply_rotation: bool = True,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        assert input_tokens.dim() == 3, "input_tokens must be [B, L, 3]"
        B, L, A = input_tokens.shape
        assert A == self.config.num_attributes, (
            f"Last dim must be num_attributes={self.config.num_attributes}, got {A}"
        )

        device = input_tokens.device

        if attention_mask is None:
            attention_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

        tokens = input_tokens.clone()
        attn = attention_mask.clone()

        if apply_masking and self.config.enable_masking and self.config.masking_noise_density > 0.0:
            tokens, attn = self._apply_masking(tokens, attn)

        if apply_deletion and self.config.enable_deletion and self.config.deletion_prob > 0.0:
            tokens, attn = self._apply_deletion(tokens, attn)

        if apply_rotation and self.config.enable_rotation and self.config.rotation_prob > 0.0:
            tokens, attn = self._apply_rotation(tokens, attn)

        return tokens, attn

    # ------------------------------------------------------------------
    # Small utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _build_prob_tensor(prob_dict, keys) -> torch.Tensor:
        probs = torch.tensor(
            [max(0.0, float(prob_dict.get(k, 0.0))) for k in keys],
            dtype=torch.float32,
        )
        s = float(probs.sum().item())
        if s <= 0.0:
            probs.fill_(1.0 / len(keys))
        else:
            probs /= s
        return probs

    def _get_positions(self, n: int, device: torch.device) -> torch.Tensor:
        key = (device.type, -1 if device.index is None else int(device.index), int(n))
        t = self._arange_cache.get(key, None)
        if t is None:
            t = torch.arange(n, device=device)
            self._arange_cache[key] = t
        return t

    def _prefix_mask(self, valid_len: torch.LongTensor, L: int, device: torch.device) -> torch.BoolTensor:
        pos = self._get_positions(L, device)
        return pos.unsqueeze(0) < valid_len.unsqueeze(1)

    @staticmethod
    def _count_from_ratio(
        valid_len: torch.LongTensor,
        ratio: float,
        min_if_positive: int = 1,
    ) -> torch.LongTensor:
        ratio = max(0.0, float(ratio))
        out = torch.round(valid_len.to(torch.float32) * ratio).to(torch.long)
        if min_if_positive > 0:
            out = torch.where(valid_len > 0, out.clamp(min=min_if_positive), torch.zeros_like(out))
        else:
            out = torch.where(valid_len > 0, out, torch.zeros_like(out))
        out = torch.minimum(out, valid_len)
        return out

    def _sample_exact_k_mask(
        self,
        valid_len: torch.LongTensor,        # [G]
        k: torch.LongTensor,                # [G]
        L: int,
        available_mask: Optional[torch.BoolTensor] = None,   # [G, L]
    ) -> torch.BoolTensor:
        """
        For each row, sample exactly k positions from the valid / available set.
        """
        device = valid_len.device
        G = int(valid_len.shape[0])

        out = torch.zeros(G, L, dtype=torch.bool, device=device)
        if G == 0:
            return out

        if available_mask is None:
            avail = self._prefix_mask(valid_len, L, device)
        else:
            avail = available_mask.to(device=device, dtype=torch.bool)

        avail_count = avail.sum(dim=1, dtype=torch.long)
        kk = torch.minimum(k.to(torch.long).clamp(min=0), avail_count)

        max_k = int(kk.max().item()) if kk.numel() > 0 else 0
        if max_k <= 0:
            return out

        scores = torch.rand(G, L, device=device)
        scores.masked_fill_(~avail, 2.0)  # valid random in [0,1), invalid = 2.0
        top_idx = scores.topk(k=max_k, dim=1, largest=False).indices  # [G, max_k]

        take = self._get_positions(max_k, device).unsqueeze(0) < kk.unsqueeze(1)  # [G, max_k]
        out.scatter_(1, top_idx, take)
        out &= avail
        return out

    def _sample_span_mask_batch(
        self,
        valid_len: torch.LongTensor,    # [G]
        target_num: torch.LongTensor,   # [G]
        span_lambda: float,
        L: int,
    ) -> torch.BoolTensor:
        """
        Faster batch span-mask sampler.

        Strategy:
        1) Estimate number of spans from target_num / avg_span_len
        2) Sample span starts + Poisson lengths in batch
        3) Turn intervals into masks via diff + cumsum
        4) If covered tokens are still fewer than target_num, fill deficit with exact-k singleton picks

        This preserves the overall semantics well enough for pretraining while
        avoiding the old per-sample Python while-loop bottleneck.
        """
        device = valid_len.device
        G = int(valid_len.shape[0])

        out = torch.zeros(G, L, dtype=torch.bool, device=device)
        if G == 0:
            return out

        target_num = torch.minimum(target_num.clamp(min=0), valid_len)
        max_target = int(target_num.max().item()) if target_num.numel() > 0 else 0
        if max_target <= 0:
            return out

        lam = max(1e-3, float(span_lambda))
        # E[max(1, Poisson(lam))] = lam + exp(-lam)
        avg_span_len = lam + math.exp(-lam)

        n_spans = torch.floor(target_num.to(torch.float32) / float(avg_span_len)).to(torch.long)
        n_spans = torch.where(target_num > 0, n_spans.clamp(min=1), torch.zeros_like(n_spans))
        n_spans = torch.minimum(n_spans, target_num)

        max_spans = int(n_spans.max().item()) if n_spans.numel() > 0 else 0
        if max_spans <= 0:
            return out

        safe_vlen = valid_len.clamp(min=1)
        span_pos = self._get_positions(max_spans, device).unsqueeze(0)
        active_span = span_pos < n_spans.unsqueeze(1)  # [G, max_spans]

        starts = torch.floor(
            torch.rand(G, max_spans, device=device) * safe_vlen.unsqueeze(1).to(torch.float32)
        ).to(torch.long)

        lengths = torch.poisson(torch.full((G, max_spans), lam, device=device)).to(torch.long)
        lengths.clamp_(min=1)

        ends = torch.minimum(starts + lengths, valid_len.unsqueeze(1))

        rows = self._get_positions(G, device).unsqueeze(1).expand(G, max_spans)
        rows = rows[active_span]
        if rows.numel() == 0:
            return out

        diff = torch.zeros(G, L + 1, dtype=torch.int32, device=device)
        flat = diff.view(-1)
        stride = L + 1

        idx_start = rows * stride + starts[active_span]
        idx_end = rows * stride + ends[active_span]

        ones = torch.ones_like(idx_start, dtype=flat.dtype)
        flat.index_add_(0, idx_start, ones)
        flat.index_add_(0, idx_end, -ones)

        mask = diff[:, :L].cumsum(dim=1) > 0
        valid = self._prefix_mask(valid_len, L, device)
        mask &= valid

        # fill deficit exactly if needed
        covered = mask.sum(dim=1, dtype=torch.long)
        deficit = (target_num - covered).clamp(min=0)
        add_mask = self._sample_exact_k_mask(
            valid_len=valid_len,
            k=deficit,
            L=L,
            available_mask=(valid & ~mask),
        )
        mask |= add_mask
        return mask

    def _left_pack(
        self,
        seq: torch.LongTensor,         # [G, L, A]
        keep_mask: torch.BoolTensor,   # [G, L]
        pad_token_id: int,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Left-pack kept tokens to the front; pad the rest.
        """
        G, L, A = seq.shape
        device = seq.device

        out = torch.full_like(seq, pad_token_id)
        dest = keep_mask.to(torch.long).cumsum(dim=1) - 1  # [G, L]
        rows = self._get_positions(G, device).unsqueeze(1).expand(G, L)

        out[rows[keep_mask], dest[keep_mask]] = seq[keep_mask]

        new_len = keep_mask.sum(dim=1, dtype=torch.long)
        attn = self._get_positions(L, device).unsqueeze(0) < new_len.unsqueeze(1)
        return out, attn

    def _compress_spans_with_mask_token(
        self,
        seq: torch.LongTensor,          # [G, L, A]
        valid_len: torch.LongTensor,    # [G]
        span_mask: torch.BoolTensor,    # [G, L]
        mask_token_id: int,
        pad_token_id: int,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Replace each contiguous True span with a single [MASK] token triple.
        """
        G, L, A = seq.shape
        device = seq.device

        valid = self._prefix_mask(valid_len, L, device)
        span_mask = span_mask & valid

        prev = torch.zeros_like(span_mask)
        prev[:, 1:] = span_mask[:, :-1]
        span_start = span_mask & ~prev
        keep_original = valid & ~span_mask
        emit = keep_original | span_start

        out = torch.full_like(seq, pad_token_id)
        dest = emit.to(torch.long).cumsum(dim=1) - 1
        rows = self._get_positions(G, device).unsqueeze(1).expand(G, L)

        out[rows[keep_original], dest[keep_original]] = seq[keep_original]
        out[rows[span_start], dest[span_start]] = mask_token_id

        new_len = emit.sum(dim=1, dtype=torch.long)
        attn = self._get_positions(L, device).unsqueeze(0) < new_len.unsqueeze(1)
        return out, attn

    def _mask_one_random_attribute_(
        self,
        seq: torch.LongTensor,          # [G, L, A]
        pos_mask: torch.BoolTensor,     # [G, L]
        mask_token_id: int,
    ) -> torch.LongTensor:
        G, L, A = seq.shape
        if G == 0:
            return seq

        attr_idx = torch.randint(0, A, (G, L), device=seq.device)
        for a in range(A):
            seq[:, :, a].masked_fill_(pos_mask & (attr_idx == a), mask_token_id)
        return seq

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------
    def _apply_masking(
        self,
        tokens: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        cfg = self.config
        B, L, A = tokens.shape
        device = tokens.device

        valid_len = attention_mask.sum(dim=1, dtype=torch.long)
        target_num = self._count_from_ratio(valid_len, cfg.masking_noise_density, min_if_positive=1)

        probs = self._build_prob_tensor(cfg.masking_granularity_probs, self._MASKING_KEYS).to(device=device)
        gran = torch.multinomial(probs, num_samples=B, replacement=True)  # [B]

        # 0) token@attribute
        idx = torch.where(gran == 0)[0]
        if idx.numel() > 0:
            sub = tokens.index_select(0, idx)
            vl = valid_len.index_select(0, idx)
            kk = target_num.index_select(0, idx)

            pos_mask = self._sample_exact_k_mask(vl, kk, L)
            sub = self._mask_one_random_attribute_(sub, pos_mask, cfg.mask_token_id)
            tokens.index_copy_(0, idx, sub)

        # 1) token@token
        idx = torch.where(gran == 1)[0]
        if idx.numel() > 0:
            sub = tokens.index_select(0, idx)
            vl = valid_len.index_select(0, idx)
            kk = target_num.index_select(0, idx)

            pos_mask = self._sample_exact_k_mask(vl, kk, L)
            sub[pos_mask] = cfg.mask_token_id
            tokens.index_copy_(0, idx, sub)

        # 2) n-token@attribute
        idx = torch.where(gran == 2)[0]
        if idx.numel() > 0:
            sub = tokens.index_select(0, idx)
            vl = valid_len.index_select(0, idx)
            kk = target_num.index_select(0, idx)

            span_mask = self._sample_span_mask_batch(vl, kk, cfg.masking_span_lambda, L)
            sub = self._mask_one_random_attribute_(sub, span_mask, cfg.mask_token_id)
            tokens.index_copy_(0, idx, sub)

        # 3) n-token@token
        idx = torch.where(gran == 3)[0]
        if idx.numel() > 0:
            sub = tokens.index_select(0, idx)
            vl = valid_len.index_select(0, idx)
            kk = target_num.index_select(0, idx)

            span_mask = self._sample_span_mask_batch(vl, kk, cfg.masking_span_lambda, L)

            if cfg.masking_compress_n_token_token_level:
                sub, sub_attn = self._compress_spans_with_mask_token(
                    seq=sub,
                    valid_len=vl,
                    span_mask=span_mask,
                    mask_token_id=cfg.mask_token_id,
                    pad_token_id=cfg.pad_token_id,
                )
            else:
                sub[span_mask] = cfg.mask_token_id
                sub_attn = self._prefix_mask(vl, L, device)

            tokens.index_copy_(0, idx, sub)
            attention_mask.index_copy_(0, idx, sub_attn)

        return tokens, attention_mask

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------
    def _apply_deletion(
        self,
        tokens: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        cfg = self.config
        B, L, A = tokens.shape
        device = tokens.device

        valid_len = attention_mask.sum(dim=1, dtype=torch.long)
        probs = self._build_prob_tensor(cfg.deletion_granularity_probs, self._DELETION_KEYS).to(device=device)
        gran = torch.multinomial(probs, num_samples=B, replacement=True)  # [B]

        # 0) token@token
        idx = torch.where(gran == 0)[0]
        if idx.numel() > 0:
            sub = tokens.index_select(0, idx)
            vl = valid_len.index_select(0, idx)

            valid = self._prefix_mask(vl, L, device)
            keep = (torch.rand(sub.size(0), L, device=device) > float(cfg.deletion_prob)) & valid

            # ensure at least one token remains
            fix = self._sample_exact_k_mask(vl, torch.ones_like(vl), L, available_mask=valid)
            empty = keep.sum(dim=1, dtype=torch.long) == 0
            keep |= (fix & empty.unsqueeze(1))

            sub, sub_attn = self._left_pack(sub, keep, cfg.pad_token_id)
            tokens.index_copy_(0, idx, sub)
            attention_mask.index_copy_(0, idx, sub_attn)

        # 1) n-token@token
        idx = torch.where(gran == 1)[0]
        if idx.numel() > 0:
            sub = tokens.index_select(0, idx)
            vl = valid_len.index_select(0, idx)

            target_del = self._count_from_ratio(vl, cfg.deletion_prob, min_if_positive=1)
            del_mask = self._sample_span_mask_batch(vl, target_del, cfg.deletion_span_lambda, L)

            valid = self._prefix_mask(vl, L, device)
            keep = valid & ~del_mask

            # ensure at least one token remains
            fix = self._sample_exact_k_mask(vl, torch.ones_like(vl), L, available_mask=valid)
            empty = keep.sum(dim=1, dtype=torch.long) == 0
            keep |= (fix & empty.unsqueeze(1))

            sub, sub_attn = self._left_pack(sub, keep, cfg.pad_token_id)
            tokens.index_copy_(0, idx, sub)
            attention_mask.index_copy_(0, idx, sub_attn)

        return tokens, attention_mask

    # ------------------------------------------------------------------
    # Rotation
    # ------------------------------------------------------------------
    def _apply_rotation(
        self,
        tokens: torch.LongTensor,
        attention_mask: torch.BoolTensor,
    ) -> Tuple[torch.LongTensor, torch.BoolTensor]:
        cfg = self.config
        B, L, A = tokens.shape
        device = tokens.device

        valid_len = attention_mask.sum(dim=1, dtype=torch.long)
        apply = (torch.rand(B, device=device) < float(cfg.rotation_prob)) & (valid_len > 1)

        idx = torch.where(apply)[0]
        if idx.numel() == 0:
            return tokens, attention_mask

        sub = tokens.index_select(0, idx)
        vl = valid_len.index_select(0, idx)
        G = int(sub.shape[0])

        pivot = torch.floor(torch.rand(G, device=device) * vl.to(torch.float32)).to(torch.long)

        pos = self._get_positions(L, device).unsqueeze(0).expand(G, L)
        valid = pos < vl.unsqueeze(1)
        safe_vl = vl.clamp(min=1)

        src_idx = (pos + pivot.unsqueeze(1)) % safe_vl.unsqueeze(1)
        src_idx = torch.where(valid, src_idx, torch.zeros_like(src_idx))

        gathered = sub.gather(1, src_idx.unsqueeze(-1).expand(-1, -1, A))

        out = torch.full_like(sub, cfg.pad_token_id)
        out[valid] = gathered[valid]

        tokens.index_copy_(0, idx, out)
        return tokens, attention_mask
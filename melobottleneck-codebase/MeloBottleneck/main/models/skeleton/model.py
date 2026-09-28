# models/skeleton/model.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Literal, Dict, Any

import torch
import torch.nn as nn

import math

from ...nn_modules import MusicBartBackbone
from ...utils import infer_attention_mask_from_tokens
from ...quantization import MusicQuantizationTables, build_quantizers
from .compressor_pointer import MusicSkeletonPointerCompressor, PointerCompressorOutput
from .postprocess import SkeletonForwardExtend
from .reconstructor_from_embeds import MusicBartSeq2SeqReconstructorFromEmbeds
from .lm_prior_decoder_only import MusicBartDecoderOnlyLM
from .losses import (
    build_recon_token_weights_from_duration,
    lm_prior_kl_loss,
    guided_attention_loss,
    pointer_soft_sharpness_loss,
)


@dataclass
class MusicSkeletonIIIConfig:
    # compressor
    rho: float = 2.0 / 3.0
    max_z_len: int = 514
    tau: float = 1.0
    pointer_d_head: int = 256
    mask_eos_before_last: bool = True
    use_gumbel_in_train: bool = False

    # ---- dynamic rho (per-sequence) ----
    dynamic_rho: bool = True
    rho_min: float = 1.0 / 3.0
    rho_max: float = 1.0
    z_len_min: int = 2                 # 建议 >=2：至少(一个内容token + EOS)，否则容易退化成只指EOS
    rho_gate_temp: float = 0.50        # 软门控温度，越小越接近hard mask（但梯度更不稳定）

    # ---- rho length regularization (fix collapse) ----
    rho_len_loss: Literal["none", "minimize", "target_mean", "quantile_normal"] = "quantile_normal"
    rho_target: float = 2.0 / 3.0
    rho_target_std: float = 0.20
    rho_len_clip_targets: bool = True
    # 是否让 “长度正则 loss” 的梯度回传进 backbone encoder
    # 建议 True：只更新 rho_pred_head，不要让 backbone 因为长度正则而漂
    rho_len_detach_backbone: bool = True

    # reconstructor
    recon_attr_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    recon_z_mode: Literal["onfly", "final"] = "final"  # 你提到的两种消融

    # loss weights
    lambda_R: float = 1.0
    lambda_P: float = 1.0
    lambda_L: float = 1.0

    # lm prior: pitch-only / rhythm-only / all
    lm_prior_attr_weights: Tuple[float, float, float] = (0.7, 0.2, 0.1)

    ignore_index: int = -100

    # guided attention (pointer alignment)
    lambda_GA: float = 0.0
    ga_sigma: float = 0.075
    ga_use_time_pos: bool = True

    # ---- reconstructor scheduled sampling ----
    recon_ss_prob: float = 0.0
    recon_ss_temperature: float = 1.0
    recon_ss_use_greedy: bool = False
    recon_ss_sync_special: bool = True

    # ---- reconstructor decoder-input mask dropout ----
    recon_input_mask_prob: float = 0.0
    recon_input_mask_keep_special: bool = True  # True: 不 mask BOS/EOS/PAD/MASK 等 special（pitch < special_n）

    # ---- NEW: leaky soft mask for pointer_soft (escape forced-size-1 trap) ----
    pointer_soft_penalty: float = 0.7    # γ：有效惩罚（log-prob space）
    pointer_soft_apply_k: int = 65536         # 仅当 #allowed <= k 时启用 soft barrier（1=只处理 singleton）
    pointer_soft_use_gumbel: bool = False # p_soft 是否也加同一个 gumbel（建议先 False 降低噪声）

    # ---- Gumbel noise scale ----
    gumbel_scale: float = 0.2  # 乘到 Gumbel 噪声上的缩放系数；1.0=标准 Gumbel，<1 降低随机性

    # --- 
    pointer_soft_for_aux_losses: Literal["train", "strict"] = "train"
    pointer_soft_for_export: Literal["train", "strict"] = "train"

    pointer_normalize: bool = True
    pointer_logit_scale: float = 8 # 1/16 = 0.0625

    recon_token_weight_mode: Literal["duration", "uniform", "sqrt_duration", "log_duration"] = "duration"

    ga_loss_form: Literal["quad", "bounded_exp"] = "quad"

    rho_use_soft_gate: bool = True

    # ---
    compressor_mode: Literal["pointer_decoder", "encoder_topk", "encoder_topk_sampling"] = "encoder_topk_sampling"

    # ---- pointer-soft sharpness regularization ----
    lambda_sharp: float = 0.0
    # "hard_ce" = -log p_soft[t, hard_idx]
    # "entropy" = minimize H(p_soft)
    # "hard_ce+entropy" = 两者相加（entropy 可再乘权重）
    pointer_soft_reg: Literal["none", "hard_ce", "entropy", "hard_ce+entropy"] = "hard_ce"
    # 对 p_soft 做额外的“温度缩放”(只用于这个正则，不改变 ST 的 p_soft)
    # <1 更尖锐，=1 不变，>1 更平
    pointer_soft_reg_temperature: float = 1.0
    # 一般建议 True：排除最后一步（EOS 强制步）不参与 sharpness loss
    pointer_soft_reg_exclude_last_step: bool = True
    # 可选：用 strict 版 p_soft（不含 leaky），否则用 train 版
    # 你现在 compressor 已支持 pointer_soft_strict；不开也行
    pointer_soft_reg_use_strict: bool = False
    # 只有在 "hard_ce+entropy" 时用
    pointer_soft_reg_entropy_weight: float = 1.0

    # ---- reconstructor conditioning on compression (rho) ----
    recon_cond: Literal["none", "film_rho"] = "film_rho"
    recon_cond_detach: bool = True          # 强烈建议 True：不让 recon loss 直接推 rho_pred_head
    recon_cond_scale: float = 0.10          # 调制幅度上限（tanh 后再乘）
    recon_cond_use_Lx: bool = True          # 是否额外提供 Lx 信息（归一化后）

    # ---- tokenwise MLP scorer for encoder_topk* ----
    score_head_hidden: int = 256
    score_head_dropout: float = 0.10

    # True: top-k 只在 note token 上选，不让 BOS / EOS / 其它 special 参与竞争
    topk_select_note_only: bool = True

@dataclass
class CompressionLengthOutput:
    loss_len: torch.Tensor
    rho_pred: torch.Tensor            # [B]
    z_len_cont: torch.Tensor          # [B]
    z_len_hard: torch.LongTensor      # [B]
    eos_pos: torch.LongTensor         # [B]
    L_x: torch.LongTensor             # [B]


@dataclass
class MusicSkeletonIIIOutput:
    loss: torch.Tensor
    loss_recon: torch.Tensor
    loss_prior: torch.Tensor
    loss_guided_attn: torch.Tensor
    loss_len: torch.Tensor
    loss_ptr_sharp: torch.Tensor

    rho_pred: torch.Tensor
    z_len_cont: torch.Tensor
    z_len_hard: torch.LongTensor

    # debug / analysis
    z_tokens_final: torch.LongTensor
    z_mask: torch.Tensor
    hard_indices: torch.LongTensor
    pointer_soft: torch.Tensor
    pointer_soft_strict: Optional[torch.Tensor] = None

    # ---- encoder_topk static scoring (for soft importance) ----
    score_logits: Optional[torch.Tensor] = None
    score_mask: Optional[torch.Tensor] = None

    # ---

    recon_loss_dict: Dict[str, torch.Tensor] = field(default_factory=dict)
    diag: Dict[str, torch.Tensor] = field(default_factory=dict)




class MusicSkeletonModelIII(nn.Module):
    """
    Model III: Pointer Compressor (enc-dec) + Reconstructor (enc-dec) + optional LM prior.
    """

    def __init__(
        self,
        *,
        backbone: MusicBartBackbone,
        quant: MusicQuantizationTables,
        cfg: MusicSkeletonIIIConfig = MusicSkeletonIIIConfig(),
        lm_prior: Optional[MusicBartDecoderOnlyLM] = None,
        use_bias_in_lm_head: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg

        dur_q, dt_q = build_quantizers(quant)
        self.duration_q = dur_q
        self.dt_q = dt_q

        self.compressor = MusicSkeletonPointerCompressor(
            backbone=backbone,
            pointer_d_head=cfg.pointer_d_head,
            tau=cfg.tau,
            max_z_len=cfg.max_z_len,
            mask_eos_before_last=cfg.mask_eos_before_last,
            use_gumbel_in_train=cfg.use_gumbel_in_train,
            soft_mask_penalty=cfg.pointer_soft_penalty,
            soft_mask_apply_k=cfg.pointer_soft_apply_k,
            soft_mask_use_gumbel=cfg.pointer_soft_use_gumbel,
            gumbel_scale=cfg.gumbel_scale,
            pointer_normalize=cfg.pointer_normalize,
            pointer_logit_scale=cfg.pointer_logit_scale,
            compressor_mode=cfg.compressor_mode,
            score_head_hidden=cfg.score_head_hidden,
            score_head_dropout=cfg.score_head_dropout,
            topk_select_note_only=cfg.topk_select_note_only,
        )

        self.forward_extend = SkeletonForwardExtend(
            duration_q=self.duration_q,
            dt_q=self.dt_q,
            pad_id=backbone.pad_id,
            eos_id=backbone.eos_id,
        )

        self.reconstructor = MusicBartSeq2SeqReconstructorFromEmbeds(
            backbone=backbone,
            use_bias_in_lm_head=use_bias_in_lm_head,
        )

        self.lm_prior = lm_prior  # 可为空

        # ---- rho predictor ----
        self.rho_pred_head: Optional[nn.Module] = None
        if self.cfg.dynamic_rho:
            d = int(backbone.cfg.backbone.d_model)
            h = max(64, d // 4)
            self.rho_pred_head = nn.Sequential(
                nn.Linear(d, h),
                nn.Tanh(),
                nn.Linear(h, 1),
            )

            # init: 让初始 rho_pred ≈ cfg.rho（你原来的固定 rho）
            rho0 = float(self.cfg.rho)
            rmin, rmax = float(self.cfg.rho_min), float(self.cfg.rho_max)
            p0 = (rho0 - rmin) / max(rmax - rmin, 1e-8)
            p0 = max(1e-3, min(1.0 - 1e-3, p0))
            bias = math.log(p0 / (1.0 - p0))
            nn.init.zeros_(self.rho_pred_head[-1].weight)
            nn.init.constant_(self.rho_pred_head[-1].bias, bias)

        # ---- reconstructor rho-conditioning (FiLM) ----
        self.recon_cond_proj: Optional[nn.Module] = None
        if self.cfg.recon_cond != "none":
            d = int(backbone.cfg.backbone.d_model)
            h = max(64, d // 4)
            in_dim = 2 if self.cfg.recon_cond_use_Lx else 1

            self.recon_cond_proj = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.Tanh(),
                nn.Linear(h, 2 * d),   # -> gamma|beta
            )

            # 关键：初始化为恒等（gamma=0,beta=0），避免一上来扰乱训练
            nn.init.zeros_(self.recon_cond_proj[-1].weight)
            nn.init.zeros_(self.recon_cond_proj[-1].bias)

    @property
    def vocab(self):
        return self.backbone.cfg.vocab
    

    def predict_compression_length(
        self,
        *,
        x_tokens: torch.LongTensor,
        x_attention_mask: Optional[torch.Tensor] = None,
        rho: Optional[float] = None,
        src_embeds: Optional[torch.Tensor] = None,        # NEW
        encoder_outputs: Optional[Any] = None,            # NEW (HF BaseModelOutput)
    ) -> CompressionLengthOutput:
        cfg = self.cfg

        if x_attention_mask is None:
            x_attention_mask = infer_attention_mask_from_tokens(x_tokens, pad_id=self.backbone.pad_id)

        B, L, _ = x_tokens.shape
        device = x_tokens.device

        # find eos_pos & L_x
        pitch = x_tokens[..., 0]
        is_eos = (pitch == self.backbone.eos_id) & x_attention_mask.to(torch.bool)
        has = is_eos.any(dim=1)
        eos_pos = is_eos.to(torch.long).argmax(dim=1)
        last_valid = x_attention_mask.to(torch.long).sum(dim=1).clamp_min(1) - 1
        eos_pos = torch.where(has, eos_pos, last_valid)
        L_x = eos_pos + 1  # [B]

        if cfg.dynamic_rho:
            assert self.rho_pred_head is not None

            if encoder_outputs is None:
                src_embeds = self.backbone.embed(x_tokens) if src_embeds is None else src_embeds
                encoder_outputs = self.backbone.encode(src_embeds=src_embeds, src_attention_mask=x_attention_mask)
            memory = encoder_outputs.last_hidden_state

            m = x_attention_mask.to(memory.dtype).unsqueeze(-1)  # [B,L,1]
            pooled = (memory * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

            rmin, rmax = float(cfg.rho_min), float(cfg.rho_max)
            logits_main = self.rho_pred_head(pooled).squeeze(-1)
            rho_pred = rmin + (rmax - rmin) * torch.sigmoid(logits_main)

            z_len_cont = (L_x.to(torch.float32) * rho_pred).clamp(
                min=float(cfg.z_len_min),
                max=float(cfg.max_z_len),
            )
            z_len_hard = torch.ceil(z_len_cont).to(torch.long).clamp(
                min=int(cfg.z_len_min),
                max=int(cfg.max_z_len),
            )

            # ------------------------------------------------------------
            # NEW: length regularizer on rho (distribution matching)
            #   - "minimize": old behavior -> collapse to rho_min
            #   - "target_mean": constrain E[rho]
            #   - "quantile_normal": match rho distribution (recommended)
            # ------------------------------------------------------------
            if cfg.rho_len_loss == "none":
                loss_len = torch.zeros((), device=device, dtype=torch.float32)

            else:
                # Optionally detach backbone for length-loss branch
                if cfg.rho_len_detach_backbone:
                    logits_len = self.rho_pred_head(pooled.detach()).squeeze(-1)
                    rho_for_loss = rmin + (rmax - rmin) * torch.sigmoid(logits_len)
                else:
                    rho_for_loss = rho_pred

                # use effective rho after clamp (more faithful to actual length)
                z_len_cont_for_loss = (L_x.to(torch.float32) * rho_for_loss).clamp(
                    min=float(cfg.z_len_min),
                    max=float(cfg.max_z_len),
                )
                rho_eff = z_len_cont_for_loss / L_x.to(torch.float32).clamp_min(1.0)  # [B]

                if cfg.rho_len_loss == "minimize":
                    loss_len = rho_eff.mean()

                elif cfg.rho_len_loss == "target_mean":
                    mu = float(cfg.rho_target)
                    loss_len = (rho_eff.mean() - mu).pow(2)

                elif cfg.rho_len_loss == "quantile_normal":
                    Bq = int(rho_eff.numel())
                    sig = float(cfg.rho_target_std)
                    mu = float(cfg.rho_target)
                    if Bq <= 1:
                        loss_len = (rho_eff.mean() - mu).pow(2)
                    else:
                        if sig <= 0:
                            raise ValueError("rho_target_std must be > 0 when rho_len_loss='quantile_normal'.")

                        # u in (0,1): (i+0.5)/B
                        u = (torch.arange(Bq, device=device, dtype=torch.float32) + 0.5) / float(Bq)
                        # standard normal quantile: Phi^{-1}(u) = sqrt(2)*erfinv(2u-1)
                        z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
                        q = mu + sig * z  # target sorted rhos

                        if cfg.rho_len_clip_targets:
                            q = q.clamp(min=float(cfg.rho_min), max=float(cfg.rho_max))

                        rho_sorted = torch.sort(rho_eff.to(torch.float32), dim=0).values
                        loss_len = torch.mean((rho_sorted - q).pow(2))

                else:
                    raise ValueError(f"Unknown rho_len_loss: {cfg.rho_len_loss}")

            # keep loss_len fp32 for stability
            loss_len = loss_len.to(torch.float32)

        else:
            rho_fixed = float(cfg.rho if rho is None else rho)
            rho_pred = torch.full((B,), rho_fixed, device=device)
            z_len_cont = torch.ceil(L_x.to(torch.float32) * rho_fixed)
            z_len_hard = z_len_cont.to(torch.long).clamp(min=1, max=int(cfg.max_z_len))
            loss_len = torch.zeros((), device=device)

        return CompressionLengthOutput(
            loss_len=loss_len,
            rho_pred=rho_pred,
            z_len_cont=z_len_cont,
            z_len_hard=z_len_hard,
            eos_pos=eos_pos,
            L_x=L_x,
        )


    def forward(
        self,
        *,
        x_tokens: torch.LongTensor,                      # [B,L,3]
        x_attention_mask: Optional[torch.Tensor] = None, # [B,L]
        rho: Optional[float] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> MusicSkeletonIIIOutput:
        cfg = self.cfg
        use_dyn = bool(cfg.dynamic_rho)

        need_strict = (
            cfg.pointer_soft_for_aux_losses == "strict"
            or cfg.pointer_soft_for_export == "strict"
            or (diagnostics is not None and diagnostics.get("pointer_soft_gap", False))
            or (cfg.lambda_sharp != 0.0 and cfg.pointer_soft_reg_use_strict)
        )

        if x_attention_mask is None:
            x_attention_mask = infer_attention_mask_from_tokens(x_tokens, pad_id=self.backbone.pad_id)

        src_embeds = None
        enc_out = None
        if use_dyn:
            src_embeds = self.backbone.embed(x_tokens)
            enc_out = self.backbone.encode(src_embeds=src_embeds, src_attention_mask=x_attention_mask)

        B, L, _ = x_tokens.shape
        device = x_tokens.device

        len_out = self.predict_compression_length(
            x_tokens=x_tokens,
            x_attention_mask=x_attention_mask,
            rho=rho,
            src_embeds=src_embeds,
            encoder_outputs=enc_out,
        )

        loss_len = len_out.loss_len
        rho_pred = len_out.rho_pred
        z_len_cont = len_out.z_len_cont
        z_len_hard = len_out.z_len_hard

        # -----------------------------
        # 1) Compressor: pointer decode -> z (on-the-fly)
        # -----------------------------
        if use_dyn:
            comp: PointerCompressorOutput = self.compressor(
                src_tokens=x_tokens,
                src_attention_mask=x_attention_mask,
                z_len=z_len_hard,
                tau=cfg.tau,
                src_embeds=src_embeds,
                encoder_outputs=enc_out,
                return_pointer_logits=False,
                return_z_embeds_hard=False,
                return_pointer_soft_strict=need_strict,
            )
        else:
            # 旧逻辑保留：外部传 rho 或用 cfg.rho
            rho_fixed = cfg.rho if rho is None else float(rho)
            comp: PointerCompressorOutput = self.compressor(
                src_tokens=x_tokens,
                src_attention_mask=x_attention_mask,
                rho=rho_fixed,
                tau=cfg.tau,
                return_pointer_logits=False,
                return_z_embeds_hard=False,
                return_pointer_soft_strict=need_strict,
            )

        pointer_soft_aux = (
            comp.pointer_soft_strict
            if cfg.pointer_soft_for_aux_losses == "strict"
            else comp.pointer_soft
        )

        pointer_soft_export = (
            comp.pointer_soft_strict
            if cfg.pointer_soft_for_export == "strict"
            else comp.pointer_soft
        )

        # def _chk(name, x):
        #     if not torch.isfinite(x).all():
        #         raise RuntimeError(f"[NaN/Inf] {name}: dtype={x.dtype}, "
        #                         f"min={torch.nan_to_num(x).min().item()}, "
        #                         f"max={torch.nan_to_num(x).max().item()}")

        # _chk("pointer_soft", comp.pointer_soft)
        # _chk("z_embeds_soft", comp.z_embeds_soft)
        # _chk("z_embeds_st", comp.z_embeds_st)

        # -----------------------------
        # 2) Postprocess: forward-extend duration -> z_final_tokens
        # -----------------------------
        z_tokens_final = self.forward_extend(
            src_tokens=x_tokens,
            z_tokens=comp.z_tokens_hard,
            hard_indices=comp.hard_indices,
            z_mask=comp.z_mask,
        )  # [B,T,3]

        # hard_final embeds (pos 以 z 序列为准)
        z_hard_final = self.backbone.embed(z_tokens_final)  # [B,T,D]

        # soft 仍用 on-the-fly (来自 E_ori 的 soft gather)，以保证“雨露均沾”的梯度
        z_soft_onfly = comp.z_embeds_soft  # [B,T,D]

        # final ST: forward=hard_final, backward=soft_onfly
        z_st_final = (z_hard_final - z_soft_onfly).detach() + z_soft_onfly
        z_st_onfly = comp.z_embeds_st

        z_src_embeds = z_st_final if cfg.recon_z_mode == "final" else z_st_onfly
        # _chk("z_src_embeds", z_src_embeds)

        z_attention_mask = comp.z_mask.to(dtype=torch.long)  # 仍用 hard mask

        if use_dyn:
            # soft gate: [B,T]
            T = z_src_embeds.size(1)
            t = (torch.arange(T, device=device, dtype=torch.float32) + 0.5)[None, :]  # [1,T]
            temp = max(float(cfg.rho_gate_temp), 1e-3)
            gate = torch.sigmoid((z_len_cont[:, None].to(torch.float32) - t) / temp)  # [B,T]
            gate = gate.to(z_src_embeds.dtype) * comp.z_mask.to(z_src_embeds.dtype)   # safety
            z_src_embeds = z_src_embeds * gate.unsqueeze(-1)

        # ------------------------------------------------------------
        # condition reconstructor on compression regime (rho)
        # via FiLM on encoder source embeds (z_src_embeds)
        # ------------------------------------------------------------
        if self.recon_cond_proj is not None and cfg.recon_cond == "film_rho":
            Lx = len_out.L_x.to(torch.float32).clamp_min(1.0)  # [B]
            rho_eff = (z_len_cont.to(torch.float32) / Lx).clamp(
                min=float(cfg.rho_min),
                max=float(cfg.rho_max),
            )  # [B]

            if cfg.recon_cond_use_Lx:
                Lx_norm = (Lx / float(self.backbone.cfg.backbone.max_seq_len)).clamp(0.0, 1.0)
                cond = torch.stack([rho_eff, Lx_norm], dim=-1)  # [B,2]
            else:
                cond = rho_eff[:, None]  # [B,1]

            if cfg.recon_cond_detach:
                cond = cond.detach()

            gb = self.recon_cond_proj(cond)  # [B, 2D]
            gamma, beta = gb.chunk(2, dim=-1)

            s = float(cfg.recon_cond_scale)
            gamma = torch.tanh(gamma) * s
            beta  = torch.tanh(beta)  * s

            gamma = gamma.to(dtype=z_src_embeds.dtype)
            beta  = beta.to(dtype=z_src_embeds.dtype)

            z_src_embeds = z_src_embeds * (1.0 + gamma[:, None, :]) + beta[:, None, :]

            # safety：把 mask 外位置压回 0，避免 beta 在 pad 区“发光”
            z_src_embeds = z_src_embeds * z_attention_mask[:, :, None].to(z_src_embeds.dtype)

        # -----------------------------
        # 3) Reconstructor: reconstruct x
        # -----------------------------
        token_weights = build_recon_token_weights_from_duration(
            tgt_tokens=x_tokens,
            tgt_attention_mask=x_attention_mask,
            duration_q=self.duration_q,
            mode=cfg.recon_token_weight_mode,
        )

        rec_out = self.reconstructor(
            src_embeds=z_src_embeds,
            src_attention_mask=z_attention_mask,
            tgt_tokens=x_tokens,
            tgt_attention_mask=x_attention_mask,
            attr_weights=cfg.recon_attr_weights,
            token_weights=token_weights,
            ignore_index=cfg.ignore_index,
            return_hidden=False,
            scheduled_sampling_prob=cfg.recon_ss_prob,
            scheduled_sampling_temperature=cfg.recon_ss_temperature,
            scheduled_sampling_use_greedy=cfg.recon_ss_use_greedy,
            scheduled_sampling_sync_special=cfg.recon_ss_sync_special,
            decoder_input_mask_prob=cfg.recon_input_mask_prob,
            decoder_input_mask_keep_special=cfg.recon_input_mask_keep_special,
        )
        loss_recon = rec_out.loss
        # _chk("loss_recon", loss_recon)

        recon_loss_dict = {}
        if rec_out.loss_dict is not None:
            recon_loss_dict = {
                k: v.detach().float()
                for k, v in rec_out.loss_dict.items()
            }

        # -----------------------------
        # 4) LM prior loss (optional): KL(r || p_LM)
        # -----------------------------
        loss_prior = torch.zeros((), device=x_tokens.device, dtype=loss_recon.dtype)
        if self.lm_prior is not None and cfg.lambda_P >= 1e-4:
            with torch.inference_mode():
                lm_logits = self.lm_prior.logits(
                    tokens=z_tokens_final,
                    attention_mask=z_attention_mask,
                )

            loss_prior = lm_prior_kl_loss(
                pointer_soft=pointer_soft_aux,
                src_tokens=x_tokens,
                lm_logits=lm_logits,
                z_mask=comp.z_mask,
                n_pitch=self.vocab.n_pitch,
                n_duration=self.vocab.n_duration,
                n_dt=self.vocab.n_dt,
                attr_weights=cfg.lm_prior_attr_weights,
            ).to(loss_recon.dtype)
        else:
            loss_prior = torch.zeros((), device=x_tokens.device, dtype=loss_recon.dtype)

        # _chk("loss_prior", loss_prior)

        # -----------------------------
        # 4a) Guided Attention loss (optional)
        # -----------------------------
        loss_guided = torch.zeros((), device=x_tokens.device, dtype=loss_recon.dtype)
        if cfg.lambda_GA != 0.0:
            loss_guided = guided_attention_loss(
                pointer_soft=pointer_soft_aux,            # [B,T,L]
                z_mask=comp.z_mask,                        # [B,T]
                src_attention_mask=x_attention_mask,        # [B,L]
                eos_pos=comp.eos_pos,                      # [B]
                z_len=comp.z_len,                          # [B]
                sigma=cfg.ga_sigma,
                use_time_pos=cfg.ga_use_time_pos,
                src_tokens=x_tokens,                       # needed if use_time_pos
                duration_q=self.duration_q,
                dt_q=self.dt_q,
                form=cfg.ga_loss_form,
            ).to(loss_recon.dtype)
        # _chk("loss_guided", loss_guided)

        # -----------------------------
        # 4b) Pointer sharpness loss (optional)
        # -----------------------------

        loss_ptr_sharp = torch.zeros((), device=x_tokens.device, dtype=loss_recon.dtype)
        if cfg.lambda_sharp != 0.0 and cfg.pointer_soft_reg != "none":
            p_for_reg = comp.pointer_soft
            if cfg.pointer_soft_reg_use_strict and (comp.pointer_soft_strict is not None):
                p_for_reg = comp.pointer_soft_strict

            loss_ptr_sharp = pointer_soft_sharpness_loss(
                pointer_soft=p_for_reg,
                hard_indices=comp.hard_indices,
                z_mask=comp.z_mask,
                kind=cfg.pointer_soft_reg,
                temperature=cfg.pointer_soft_reg_temperature,
                exclude_last_step=cfg.pointer_soft_reg_exclude_last_step,
                entropy_weight=cfg.pointer_soft_reg_entropy_weight,
            ).to(loss_recon.dtype)
        else:
            loss_ptr_sharp = torch.zeros((), device=x_tokens.device, dtype=loss_recon.dtype)

        # -----------------------------
        # 5) Diagnostics (optional)
        # -----------------------------
        diag: Dict[str, torch.Tensor] = {}
        diagnostics = diagnostics or {}
        if diagnostics.get("recon_wo_z", False):
            diag.update(self._diag_recon_without_z(
                z_src_embeds=z_src_embeds,
                z_attention_mask=z_attention_mask,
                x_tokens=x_tokens,
                x_attention_mask=x_attention_mask,
                token_weights=token_weights,
                loss_recon_train=loss_recon,
            ))
        if diagnostics.get("z_start_time", False):
            diag.update(self._diag_z_start_time_norm(
                src_tokens=x_tokens,
                src_attention_mask=x_attention_mask,
                hard_indices=comp.hard_indices,
                eos_pos=comp.eos_pos,
                pointer_soft=pointer_soft_export,
            ))
        if diagnostics.get("pointer_soft_gap", False):
            diag.update(self._diag_pointer_soft_gap(comp=comp))

        # -----------------------------
        # total
        # -----------------------------
        loss = (
            cfg.lambda_R * loss_recon
            + cfg.lambda_P * loss_prior
            + cfg.lambda_GA * loss_guided
            + cfg.lambda_L * loss_len.to(loss_recon.dtype)
            + cfg.lambda_sharp * loss_ptr_sharp.to(loss_recon.dtype)
        )

        return MusicSkeletonIIIOutput(
            loss=loss,
            loss_recon=loss_recon.detach(),
            loss_prior=loss_prior.detach(),
            loss_guided_attn=loss_guided.detach(),
            loss_len=loss_len.detach(),
            loss_ptr_sharp=loss_ptr_sharp.detach(),

            rho_pred=rho_pred.detach(),
            z_len_cont=z_len_cont.detach(),
            z_len_hard=z_len_hard.detach(),

            z_tokens_final=z_tokens_final.detach(),
            z_mask=comp.z_mask,
            hard_indices=comp.hard_indices.detach(),
            pointer_soft=pointer_soft_export.detach(),
            pointer_soft_strict=comp.pointer_soft_strict.detach() if need_strict else None,
            score_logits=comp.score_logits.detach() if comp.score_logits is not None else None,
            score_mask=comp.score_mask.detach() if comp.score_mask is not None else None,

            recon_loss_dict=recon_loss_dict,
            diag=diag,
        )

    def _diag_recon_without_z(
        self,
        *,
        z_src_embeds: torch.Tensor,          # [B,S,D]
        z_attention_mask: torch.Tensor,      # [B,S]
        x_tokens: torch.LongTensor,          # [B,L,3]
        x_attention_mask: torch.Tensor,      # [B,L]
        token_weights: torch.Tensor,         # [B,L]
        loss_recon_train: torch.Tensor,      # scalar (requires grad)
    ) -> Dict[str, torch.Tensor]:
        """
        Sanity check: compare recon loss with normal z vs. zeroed z.
        Uses eval + teacher forcing (no scheduled sampling) for stability.
        """
        cfg = self.cfg
        eps = 1e-8

        metrics: Dict[str, torch.Tensor] = {}

        # IMPORTANT: backbone is shared; toggling reconstructor toggles backbone too.
        was_training = self.reconstructor.training
        self.reconstructor.eval()
        try:
            with torch.inference_mode():
                def recon_loss_tf(src_embeds: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
                    rec = self.reconstructor(
                        src_embeds=src_embeds,
                        src_attention_mask=src_mask,
                        tgt_tokens=x_tokens,
                        tgt_attention_mask=x_attention_mask,
                        attr_weights=cfg.recon_attr_weights,
                        token_weights=token_weights,
                        ignore_index=cfg.ignore_index,
                        return_hidden=False,
                        scheduled_sampling_prob=0.0,   # teacher forcing only
                    )
                    return rec.loss

                loss_w_z = recon_loss_tf(z_src_embeds, z_attention_mask)

                # z0 = torch.zeros_like(z_src_embeds)
                # loss_wo_z = recon_loss_tf(z0, z_attention_mask)

                B, S, D = z_src_embeds.shape
                dummy = torch.zeros((B, 1, D), device=z_src_embeds.device, dtype=z_src_embeds.dtype)
                dummy_mask = torch.ones((B, 1), device=z_attention_mask.device, dtype=z_attention_mask.dtype)
                loss_wo_z = recon_loss_tf(dummy, dummy_mask)

                metrics["recon_wo_z/loss_recon_tf_w_z"] = loss_w_z
                metrics["recon_wo_z/loss_recon_tf_wo_z"] = loss_wo_z
                metrics["recon_wo_z/ratio_wo_over_w_tf"] = loss_wo_z / loss_w_z.clamp_min(eps)
                metrics["recon_wo_z/delta_wo_minus_w_tf"] = loss_wo_z - loss_w_z

                # optional: compare to the *training* recon loss (might include scheduled sampling)
                metrics["recon_wo_z/ratio_wo_over_train"] = (
                    loss_wo_z / loss_recon_train.detach().clamp_min(eps)
                )
        finally:
            self.reconstructor.train(was_training)

        # detach to be safe (even though inference_mode already does)
        return {k: v.detach() for k, v in metrics.items()}
    
    def _diag_z_start_time_norm(
        self,
        *,
        src_tokens: torch.LongTensor,          # [B,L,3]
        src_attention_mask: torch.Tensor,      # [B,L]
        hard_indices: torch.LongTensor,        # [B,T]
        eos_pos: torch.LongTensor,             # [B]
        pointer_soft: torch.Tensor,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """
        start_time_norm = onset[l0] / onset[eos]
        onset 的定义与 guided_attention_loss 内一致：
            span = dur_pos + dt_pos
            prefix = cumsum(span)
            onset = prefix - (dur_pos + dt_pos)
        """
        with torch.inference_mode():
            m = src_attention_mask.to(torch.float32)  # [B,L]

            dur = self.duration_q.decode_local_to_pos(src_tokens[..., 1]).to(torch.float32) * m
            dt  = self.dt_q.decode_local_to_pos(src_tokens[..., 2]).to(torch.float32) * m

            span = dur + dt
            prefix = torch.cumsum(span, dim=1)   # prefix[l] = onset_{l+1}
            onset = prefix - span                # onset[l]  = sum_{k<l}(dur_k+dt_k)

            # l0: z 的第一个 token 对应的 src index（hard pointer）
            l0 = hard_indices[:, 0].clamp(min=0, max=src_tokens.size(1) - 1)  # [B]

            onset_l0  = onset.gather(1, l0[:, None]).squeeze(1)               # [B]
            onset_eos = onset.gather(1, eos_pos[:, None]).squeeze(1)          # [B]

            # 为了和 guided_attention_loss 的稳定性一致，可以用 clamp_min(1.0)
            denom = onset_eos.clamp_min(1.0)

            start_time_norm = onset_l0 / denom                                # [B]

            # pointer soft version
            p0 = pointer_soft[:, 0, :].to(torch.float32) * src_attention_mask[:, None, :].to(torch.float32)  # [B,L]
            soft_onset0 = (p0 * onset.to(torch.float32)).sum(dim=-1)  # [B]
            soft_start_time_norm = soft_onset0 / denom

            # 下面这些必须都是 “标量 tensor”，因为你 wandb 那边在 v.item()
            metrics: Dict[str, torch.Tensor] = {}
            metrics["z_start_time/start_time_norm_mean"] = start_time_norm.mean()
            metrics["z_start_time/start_time_norm_max"]  = start_time_norm.max()
            metrics["z_start_time/fraction_gt_0.2"]      = (start_time_norm > 0.2).to(torch.float32).mean()
            metrics["z_start_time/soft_start_time_norm_mean"] = soft_start_time_norm.mean()
            metrics["z_start_time/soft_start_time_norm_max"]  = soft_start_time_norm.max()

            return metrics
        
    def _diag_pointer_soft_gap(
        self,
        *,
        comp: PointerCompressorOutput,
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        with torch.inference_mode():
            p_train = comp.pointer_soft.to(torch.float32)
            p_strict = comp.pointer_soft_strict.to(torch.float32)
            idx = comp.hard_indices.unsqueeze(-1)

            tv = 0.5 * (p_train - p_strict).abs().sum(dim=-1).mean()
            p_train_at_hard = p_train.gather(dim=-1, index=idx).mean()
            p_strict_at_hard = p_strict.gather(dim=-1, index=idx).mean()

            ent_train = -(p_train * torch.log(p_train.clamp_min(eps))).sum(dim=-1).mean()
            ent_strict = -(p_strict * torch.log(p_strict.clamp_min(eps))).sum(dim=-1).mean()

            return {
                "pointer_soft_gap/tv_mean": tv,
                "pointer_soft_gap/p_train_at_hard": p_train_at_hard,
                "pointer_soft_gap/p_strict_at_hard": p_strict_at_hard,
                "pointer_soft_gap/entropy_train": ent_train,
                "pointer_soft_gap/entropy_strict": ent_strict,
            }
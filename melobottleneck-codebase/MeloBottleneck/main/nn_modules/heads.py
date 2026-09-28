from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelHead(nn.Module):
    """
    Language model head for composite tokens (pitch, duration, dt).

    输入:
        h: FloatTensor, [B, S, d_model]

    输出:
        pitch_logits:    [B, S, n_pitch]
        duration_logits: [B, S, n_duration]
        dt_logits:       [B, S, n_dt]
    """

    def __init__(
        self,
        d_model: int,
        d_embed: int,
        pitch_embed: nn.Embedding,
        duration_embed: nn.Embedding,
        dt_embed: nn.Embedding,
        use_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_embed = d_embed

        # 持有与 InputEmbedding 相同的 embedding（参数共享）
        self.pitch_embed = pitch_embed
        self.duration_embed = duration_embed
        self.dt_embed = dt_embed

        # W↓: [d_model -> 3*d_embed]
        self.proj_down = nn.Linear(d_model, 3 * d_embed)

        # 可选的三个 bias（不与 embedding tying）
        if use_bias:
            self.pitch_bias = nn.Parameter(
                torch.zeros(pitch_embed.num_embeddings)
            )
            self.duration_bias = nn.Parameter(
                torch.zeros(duration_embed.num_embeddings)
            )
            self.dt_bias = nn.Parameter(
                torch.zeros(dt_embed.num_embeddings)
            )
        else:
            self.register_parameter("pitch_bias", None)
            self.register_parameter("duration_bias", None)
            self.register_parameter("dt_bias", None)

    def forward(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h: [B, S, d_model]
        returns:
            pitch_logits:    [B, S, n_pitch]
            duration_logits: [B, S, n_duration]
            dt_logits:       [B, S, n_dt]
        """
        # [B, S, d_model] -> [B, S, 3*d_embed]
        x = self.proj_down(h)

        # 3 份: [B, S, d_embed] * 3
        pitch_h, dur_h, dt_h = x.chunk(3, dim=-1)

        # D_P = E_P^T:  F.linear(input, weight) = input @ weight.T
        pitch_logits = F.linear(pitch_h, self.pitch_embed.weight, self.pitch_bias)
        duration_logits = F.linear(
            dur_h, self.duration_embed.weight, self.duration_bias
        )
        dt_logits = F.linear(dt_h, self.dt_embed.weight, self.dt_bias)

        # 不在这里做 softmax，留给 loss / 推理阶段使用
        return pitch_logits, duration_logits, dt_logits

class MusicBartMultiAttrLMHead(nn.Module):
    """
    你的 LanguageModelHead 的轻包装：强调“多属性 logits”输出语义。
    """

    def __init__(
        self,
        *,
        d_model: int,
        d_embed: int,
        pitch_embed: nn.Embedding,
        duration_embed: nn.Embedding,
        dt_embed: nn.Embedding,
        use_bias: bool = True,
    ):
        super().__init__()
        self.head = LanguageModelHead(
            d_model=d_model,
            d_embed=d_embed,
            pitch_embed=pitch_embed,
            duration_embed=duration_embed,
            dt_embed=dt_embed,
            use_bias=use_bias,
        )

    def forward(self, hidden: torch.Tensor):
        # return: pitch_logits, duration_logits, dt_logits
        return self.head(hidden)


class TokenClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, num_classes)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(hidden))


class AWALPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.W = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        H:    [B, T, D]
        mask: [B, T]  1 for valid, 0 for pad
        """
        e = self.v(torch.tanh(self.W(H))).squeeze(-1)  # [B, T]
        pad_mask = (mask == 0)
        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(pad_mask, neg_inf)
        alpha = torch.softmax(e, dim=-1)               # [B, T]
        s = torch.sum(alpha.unsqueeze(-1) * H, dim=1)  # [B, D]
        return s


class SequenceClassificationHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1, pooling: str = "awal"):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        if pooling == "awal":
            self.pool = AWALPooling(d_model)
        elif pooling == "mean":
            self.pool = None
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor):
        if self.pool is None:
            # masked mean
            m = mask.to(hidden.dtype).unsqueeze(-1)  # [B,T,1]
            pooled = (hidden * m).sum(dim=1) / (m.sum(dim=1).clamp_min(1.0))
        else:
            pooled = self.pool(hidden, mask)
        logits = self.classifier(self.dropout(pooled))
        return logits, pooled

class SkeletonPointerHead(nn.Module):
    """
    logits_{t,l} = <Wq h_t, Wk m_l> / sqrt(d_head)
    """
    def __init__(self, d_model: int, d_head: int, normalize: bool = False):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.scale = float(d_head) ** -0.5
        self.normalize = normalize
    def forward(self, h_t: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        h_t:   [B, D]
        memory:[B, L, D]
        return logits: [B, L]
        """
        q = self.q_proj(h_t)            # [B, H]
        k = self.k_proj(memory)         # [B, L, H]
        if self.normalize:
            q = F.normalize(q, dim=-1)                   # [B, H]
            k = F.normalize(k, dim=-1)                   # [B, L, H]
        logits = torch.einsum("bh,blh->bl", q, k)
        return logits

class SkeletonTokenScoreHead(nn.Module):
    """
    Shared tokenwise scorer for:
      - O2B-Learner keep baseline
      - full model encoder_topk / encoder_topk_sampling

    Input:
        hidden: [B, L, D]
    Output:
        score_logits: [B, L]
    """
    def __init__(self, d_model: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        hidden = int(hidden)
        dropout = float(dropout)

        if hidden <= 0:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
        else:
            self.net = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)   # [B,L]
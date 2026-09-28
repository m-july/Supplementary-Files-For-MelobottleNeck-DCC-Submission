import torch
from torch import nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    """
    Input embedding for composite tokens (pitch, duration, dt).

    输入:
        tokens: LongTensor, shape [B, S, 3]
                最后一维分别为 (pitch_id, duration_id, dt_id)

    输出:
        x: FloatTensor, shape [B, S, d_model]
    """

    def __init__(
        self,
        n_pitch: int,
        n_duration: int,
        n_dt: int,
        d_embed: int,
        d_model: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.d_embed = d_embed
        self.d_model = d_model

        # E_P, E_D, E_Δ: [n_x, d_embed]
        self.pitch_embed = nn.Embedding(n_pitch, d_embed)
        self.duration_embed = nn.Embedding(n_duration, d_embed)
        self.dt_embed = nn.Embedding(n_dt, d_embed)

        # 位置编码: [max_seq_len, d_model]
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

        # W↑: [3*d_embed -> d_model]
        self.proj_up = nn.Linear(3 * d_embed, d_model)

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        """
        tokens: [B, S, 3]
        return: [B, S, d_model]
        """
        B, S, three = tokens.shape
        assert three == 3, "Last dim of tokens must be 3 (pitch, duration, dt)."

        # 拆出三个属性: [B, S]
        pitch_ids = tokens[..., 0]
        dur_ids = tokens[..., 1]
        dt_ids = tokens[..., 2]

        # 分别 lookup：得到 [B, S, d_embed]
        pitch_emb = self.pitch_embed(pitch_ids)
        dur_emb = self.duration_embed(dur_ids)
        dt_emb = self.dt_embed(dt_ids)

        # concat -> [B, S, 3*d_embed]
        x = torch.cat([pitch_emb, dur_emb, dt_emb], dim=-1)

        # W↑ 投影到 d_model -> [B, S, d_model]
        x = self.proj_up(x)

        # 位置编码: [S] -> [1, S, d_model] -> broadcast 到 [B, S, d_model]
        pos_ids = torch.arange(S, device=tokens.device)
        pos = self.pos_embed(pos_ids)[None, :, :]

        x = x + pos  # Add positional embedding
        return x

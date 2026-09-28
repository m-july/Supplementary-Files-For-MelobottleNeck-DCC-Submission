import torch
from torch import nn
import torch.nn.functional as F


def multi_attribute_ce_loss(
    labels: torch.LongTensor,
    pitch_logits: torch.Tensor,
    duration_logits: torch.Tensor,
    dt_logits: torch.Tensor,
    attr_weights=(1.0, 1.0, 1.0),
    ignore_index: int = -100,
    reduction: str = "mean",
):
    """
    计算 (pitch, duration, dt) 三个属性的加权 cross-entropy loss。
    参数:
        labels: LongTensor, [B, S, 3]
        pitch_logits:    [B, S, n_pitch]
        duration_logits: [B, S, n_duration]
        dt_logits:       [B, S, n_dt]
        attr_weights: (w_pitch, w_duration, w_dt)，会自动归一化使和为 1
        ignore_index: 与 nn.CrossEntropyLoss 一致，用于 padding 等
        reduction: 'mean' | 'sum' | 'none'
    返回:
        total_loss: 标量或张量（取决于 reduction）
        loss_dict:  dict，包含三个单独 loss，方便监控
    """
    assert labels.shape[:2] == pitch_logits.shape[:2], "B, S must match."
    B, S, three = labels.shape
    assert three == 3, "Last dim of labels must be 3."
    # 拆 label: [B, S]
    pitch_target = labels[..., 0].contiguous()
    duration_target = labels[..., 1].contiguous()
    dt_target = labels[..., 2].contiguous()
    # 展平 batch & seq，以便用 F.cross_entropy
    pitch_loss = F.cross_entropy(
        pitch_logits.view(-1, pitch_logits.size(-1)),
        pitch_target.view(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )
    duration_loss = F.cross_entropy(
        duration_logits.view(-1, duration_logits.size(-1)),
        duration_target.view(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )
    dt_loss = F.cross_entropy(
        dt_logits.view(-1, dt_logits.size(-1)),
        dt_target.view(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )
    # 归一化属性权重
    w = torch.tensor(
        attr_weights,
        dtype=pitch_loss.dtype,
        device=pitch_loss.device,
    )
    w = w / w.sum()
    total_loss = w[0] * pitch_loss + w[1] * duration_loss + w[2] * dt_loss
    loss_dict = {
        "loss_total": total_loss,
        "loss_pitch": pitch_loss,
        "loss_duration": duration_loss,
        "loss_dt": dt_loss,
        "w_pitch": w[0].item(),
        "w_duration": w[1].item(),
        "w_dt": w[2].item(),
    }
    return total_loss, loss_dict
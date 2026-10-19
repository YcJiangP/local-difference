"""Temporal readouts with the reference checkpoint parameterization."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TaskBackbone(nn.Module):
    def __init__(self, hidden=32, dropout=0.2):
        super().__init__()
        self.project = nn.Linear(768, hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        initial = nn.Linear(hidden, 1)
        self.task_weight = nn.Parameter(initial.weight.detach().repeat(3, 1))
        self.task_bias = nn.Parameter(initial.bias.detach().repeat(3))


def encode_bins(module, x, duration):
    if x.ndim != 4 or x.shape[1] != 3 or x.shape[-1] != 768:
        raise ValueError("Expected [participants, 3, bins, 768] neural inputs.")
    if duration.shape != x.shape[:3] or not torch.isfinite(x).all():
        raise ValueError("Invalid neural input shapes or values.")
    if not torch.isfinite(duration).all() or (duration < 0).any():
        raise ValueError("Durations must be finite and nonnegative.")
    mask = duration > 0
    lengths = mask.sum(-1)
    expected = torch.arange(x.shape[2], device=duration.device) < lengths.unsqueeze(-1)
    if not (lengths > 0).all() or not torch.equal(mask, expected):
        raise ValueError("Each task needs positive-duration bins followed only by padding.")
    batch, tasks, length, _ = x.shape
    hidden = F.gelu(module.project(x)) * mask.unsqueeze(-1)
    context = module.depthwise(hidden.reshape(batch * tasks, length, -1).transpose(1, 2))
    context = context.transpose(1, 2).reshape(batch, tasks, length, -1)
    hidden = module.dropout(module.norm(hidden + module.mix(F.gelu(context))))
    hidden = hidden * mask.unsqueeze(-1)
    weights = duration / duration.sum(-1, keepdim=True)
    pooled = (weights.unsqueeze(-1) * hidden).sum(-2)
    return pooled, hidden, weights


class BinaryReadout(nn.Module):
    def __init__(self, hidden=32, dropout=0.2):
        super().__init__()
        self.backbone = TaskBackbone(hidden, dropout)
        # Preserve the two-parameter mean head and its optimizer trajectory.
        self.stat_weight = nn.Parameter(torch.zeros(3, hidden))

    def forward(self, x, duration):
        pooled, _, weights = encode_bins(self.backbone, x, duration)
        task_logits = (pooled * self.backbone.task_weight[None]).sum(-1)
        task_logits = task_logits + self.backbone.task_bias
        task_logits = task_logits + (pooled * self.stat_weight[None]).sum(-1)
        return {"logits": task_logits.mean(1), "task_logits": task_logits, "weights": weights}


class MulticlassReadout(nn.Module):
    def __init__(self, hidden=64, dropout=0.2, classes=3):
        super().__init__()
        self.classes = classes
        self.project = nn.Linear(768, hidden)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.mix = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        initial = nn.Linear(hidden, classes)
        self.task_weight = nn.Parameter(initial.weight.detach()[None].repeat(3, 1, 1))
        self.task_bias = nn.Parameter(initial.bias.detach()[None].repeat(3, 1))
        self.stat_weight = nn.Parameter(torch.zeros(3, classes, hidden))

    def forward(self, x, duration):
        pooled, _, weights = encode_bins(self, x, duration)
        task_logits = torch.einsum(
            "bth,tch->btc", pooled, self.task_weight + self.stat_weight
        ) + self.task_bias
        return {"logits": task_logits.mean(1), "task_logits": task_logits, "weights": weights}


def class_weights(labels, classes=3):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=classes)
    if (counts == 0).any():
        raise ValueError("A fitting partition is missing a class.")
    return len(labels) / (classes * counts)


def neural_loss(output, labels, weight, classes=2):
    if classes == 2:
        person = F.binary_cross_entropy_with_logits(output["logits"], labels, pos_weight=weight)
        task = F.binary_cross_entropy_with_logits(
            output["task_logits"], labels[:, None].expand(-1, 3), pos_weight=weight
        )
    else:
        person = F.cross_entropy(output["logits"], labels, weight=weight)
        task = F.cross_entropy(
            output["task_logits"].reshape(-1, classes),
            labels[:, None].expand(-1, 3).reshape(-1),
            weight=weight,
        )
    return person + 0.5 * task


def batch(items, device="cpu"):
    length = max(len(item["x"][task]) for item in items for task in range(3))
    x = torch.zeros(len(items), 3, length, 768)
    duration = torch.zeros(len(items), 3, length)
    for i, item in enumerate(items):
        for task in range(3):
            count = len(item["x"][task])
            x[i, task, :count] = torch.from_numpy(item["x"][task])
            duration[i, task, :count] = torch.from_numpy(item["duration"][task])
    return x.to(device), duration.to(device)

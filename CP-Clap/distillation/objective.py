from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .feature import normalized_huber_feature_loss
from .logits import (
    kl_from_teacher_probability,
    softened_distribution,
)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    label: torch.Tensor
    logit: torch.Tensor
    feature: torch.Tensor

    def detached(self) -> dict:
        return {
            "total": float(self.total.detach().cpu()),
            "label": float(self.label.detach().cpu()),
            "logit": float(self.logit.detach().cpu()),
            "feature": float(self.feature.detach().cpu()),
        }


def stage2_objective(
    student_logits: torch.Tensor,
    aligned_logits: torch.Tensor,
    labels: torch.Tensor,
    student_features: Sequence[torch.Tensor],
    aligned_features: Sequence[torch.Tensor],
    alpha: float = 0.60,
    beta: float = 0.80,
    temperature: float = 5.0,
    huber_delta: float = 1.0,
) -> LossBreakdown:
    ce = F.cross_entropy(student_logits, labels)
    teacher_prob = softened_distribution(aligned_logits.detach(), temperature)
    kd = kl_from_teacher_probability(student_logits, teacher_prob, temperature)
    feat = normalized_huber_feature_loss(
        student_features,
        aligned_features,
        delta=huber_delta,
        expected_points=4,
    )
    total = (1.0 - alpha) * ce + alpha * kd + beta * feat
    return LossBreakdown(total, ce, kd, feat)


def stage3_objective(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.95,
    temperature: float = 8.0,
) -> LossBreakdown:
    ce = F.cross_entropy(student_logits, labels)
    kd = kl_from_teacher_probability(
        student_logits,
        teacher_probability,
        temperature,
    )
    zero = student_logits.new_zeros(())
    total = (1.0 - alpha) * ce + alpha * kd
    return LossBreakdown(total, ce, kd, zero)

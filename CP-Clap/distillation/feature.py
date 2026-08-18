from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


def l2_normalize_feature_map(
    feature: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L2 normalize each example across the complete feature map."""
    shape = feature.shape
    flat = feature.reshape(feature.size(0), -1)
    norm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return (flat / norm).reshape(shape)


def normalized_huber_feature_loss(
    student_features: Sequence[torch.Tensor],
    teacher_features: Sequence[torch.Tensor],
    delta: float = 1.0,
    expected_points: int | None = 4,
) -> torch.Tensor:
    """Average normalized Huber matching over aligned feature positions."""
    if len(student_features) != len(teacher_features):
        raise ValueError("student and teacher feature lists must have equal length")
    if expected_points is not None and len(student_features) != expected_points:
        raise ValueError(
            f"expected {expected_points} aligned feature points, got {len(student_features)}"
        )
    if not student_features:
        raise ValueError("at least one aligned feature pair is required")

    losses = []
    for student, teacher in zip(student_features, teacher_features):
        if student.shape != teacher.shape:
            raise ValueError(
                f"aligned feature shape mismatch: {student.shape} vs {teacher.shape}"
            )
        s = l2_normalize_feature_map(student)
        t = l2_normalize_feature_map(teacher.detach())
        losses.append(F.huber_loss(s, t, delta=delta, reduction="mean"))
    return torch.stack(losses).mean()

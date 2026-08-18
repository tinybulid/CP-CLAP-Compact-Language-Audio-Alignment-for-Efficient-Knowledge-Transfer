from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def softened_distribution(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits / temperature, dim=1)


def student_log_distribution(
    logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return F.log_softmax(logits / temperature, dim=1)


def kl_from_teacher_probability(
    student_logits: torch.Tensor,
    teacher_probability: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return (
        F.kl_div(
            student_log_distribution(student_logits, temperature),
            teacher_probability.detach(),
            reduction="batchmean",
        )
        * (temperature ** 2)
    )


@torch.no_grad()
def ensemble_teacher_probability(
    teachers: Sequence[nn.Module],
    inputs: torch.Tensor,
    temperature: float,
    weights: Sequence[float] | None = None,
) -> torch.Tensor:
    """Average temperature-scaled probability distributions."""
    if not teachers:
        raise ValueError("at least one external teacher is required")
    if weights is None:
        weights = [1.0 / len(teachers)] * len(teachers)
    if len(weights) != len(teachers):
        raise ValueError("teacher weights must match the number of teachers")

    total = float(sum(weights))
    if total <= 0:
        raise ValueError("teacher weights must sum to a positive value")
    weights = [float(w) / total for w in weights]

    ensemble = None
    for model, weight in zip(teachers, weights):
        logits = model(inputs)
        probs = softened_distribution(logits, temperature)
        ensemble = probs * weight if ensemble is None else ensemble + probs * weight
    return ensemble

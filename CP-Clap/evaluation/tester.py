from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..batches import move_batch, parse_batch
from .metrics import confusion_matrix, macro_recall_from_confusion


@dataclass
class EvaluationReport:
    loss: float
    accuracy: float
    macro_recall: float
    samples: int
    confusion: torch.Tensor
    device_accuracy: dict[int, float] = field(default_factory=dict)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    batches: Iterable,
    device: str | torch.device,
    num_classes: int,
) -> EvaluationReport:
    device = torch.device(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)
    device_stats = defaultdict(lambda: [0, 0])

    for raw in batches:
        batch = move_batch(parse_batch(raw), device)
        logits = model(batch.inputs)
        labels = batch.labels

        total_loss += float(F.cross_entropy(logits, labels, reduction="sum").cpu())
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total_samples += labels.numel()
        matrix += confusion_matrix(
            preds.detach().cpu(),
            labels.detach().cpu(),
            num_classes,
        )

        if batch.device_ids is not None:
            for pred, label, dev in zip(preds, labels, batch.device_ids):
                key = int(dev.detach().cpu().item())
                device_stats[key][0] += int(pred.item() == label.item())
                device_stats[key][1] += 1

    if total_samples == 0:
        raise ValueError("evaluation iterable produced no samples")

    per_device = {
        key: correct / max(total, 1)
        for key, (correct, total) in sorted(device_stats.items())
    }
    return EvaluationReport(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        macro_recall=macro_recall_from_confusion(matrix),
        samples=total_samples,
        confusion=matrix,
        device_accuracy=per_device,
    )


@torch.no_grad()
def predict(
    model: nn.Module,
    inputs: torch.Tensor,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    inputs = inputs.to(device)
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.argmax(dim=1), probabilities

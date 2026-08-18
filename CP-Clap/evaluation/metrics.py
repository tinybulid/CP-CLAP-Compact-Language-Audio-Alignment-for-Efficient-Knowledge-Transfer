from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().detach().cpu())


def confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    predictions = predictions.view(-1).to(torch.long)
    labels = labels.view(-1).to(torch.long)
    bins = labels * num_classes + predictions
    counts = torch.bincount(bins, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def macro_recall_from_confusion(matrix: torch.Tensor) -> float:
    matrix = matrix.float()
    recall = matrix.diag() / matrix.sum(dim=1).clamp_min(1.0)
    return float(recall.mean().cpu())


def per_device_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    device_ids: torch.Tensor,
) -> Dict[int, float]:
    stats = defaultdict(lambda: [0, 0])
    for pred, label, dev in zip(
        predictions.detach().cpu(),
        labels.detach().cpu(),
        device_ids.detach().cpu(),
    ):
        key = int(dev.item())
        stats[key][0] += int(pred.item() == label.item())
        stats[key][1] += 1
    return {k: correct / max(total, 1) for k, (correct, total) in stats.items()}

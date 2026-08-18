from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import torch
import torch.nn as nn

from ..batches import move_batch, parse_batch
from ..checkpoints import save_training_checkpoint


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_optimizer(model: nn.Module, learning_rate: float, weight_decay: float = 0.0):
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def make_cosine_scheduler(optimizer, epochs: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs),
    )


@dataclass
class EpochStats:
    loss: float
    accuracy: float
    samples: int
    pieces: Dict[str, float] = field(default_factory=dict)


@dataclass
class TrainingHistory:
    train: list[EpochStats] = field(default_factory=list)
    validation: list[EpochStats] = field(default_factory=list)
    best_accuracy: float = float("-inf")
    best_epoch: int = -1


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.bad_epochs = 0
        self.best = float("-inf")

    def update(self, metric: float) -> bool:
        if metric > self.best:
            self.best = metric
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@torch.no_grad()
def simple_validation(
    model: nn.Module,
    batches: Iterable,
    device: torch.device,
) -> EpochStats:
    model.eval()
    loss_sum = 0.0
    correct = 0
    samples = 0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for raw in batches:
        batch = move_batch(parse_batch(raw), device)
        logits = model(batch.inputs)
        loss_sum += float(criterion(logits, batch.labels).cpu())
        correct += int((logits.argmax(1) == batch.labels).sum().item())
        samples += batch.labels.numel()

    if samples == 0:
        raise ValueError("validation iterable produced no samples")
    return EpochStats(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
    )


def maybe_save_best(
    history: TrainingHistory,
    validation: EpochStats,
    epoch: int,
    path: str | Path,
    model: nn.Module,
    optimizer,
    scheduler,
    config,
    extra: Optional[dict] = None,
) -> bool:
    if validation.accuracy <= history.best_accuracy:
        return False
    history.best_accuracy = validation.accuracy
    history.best_epoch = epoch
    save_training_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        best_metric=validation.accuracy,
        config=config,
        extra=extra,
    )
    return True

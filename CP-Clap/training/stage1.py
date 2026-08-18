from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..augmentations import SpectrogramAugment
from ..batches import move_batch, parse_batch
from ..config import Stage1Config
from .engine import (
    EarlyStopping,
    EpochStats,
    TrainingHistory,
    make_cosine_scheduler,
    make_optimizer,
    maybe_save_best,
    simple_validation,
)


def train_stage1(
    aligned_teacher: nn.Module,
    train_batches: Iterable,
    validation_batches: Iterable,
    device: str | torch.device,
    config: Stage1Config = Stage1Config(),
    checkpoint_path: str | Path | None = None,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> TrainingHistory:
    """Stage I: train the aligned higher-capacity model with label CE only."""
    device = torch.device(device)
    model = aligned_teacher.to(device)
    optimizer = make_optimizer(
        model,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
    )
    scheduler = make_cosine_scheduler(optimizer, config.epochs)
    checkpoint_path = checkpoint_path or config.checkpoint_name
    stopper = EarlyStopping(config.early_stopping_patience)
    history = TrainingHistory()

    for epoch in range(config.epochs):
        model.train()
        loss_sum = 0.0
        correct = 0
        samples = 0

        for raw in train_batches:
            batch = move_batch(parse_batch(raw), device)
            x = batch.inputs
            if augment is not None:
                x = augment(x)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, batch.labels)
            loss.backward()
            optimizer.step()

            n = batch.labels.numel()
            loss_sum += float(loss.detach().cpu()) * n
            correct += int((logits.argmax(1) == batch.labels).sum().item())
            samples += n

        scheduler.step()
        if samples == 0:
            raise ValueError("training iterable produced no samples")

        train_stats = EpochStats(loss_sum / samples, correct / samples, samples)
        val_stats = simple_validation(model, validation_batches, device)
        history.train.append(train_stats)
        history.validation.append(val_stats)

        improved = maybe_save_best(
            history,
            val_stats,
            epoch,
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            config,
            extra={"stage": 1},
        )
        if not improved and stopper.update(val_stats.accuracy):
            break
        if improved:
            stopper.best = val_stats.accuracy
            stopper.bad_epochs = 0

    return history

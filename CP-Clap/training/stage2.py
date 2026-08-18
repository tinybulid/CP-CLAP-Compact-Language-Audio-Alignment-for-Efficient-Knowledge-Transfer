from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from ..batches import move_batch, parse_batch
from ..config import Stage2Config
from ..distillation.objective import stage2_objective
from .engine import (
    EarlyStopping,
    EpochStats,
    TrainingHistory,
    make_cosine_scheduler,
    make_optimizer,
    maybe_save_best,
    simple_validation,
)


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def train_stage2(
    student: nn.Module,
    aligned_teacher: nn.Module,
    train_batches: Iterable,
    validation_batches: Iterable,
    device: str | torch.device,
    config: Stage2Config = Stage2Config(),
    checkpoint_path: str | Path | None = None,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> TrainingHistory:
    """Stage II: representation and aligned-logit transfer into the student."""
    device = torch.device(device)
    student = student.to(device)
    aligned_teacher = _freeze(aligned_teacher.to(device))

    optimizer = make_optimizer(
        student,
        config.optimizer.learning_rate,
        config.optimizer.weight_decay,
    )
    scheduler = make_cosine_scheduler(optimizer, config.epochs)
    checkpoint_path = checkpoint_path or config.checkpoint_name
    stopper = EarlyStopping(config.early_stopping_patience)
    history = TrainingHistory()

    for epoch in range(config.epochs):
        student.train()
        loss_sum = 0.0
        correct = 0
        samples = 0
        piece_sums = {"label": 0.0, "logit": 0.0, "feature": 0.0}

        for raw in train_batches:
            batch = move_batch(parse_batch(raw), device)
            x = batch.inputs
            if augment is not None:
                x = augment(x)

            with torch.no_grad():
                teacher_logits, teacher_features = aligned_teacher.forward_with_features(x)

            student_logits, student_features = student.forward_with_features(x)
            breakdown = stage2_objective(
                student_logits=student_logits,
                aligned_logits=teacher_logits,
                labels=batch.labels,
                student_features=student_features,
                aligned_features=teacher_features,
                alpha=config.alpha,
                beta=config.beta,
                temperature=config.temperature,
                huber_delta=config.huber_delta,
            )

            optimizer.zero_grad(set_to_none=True)
            breakdown.total.backward()
            optimizer.step()

            n = batch.labels.numel()
            loss_sum += float(breakdown.total.detach().cpu()) * n
            correct += int((student_logits.argmax(1) == batch.labels).sum().item())
            samples += n
            piece_sums["label"] += float(breakdown.label.detach().cpu()) * n
            piece_sums["logit"] += float(breakdown.logit.detach().cpu()) * n
            piece_sums["feature"] += float(breakdown.feature.detach().cpu()) * n

        scheduler.step()
        if samples == 0:
            raise ValueError("training iterable produced no samples")

        train_stats = EpochStats(
            loss=loss_sum / samples,
            accuracy=correct / samples,
            samples=samples,
            pieces={k: v / samples for k, v in piece_sums.items()},
        )
        val_stats = simple_validation(student, validation_batches, device)
        history.train.append(train_stats)
        history.validation.append(val_stats)

        improved = maybe_save_best(
            history,
            val_stats,
            epoch,
            checkpoint_path,
            student,
            optimizer,
            scheduler,
            config,
            extra={"stage": 2},
        )
        if not improved and stopper.update(val_stats.accuracy):
            break
        if improved:
            stopper.best = val_stats.accuracy
            stopper.bad_epochs = 0

    return history

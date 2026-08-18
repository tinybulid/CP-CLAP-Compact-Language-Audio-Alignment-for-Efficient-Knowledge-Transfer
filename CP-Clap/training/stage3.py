from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from ..batches import move_batch, parse_batch
from ..config import Stage3Config
from ..distillation.logits import ensemble_teacher_probability
from ..distillation.objective import stage3_objective
from .engine import (
    EarlyStopping,
    EpochStats,
    TrainingHistory,
    make_cosine_scheduler,
    make_optimizer,
    maybe_save_best,
    simple_validation,
)


def _freeze_all(models: Sequence[nn.Module]) -> list[nn.Module]:
    result = []
    for model in models:
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        result.append(model)
    return result


def train_stage3(
    student: nn.Module,
    external_teachers: Sequence[nn.Module],
    train_batches: Iterable,
    validation_batches: Iterable,
    device: str | torch.device,
    config: Stage3Config = Stage3Config(),
    checkpoint_path: str | Path | None = None,
    teacher_weights: Sequence[float] | None = None,
    augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> TrainingHistory:
    """Stage III: strong probability-ensemble distillation into the student."""
    if not external_teachers:
        raise ValueError("Stage III requires at least one external teacher")

    device = torch.device(device)
    student = student.to(device)
    teachers = _freeze_all([model.to(device) for model in external_teachers])

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
        piece_sums = {"label": 0.0, "logit": 0.0}

        for raw in train_batches:
            batch = move_batch(parse_batch(raw), device)
            x = batch.inputs
            if augment is not None:
                x = augment(x)

            teacher_probability = ensemble_teacher_probability(
                teachers=teachers,
                inputs=x,
                temperature=config.temperature,
                weights=teacher_weights,
            )
            student_logits = student(x)
            breakdown = stage3_objective(
                student_logits=student_logits,
                teacher_probability=teacher_probability,
                labels=batch.labels,
                alpha=config.alpha,
                temperature=config.temperature,
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
            extra={"stage": 3, "teacher_count": len(teachers)},
        )
        if not improved and stopper.update(val_stats.accuracy):
            break
        if improved:
            stopper.best = val_stats.accuracy
            stopper.bad_epochs = 0

    return history

"""Reference CP-CLAP training loop separated from the model definition.

This module is intentionally a structured reference rather than a turnkey
script.  Dataset construction, exact checkpoint locations, and project-specific
logging remain external to the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from ...checkpoints import save_training_checkpoint
from ...models.teachers.cp_clap.model import CPCLAPTeacher
from .config import CPCLAPTrainingConfig
from .losses import CPCLAPLossBreakdown, cpclap_objective


@dataclass
class CPCLAPEpochStats:
    loss: float
    accuracy: float
    samples: int
    pieces: dict[str, float] = field(default_factory=dict)


def build_optimizer(
    model: CPCLAPTeacher,
    config: CPCLAPTrainingConfig,
) -> torch.optim.Optimizer:
    """Use the paper-level Adam configuration, not the notebook's AdamW setup."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=config.learning_rate)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: CPCLAPTrainingConfig,
):
    if config.epochs is None:
        return None
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
    )


def _parse_batch(batch):
    """Accept (inputs, devices, labels) or (inputs, labels) reference batches."""
    if isinstance(batch, Mapping):
        return batch["inputs"], batch.get("devices"), batch["labels"]
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    if len(batch) == 2:
        return batch[0], None, batch[1]
    raise ValueError("expected batch as mapping, (x,y), or (x,device,y)")


def _active_class_prototypes(
    model: CPCLAPTeacher,
    labels: torch.Tensor,
    device: torch.device,
):
    class_ids = torch.unique(labels, sorted=True)
    text = model.encode_class_prototypes(class_ids, device=device)
    return class_ids, text


def train_cpclap_epoch(
    model: CPCLAPTeacher,
    batches: Iterable,
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    config: CPCLAPTrainingConfig = CPCLAPTrainingConfig(),
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> CPCLAPEpochStats:
    """Train one epoch using the paper's CP-CLAP objective."""
    device = torch.device(device)
    model.to(device)
    model.train()
    if not model.has_fixed_class_embeddings:
        model.build_fixed_class_embeddings(device=device)

    loss_sum = 0.0
    correct = 0
    samples = 0
    pieces = {
        "a2t": 0.0,
        "t2a": 0.0,
        "contrastive": 0.0,
        "classification": 0.0,
    }

    for raw_batch in batches:
        inputs, _, labels = _parse_batch(raw_batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        if augment is not None:
            inputs = augment(inputs)

        audio_embeddings = model.encode_audio(inputs)
        active_ids, active_text = _active_class_prototypes(model, labels, device)
        breakdown = cpclap_objective(
            audio_embeddings=audio_embeddings,
            labels=labels,
            active_class_ids=active_ids,
            active_text_embeddings=active_text,
            fixed_text_embeddings=model.fixed_class_embeddings,
            tau_ta=model.tau_ta,
            tau_cls=model.tau_cls,
            alpha=config.alpha,
            beta=config.beta,
        )

        optimizer.zero_grad(set_to_none=True)
        breakdown.total.backward()
        optimizer.step()

        logits = model.classification_similarity(
            audio_embeddings,
            model.fixed_class_embeddings,
        )
        batch_n = labels.numel()
        samples += batch_n
        loss_sum += float(breakdown.total.detach().cpu()) * batch_n
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        pieces["a2t"] += float(breakdown.audio_to_text.detach().cpu()) * batch_n
        pieces["t2a"] += float(breakdown.text_to_audio.detach().cpu()) * batch_n
        pieces["contrastive"] += float(breakdown.contrastive.detach().cpu()) * batch_n
        pieces["classification"] += float(breakdown.classification.detach().cpu()) * batch_n

    if samples == 0:
        raise ValueError("training iterable produced no samples")
    return CPCLAPEpochStats(
        loss=loss_sum / samples,
        accuracy=correct / samples,
        samples=samples,
        pieces={key: value / samples for key, value in pieces.items()},
    )


@torch.no_grad()
def evaluate_cpclap(
    model: CPCLAPTeacher,
    batches: Iterable,
    device: str | torch.device,
) -> CPCLAPEpochStats:
    """Evaluate class-similarity logits using fixed CLAP class embeddings."""
    device = torch.device(device)
    model.to(device)
    model.eval()
    if not model.has_fixed_class_embeddings:
        model.build_fixed_class_embeddings(device=device)

    correct = 0
    samples = 0
    for raw_batch in batches:
        inputs, _, labels = _parse_batch(raw_batch)
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        samples += labels.numel()

    if samples == 0:
        raise ValueError("evaluation iterable produced no samples")
    return CPCLAPEpochStats(
        loss=float("nan"),
        accuracy=correct / samples,
        samples=samples,
    )


def train_cpclap_reference(
    model: CPCLAPTeacher,
    train_batches: Iterable,
    validation_batches: Iterable,
    device: str | torch.device,
    config: CPCLAPTrainingConfig = CPCLAPTrainingConfig(),
    augment: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    """High-level reference orchestration for CP-CLAP teacher training.

    The paper does not specify a standalone CP-CLAP epoch count in the supplied
    section.  Set ``config.epochs`` explicitly before using this routine.
    """
    if config.epochs is None:
        raise ValueError("Set CPCLAPTrainingConfig.epochs from your final experiment")

    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    history = {"train": [], "validation": []}
    best_accuracy = float("-inf")

    for epoch in range(config.epochs):
        train_stats = train_cpclap_epoch(
            model,
            train_batches,
            optimizer,
            device,
            config=config,
            augment=augment,
        )
        validation_stats = evaluate_cpclap(model, validation_batches, device)
        history["train"].append(train_stats)
        history["validation"].append(validation_stats)

        if validation_stats.accuracy > best_accuracy:
            best_accuracy = validation_stats.accuracy
            save_training_checkpoint(
                Path(config.checkpoint_path),
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_accuracy,
                config=config,
                extra={"component": "cp_clap_teacher"},
            )
        if scheduler is not None:
            scheduler.step()

    return history

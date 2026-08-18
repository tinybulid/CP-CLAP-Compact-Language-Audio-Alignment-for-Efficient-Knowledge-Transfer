from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch.nn as nn

from ..checkpoints import load_model_checkpoint
from ..config import TrainingConfig
from ..models.aligned_teacher import AlignedLightEchoNet
from ..models.student import StudentLightEchoNet
from .engine import choose_device, set_seed
from .stage1 import train_stage1
from .stage2 import train_stage2
from .stage3 import train_stage3


@dataclass
class PipelineArtifacts:
    aligned_teacher: nn.Module
    student: nn.Module
    stage1_history: object
    stage2_history: object
    stage3_history: object


def run_three_stage_reference_pipeline(
    train_batches: Iterable,
    validation_batches: Iterable,
    external_teachers: Sequence[nn.Module],
    config: TrainingConfig = TrainingConfig(),
) -> PipelineArtifacts:
    """Approximate end-to-end orchestration.

    The caller owns the batch iterables and all external teacher checkpoint
    preparation.  This function only wires the three optimization stages.
    """
    set_seed(config.seed)
    device = choose_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    aligned = AlignedLightEchoNet(config.num_classes)
    student = StudentLightEchoNet(config.num_classes)

    stage1_path = config.output_dir / config.stage1.checkpoint_name
    h1 = train_stage1(
        aligned,
        train_batches,
        validation_batches,
        device,
        config.stage1,
        stage1_path,
    )
    load_model_checkpoint(aligned, stage1_path, device=device, strict=False)

    stage2_path = config.output_dir / config.stage2.checkpoint_name
    h2 = train_stage2(
        student,
        aligned,
        train_batches,
        validation_batches,
        device,
        config.stage2,
        stage2_path,
    )
    load_model_checkpoint(student, stage2_path, device=device, strict=False)

    stage3_path = config.output_dir / config.stage3.checkpoint_name
    h3 = train_stage3(
        student,
        external_teachers,
        train_batches,
        validation_batches,
        device,
        config.stage3,
        stage3_path,
    )
    load_model_checkpoint(student, stage3_path, device=device, strict=False)

    return PipelineArtifacts(
        aligned_teacher=aligned,
        student=student,
        stage1_history=h1,
        stage2_history=h2,
        stage3_history=h3,
    )

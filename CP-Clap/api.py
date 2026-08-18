from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .config import SpectrumConfig, get_stage_config
from .models.aligned_teacher import AlignedLightEchoNet
from .models.student import StudentLightEchoNet
from .models.teachers import build_external_teacher
from .spectrum import audio_file_to_logmel, waveform_to_logmel


def build_student(num_classes: int = 10) -> StudentLightEchoNet:
    return StudentLightEchoNet(num_classes=num_classes)


def build_aligned_teacher(num_classes: int = 10) -> AlignedLightEchoNet:
    return AlignedLightEchoNet(num_classes=num_classes)


def extract_logmel(
    source,
    source_rate: Optional[int] = None,
    target_frames: Optional[int] = None,
    config: SpectrumConfig = SpectrumConfig(),
) -> torch.Tensor:
    """Convenience wrapper for either a waveform tensor or one audio path."""
    if torch.is_tensor(source):
        if source_rate is None:
            raise ValueError("source_rate is required when source is a waveform tensor")
        return waveform_to_logmel(
            source,
            source_rate=source_rate,
            config=config,
            target_frames=target_frames,
        )
    return audio_file_to_logmel(
        Path(source),
        config=config,
        target_frames=target_frames,
    )

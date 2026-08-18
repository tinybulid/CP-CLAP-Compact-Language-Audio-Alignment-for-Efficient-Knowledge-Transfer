from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SpectrumConfig:
    """Log-Mel settings used by the audio front end."""

    sample_rate: int = 32_000
    n_fft: int = 4_096
    hop_length: int = 502
    n_mels: int = 256
    power: float = 2.0
    top_db: float = 80.0
    center: bool = True
    normalized: bool = False


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    batch_size_reference: int = 64
    scheduler: str = "cosine"


@dataclass(frozen=True)
class Stage1Config:
    """Stage I: fit the higher-capacity aligned network with labels."""

    epochs: int = 60
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping_patience: int = 15
    checkpoint_name: str = "aligned_stage1_best.pth"


@dataclass(frozen=True)
class Stage2Config:
    """Stage II: labels + aligned logits + normalized Huber features."""

    epochs: int = 66
    alpha: float = 0.60
    beta: float = 0.80
    temperature: float = 5.0
    huber_delta: float = 1.0
    expected_feature_points: int = 4
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping_patience: int = 15
    checkpoint_name: str = "student_stage2_best.pth"


@dataclass(frozen=True)
class Stage3Config:
    """Stage III: labels + probability-ensemble teacher distillation."""

    epochs: int = 51
    alpha: float = 0.95
    beta: float = 0.0
    temperature: float = 8.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    early_stopping_patience: int = 15
    checkpoint_name: str = "student_stage3_best.pth"


@dataclass(frozen=True)
class TrainingConfig:
    num_classes: int = 10
    seed: int = 42
    device: Optional[str] = None
    output_dir: Path = Path("checkpoints")
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)


STAGE_CONFIGS = {
    1: Stage1Config(),
    2: Stage2Config(),
    3: Stage3Config(),
}


def get_stage_config(stage: int):
    if stage not in STAGE_CONFIGS:
        raise ValueError(f"stage must be one of {sorted(STAGE_CONFIGS)}")
    return STAGE_CONFIGS[stage]

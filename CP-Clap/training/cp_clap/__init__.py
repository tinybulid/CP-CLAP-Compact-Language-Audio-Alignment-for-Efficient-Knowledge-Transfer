"""Training utilities for the CP-CLAP teacher."""

from .config import CPCLAPTrainingConfig
from .losses import CPCLAPLossBreakdown, cpclap_objective
from .trainer import (
    CPCLAPEpochStats,
    build_optimizer,
    build_scheduler,
    evaluate_cpclap,
    train_cpclap_epoch,
    train_cpclap_reference,
)

__all__ = [
    "CPCLAPTrainingConfig",
    "CPCLAPLossBreakdown",
    "cpclap_objective",
    "CPCLAPEpochStats",
    "build_optimizer",
    "build_scheduler",
    "evaluate_cpclap",
    "train_cpclap_epoch",
    "train_cpclap_reference",
]

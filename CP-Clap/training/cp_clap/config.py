"""Training configuration for CP-CLAP.

Paper-reported hyperparameters take precedence over notebook experiments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CPCLAPTrainingConfig:
    # CPCLAP loss coefficients reported in the paper.
    alpha: float = 0.50
    beta: float = 0.50

    # Shared training-pipeline settings reported for the project.
    learning_rate: float = 1e-3
    batch_size: int = 64
    optimizer: str = "Adam"
    scheduler: str = "CosineAnnealingLR"

    # A standalone CP-CLAP teacher epoch count is not explicitly specified in
    # the supplied paper text.  Keep it user-owned rather than copying the
    # notebook's experimental value.
    epochs: int | None = None

    checkpoint_path: str = "<FILL_CP_CLAP_CHECKPOINT>"

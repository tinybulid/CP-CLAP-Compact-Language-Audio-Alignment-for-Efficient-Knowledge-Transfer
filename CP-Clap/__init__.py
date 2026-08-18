"""Lightweight audio distillation reference package.

The code intentionally contains no data-source definitions.  All training and
evaluation entry points consume externally supplied batches or provider
functions.
"""

from .api import (
    build_aligned_teacher,
    build_student,
    build_external_teacher,
    extract_logmel,
    get_stage_config,
)
from .config import (
    SpectrumConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    TrainingConfig,
)

__all__ = [
    "build_aligned_teacher",
    "build_student",
    "build_external_teacher",
    "extract_logmel",
    "get_stage_config",
    "SpectrumConfig",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
    "TrainingConfig",
]

__version__ = "0.1.0"

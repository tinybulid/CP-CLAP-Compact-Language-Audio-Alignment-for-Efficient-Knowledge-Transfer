"""CP-CLAP heterogeneous teacher components."""

from .audio_encoder import CPMobileCLAPAudioEncoder
from .config import (
    CPCLAPFootprintPlaceholders,
    CPCLAPModelConfig,
    DEFAULT_PROMPT_TEMPLATES,
)
from .model import (
    CPCLAPTeacher,
    CPCLAPTextComponents,
    build_cpclap_from_transformers,
    configure_text_encoder_trainability,
)
from .prompts import build_prompt_bank, prompts_for_class

__all__ = [
    "CPMobileCLAPAudioEncoder",
    "CPCLAPFootprintPlaceholders",
    "CPCLAPModelConfig",
    "DEFAULT_PROMPT_TEMPLATES",
    "CPCLAPTeacher",
    "CPCLAPTextComponents",
    "build_cpclap_from_transformers",
    "configure_text_encoder_trainability",
    "build_prompt_bank",
    "prompts_for_class",
]

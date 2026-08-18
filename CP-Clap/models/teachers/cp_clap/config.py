"""Configuration objects for the CP-CLAP teacher.

The values that are explicitly reported in the project write-up are kept here.
Checkpoint paths and footprint/size figures are intentionally not populated;
those are user-owned deployment artifacts and are represented by placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


DEFAULT_PROMPT_TEMPLATES: tuple[str, ...] = (
    "this is an audio of [class]",
    "[class]",
    "this is [class]",
    "this is a sound of [class]",
)


@dataclass(frozen=True)
class CPCLAPModelConfig:
    """Model-side configuration for CP-CLAP.

    Notes
    -----
    * The paper uses a CP-Mobile audio encoder and CLAP text encoder.
    * Four prompts are averaged for every class.
    * The paper states that more than 80% of the CLAP text encoder is trainable,
      but does not give an exact percentage.  The exact fraction is therefore a
      required integration choice instead of being copied from the notebook.
    """

    num_classes: int = 10
    embedding_dim: int = 512
    prompt_templates: Sequence[str] = field(default_factory=lambda: DEFAULT_PROMPT_TEMPLATES)
    minimum_text_trainable_fraction: float = 0.80

    # User-supplied placeholders.
    cp_mobile_checkpoint: str = "<FILL_CP_MOBILE_CHECKPOINT>"
    clap_text_checkpoint: str = "<FILL_CLAP_TEXT_CHECKPOINT_OR_MODEL_ID>"
    cp_clap_checkpoint: str = "<FILL_CP_CLAP_CHECKPOINT>"


@dataclass(frozen=True)
class CPCLAPFootprintPlaceholders:
    """Intentionally blank footprint values to be completed by the user."""

    audio_encoder_parameters: str = "<FILL_PARAMS>"
    audio_encoder_macs_per_sample: str = "<FILL_MACS_PER_SAMPLE>"
    model_size: str = "<FILL_MODEL_SIZE>"
    peak_gpu_memory: str = "<FILL_PEAK_GPU_MEMORY>"
    latency_per_batch: str = "<FILL_LATENCY_PER_BATCH>"

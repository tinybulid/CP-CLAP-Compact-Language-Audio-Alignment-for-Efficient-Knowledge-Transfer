"""CP-CLAP model definition.

This file isolates the model-side code from the training implementation.  The
teacher combines a compact CP-Mobile audio branch with CLAP text semantics and
can return class logits directly, making it suitable as one heterogeneous
teacher in the package's existing outer-distillation stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_encoder import CPMobileCLAPAudioEncoder
from .config import CPCLAPModelConfig
from .prompts import build_prompt_bank


@dataclass
class CPCLAPTextComponents:
    """Container for text-side components supplied by a CLAP implementation."""

    encoder: nn.Module
    projection: nn.Module
    processor: Any


class CPCLAPTeacher(nn.Module):
    """Compact language-audio teacher using CP-Mobile + CLAP text semantics.

    Two text representations are maintained conceptually:

    1. trainable/partially-trainable class prototypes for bidirectional
       audio-text alignment;
    2. fixed CLAP class prototypes for the semantic classification objective.

    The fixed class embeddings are registered as a buffer after they are built.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        text: CPCLAPTextComponents,
        config: CPCLAPModelConfig | None = None,
        audio_encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config or CPCLAPModelConfig(num_classes=len(class_names))
        self.class_names = tuple(class_names)
        if len(self.class_names) != self.config.num_classes:
            raise ValueError("class_names must match config.num_classes")

        self.audio_encoder = audio_encoder or CPMobileCLAPAudioEncoder(
            embedding_dim=self.config.embedding_dim
        )
        self.text_encoder = text.encoder
        self.text_projection = text.projection
        self.processor = text.processor
        self.prompt_bank = build_prompt_bank(
            self.class_names,
            templates=self.config.prompt_templates,
        )

        # Independent learnable log-temperature parameters, matching the
        # formulation's distinct sim_ta and sim_cls scales.  Their exact
        # initialization is an implementation choice rather than a reported
        # project hyperparameter.
        self.tau_ta = nn.Parameter(torch.zeros(1))
        self.tau_cls = nn.Parameter(torch.zeros(1))

        # Populated through build_fixed_class_embeddings().
        self.register_buffer(
            "fixed_class_embeddings",
            torch.empty(0, self.config.embedding_dim),
            persistent=True,
        )

    @property
    def has_fixed_class_embeddings(self) -> bool:
        return self.fixed_class_embeddings.numel() > 0

    def encode_audio(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.audio_encoder(inputs)
        return F.normalize(embeddings, dim=-1)

    def _tokenize(self, prompts: Sequence[str], device: torch.device):
        tokens = self.processor(
            text=list(prompts),
            return_tensors="pt",
            padding=True,
        )
        if hasattr(tokens, "to"):
            return tokens.to(device)
        return {key: value.to(device) for key, value in tokens.items()}

    def encode_prompts(self, prompts: Sequence[str], device: torch.device) -> torch.Tensor:
        tokens = self._tokenize(prompts, device)
        outputs = self.text_encoder(**tokens)
        pooled = outputs.pooler_output
        embeddings = self.text_projection(pooled)
        return F.normalize(embeddings, dim=-1)

    def encode_class_prototypes(
        self,
        class_indices: Sequence[int] | torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Average four prompt embeddings for every requested class."""
        if torch.is_tensor(class_indices):
            class_indices = [int(v) for v in class_indices.detach().cpu().tolist()]

        class_embeddings = []
        for index in class_indices:
            prompt_embeddings = self.encode_prompts(self.prompt_bank[index], device)
            class_embeddings.append(prompt_embeddings.mean(dim=0, keepdim=True))
        return F.normalize(torch.cat(class_embeddings, dim=0), dim=-1)

    @torch.no_grad()
    def build_fixed_class_embeddings(self, device: torch.device | None = None) -> torch.Tensor:
        """Create fixed class embeddings from the initial CLAP text representation.

        The resulting embeddings are detached and stored as a persistent buffer
        for the semantic classification branch.
        """
        device = device or next(self.parameters()).device
        was_training_encoder = self.text_encoder.training
        was_training_projection = self.text_projection.training
        self.text_encoder.eval()
        self.text_projection.eval()
        embeddings = self.encode_class_prototypes(
            range(self.config.num_classes),
            device=device,
        ).detach()
        self.fixed_class_embeddings = embeddings
        self.text_encoder.train(was_training_encoder)
        self.text_projection.train(was_training_projection)
        return embeddings

    def alignment_similarity(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return audio_embeddings @ text_embeddings.transpose(0, 1) * torch.exp(self.tau_ta)

    def classification_similarity(
        self,
        audio_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_embeddings is None:
            if not self.has_fixed_class_embeddings:
                raise RuntimeError(
                    "fixed class embeddings are not initialized; call "
                    "build_fixed_class_embeddings() first"
                )
            text_embeddings = self.fixed_class_embeddings
        return audio_embeddings @ text_embeddings.transpose(0, 1) * torch.exp(self.tau_cls)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return semantic class logits for use as a standalone/ensemble teacher."""
        audio_embeddings = self.encode_audio(inputs)
        return self.classification_similarity(audio_embeddings)


def configure_text_encoder_trainability(
    text_encoder: nn.Module,
    trainable_fraction: float,
    minimum_reported_fraction: float = 0.80,
) -> list[str]:
    """Unfreeze a tail fraction of the text encoder.

    The paper reports *more than 80%* trainable text-encoder parameters but does
    not specify an exact percentage.  The caller must therefore supply an exact
    fraction greater than 0.80; the notebook's 20% value is intentionally not
    used here.
    """
    if not (minimum_reported_fraction < trainable_fraction <= 1.0):
        raise ValueError(
            "trainable_fraction must be > 0.80 and <= 1.0 to match the reported setup"
        )

    for parameter in text_encoder.parameters():
        parameter.requires_grad = False

    parameters = list(text_encoder.named_parameters())
    total = sum(parameter.numel() for _, parameter in parameters)
    target = total * trainable_fraction
    selected: list[str] = []
    count = 0
    for name, parameter in reversed(parameters):
        selected.append(name)
        count += parameter.numel()
        if count >= target:
            break

    selected_set = set(selected)
    for name, parameter in parameters:
        if name in selected_set:
            parameter.requires_grad = True
    return selected


def build_cpclap_from_transformers(
    class_names: Sequence[str],
    config: CPCLAPModelConfig | None = None,
    text_trainable_fraction: float | None = None,
) -> CPCLAPTeacher:
    """Reference builder using Hugging Face CLAP components.

    This function is intentionally integration-oriented: the user should fill
    the model/checkpoint identifier in CPCLAPModelConfig.  It is not expected to
    run until project-specific assets and dependencies are supplied.
    """
    config = config or CPCLAPModelConfig(num_classes=len(class_names))

    from transformers import ClapModel, ClapProcessor  # optional project dependency

    clap = ClapModel.from_pretrained(config.clap_text_checkpoint)
    processor = ClapProcessor.from_pretrained(config.clap_text_checkpoint)

    if text_trainable_fraction is None:
        raise ValueError(
            "text_trainable_fraction must be supplied; the paper only specifies >80%"
        )
    configure_text_encoder_trainability(
        clap.text_model,
        text_trainable_fraction,
        minimum_reported_fraction=config.minimum_text_trainable_fraction,
    )
    for parameter in clap.text_projection.parameters():
        parameter.requires_grad = True

    teacher = CPCLAPTeacher(
        class_names=class_names,
        config=config,
        text=CPCLAPTextComponents(
            encoder=clap.text_model,
            projection=clap.text_projection,
            processor=processor,
        ),
    )
    return teacher

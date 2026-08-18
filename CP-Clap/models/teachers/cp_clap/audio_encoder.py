"""CP-Mobile audio encoder adapted to the CLAP shared embedding space."""

from __future__ import annotations

import torch

from ..cp_mobile import CPMobileTeacher


class CPMobileCLAPAudioEncoder(CPMobileTeacher):
    """Use the shared CP-Mobile backbone as a compact audio embedding encoder.

    The existing CP-Mobile implementation is intentionally reused instead of
    duplicated.  Its final 1x1 head is configured to emit the CLAP embedding
    dimension rather than task-class logits.
    """

    def __init__(self, embedding_dim: int = 512):
        super().__init__(num_classes=embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)

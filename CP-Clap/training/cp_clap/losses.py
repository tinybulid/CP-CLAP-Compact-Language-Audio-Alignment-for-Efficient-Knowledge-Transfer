"""CP-CLAP objective from the project formulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class CPCLAPLossBreakdown:
    total: torch.Tensor
    audio_to_text: torch.Tensor
    text_to_audio: torch.Tensor
    contrastive: torch.Tensor
    classification: torch.Tensor


def class_mean_audio_embeddings(
    audio_embeddings: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean normalized audio embedding for each class in class_ids."""
    means = []
    for class_id in class_ids:
        mask = labels == class_id
        if not torch.any(mask):
            raise ValueError("every class_id must occur in the batch")
        means.append(audio_embeddings[mask].mean(dim=0, keepdim=True))
    return F.normalize(torch.cat(means, dim=0), dim=-1)


def cpclap_objective(
    audio_embeddings: torch.Tensor,
    labels: torch.Tensor,
    active_class_ids: torch.Tensor,
    active_text_embeddings: torch.Tensor,
    fixed_text_embeddings: torch.Tensor,
    tau_ta: torch.Tensor,
    tau_cls: torch.Tensor,
    alpha: float = 0.50,
    beta: float = 0.50,
) -> CPCLAPLossBreakdown:
    """Compute the CP-CLAP composite loss.

    Implements

        L_cpcl = alpha * (beta * L_a2t + (1-beta) * L_t2a)
                 + (1-alpha) * L_cls

    with alpha=beta=0.5 by default, as reported in the paper.
    """
    audio = F.normalize(audio_embeddings, dim=-1)
    active_text = F.normalize(active_text_embeddings, dim=-1)
    fixed_text = F.normalize(fixed_text_embeddings, dim=-1)

    # Map original labels to positions inside active_class_ids.
    equality = labels[:, None] == active_class_ids[None, :]
    if not torch.all(equality.any(dim=1)):
        raise ValueError("all batch labels must be represented in active_class_ids")
    active_targets = equality.float().argmax(dim=1)

    # Audio -> text, corresponding to Eq. (8).
    sim_a2t = audio @ active_text.transpose(0, 1) * torch.exp(tau_ta)
    loss_a2t = F.cross_entropy(sim_a2t, active_targets)

    # Text -> audio, corresponding to Eq. (9).  The positive numerator uses
    # the mean audio embedding for a class, while the denominator spans all
    # sample-level audio embeddings in the batch.
    class_audio = class_mean_audio_embeddings(audio, labels, active_class_ids)
    positive = torch.sum(class_audio * active_text, dim=1) * torch.exp(tau_ta)
    all_scores = active_text @ audio.transpose(0, 1) * torch.exp(tau_ta)
    loss_t2a = -(positive - torch.logsumexp(all_scores, dim=1)).mean()

    contrastive = beta * loss_a2t + (1.0 - beta) * loss_t2a

    # Semantic classification, corresponding to Eq. (10).
    sim_cls = audio @ fixed_text.transpose(0, 1) * torch.exp(tau_cls)
    loss_cls = F.cross_entropy(sim_cls, labels)

    total = alpha * contrastive + (1.0 - alpha) * loss_cls
    return CPCLAPLossBreakdown(
        total=total,
        audio_to_text=loss_a2t,
        text_to_audio=loss_t2a,
        contrastive=contrastive,
        classification=loss_cls,
    )

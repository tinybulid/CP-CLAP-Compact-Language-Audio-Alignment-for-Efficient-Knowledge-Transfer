from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn


_STATE_KEYS = (
    "model_state_dict",
    "state_dict",
    "model",
    "student_state_dict",
    "teacher_state_dict",
)


def extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
        for key in _STATE_KEYS:
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
    raise ValueError("Could not locate a model state dictionary in checkpoint")


def strip_prefix(
    state_dict: Mapping[str, torch.Tensor],
    prefix: str = "module.",
) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict):
        return dict(state_dict)
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }


def load_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> dict:
    payload = torch.load(str(path), map_location=device)
    state = strip_prefix(extract_state_dict(payload))
    incompatible = model.load_state_dict(state, strict=strict)
    return {
        "payload": payload,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: Optional[int] = None,
    best_metric: Optional[float] = None,
    config: Optional[Any] = None,
    extra: Optional[dict] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        payload["config"] = asdict(config) if is_dataclass(config) else config
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))

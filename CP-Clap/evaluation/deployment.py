from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def make_fp16_copy(model: nn.Module) -> nn.Module:
    copied = deepcopy(model).eval()
    return copied.half()


def save_fp16_state_dict(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp16_state = {
        k: (v.half() if torch.is_floating_point(v) else v)
        for k, v in model.state_dict().items()
    }
    torch.save(fp16_state, path)


def model_summary(model: nn.Module) -> dict:
    params = count_parameters(model)
    bytes_ = parameter_size_bytes(model)
    return {
        "parameters": params,
        "parameter_k": params / 1_000,
        "size_bytes": bytes_,
        "size_kib": bytes_ / 1024,
    }

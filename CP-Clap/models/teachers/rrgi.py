from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .dsf_amir import DSFAmirTeacher
from .dsf_mehdy import DSFMehdyTeacher
from .dsf_mehdy2 import DSFMehdy2Teacher
from .cp_mobile import CPMobileTeacher
from .cp_resnet import CPResNetTeacher


_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "dsf_amir": DSFAmirTeacher,
    "dsf_mehdy": DSFMehdyTeacher,
    "dsf_mehdy2": DSFMehdy2Teacher,
    "cp_mobile": CPMobileTeacher,
    "cp_resnet": CPResNetTeacher,
}


def available_external_teachers():
    return tuple(sorted(_BUILDERS))


def build_external_teacher(name: str, num_classes: int = 10, **kwargs) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key not in _BUILDERS:
        raise KeyError(
            f"Unknown external teacher {name!r}. "
            f"Available: {', '.join(available_external_teachers())}"
        )
    return _BUILDERS[key](num_classes=num_classes, **kwargs)

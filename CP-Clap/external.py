from __future__ import annotations

import importlib
from typing import Any, Callable


def resolve_object(spec: str) -> Any:
    """Resolve `module.submodule:attribute` from user-owned integration code."""
    if ":" not in spec:
        raise ValueError("provider must be written as module.path:attribute")
    module_name, attr_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in attr_name.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_provider(spec: str) -> Callable[[], object]:
    obj = resolve_object(spec)
    if not callable(obj):
        raise TypeError(f"{spec!r} resolved to a non-callable object")
    return obj

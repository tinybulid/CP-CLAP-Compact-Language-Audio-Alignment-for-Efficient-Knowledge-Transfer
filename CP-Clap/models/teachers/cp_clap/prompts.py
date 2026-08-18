"""Prompt construction utilities used by CP-CLAP."""

from __future__ import annotations

from typing import Iterable, Sequence

from .config import DEFAULT_PROMPT_TEMPLATES


def prompts_for_class(
    class_name: str,
    templates: Sequence[str] = DEFAULT_PROMPT_TEMPLATES,
) -> list[str]:
    """Expand the four class prompts described in the CP-CLAP formulation."""
    return [template.replace("[class]", class_name) for template in templates]


def build_prompt_bank(
    class_names: Iterable[str],
    templates: Sequence[str] = DEFAULT_PROMPT_TEMPLATES,
) -> list[list[str]]:
    """Return one prompt list per class."""
    return [prompts_for_class(name, templates) for name in class_names]

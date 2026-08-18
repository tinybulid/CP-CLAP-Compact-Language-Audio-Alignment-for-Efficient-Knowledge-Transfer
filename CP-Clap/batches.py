from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass
class ParsedBatch:
    inputs: torch.Tensor
    labels: torch.Tensor
    device_ids: Optional[torch.Tensor] = None
    sample_ids: Optional[Any] = None


def parse_batch(batch: Any) -> ParsedBatch:
    """Normalize common external batch formats without defining a data source.

    Accepted shapes:
      * (inputs, labels)
      * (inputs, device_ids, labels)
      * mapping with keys inputs/x, labels/y and optional device_ids/devices
    """
    if isinstance(batch, Mapping):
        x = batch.get("inputs", batch.get("x"))
        y = batch.get("labels", batch.get("y"))
        dev = batch.get("device_ids", batch.get("devices"))
        ids = batch.get("sample_ids", batch.get("ids"))
        if x is None or y is None:
            raise KeyError("mapping batch must contain inputs/x and labels/y")
        return ParsedBatch(x, y, dev, ids)

    if isinstance(batch, Sequence):
        if len(batch) == 2:
            return ParsedBatch(batch[0], batch[1])
        if len(batch) == 3:
            return ParsedBatch(batch[0], batch[2], batch[1])
        if len(batch) >= 4:
            return ParsedBatch(batch[0], batch[-1], batch[1], batch[2])

    raise TypeError(
        "Unsupported batch. Supply (inputs, labels), (inputs, device_ids, labels), "
        "or a mapping with inputs and labels."
    )


def move_batch(batch: ParsedBatch, device: torch.device) -> ParsedBatch:
    return ParsedBatch(
        inputs=batch.inputs.to(device),
        labels=batch.labels.to(device, dtype=torch.long),
        device_ids=None if batch.device_ids is None else batch.device_ids.to(device),
        sample_ids=batch.sample_ids,
    )

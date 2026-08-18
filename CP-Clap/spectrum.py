from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from .config import SpectrumConfig

try:
    import torchaudio
except Exception:  # pragma: no cover - optional dependency
    torchaudio = None


TensorOrPath = Union[torch.Tensor, str, Path]


def _require_torchaudio() -> None:
    if torchaudio is None:
        raise RuntimeError(
            "torchaudio is required for audio-file loading and Mel filter-bank extraction."
        )


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Return a [1, time] waveform."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [time] or [channels, time]")
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def resample_if_needed(
    waveform: torch.Tensor,
    source_rate: int,
    target_rate: int,
) -> torch.Tensor:
    if source_rate == target_rate:
        return waveform
    _require_torchaudio()
    return torchaudio.functional.resample(waveform, source_rate, target_rate)


def waveform_to_logmel(
    waveform: torch.Tensor,
    source_rate: int,
    config: SpectrumConfig = SpectrumConfig(),
    target_frames: Optional[int] = None,
) -> torch.Tensor:
    """Convert a waveform to the log-Mel tensor expected by the models.

    Returns [1, n_mels, frames].  `target_frames` is optional and only pads or
    crops the time axis; it does not imply any particular data source.
    """
    _require_torchaudio()
    waveform = ensure_mono(waveform.float())
    waveform = resample_if_needed(waveform, source_rate, config.sample_rate)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        power=config.power,
        center=config.center,
        normalized=config.normalized,
    )(waveform)

    logmel = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=config.top_db,
    )(mel)

    if target_frames is not None:
        logmel = fit_time_frames(logmel, target_frames)
    return logmel


def audio_file_to_logmel(
    path: Union[str, Path],
    config: SpectrumConfig = SpectrumConfig(),
    target_frames: Optional[int] = None,
) -> torch.Tensor:
    """Load one audio file and return a [1, n_mels, frames] log-Mel tensor."""
    _require_torchaudio()
    waveform, source_rate = torchaudio.load(str(path))
    return waveform_to_logmel(
        waveform=waveform,
        source_rate=source_rate,
        config=config,
        target_frames=target_frames,
    )


def fit_time_frames(spec: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Right-pad or crop only the final time dimension."""
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    current = spec.size(-1)
    if current == target_frames:
        return spec
    if current > target_frames:
        return spec[..., :target_frames]
    return F.pad(spec, (0, target_frames - current))


def standardize_per_example(
    spec: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Optional per-example standardization over frequency and time."""
    dims = tuple(range(spec.ndim - 2, spec.ndim))
    mean = spec.mean(dim=dims, keepdim=True)
    std = spec.std(dim=dims, keepdim=True).clamp_min(eps)
    return (spec - mean) / std


def spectrum_summary(spec: torch.Tensor) -> dict:
    return {
        "shape": tuple(spec.shape),
        "dtype": str(spec.dtype),
        "min": float(spec.min().detach().cpu()),
        "max": float(spec.max().detach().cpu()),
        "mean": float(spec.mean().detach().cpu()),
        "std": float(spec.std().detach().cpu()),
    }

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class SpectrogramAugmentConfig:
    time_mask_width: int = 16
    frequency_mask_width: int = 16
    time_masks: int = 1
    frequency_masks: int = 1
    gain_db: float = 4.0
    noise_std: float = 0.01
    mixstyle_probability: float = 0.5
    mixstyle_alpha: float = 0.3


def _mask_along_axis(
    x: torch.Tensor,
    axis: int,
    max_width: int,
    count: int,
) -> torch.Tensor:
    if max_width <= 0 or count <= 0:
        return x
    out = x.clone()
    length = x.size(axis)
    if length <= 1:
        return out

    for _ in range(count):
        width = int(torch.randint(0, min(max_width, length) + 1, (1,)).item())
        if width == 0:
            continue
        start_max = max(1, length - width + 1)
        start = int(torch.randint(0, start_max, (1,)).item())
        sl = [slice(None)] * out.ndim
        sl[axis] = slice(start, start + width)
        out[tuple(sl)] = 0
    return out


def random_gain(x: torch.Tensor, max_db: float) -> torch.Tensor:
    if max_db <= 0:
        return x
    db = torch.empty(x.size(0), 1, 1, 1, device=x.device).uniform_(-max_db, max_db)
    scale = torch.pow(10.0, db / 20.0)
    return x * scale


def gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    return x + torch.randn_like(x) * std


def mixstyle(
    x: torch.Tensor,
    probability: float = 0.5,
    alpha: float = 0.3,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Mix per-example feature statistics across the current batch."""
    if x.ndim != 4 or x.size(0) < 2:
        return x
    if torch.rand(()) > probability:
        return x

    mu = x.mean(dim=(2, 3), keepdim=True)
    var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
    sig = (var + eps).sqrt()
    normalized = (x - mu) / sig

    perm = torch.randperm(x.size(0), device=x.device)
    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = beta_dist.sample((x.size(0), 1, 1, 1)).to(x.device, x.dtype)

    mu_mix = lam * mu + (1.0 - lam) * mu[perm]
    sig_mix = lam * sig + (1.0 - lam) * sig[perm]
    return normalized * sig_mix + mu_mix


def apply_impulse_response(
    waveform: torch.Tensor,
    impulse_response: Optional[torch.Tensor],
) -> torch.Tensor:
    """Convolve waveforms with an externally supplied impulse response tensor."""
    if impulse_response is None:
        return waveform
    if waveform.ndim == 1:
        waveform = waveform[None, None, :]
    elif waveform.ndim == 2:
        waveform = waveform[:, None, :]
    if impulse_response.ndim == 1:
        impulse_response = impulse_response[None, None, :]
    elif impulse_response.ndim == 2:
        impulse_response = impulse_response[:, None, :]

    kernel = impulse_response.flip(-1).to(waveform.device, waveform.dtype)
    if kernel.size(0) == 1 and waveform.size(0) > 1:
        kernel = kernel.expand(waveform.size(0), -1, -1)

    outputs = []
    for w, k in zip(waveform, kernel):
        pad = k.size(-1) - 1
        y = F.conv1d(w.unsqueeze(0), k.unsqueeze(0), padding=pad)
        outputs.append(y[..., : w.size(-1)])
    return torch.cat(outputs, dim=0)


class SpectrogramAugment:
    def __init__(self, config: SpectrogramAugmentConfig = SpectrogramAugmentConfig()):
        self.config = config

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4:
            raise ValueError("spectrogram batch must have shape [B, C, F, T]")

        x = _mask_along_axis(
            x,
            axis=3,
            max_width=self.config.time_mask_width,
            count=self.config.time_masks,
        )
        x = _mask_along_axis(
            x,
            axis=2,
            max_width=self.config.frequency_mask_width,
            count=self.config.frequency_masks,
        )
        x = random_gain(x, self.config.gain_db)
        x = gaussian_noise(x, self.config.noise_std)
        x = mixstyle(
            x,
            probability=self.config.mixstyle_probability,
            alpha=self.config.mixstyle_alpha,
        )
        return x

# ==============================================================================
# Project: ADI-LoRA / ADI-DoRA
# File:    frequency_metrics.py
# Desc:    Minimal diagnostics for the ADI diagnostic evaluation: LFER and linear CKA.
# ============================================================================== 

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

SpatialSize = Optional[Tuple[int, int]]


def infer_spatial_size(num_tokens: int, spatial_size: SpatialSize = None) -> Tuple[int, int]:
    if spatial_size is not None:
        h, w = int(spatial_size[0]), int(spatial_size[1])
        if h * w != num_tokens:
            raise ValueError(f"spatial_size={spatial_size} does not match {num_tokens} tokens.")
        return h, w
    h = int(math.sqrt(num_tokens))
    if h * h != num_tokens:
        raise ValueError("Pass spatial_size=(H, W) for non-square token grids.")
    return h, h


def tokens_to_2d(x: torch.Tensor, spatial_size: SpatialSize = None, num_prefix_tokens: int = 1) -> torch.Tensor:
    if x.ndim == 4:
        return x
    b, n_total, c = x.shape
    x_spatial = x[:, num_prefix_tokens:, :]
    h, w = infer_spatial_size(x_spatial.shape[1], spatial_size)
    return x_spatial.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def low_frequency_energy_ratio(
    x: torch.Tensor,
    spatial_size: SpatialSize = None,
    num_prefix_tokens: int = 1,
    cutoff_ratio: float = 0.35,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return scalar LFER over a batch of tokens/features."""
    x2d = tokens_to_2d(x, spatial_size=spatial_size, num_prefix_tokens=num_prefix_tokens).float()
    b, c, h, w = x2d.shape
    fft = torch.fft.rfft2(x2d, norm="ortho")
    energy = torch.abs(fft).pow(2)

    fy = torch.fft.fftfreq(h, device=x2d.device, dtype=torch.float32) * h
    fx = torch.fft.rfftfreq(w, device=x2d.device, dtype=torch.float32) * w
    radius = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
    radius = radius / radius.max().clamp_min(eps)
    low_mask = (radius <= cutoff_ratio).view(1, 1, h, w // 2 + 1)

    low_energy = (energy * low_mask).sum(dim=(-2, -1))
    total_energy = energy.sum(dim=(-2, -1)).clamp_min(eps)
    return (low_energy / total_energy).mean().clamp(0.0, 1.0)


@torch.no_grad()
def linear_cka(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Linear CKA between two feature tensors. Flattens all non-feature dimensions.
    Input can be [B, N, C] or [M, C].
    """
    if x.shape != y.shape:
        raise ValueError(f"CKA expects matching shapes, got {x.shape} vs {y.shape}.")
    x = x.reshape(-1, x.shape[-1]).float()
    y = y.reshape(-1, y.shape[-1]).float()
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    xty = x.T @ y
    xtx = x.T @ x
    yty = y.T @ y
    hsic = xty.pow(2).sum()
    norm = torch.sqrt(xtx.pow(2).sum() * yty.pow(2).sum()).clamp_min(eps)
    return (hsic / norm).clamp(0.0, 1.0)


@torch.no_grad()
def spectral_log_amplitude_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    spatial_size: SpatialSize = None,
    num_prefix_tokens: int = 1,
    cutoff_ratio: float = 0.35,
    high_pass_only: bool = False,
    min_clamp: float = 1e-6,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    SpecDist: log-amplitude spectral distance between two feature tensors.

    This metric compares the 2D Fourier log-amplitude spectra of two feature maps
    or token sequences. It can be computed on the full spectrum or only the
    high-frequency region.

    Args:
        x, y:
            Feature tensors with shape [B, N, C] or [B, C, H, W].
        spatial_size:
            Explicit (H, W) grid size when using token features.
        num_prefix_tokens:
            Number of prefix tokens such as CLS.
        cutoff_ratio:
            Normalized radial cutoff. Used only when high_pass_only=True.
        high_pass_only:
            If True, compute distance only on the high-frequency region.
        min_clamp:
            Lower bound before log to avoid numerical underflow.
        eps:
            Numerical stability constant.

    Returns:
        Scalar spectral distance. Lower means more similar spectra.
    """
    if x.shape != y.shape:
        raise ValueError(f"SpecDist expects matching shapes, got {x.shape} vs {y.shape}.")

    x2d = tokens_to_2d(x, spatial_size=spatial_size, num_prefix_tokens=num_prefix_tokens).float()
    y2d = tokens_to_2d(y, spatial_size=spatial_size, num_prefix_tokens=num_prefix_tokens).float()

    x_fft = torch.fft.rfft2(x2d, norm="ortho")
    y_fft = torch.fft.rfft2(y2d, norm="ortho")

    x_log_amp = torch.log(torch.abs(x_fft).clamp_min(min_clamp))
    y_log_amp = torch.log(torch.abs(y_fft).clamp_min(min_clamp))

    diff = x_log_amp - y_log_amp

    if high_pass_only:
        _, _, h, w = x2d.shape
        fy = torch.fft.fftfreq(h, device=x2d.device, dtype=torch.float32) * h
        fx = torch.fft.rfftfreq(w, device=x2d.device, dtype=torch.float32) * w
        radius = torch.sqrt(fy[:, None].pow(2) + fx[None, :].pow(2))
        radius = radius / radius.max().clamp_min(eps)
        high_mask = (radius >= cutoff_ratio).view(1, 1, h, w // 2 + 1).to(
            device=x2d.device,
            dtype=diff.dtype,
        )

        diff = diff * high_mask
        denom = high_mask.sum().clamp_min(eps) * x2d.shape[0] * x2d.shape[1]
        return diff.pow(2).sum().div(denom).sqrt()

    return diff.pow(2).mean().sqrt()


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 197, 64)
    y = x + 0.1 * torch.randn_like(x)
    print("LFER=", float(low_frequency_energy_ratio(x, spatial_size=(14, 14))))
    print("CKA=", float(linear_cka(x, y)))
    print("SpecDist=", float(spectral_log_amplitude_distance(x, y, spatial_size=(14, 14))))
    print(
        "HighSpecDist=",
        float(
            spectral_log_amplitude_distance(
                x,
                y,
                spatial_size=(14, 14),
                high_pass_only=True,
            )
        ),
    )

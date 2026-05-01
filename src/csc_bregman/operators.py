from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_dictionary(dictionary: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize each convolutional atom to unit Euclidean norm."""
    flat = dictionary.flatten(start_dim=1)
    norms = flat.norm(dim=1).clamp_min(eps)
    return dictionary / norms.view(-1, 1, 1, 1)


def synthesize(coefficients: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    """Compute sum_k D_k * X_k with same spatial size via transposed convolution."""
    padding = dictionary.shape[-1] // 2
    return F.conv_transpose2d(coefficients, dictionary, padding=padding)


def analysis_gradient(residual: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    """Gradient of 0.5||synthesize(X, D)-Y||^2 with respect to coefficient maps."""
    padding = dictionary.shape[-1] // 2
    return F.conv2d(residual, dictionary, padding=padding)


def soft_threshold(x: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.clamp(torch.abs(x) - threshold, min=0)


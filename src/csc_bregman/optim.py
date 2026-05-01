from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import torch

from csc_bregman.metrics import nmse, psnr
from csc_bregman.operators import (
    analysis_gradient,
    normalize_dictionary,
    soft_threshold,
    synthesize,
)


@dataclass(frozen=True)
class LearningConfig:
    lmbda: float = 0.05
    outer_iters: int = 40
    sparse_iters: int = 50
    sparse_step: float = 0.18
    dict_step: float = 0.8
    bregman_rho: float = 0.25


@dataclass
class LearningResult:
    dictionary: torch.Tensor
    coefficients: torch.Tensor
    history: list[dict[str, float]] = field(default_factory=list)
    runtime_seconds: float = 0.0


def objective(images: torch.Tensor, coefficients: torch.Tensor, dictionary: torch.Tensor, lmbda: float) -> float:
    residual = synthesize(coefficients, dictionary) - images
    value = 0.5 * residual.square().sum() + lmbda * coefficients.abs().sum()
    return float(value.detach().cpu())


def sparse_code_pgd(
    images: torch.Tensor,
    dictionary: torch.Tensor,
    coefficients: torch.Tensor,
    lmbda: float,
    step: float,
    iters: int,
) -> torch.Tensor:
    x = coefficients
    for _ in range(iters):
        residual = synthesize(x, dictionary) - images
        grad = analysis_gradient(residual, dictionary)
        x = soft_threshold(x - step * grad, step * lmbda)
    return x


def _solve_quartic_mirror(a: torch.Tensor, rho: float, newton_iters: int = 20) -> torch.Tensor:
    """Solve u + rho*u^3 = a elementwise for rho >= 0."""
    if rho == 0:
        return a
    u = a.clone()
    for _ in range(newton_iters):
        numerator = u + rho * u.pow(3) - a
        denominator = 1 + 3 * rho * u.square()
        u = u - numerator / denominator.clamp_min(1e-12)
    return u


def sparse_code_bregman(
    images: torch.Tensor,
    dictionary: torch.Tensor,
    coefficients: torch.Tensor,
    lmbda: float,
    step: float,
    iters: int,
    rho: float,
) -> torch.Tensor:
    x = coefficients
    for _ in range(iters):
        residual = synthesize(x, dictionary) - images
        grad = analysis_gradient(residual, dictionary)
        dual = x + rho * x.pow(3) - step * grad
        shrunk_dual = soft_threshold(dual, step * lmbda)
        x = _solve_quartic_mirror(shrunk_dual, rho)
    return x


def dictionary_step(
    images: torch.Tensor,
    coefficients: torch.Tensor,
    dictionary: torch.Tensor,
    step: float,
) -> torch.Tensor:
    d = dictionary.detach().clone().requires_grad_(True)
    residual = synthesize(coefficients.detach(), d) - images
    loss = 0.5 * residual.square().mean()
    (grad,) = torch.autograd.grad(loss, d)
    with torch.no_grad():
        d_next = normalize_dictionary(d - step * grad)
    return d_next.detach()


def learn_dictionary(
    images: torch.Tensor,
    initial_dictionary: torch.Tensor,
    config: LearningConfig,
    method: str = "pgd",
) -> LearningResult:
    """Alternating convolutional sparse coding and dictionary learning."""
    if method not in {"pgd", "bregman"}:
        raise ValueError(f"Unknown method: {method}")

    dictionary = normalize_dictionary(initial_dictionary.detach().clone())
    coefficients = torch.zeros(
        images.shape[0],
        dictionary.shape[0],
        images.shape[-2],
        images.shape[-1],
        dtype=images.dtype,
        device=images.device,
    )

    history: list[dict[str, float]] = []
    start = perf_counter()
    for outer in range(config.outer_iters):
        if method == "pgd":
            coefficients = sparse_code_pgd(
                images,
                dictionary,
                coefficients,
                config.lmbda,
                config.sparse_step,
                config.sparse_iters,
            )
        else:
            coefficients = sparse_code_bregman(
                images,
                dictionary,
                coefficients,
                config.lmbda,
                config.sparse_step,
                config.sparse_iters,
                config.bregman_rho,
            )

        dictionary = dictionary_step(images, coefficients, dictionary, config.dict_step)
        recon = synthesize(coefficients, dictionary)
        history.append(
            {
                "iter": float(outer),
                "objective": objective(images, coefficients, dictionary, config.lmbda),
                "nmse": nmse(images, recon),
                "psnr": psnr(images, recon),
                "sparsity": float((coefficients.abs() > 1e-8).float().mean().detach().cpu()),
            }
        )

    return LearningResult(
        dictionary=dictionary.detach(),
        coefficients=coefficients.detach(),
        history=history,
        runtime_seconds=perf_counter() - start,
    )

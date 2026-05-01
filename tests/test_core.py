from __future__ import annotations

import torch

from csc_bregman.operators import analysis_gradient, normalize_dictionary, soft_threshold, synthesize


def test_synthesis_and_analysis_shapes_match() -> None:
    coefficients = torch.randn(2, 4, 16, 16)
    dictionary = normalize_dictionary(torch.randn(4, 1, 5, 5))
    images = synthesize(coefficients, dictionary)
    grad = analysis_gradient(images, dictionary)
    assert images.shape == (2, 1, 16, 16)
    assert grad.shape == coefficients.shape


def test_dictionary_normalization() -> None:
    dictionary = normalize_dictionary(torch.randn(6, 1, 7, 7))
    norms = dictionary.flatten(start_dim=1).norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_soft_threshold() -> None:
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    actual = soft_threshold(x, 1.0)
    expected = torch.tensor([-1.0, -0.0, 0.0, 0.0, 1.0])
    assert torch.allclose(actual, expected)


def test_analysis_gradient_finite_difference() -> None:
    torch.manual_seed(0)
    coefficients = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    dictionary = normalize_dictionary(torch.randn(2, 1, 3, 3, dtype=torch.float64))
    target = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    residual = synthesize(coefficients, dictionary) - target
    grad = analysis_gradient(residual, dictionary)

    direction = torch.randn_like(coefficients)
    direction = direction / direction.norm()
    eps = 1e-6
    plus = 0.5 * (synthesize(coefficients + eps * direction, dictionary) - target).square().sum()
    minus = 0.5 * (synthesize(coefficients - eps * direction, dictionary) - target).square().sum()
    finite_difference = (plus - minus) / (2 * eps)
    directional = (grad * direction).sum()
    assert torch.allclose(finite_difference, directional, rtol=1e-4, atol=1e-5)


def test_dictionary_gradient_finite_difference() -> None:
    torch.manual_seed(1)
    coefficients = torch.randn(1, 2, 8, 8, dtype=torch.float64)
    dictionary = normalize_dictionary(torch.randn(2, 1, 3, 3, dtype=torch.float64))
    dictionary.requires_grad_(True)
    target = torch.randn(1, 1, 8, 8, dtype=torch.float64)
    loss = 0.5 * (synthesize(coefficients, dictionary) - target).square().sum()
    (grad,) = torch.autograd.grad(loss, dictionary)

    direction = torch.randn_like(dictionary)
    direction = direction / direction.norm()
    eps = 1e-6
    plus = 0.5 * (synthesize(coefficients, dictionary + eps * direction) - target).square().sum()
    minus = 0.5 * (synthesize(coefficients, dictionary - eps * direction) - target).square().sum()
    finite_difference = (plus - minus) / (2 * eps)
    directional = (grad * direction).sum()
    assert torch.allclose(finite_difference, directional, rtol=1e-4, atol=1e-5)

from __future__ import annotations

from dataclasses import dataclass

import torch

from csc_bregman.operators import normalize_dictionary, synthesize


@dataclass(frozen=True)
class SimulationData:
    images: torch.Tensor
    dictionary: torch.Tensor
    coefficients: torch.Tensor
    clean_images: torch.Tensor


def make_simulation(
    num_images: int = 10,
    image_size: int = 32,
    num_atoms: int = 8,
    kernel_size: int = 11,
    sparsity: float = 0.04,
    noise_std: float = 0.01,
    coefficient_scale: float = 1.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SimulationData:
    """Generate normalized filters, sparse maps, and 2D image signals."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    device = torch.device(device)

    dictionary = torch.randn(num_atoms, 1, kernel_size, kernel_size, generator=generator, dtype=dtype)
    dictionary = normalize_dictionary(dictionary).to(device)

    coeff_shape = (num_images, num_atoms, image_size, image_size)
    support = torch.rand(coeff_shape, generator=generator, dtype=dtype) < sparsity
    values = coefficient_scale * torch.randn(coeff_shape, generator=generator, dtype=dtype)
    coefficients = (support * values).to(device)

    clean_images = synthesize(coefficients, dictionary)
    if noise_std > 0:
        noise = noise_std * torch.randn(clean_images.shape, generator=generator, dtype=dtype).to(device)
        images = clean_images + noise
    else:
        images = clean_images.clone()

    return SimulationData(
        images=images,
        dictionary=dictionary,
        coefficients=coefficients,
        clean_images=clean_images,
    )


def random_dictionary(
    num_atoms: int,
    kernel_size: int,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    dictionary = torch.randn(num_atoms, 1, kernel_size, kernel_size, generator=generator, dtype=dtype)
    return normalize_dictionary(dictionary).to(device)


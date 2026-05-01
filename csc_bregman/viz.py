from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from csc_bregman.operators import synthesize


def plot_objective(histories: dict[str, list[dict[str, float]]], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, history in histories.items():
        ax.plot([h["iter"] for h in history], [h["objective"] for h in history], label=label)
    ax.set_xlabel("Outer iteration")
    ax.set_ylabel("Objective")
    ax.set_title("Dictionary learning objective")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_dictionary_compare(
    true_dictionary: torch.Tensor,
    learned: dict[str, torch.Tensor],
    path: str | Path,
) -> None:
    labels = ["true", *learned.keys()]
    dictionaries = [true_dictionary, *learned.values()]
    num_atoms = true_dictionary.shape[0]
    fig, axes = plt.subplots(
        len(dictionaries),
        num_atoms,
        figsize=(1.5 * num_atoms, 1.8 * len(dictionaries)),
        squeeze=False,
    )
    for row, (label, dictionary) in enumerate(zip(labels, dictionaries, strict=True)):
        for col in range(num_atoms):
            ax = axes[row, col]
            atom = dictionary[col, 0].detach().cpu()
            vmax = float(atom.abs().max())
            ax.imshow(atom, cmap="gray", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_reconstruction_compare(
    images: torch.Tensor,
    learned: dict[str, tuple[torch.Tensor, torch.Tensor]],
    path: str | Path,
    image_index: int = 0,
) -> None:
    rows = 1 + 2 * len(learned)
    fig, axes = plt.subplots(rows, 1, figsize=(4, 3 * rows))
    axes[0].imshow(images[image_index, 0].detach().cpu(), cmap="gray")
    axes[0].set_title("Observed image")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    row = 1
    for label, (coefficients, dictionary) in learned.items():
        recon = synthesize(coefficients, dictionary)
        residual = images - recon
        axes[row].imshow(recon[image_index, 0].detach().cpu(), cmap="gray")
        axes[row].set_title(f"{label} reconstruction")
        axes[row].set_xticks([])
        axes[row].set_yticks([])
        row += 1
        axes[row].imshow(residual[image_index, 0].detach().cpu(), cmap="gray")
        axes[row].set_title(f"{label} residual")
        axes[row].set_xticks([])
        axes[row].set_yticks([])
        row += 1
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_sparse_maps_compare(
    true_coefficients: torch.Tensor,
    learned_coefficients: torch.Tensor,
    path: str | Path,
    image_index: int = 0,
    max_atoms: int = 8,
) -> None:
    atoms = min(max_atoms, true_coefficients.shape[1])
    fig, axes = plt.subplots(2, atoms, figsize=(1.6 * atoms, 3.4), squeeze=False)
    for col in range(atoms):
        for row, tensor in enumerate([true_coefficients, learned_coefficients]):
            ax = axes[row, col]
            coeff = tensor[image_index, col].detach().cpu()
            vmax = float(coeff.abs().max().clamp_min(1e-12))
            ax.imshow(coeff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel("true" if row == 0 else "learned")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

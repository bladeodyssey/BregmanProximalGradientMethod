from __future__ import annotations

import numpy as np
import torch


def nmse(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12) -> float:
    numerator = (reference - estimate).square().sum()
    denominator = reference.square().sum().clamp_min(eps)
    return float((numerator / denominator).detach().cpu())


def psnr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12) -> float:
    mse = (reference - estimate).square().mean().clamp_min(eps)
    peak = reference.max() - reference.min()
    if float(peak.detach().cpu()) <= eps:
        peak = torch.tensor(1.0, dtype=reference.dtype, device=reference.device)
    return float((20 * torch.log10(peak / torch.sqrt(mse))).detach().cpu())


def dictionary_recovery(
    true_dictionary: torch.Tensor,
    learned_dictionary: torch.Tensor,
) -> dict[str, float | list[int]]:
    """Match learned atoms to true atoms by absolute correlation."""
    true_flat = true_dictionary.detach().cpu().flatten(start_dim=1).numpy()
    learned_flat = learned_dictionary.detach().cpu().flatten(start_dim=1).numpy()
    true_flat /= np.linalg.norm(true_flat, axis=1, keepdims=True).clip(min=1e-12)
    learned_flat /= np.linalg.norm(learned_flat, axis=1, keepdims=True).clip(min=1e-12)
    corr = np.abs(true_flat @ learned_flat.T)

    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(-corr)
    except Exception:
        rows, cols = _greedy_match(corr)

    matched = corr[rows, cols]
    return {
        "mean_abs_corr": float(matched.mean()),
        "min_abs_corr": float(matched.min()),
        "max_abs_corr": float(matched.max()),
        "true_indices": rows.tolist(),
        "learned_indices": cols.tolist(),
    }


def _greedy_match(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    cols = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    for flat_idx in np.argsort(corr.ravel())[::-1]:
        row, col = np.unravel_index(flat_idx, corr.shape)
        if row not in used_rows and col not in used_cols:
            rows.append(row)
            cols.append(col)
            used_rows.add(row)
            used_cols.add(col)
        if len(rows) == min(corr.shape):
            break
    return np.array(rows), np.array(cols)


def support_recovery(
    true_coefficients: torch.Tensor,
    learned_coefficients: torch.Tensor,
    threshold: float = 1e-3,
) -> dict[str, float]:
    true_support = true_coefficients.abs() > threshold
    learned_support = learned_coefficients.abs() > threshold
    tp = (true_support & learned_support).sum().float()
    fp = (~true_support & learned_support).sum().float()
    fn = (true_support & ~learned_support).sum().float()
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {
        "support_precision": float(precision.detach().cpu()),
        "support_recall": float(recall.detach().cpu()),
        "support_f1": float(f1.detach().cpu()),
    }


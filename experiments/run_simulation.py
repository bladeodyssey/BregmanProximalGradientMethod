from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from csc_bregman.data import make_simulation, random_dictionary
from csc_bregman.metrics import dictionary_recovery, nmse, psnr, support_recovery
from csc_bregman.operators import synthesize
from csc_bregman.optim import LearningConfig, learn_dictionary
from csc_bregman.viz import (
    plot_dictionary_compare,
    plot_objective,
    plot_reconstruction_compare,
    plot_sparse_maps_compare,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/simulation.json"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def flatten_metrics(method: str, result, data) -> dict[str, float | str]:
    recon = synthesize(result.coefficients, result.dictionary)
    dict_metrics = dictionary_recovery(data.dictionary, result.dictionary)
    support_metrics = support_recovery(data.coefficients, result.coefficients)
    return {
        "method": method,
        "runtime_seconds": result.runtime_seconds,
        "final_objective": result.history[-1]["objective"],
        "image_nmse": nmse(data.images, recon),
        "image_psnr": psnr(data.images, recon),
        "dict_mean_abs_corr": dict_metrics["mean_abs_corr"],
        "dict_min_abs_corr": dict_metrics["min_abs_corr"],
        **support_metrics,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    device = resolve_device(config["device"])
    dtype = resolve_dtype(config["dtype"])
    seed = int(config["seed"])

    data = make_simulation(
        **config["data"],
        seed=seed,
        device=device,
        dtype=dtype,
    )
    initial_dictionary = random_dictionary(
        config["data"]["num_atoms"],
        config["data"]["kernel_size"],
        seed=seed + 1000,
        device=device,
        dtype=dtype,
    )

    learning_config = LearningConfig(
        lmbda=config["learning"]["lambda"],
        outer_iters=config["learning"]["outer_iters"],
        sparse_iters=config["learning"]["sparse_iters"],
        sparse_step=config["learning"]["sparse_step"],
        dict_step=config["learning"]["dict_step"],
        bregman_rho=config["learning"]["bregman_rho"],
    )

    results = {
        "pgd": learn_dictionary(data.images, initial_dictionary, learning_config, method="pgd"),
        "bregman": learn_dictionary(data.images, initial_dictionary, learning_config, method="bregman"),
    }

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [flatten_metrics(method, result, data) for method, result in results.items()]
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    plot_objective({name: result.history for name, result in results.items()}, output_dir / "objective_curve.png")
    plot_dictionary_compare(
        data.dictionary,
        {name: result.dictionary for name, result in results.items()},
        output_dir / "dictionary_compare.png",
    )
    plot_reconstruction_compare(
        data.images,
        {name: (result.coefficients, result.dictionary) for name, result in results.items()},
        output_dir / "reconstruction_compare.png",
    )
    plot_sparse_maps_compare(
        data.coefficients,
        results["bregman"].coefficients,
        output_dir / "sparse_maps_compare.png",
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()


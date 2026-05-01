# CSC Bregman PGD

This repository implements a small, reproducible simulation for 2D convolutional sparse coding and convolutional dictionary learning:

- generate normalized convolutional dictionaries and sparse coefficient maps;
- synthesize 10 image signals;
- learn dictionaries and coefficients with Euclidean PGD/ISTA;
- compare against a Bregman proximal gradient variant using
  `h(x)=0.5||x||_2^2 + rho/4||x||_4^4`;
- report quantitative metrics and qualitative figures.

## Recommended Local Workflow

Use Windows as the editor machine and run code on the Linux GPU host:

1. Install VS Code extensions on Windows: Remote SSH, Python, Jupyter, GitHub Pull Requests.
2. SSH into the Linux GPU host from VS Code.
3. Keep the project on Linux, for example `~/projects/csc-bregman/`.
4. Create the environment:

```bash
conda create -n csc-bregman python=3.11
conda activate csc-bregman
pip install -e ".[dev]"
```

Install the CUDA-enabled PyTorch build that matches the GPU host from the official selector:
https://pytorch.org/get-started/locally/

## Run the Simulation

```bash
python -m experiments.run_simulation --config configs/simulation.json
```

Outputs are written to `results/simulation/`:

- `metrics.json`
- `metrics.csv`
- `objective_curve.png`
- `dictionary_compare.png`
- `reconstruction_compare.png`
- `sparse_maps_compare.png`

## Run Tests

```bash
pytest
```

The tests cover operator shapes, dictionary normalization, soft thresholding, and finite-difference gradient checks.

## Notes

The first implementation favors clarity and reproducibility over maximum speed. It uses `torch.nn.functional.conv2d` and `conv_transpose2d`; FFT-based convolutions can be added later for larger experiments.

See `docs/learning_plan_zh.md` for the staged study and experiment plan in Chinese.

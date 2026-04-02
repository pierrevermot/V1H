# V1H

V1H is a research codebase for physics-informed adaptive-optics image reconstruction. It generates synthetic observations, trains staged neural networks to recover image, PSF, and noise components, and then fine-tunes a joint PINN model that enforces physical consistency.

At a high level, the model learns to satisfy:

$$
\mathrm{observation} \approx \mathrm{image} \otimes \mathrm{PSF} - \mathrm{noise}
$$

with uncertainty estimates for the recovered quantities.

## Pipeline

1. Generate synthetic sky, phase, PSF, and noise data and write TFRecords.
2. Train independent heads for the image, noise, and PSF mean.
3. Train a second-stage PSF uncertainty head.
4. Jointly fine-tune the full four-head PINN.
5. Plot validation or inference results.

The full pipeline is orchestrated by `workflow/run_experiment.sh`, which submits the stages as dependent SLURM jobs.

## Repository Layout

- `configs/`: experiment configuration template and config loader
- `instruments/`: telescope and AO instrument models
- `phases/`: phase screen generation (power-law, Zernike, LWE)
- `psfs/`: PSF generation and centering
- `skies/`: synthetic sky and object generation
- `noises/`: noise models and random noise parameter samplers
- `neural_networks/`: model builders, datasets, losses, training, plotting
- `utils/`: shared I/O, normalization, convolution, metrics, model helpers
- `workflow/`: dataset generation, staged training, plotting, and SLURM orchestration

## Installation

Python 3.10+ is expected.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

If you want the optional CuPy backend:

```bash
pip install -e .[gpu]
```

You are still responsible for matching TensorFlow and CuPy to your CUDA runtime.

## Configuration

All main workflows consume a single Python config file. Use `configs/experiment_template.py` as the starting point.

The config covers:

- instrument and wavelength sampling
- atmospheric phase generation
- synthetic sky generation
- noise generation
- dataset generation and loading
- loss configuration
- independent head architectures and training hyperparameters
- PSF uncertainty stage-2 settings
- joint PINN training settings
- SLURM settings and output directories

## Quick Start

Generate a dataset:

```bash
python workflow/create_dataset.py --config configs/experiment_template.py
```

Train one independent head:

```bash
python workflow/independent_training.py --config configs/experiment_template.py --head-target im
python workflow/independent_training.py --config configs/experiment_template.py --head-target noise
python workflow/independent_training.py --config configs/experiment_template.py --head-target psf
```

Train the remaining stages:

```bash
python workflow/psf_uncertainty_stage2_training.py --config configs/experiment_template.py
python workflow/joint_pinn_fourhead_training.py --config configs/experiment_template.py
```

Run the full SLURM pipeline:

```bash
bash workflow/run_experiment.sh configs/experiment_template.py --gpu h100
```

Preview the submitted jobs without launching them:

```bash
bash workflow/run_experiment.sh configs/experiment_template.py --gpu h100 --dry-run
```

Plot results from a run directory:

```bash
python workflow/plot_results.py --run-dir <run_dir> --checkpoint best --tfrecord <validation_tfrecord>
```

## Outputs

By default, outputs are rooted at `OUTPUT_BASE_DIR` from the experiment config.

Typical outputs include:

- `dataset/`: training and validation TFRecords
- per-head run directories with checkpoints, metrics, and plots
- `joint_pinn_fourhead/`: final joint model, metrics, and plots
- `slurm_logs/`: generated job scripts and SLURM stdout/stderr logs

## Notes

- The codebase is organized as Python packages and can be installed with `pip install -e .`.
- The workflow scripts still include a small `sys.path` fallback so they can also be run directly from the repository checkout.
- `workflow/plot_results.py` supports both TFRecord-based evaluation and external data inference.

## License

This project is licensed under the Mozilla Public License 2.0 (`MPL-2.0`). See `LICENSE` or visit <https://www.mozilla.org/en-US/MPL/2.0/>.
#!/usr/bin/env python3
"""Step 4: generate comparison plots and histograms from saved GalSim evaluation artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config
from testing.galsim_evaluation import _resolve_runtime_options, run_plotting


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate step4 plots for test_on_galsim outputs.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
	parser.add_argument("--run-dir", type=Path, default=None, help="Override trained model run directory")
	parser.add_argument("--model-label", choices=("best_model", "final_model"), default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--plot-examples", type=int, default=None, help="Number of examples per dataset to plot")
	parser.add_argument("--plot-dpi", type=int, default=None, help="Output DPI for saved figures")
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	cfg = load_experiment_config(args.config)
	settings = _resolve_runtime_options(
		cfg,
		run_dir=args.run_dir,
		model_label=args.model_label,
		output_dir=args.output_dir,
		plot_examples=args.plot_examples,
		plot_dpi=args.plot_dpi,
	)
	run_plotting(
		cfg=cfg,
		run_dir=settings["run_dir"],
		model_label=settings["model_label"],
		output_dir=settings["output_dir"],
		plot_examples=settings["plot_examples"],
		plot_dpi=settings["plot_dpi"],
	)


if __name__ == "__main__":
	main()
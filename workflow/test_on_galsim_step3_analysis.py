#!/usr/bin/env python3
"""Step 3: analyze saved val and GalSim inference artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config
from testing.galsim_evaluation import _resolve_runtime_options, run_analysis


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Analyze saved test_on_galsim inference artifacts.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
	parser.add_argument("--algorithm", type=str, default=None, help="Evaluation backend name")
	parser.add_argument("--run-dir", type=Path, default=None, help="Override trained model run directory")
	parser.add_argument("--model-label", choices=("best_model", "final_model"), default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--eval-batch-size", type=int, default=None)
	parser.add_argument(
		"--stats-examples",
		type=int,
		default=None,
		help="Number of saved examples per dataset to use for component statistics (0 disables)",
	)
	return parser.parse_args()


def main() -> None:
	args = _parse_args()
	cfg = load_experiment_config(args.config)
	settings = _resolve_runtime_options(
		cfg,
		algorithm=args.algorithm,
		run_dir=args.run_dir,
		model_label=args.model_label,
		output_dir=args.output_dir,
		eval_batch_size=args.eval_batch_size,
		stats_examples=args.stats_examples,
	)
	run_analysis(
		cfg=cfg,
		algorithm=settings["algorithm"],
		run_dir=settings["run_dir"],
		model_label=settings["model_label"],
		output_dir=settings["output_dir"],
		stats_examples=settings["stats_examples"],
		analysis_batch_size=settings["eval_batch_size"],
	)


if __name__ == "__main__":
	main()
#!/usr/bin/env python3
"""Step 2a: run inference on val and GalSim datasets and save artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from configs.load_config import load_experiment_config
from testing.galsim_evaluation import _resolve_runtime_options, run_inference


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run test_on_galsim inference and save artifacts.")
	parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
	parser.add_argument("--algorithm", type=str, default=None, help="Evaluation backend name")
	parser.add_argument("--run-dir", type=Path, default=None, help="Override trained model run directory")
	parser.add_argument("--model-label", choices=("best_model", "final_model"), default=None)
	parser.add_argument("--output-dir", type=Path, default=None)
	parser.add_argument("--eval-batch-size", type=int, default=None)
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
	)
	run_inference(
		cfg=cfg,
		algorithm=settings["algorithm"],
		run_dir=settings["run_dir"],
		model_label=settings["model_label"],
		output_dir=settings["output_dir"],
		eval_batch_size=settings["eval_batch_size"],
		first_batch_only=settings["first_batch_only"],
	)


if __name__ == "__main__":
	main()
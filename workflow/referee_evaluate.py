"""Stream referee evaluations, retaining per-image metrics and selected examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def evaluate(config: Path, output: Path, label: str, algorithm: str,
             run_dir: Path | None = None, rl_iterations: int | None = None,
             validation_examples: int = 1024, batch_size: int = 32) -> dict:
    import tensorflow as tf
    from configs.load_config import load_experiment_config
    from testing.galsim_evaluation import (
        BACKEND_REGISTRY, _build_dataset_specs, _make_eval_dataset,
        _evaluate_saved_inference,
    )

    if validation_examples < 1 or batch_size < 1:
        raise ValueError("Validation count and batch size must be positive")
    cfg = load_experiment_config(config)
    if rl_iterations is not None:
        if rl_iterations < 1:
            raise ValueError("RL iterations must be positive")
        cfg.RICHARDSON_LUCY_CONFIG = dict(cfg.RICHARDSON_LUCY_CONFIG, num_iter=rl_iterations)
    run_dir = run_dir or Path(cfg.OUTPUT_BASE_DIR) / cfg.JOINT_PINN_CONFIG["run_name"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Evaluation already exists: {output}")
    specs = _build_dataset_specs(cfg)
    preview, _ = next(iter(_make_eval_dataset(cfg, specs[0].path, 1)))
    backend = BACKEND_REGISTRY[algorithm].from_run(
        cfg=cfg, run_dir=run_dir, model_label="best_model", preview_obs=preview)
    crop = int(cfg.TEST_ON_GALSIM_CONFIG.get("eval_crop_border", 16))
    manifest = dict(label=label, algorithm=algorithm, run_dir=str(run_dir),
                    backend=backend.describe(), rl_iterations=rl_iterations,
                    crop_border=crop, first_batch_only=False, datasets={},
                    metric_aggregation="per-image metrics; median and IQR across images",
                    psnr_data_range="ground-truth max-minus-min for each image, shared by methods",
                    uncertainty_scope="marginal predictive intervals under simulator assumptions; no epistemic or model-mismatch guarantee")
    for spec in specs:
        limit = validation_examples if spec.name == "val" else None
        metrics: dict[str, list[np.ndarray]] = {}
        samples: dict[str, list[np.ndarray]] = {k: [] for k in ("obs", "y_true", "y_pred", "indices")}
        selected = {0, 60, 120, 265, 512, 800, 1023, 1209}
        count = 0
        for obs, truth in _make_eval_dataset(cfg, spec.path, batch_size):
            remaining = int(obs.shape[0]) if limit is None else min(int(obs.shape[0]), limit - count)
            if remaining <= 0:
                break
            obs, truth = obs[:remaining], truth[:remaining]
            pred = backend.predict_batch(obs, y_true=truth)
            obs_np, truth_np, pred_np = (np.asarray(x, dtype=np.float32) for x in (obs, truth, pred))
            batch_metrics = _evaluate_saved_inference(
                backend, obs=obs_np, y_true=truth_np, y_pred=pred_np,
                batch_size=batch_size, crop_border=crop)
            for key, value in batch_metrics.items():
                metrics.setdefault(key, []).append(value)
            local = [i for i in range(remaining) if count + i in selected]
            if local:
                for key, array in (("obs", obs_np), ("y_true", truth_np), ("y_pred", pred_np)):
                    samples[key].append(array[local])
                samples["indices"].append(np.asarray([count + i for i in local], dtype=np.int64))
            count += remaining
            if limit is not None and count >= limit:
                break
        if not count:
            raise ValueError(f"No examples in {spec.path}")
        if limit is not None and count != limit:
            raise ValueError(f"Expected {limit} validation examples, found {count}")
        if spec.name == "galsim":
            expected = int(cfg.GALSIM_TEST_CONFIG["n_scenes"]) * int(cfg.GALSIM_TEST_CONFIG["n_psfs"]) * int(cfg.GALSIM_TEST_CONFIG["n_noise_levels"])
            if count != expected:
                raise ValueError(f"Expected the full {expected}-case GalSim grid, evaluated {count}")
        metric_arrays = {key: np.concatenate(values) for key, values in metrics.items()}
        sample_arrays = {key: np.concatenate(values) for key, values in samples.items()}
        np.savez_compressed(output / f"metrics_{spec.name}.npz", **metric_arrays)
        np.savez_compressed(output / f"samples_{spec.name}.npz", **sample_arrays)
        manifest["datasets"][spec.name] = dict(
            n_examples=count, dataset_path=str(spec.path),
            metrics=f"metrics_{spec.name}.npz", samples=f"samples_{spec.name}.npz")
        print(f"{label}: evaluated {count} {spec.name} examples", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_dataset(config: Path, output: Path) -> dict:
    import tensorflow as tf
    from configs.load_config import load_experiment_config

    cfg = load_experiment_config(config)
    result = dict(config=str(config), splits={})
    for split, expected in (("train", int(cfg.DATASET_GEN_CONFIG["n_batches"])),
                            ("val", int(cfg.DATASET_GEN_CONFIG["val_n_batches"]))):
        directory = Path(cfg.DATASET_LOAD_CONFIG["data_dir"]) / split
        shards = sorted(directory.glob("*.tfrecord"))
        if len(shards) != expected:
            raise ValueError(f"{split}: expected {expected} shards, found {len(shards)}")
        for path in {shards[0], shards[-1]}:
            records = sum(1 for _ in tf.data.TFRecordDataset(str(path)))
            if records != int(cfg.DATASET_GEN_CONFIG["n_ex_per_batch"]):
                raise ValueError(f"Incomplete shard {path}: {records} records")
        result["splits"][split] = dict(path=str(directory.resolve()), shards=len(shards),
            examples=len(shards) * int(cfg.DATASET_GEN_CONFIG["n_ex_per_batch"]),
            size_bytes=sum(p.stat().st_size for p in shards),
            verification="shard count plus full record count of first and last shard")
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify-dataset")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("evaluate")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--label", required=True)
    run.add_argument("--algorithm", choices=("pretrained", "joint_pinn", "richardson_lucy", "wiener"), required=True)
    run.add_argument("--run-dir", type=Path)
    run.add_argument("--rl-iterations", type=int)
    run.add_argument("--validation-examples", type=int, default=1024)
    run.add_argument("--batch-size", type=int, default=32)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "verify-dataset":
        print(json.dumps(verify_dataset(**args), indent=2))
    else:
        evaluate(**args)


if __name__ == "__main__":
    main()

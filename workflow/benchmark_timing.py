"""Benchmark inference timing for independent heads and joint model.

Usage
-----
    python benchmark_timing.py --config <experiment.py> --mode independent \
        --head-target im --batch-size 64 --n-warmup 5 --n-repeats 50
    python benchmark_timing.py --config <experiment.py> --mode joint \
        --batch-size 64 --n-warmup 5 --n-repeats 50

Outputs a single JSON line to stdout with timing statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _detect_device_info() -> dict:
    """Gather device info (CPU count, GPU name/count if available)."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    cpus = tf.config.list_physical_devices("CPU")
    info = {
        "n_cpus_tf": len(cpus),
        "n_gpus_tf": len(gpus),
        "gpu_names": [],
        "cpu_count_os": os.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST", ""),
    }
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            info["gpu_names"].append(details.get("device_name", str(gpu)))
        except Exception:
            info["gpu_names"].append(str(gpu))
    return info


def _build_independent_model(cfg, head_target: str):
    """Build an independent head model from config (no trained weights)."""
    import tensorflow as tf
    from configs.load_config import get_head_config, extract_arch_config
    from neural_networks.unet import build_unet
    from neural_networks.gpkh import build_gpkh
    from neural_networks.gpkh_convdecoder import build_gpkh_convdecoder
    from neural_networks.skh import build_skh
    from neural_networks.dense_psf import build_dense_psf
    from utils.model_utils import _wrap_model_output_activation

    head_config = get_head_config(cfg, head_target)
    dataset_config = dict(cfg.DATASET_LOAD_CONFIG)
    model_name = str(head_config.get("model_name", "unet")).strip().lower()
    nll = str(head_config.get("loss_mode", "nll")).strip().lower() == "nll"
    output_activation = str(head_config.get("output_activation_function", "linear")).strip().lower()
    arch_config = extract_arch_config(head_config)

    n_pix = cfg.N_PIX
    crop = int(dataset_config.get("half_n_pix_crop", 16))
    input_size = n_pix - 2 * crop
    n_frames = int(cfg.INSTRUMENT_CONFIG.get("n_frames", 1))

    input_shape = (input_size, input_size, n_frames)

    HEAD_SPECS = {
        "im": {"out_channels": 1},
        "psf": {"out_channels": n_frames},
        "noise": {"out_channels": n_frames},
    }
    out_ch = HEAD_SPECS[head_target]["out_channels"]
    model_output_shape = (input_size, input_size, out_ch * (2 if nll else 1))

    builders = {
        "gpkh": build_gpkh,
        "gpkh_convdecoder": build_gpkh_convdecoder,
        "skh": build_skh,
        "dense": build_dense_psf,
    }
    kwargs = dict(arch_config)
    kwargs["input_shape"] = input_shape
    kwargs["output_shape"] = model_output_shape
    kwargs["output_activation_function"] = "linear"
    builder = builders.get(model_name, build_unet)
    model = builder(**kwargs)
    model = _wrap_model_output_activation(
        model, activation_name=output_activation,
        output_channels=out_ch, nll=nll,
    )
    return model, input_shape


def _build_joint_model(cfg):
    """Build the joint four-head PINN model from config (random weights)."""
    import tensorflow as tf
    from workflow.joint_pinn_fourhead_training import FourHeadJointPinnModel

    dataset_config = dict(cfg.DATASET_LOAD_CONFIG)
    n_pix = cfg.N_PIX
    crop = int(dataset_config.get("half_n_pix_crop", 16))
    input_size = n_pix - 2 * crop
    n_frames = int(cfg.INSTRUMENT_CONFIG.get("n_frames", 1))
    input_shape = (input_size, input_size, n_frames)

    # Build the four sub-models
    im_model, _ = _build_independent_model(cfg, "im")
    noise_model, _ = _build_independent_model(cfg, "noise")
    psf_mean_model, _ = _build_independent_model(cfg, "psf")

    # PSF uncertainty head takes obs + psf_mean as input
    from configs.load_config import get_head_config, extract_arch_config
    from neural_networks.gpkh import build_gpkh
    from neural_networks.gpkh_convdecoder import build_gpkh_convdecoder
    from neural_networks.skh import build_skh
    from neural_networks.dense_psf import build_dense_psf
    from neural_networks.unet import build_unet
    from utils.model_utils import _wrap_model_output_activation

    psf_unc_config = dict(cfg.PSF_UNC_CONFIG)
    model_name = str(psf_unc_config.get("model_name", "gpkh")).strip().lower()
    nll = True
    output_activation = str(psf_unc_config.get("output_activation_function", "linear")).strip().lower()
    arch_config = extract_arch_config(psf_unc_config)

    psf_unc_input_shape = (input_size, input_size, 2 * n_frames)
    model_output_shape = (input_size, input_size, n_frames * 2)

    builders = {
        "gpkh": build_gpkh,
        "gpkh_convdecoder": build_gpkh_convdecoder,
        "skh": build_skh,
        "dense": build_dense_psf,
    }
    kwargs = dict(arch_config)
    kwargs["input_shape"] = psf_unc_input_shape
    kwargs["output_shape"] = model_output_shape
    kwargs["output_activation_function"] = "linear"
    builder = builders.get(model_name, build_unet)
    psf_unc_model = builder(**kwargs)
    psf_unc_model = _wrap_model_output_activation(
        psf_unc_model, activation_name=output_activation,
        output_channels=n_frames, nll=nll,
    )

    joint_config = dict(cfg.JOINT_PINN_CONFIG)
    loss_config = dict(cfg.LOSS_CONFIG)
    joint_model = FourHeadJointPinnModel(
        image_model=im_model,
        noise_model=noise_model,
        psf_mean_model=psf_mean_model,
        psf_unc_model=psf_unc_model,
        pinn_weight=float(joint_config.get("pinn_weight", 1.0)),
        im_weight=float(joint_config.get("im_weight", 1.0)),
        psf_weight=float(joint_config.get("psf_weight", 1.0)),
        noise_weight=float(joint_config.get("noise_weight", 1.0)),
        log_sigma=bool(loss_config.get("log_sigma", False)),
        log_min=float(loss_config.get("log_min", -6.0)),
        log_max=float(loss_config.get("log_max", 20.0)),
        sigma2_eps=float(loss_config.get("sigma2_eps", 1e-30)),
        psf_mean_source_norm_psf=dataset_config.get("norm_psf", "npix2"),
        psf_unc_input_norm_psf=dataset_config.get("norm_psf", "npix2"),
        norm_psf=dataset_config.get("norm_psf", "npix2"),
        norm_noise=dataset_config.get("norm_noise", None),
        reconstruction_crop=int(joint_config.get("reconstruction_crop", 16)),
    )
    return joint_model, input_shape


def _benchmark_forward(model, input_shape, batch_size, n_warmup, n_repeats, device):
    """Run forward passes and measure timing."""
    import tensorflow as tf

    dummy = tf.random.normal((batch_size,) + input_shape)

    if device == "cpu":
        target_device = "/CPU:0"
    else:
        target_device = "/GPU:0"

    # Warm-up
    with tf.device(target_device):
        for _ in range(n_warmup):
            _ = model(dummy, training=False)
        if device != "cpu":
            tf.test.experimental.sync_devices()

    # Timed runs
    times = []
    with tf.device(target_device):
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = model(dummy, training=False)
            if device != "cpu":
                tf.test.experimental.sync_devices()
            t1 = time.perf_counter()
            times.append(t1 - t0)

    times = np.array(times)
    return {
        "mean_s": float(np.mean(times)),
        "std_s": float(np.std(times)),
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "per_sample_mean_ms": float(np.mean(times) / batch_size * 1000),
        "times_s": times.tolist(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model inference timing.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
    parser.add_argument("--mode", required=True, choices=["independent", "joint"],
                        help="Benchmark independent heads or the joint model")
    parser.add_argument("--head-target", choices=["im", "psf", "noise"], default=None,
                        help="Head target (required for --mode independent)")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for inference")
    parser.add_argument("--n-warmup", type=int, default=5, help="Number of warmup forward passes")
    parser.add_argument("--n-repeats", type=int, default=50, help="Number of timed forward passes")
    parser.add_argument("--device", choices=["cpu", "gpu"], default=None,
                        help="Force device (auto-detect if omitted)")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="Number of GPUs to use (for multi-GPU benchmarks)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    import tensorflow as tf

    # Configure GPU visibility
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if args.n_gpus > 0 and gpus:
            tf.config.set_visible_devices(gpus[:args.n_gpus], "GPU")
            for gpu in gpus[:args.n_gpus]:
                tf.config.experimental.set_memory_growth(gpu, True)

    device_info = _detect_device_info()

    effective_device = args.device
    if effective_device is None:
        effective_device = "gpu" if device_info["n_gpus_tf"] > 0 else "cpu"

    from configs.load_config import load_experiment_config
    cfg = load_experiment_config(args.config)

    if args.mode == "independent":
        if args.head_target is None:
            print("ERROR: --head-target is required for --mode independent", file=sys.stderr)
            sys.exit(1)
        model, input_shape = _build_independent_model(cfg, args.head_target)
        model_desc = f"independent_{args.head_target}"
    else:
        model, input_shape = _build_joint_model(cfg)
        model_desc = "joint_pinn_fourhead"

    timing = _benchmark_forward(
        model, input_shape, args.batch_size, args.n_warmup, args.n_repeats, effective_device,
    )

    n_params = int(model.count_params())

    result = {
        "mode": args.mode,
        "head_target": args.head_target,
        "model": model_desc,
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_repeats": args.n_repeats,
        "device": effective_device,
        "n_gpus_requested": args.n_gpus,
        "n_params": n_params,
        "input_shape": list(input_shape),
        "device_info": device_info,
        "timing": timing,
    }
    # Output JSON to stdout (one line)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

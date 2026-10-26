"""Benchmark inference timing for independent heads and joint model.

Usage
-----
    python benchmark_timing.py --config <experiment.py> --mode independent \
        --head-target im --batch-size 64 --n-warmup 5 --n-repeats 50
    python benchmark_timing.py --config <experiment.py> --mode joint \
        --batch-size 64 --n-warmup 5 --n-repeats 50
    python benchmark_timing.py --config <experiment.py> --run-dir <joint-run> \
        --mode independent --head-target im --object-source finetuned --batch-size 1

Outputs a single JSON line to stdout with graph-mode latency and throughput statistics.
"""

from __future__ import annotations

import argparse
import hashlib
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
    cpu_affinity_count = ""
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_affinity_count = len(os.sched_getaffinity(0))
        except OSError:
            cpu_affinity_count = ""
    info = {
        "n_cpu_devices_tf": len(cpus),
        "n_gpus_tf": len(gpus),
        "gpu_names": [],
        "cpu_count_os": os.cpu_count(),
        "cpu_affinity_count": cpu_affinity_count,
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK", ""),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_nodelist": os.environ.get("SLURM_NODELIST", ""),
        "tf_num_interop_threads": os.environ.get("TF_NUM_INTEROP_THREADS", ""),
        "tf_num_intraop_threads": os.environ.get("TF_NUM_INTRAOP_THREADS", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            info["gpu_names"].append(details.get("device_name", str(gpu)))
        except Exception:
            info["gpu_names"].append(str(gpu))
    return info


def _configure_precision(dtype_name: str, mixed_precision_policy: str) -> dict[str, str | bool]:
    """Configure runtime precision for benchmarking."""
    import tensorflow as tf

    if mixed_precision_policy != "none" and dtype_name != "float32":
        raise ValueError("--mixed-precision requires --dtype float32")

    tf.keras.backend.set_floatx(dtype_name)
    if mixed_precision_policy == "none":
        tf.keras.mixed_precision.set_global_policy(dtype_name)
    else:
        tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)

    policy = tf.keras.mixed_precision.global_policy()
    return {
        "dtype": dtype_name,
        "mixed_precision_policy": mixed_precision_policy,
        "global_policy": policy.name,
        "compute_dtype": policy.compute_dtype,
        "variable_dtype": policy.variable_dtype,
    }


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
    output_activation = str(psf_unc_config.get("output_activation_function", "linear")).strip().lower()
    arch_config = extract_arch_config(psf_unc_config)

    psf_unc_input_shape = (input_size, input_size, 2 * n_frames)
    # Stage 3 predicts only the uncertainty parameter; its mean is supplied by
    # the frozen PSF-mean head. It does not use the two-output NLL head layout.
    model_output_shape = (input_size, input_size, n_frames)

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
        output_channels=n_frames, nll=False,
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


def _checkpoint_identity(path: Path) -> dict:
    """Identify the exact saved checkpoint, outside the timed region."""
    path = path.expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _load_checkpoint_model(cfg, *, run_dir: Path, mode: str, head_target: str | None,
                           object_source: str, model_label: str):
    """Load trained weights; object-only timing runs only the selected head."""
    import tensorflow as tf
    from utils.io import _load_joint_run_config
    from utils.model_io import _load_independent_head_model, _resolve_model_paths

    run_dir = run_dir.expanduser().resolve(strict=True)
    dataset = dict(cfg.DATASET_LOAD_CONFIG)
    side = int(cfg.N_PIX) - 2 * int(dataset.get("half_n_pix_crop", 16))
    input_shape = (side, side, int(cfg.INSTRUMENT_CONFIG.get("n_frames", 1)))
    if mode == "joint" or object_source == "finetuned":
        if object_source == "finetuned" and mode == "independent" and head_target != "im":
            raise ValueError("--object-source finetuned requires --head-target im")
        from testing.galsim_evaluation import JointPinnBackend
        backend = JointPinnBackend.from_run(
            cfg=cfg, run_dir=run_dir, model_label=model_label,
            preview_obs=tf.zeros((1,) + input_shape, dtype=tf.float32),
        )
        model = backend.model if mode == "joint" else backend.model.image_model
        identity = _checkpoint_identity(backend.model_path)
        identity["component"] = "full_model" if mode == "joint" else "image_model"
        description = "joint_pinn_fourhead" if mode == "joint" else "finetuned_object"
    else:
        sections = {"im": ("image", "IMAGE_HEAD_CONFIG"),
                    "noise": ("noise", "NOISE_HEAD_CONFIG"),
                    "psf": ("psf_mean", "PSF_HEAD_CONFIG")}
        if head_target not in sections:
            raise ValueError("--head-target is required for --mode independent")
        source_key, config_key = sections[head_target]
        training_cfg = _load_joint_run_config(run_dir)
        source = dict(training_cfg.get("source_models", {}).get(source_key, {}))
        recorded = str(source.get("model_path", "")).strip()
        checkpoint = Path(recorded).expanduser() if recorded else None
        if checkpoint is None or not checkpoint.is_file():
            head_dir = run_dir.parent / str(getattr(cfg, config_key)["run_name"])
            checkpoint = _resolve_model_paths(head_dir)[model_label]
        out_channels = 1 if head_target == "im" else input_shape[-1]
        model, _ = _load_independent_head_model(
            checkpoint, fallback_input_shape=input_shape,
            fallback_output_shape=(side, side, out_channels),
        )
        identity = _checkpoint_identity(checkpoint)
        identity["component"] = source_key
        description = f"pretrained_{head_target}"
    shape = getattr(model, "input_shape", None)
    if shape is not None and len(shape) == 4 and all(v is not None for v in shape[1:]):
        input_shape = tuple(int(v) for v in shape[1:])
    return model, input_shape, description, identity


def _latency_statistics(latencies_s, batch_size: int) -> dict:
    """Keep batch latency separate from amortized time and throughput."""
    values = np.asarray(latencies_s, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Timings must be a nonempty vector of finite positive values")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    throughput_sps = batch_size / values
    return {
        "latency": {
            "batch_mean_s": float(np.mean(values)),
            "batch_std_s": float(np.std(values)),
            "batch_median_s": float(np.median(values)),
            "batch_min_s": float(np.min(values)),
            "batch_max_s": float(np.max(values)),
            "batch_times_s": values.tolist(),
        },
        "throughput": {
            "samples_per_second_mean": float(np.mean(throughput_sps)),
            "samples_per_second_std": float(np.std(throughput_sps)),
            "samples_per_second_median": float(np.median(throughput_sps)),
            "samples_per_second_min": float(np.min(throughput_sps)),
            "samples_per_second_max": float(np.max(throughput_sps)),
            "per_sample_mean_ms": float(np.mean(values) / batch_size * 1000),
            "per_sample_median_ms": float(np.median(values) / batch_size * 1000),
        },
        "single_image_latency_ms": float(np.mean(values) * 1000) if batch_size == 1 else None,
        "amortized_time_per_image_ms": float(np.mean(values) / batch_size * 1000),
    }


def _benchmark_forward(model, input_shape, batch_size, n_warmup, n_repeats, device, *, jit_compile,
                       include_transfers: bool = False):
    """Run graph-mode forward passes and measure latency and throughput."""
    import tensorflow as tf

    compute_dtype = tf.dtypes.as_dtype(getattr(model, "compute_dtype", None) or tf.keras.mixed_precision.global_policy().compute_dtype)
    if device == "cpu":
        target_device = "/CPU:0"
    else:
        target_device = "/GPU:0"

    host_dummy = np.random.default_rng(1234).normal(size=(batch_size,) + input_shape).astype(compute_dtype.as_numpy_dtype)
    with tf.device(target_device):
        dummy = tf.convert_to_tensor(host_dummy, dtype=compute_dtype)

    effective_n_warmup = n_warmup + 3

    @tf.function(jit_compile=jit_compile, reduce_retracing=True)
    def compiled_inference(batch):
        return model(batch, training=False)

    def sync_if_needed() -> None:
        if device != "cpu":
            tf.test.experimental.sync_devices()

    def run_once():
        batch = tf.convert_to_tensor(host_dummy, dtype=compute_dtype) if include_transfers else dummy
        result = compiled_inference(batch)
        if include_transfers:
            # Materialize every returned tensor on the host, including uncertainty.
            tf.nest.map_structure(lambda value: value.numpy(), result)
        sync_if_needed()

    # Build and compile the graph before warmup so trace/compile time is excluded.
    with tf.device(target_device):
        compile_start = time.perf_counter()
        _ = compiled_inference(dummy)
        sync_if_needed()
        compile_end = time.perf_counter()

    # Warm-up
    with tf.device(target_device):
        for _ in range(effective_n_warmup):
            run_once()

    # Synchronize every call; inference timing excludes the separately measured
    # transfer path, preprocessing, file I/O, tracing, and model loading.
    latencies_s = []
    with tf.device(target_device):
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            run_once()
            t1 = time.perf_counter()
            latencies_s.append(t1 - t0)

    return {
        "execution_mode": "graph",
        "jit_compile": bool(jit_compile),
        "requested_warmup_calls": int(n_warmup),
        "effective_warmup_calls": int(effective_n_warmup),
        "first_compiled_call_s": float(compile_end - compile_start),
        "scope": "input_transfer_network_output_transfer" if include_transfers else "network_only",
        "includes_host_device_transfers": bool(include_transfers),
        "includes_preprocessing": False,
        "preprocessing_note": "Inputs are already cropped and normalized; image decoding and scientific preprocessing are excluded.",
        "synchronization_mode": (
            "device_sync"
            if device != "cpu"
            else "host_blocking"
        ),
        "synchronization_note": (
            "GPU execution is synchronized after every call. Host/device transfers are included only in the separately requested transfer benchmark."
            if device != "cpu"
            else "CPU execution is host-blocking; preprocessing is excluded."
        ),
        **_latency_statistics(latencies_s, batch_size),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark model inference timing.")
    parser.add_argument("--config", type=Path, required=True, help="Path to experiment config .py file")
    parser.add_argument("--mode", required=True, choices=["independent", "joint"],
                        help="Benchmark independent heads or the joint model")
    parser.add_argument("--head-target", choices=["im", "psf", "noise"], default=None,
                        help="Head target (required for --mode independent)")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Joint run directory: load actual checkpoints rather than random architecture weights")
    parser.add_argument("--model-label", choices=["best_model", "final_model"], default="best_model")
    parser.add_argument("--object-source", choices=["pretrained", "finetuned"], default="pretrained",
                        help="For object-only timing, select the independent or fine-tuned object head")
    parser.add_argument("--end-to-end", action="store_true",
                        help="Also measure input/output transfers; excludes image decoding, cropping and normalization")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for inference")
    parser.add_argument("--n-warmup", type=int, default=5, help="Number of warmup forward passes")
    parser.add_argument("--n-repeats", type=int, default=50, help="Number of timed forward passes")
    parser.add_argument("--device", choices=["cpu", "gpu"], default=None,
                        help="Force device (auto-detect if omitted)")
    parser.add_argument("--jit-compile", action="store_true",
                        help="Enable XLA JIT compilation for the inference graph")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="Default dtype for non-mixed-precision benchmarking")
    parser.add_argument("--mixed-precision", choices=["none", "mixed_float16", "mixed_bfloat16"], default="none",
                        help="Optional mixed precision policy for supported hardware")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.n_repeats <= 0 or args.n_warmup < 0:
        raise ValueError("Batch size and repeats must be positive; warmup must be nonnegative")
    if args.object_source == "finetuned" and args.run_dir is None:
        raise ValueError("--object-source finetuned requires --run-dir")

    import tensorflow as tf

    # Configure GPU visibility
    if args.device == "cpu":
        tf.config.set_visible_devices([], "GPU")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus[:1], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
        elif args.device == "gpu":
            print("ERROR: --device gpu requested but no GPU is available", file=sys.stderr)
            sys.exit(1)

    precision_config = _configure_precision(args.dtype, args.mixed_precision)

    device_info = _detect_device_info()

    effective_device = args.device
    if effective_device is None:
        effective_device = "gpu" if device_info["n_gpus_tf"] > 0 else "cpu"

    from configs.load_config import load_experiment_config
    cfg = load_experiment_config(args.config)

    checkpoint = None
    if args.run_dir is not None:
        model, input_shape, model_desc, checkpoint = _load_checkpoint_model(
            cfg, run_dir=args.run_dir, mode=args.mode, head_target=args.head_target,
            object_source=args.object_source, model_label=args.model_label,
        )
    elif args.mode == "independent":
        if args.head_target is None:
            print("ERROR: --head-target is required for --mode independent", file=sys.stderr)
            sys.exit(1)
        model, input_shape = _build_independent_model(cfg, args.head_target)
        model_desc = f"independent_{args.head_target}"
    else:
        model, input_shape = _build_joint_model(cfg)
        model_desc = "joint_pinn_fourhead"

    timing = _benchmark_forward(
        model,
        input_shape,
        args.batch_size,
        args.n_warmup,
        args.n_repeats,
        effective_device,
        jit_compile=args.jit_compile,
    )

    n_params = int(model.count_params())

    result = {
        "mode": args.mode,
        "head_target": args.head_target,
        "model": model_desc,
        "weights": "checkpoint" if checkpoint is not None else "random_architecture",
        "checkpoint": checkpoint,
        "object_source": args.object_source if args.mode == "independent" and args.head_target == "im" else None,
        "batch_size": args.batch_size,
        "n_warmup": args.n_warmup,
        "n_repeats": args.n_repeats,
        "device": effective_device,
        "jit_compile": bool(args.jit_compile),
        "dtype": args.dtype,
        "mixed_precision": args.mixed_precision,
        "precision": precision_config,
        "model_compute_dtype": str(getattr(model, "compute_dtype", "unknown")),
        "n_params": n_params,
        "input_shape": list(input_shape),
        "device_info": device_info,
        "timing": timing,
    }
    if args.end_to_end:
        result["timing_with_transfers"] = _benchmark_forward(
            model, input_shape, args.batch_size, args.n_warmup, args.n_repeats,
            effective_device, jit_compile=args.jit_compile, include_transfers=True,
        )
    # Output JSON to stdout (one line)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

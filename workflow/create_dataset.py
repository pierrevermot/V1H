"""Generate TFRecord dataset for training and validation (sharded)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import threading
import queue as queue_mod

import numpy as np
import tensorflow as tf

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from utils.convolution import _convolve_image_with_psfs_numpy as _convolve_image_with_psfs
from utils.tfrecord_io import _bytes_feature, _int64_feature, serialize_example

# Route CuPy cache to $SCRATCH when available
_scratch = os.environ.get("SCRATCH")
if _scratch:
	os.environ.setdefault("CUPY_CACHE_DIR", str(Path(_scratch) / "cupy_cache"))

from configs.load_config import load_experiment_config
from instruments.ao_instrument import get_ao_instrument, get_obstructed_circular_ao_instrument
from utils.array_backend import to_numpy
from phases.phase_generator import generate_phase_screens
from phases.random_phase_parameters import draw_random_phase_parameters
from psfs.centering import center_psf_peak
from psfs.generate_psfs import long_exposure_psfs_vectorized, short_exposure_psf
from skies.image_generator import image_generator
from skies.random_sky_parameters import draw_random_image_parameters
from noises.noise_simulator import noise_simulator
from noises.random_noise_parameters import draw_random_noise_parameters


VLT_PRIMARY_DIAMETER_M = 8.0
VLT_CENTRAL_OBSCURATION_DIAMETER_M = 1.116


def _prepare_random_phase_kwargs(random_phase_config: dict, n_frames: int) -> dict[str, object]:
	kwargs = dict(random_phase_config)
	kwargs.pop("USE_CUPY", None)
	kwargs["n_frames"] = n_frames
	return kwargs


def _reference_psf_no_spiders(ref_ao_instru, rotated_index: int):
	xp = getattr(ref_ao_instru, "xp", np)
	zero_phase = xp.zeros_like(ref_ao_instru.pupil_array)
	psf = short_exposure_psf(ref_ao_instru, zero_phase, rotated_index=rotated_index)
	return center_psf_peak(psf)


def get_example(ao_instru, ref_ao_instru, rng=None, *, random_phase_config, random_sky_config, random_noise_config):
	"""Generate one training example.

	Returns
	-------
	image : ndarray
	obs : ndarray
	psf : ndarray
	noise : ndarray
	ref_psf : ndarray
	"""
	# a) random image
	sky_kwargs = dict(random_sky_config)
	sky_kwargs["rng"] = rng
	funcs, params, fluxes = draw_random_image_parameters(**sky_kwargs)
	image = image_generator(ao_instru, funcs, params, fluxes)

	# b) random phase cube
	n_frames = int(getattr(ao_instru, "n_frames", 2))
	phase_kwargs = _prepare_random_phase_kwargs(random_phase_config, n_frames)
	phase_kwargs.pop("rng", None)
	phase_params = draw_random_phase_parameters(
		ao_instru,
		rng=rng,
		**phase_kwargs,
	)
	exp_lf, exp_hf, cutoff, rms_lf, rms_hf = phase_params["powerlaw_params"]
	phase_screens = generate_phase_screens(
		ao_instru,
		phase_params["rotated_index"],
		phase_params["n_screens"],
		exponent_lf=exp_lf,
		exponent_hf=exp_hf,
		cutoff=cutoff,
		rms_lf=rms_lf,
		rms_hf=rms_hf,
		component_flags=phase_params["component_flags"],
		zernike_coeffs=phase_params["zernike_coeffs"],
		lwe_weights=phase_params["lwe_coeffs"],
		final_strehl=phase_params["final_strehl"],
	)

	# c) long-exposure PSF cube
	n_se_screens_per_le = phase_params["n_screens"] // n_frames
	psf = long_exposure_psfs_vectorized(
		ao_instru,
		phase_screens,
		rotated_index=phase_params["rotated_index"],
		n_se_screens_per_le=n_se_screens_per_le,
		n_le=n_frames,
	)
	psf = center_psf_peak(psf)
	ref_psf = _reference_psf_no_spiders(ref_ao_instru, rotated_index=phase_params["rotated_index"])

	# d) no-noise observation
	no_noise_obs = _convolve_image_with_psfs(ao_instru, image, psf)

	# e) add noise
	noise_kwargs = dict(random_noise_config)
	noise_kwargs["rng"] = rng
	noise_functions, noise_params, noise_rel_std, peak_snr, pixel_functions, pixel_params = (
		draw_random_noise_parameters(ao_instru, **noise_kwargs)
	)
	obs = noise_simulator(
		ao_instru,
		no_noise_obs,
		noise_functions,
		noise_params,
		noise_rel_std,
		peak_snr,
		pixel_functions,
		pixel_params,
	)

	# f) noise = (image * psf) - obs
	noise = no_noise_obs - obs

	return image, obs, psf, noise, ref_psf


def _write_batch(
	*,
	ao_instru,
	ref_ao_instru,
	batch_idx: int,
	output_dir: Path,
	n_ex_per_batch: int,
	seed: int,
	random_phase_config: dict,
	random_sky_config: dict,
	random_noise_config: dict,
):
	"""Write one TFRecord batch with a local RNG seed."""
	file_path = output_dir / f"batch_{batch_idx:04d}.tfrecord"
	if file_path.exists():
		return
	rng = np.random.default_rng(seed)
	with tf.io.TFRecordWriter(str(file_path)) as writer:
		for _ in range(n_ex_per_batch):
			image, obs, psf, noise, ref_psf = get_example(
				ao_instru, ref_ao_instru, rng=rng,
				random_phase_config=random_phase_config,
				random_sky_config=random_sky_config,
				random_noise_config=random_noise_config,
			)
			image = to_numpy(image)
			obs = to_numpy(obs)
			psf = to_numpy(psf)
			noise = to_numpy(noise)
			ref_psf = to_numpy(ref_psf)
			writer.write(serialize_example(image, obs, psf, noise, ref_psf))


def _worker_generate(queue, n_examples: int, seed: int, ao_instru, ref_ao_instru,
                     random_phase_config, random_sky_config, random_noise_config):
	"""Worker: generate examples and put serialized bytes into queue."""
	if seed is not None:
		np.random.seed(seed)
		rng = np.random.default_rng(seed)
	else:
		rng = None
	for _ in range(n_examples):
		image, obs, psf, noise, ref_psf = get_example(
			ao_instru, ref_ao_instru, rng=rng,
			random_phase_config=random_phase_config,
			random_sky_config=random_sky_config,
			random_noise_config=random_noise_config,
		)
		image = to_numpy(image)
		obs = to_numpy(obs)
		psf = to_numpy(psf)
		noise = to_numpy(noise)
		ref_psf = to_numpy(ref_psf)
		queue.put(serialize_example(image, obs, psf, noise, ref_psf))
	queue.put(None)


def _write_batch_queue(
	*,
	ao_instru,
	ref_ao_instru,
	batch_idx: int,
	output_dir: Path,
	n_ex_per_batch: int,
	seed: int,
	n_workers: int,
	random_phase_config: dict,
	random_sky_config: dict,
	random_noise_config: dict,
):
	"""Write one batch using producer workers and a single writer."""
	file_path = output_dir / f"batch_{batch_idx:04d}.tfrecord"
	if file_path.exists():
		return
	queue = queue_mod.Queue(maxsize=256)
	seeds = np.random.SeedSequence(seed).spawn(n_workers)
	worker_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in seeds]

	# Split work
	base = n_ex_per_batch // n_workers
	rem = n_ex_per_batch % n_workers
	counts = [base + (1 if i < rem else 0) for i in range(n_workers)]

	threads = []
	for i in range(n_workers):
		if counts[i] == 0:
			continue
		t = threading.Thread(
			target=_worker_generate,
			args=(queue, counts[i], worker_seeds[i], ao_instru, ref_ao_instru,
			      random_phase_config, random_sky_config, random_noise_config),
			daemon=True,
		)
		t.start()
		threads.append(t)

	# Writer loop
	finished = 0
	with tf.io.TFRecordWriter(str(file_path)) as writer:
		while finished < len(threads):
			item = queue.get()
			if item is None:
				finished += 1
				continue
			writer.write(item)

	for t in threads:
		t.join()


def write_dataset(
	ao_instru,
	ref_ao_instru,
	output_dir: Path,
	n_batches: int,
	n_ex_per_batch: int,
	*,
	parallel: bool,
	parallel_mode: str,
	n_workers: int,
	batch_offset: int = 0,
	seed_base: int | None = None,
	random_phase_config: dict,
	random_sky_config: dict,
	random_noise_config: dict,
):
	"""Write TFRecord dataset to output_dir."""
	output_dir.mkdir(parents=True, exist_ok=True)
	seed_seq = (
		np.random.SeedSequence(seed_base + batch_offset)
		if seed_base is not None
		else np.random.SeedSequence()
	)
	seeds = seed_seq.spawn(n_batches)
	batch_seeds = [int(s.generate_state(1, dtype=np.uint32)[0]) for s in seeds]

	if parallel and parallel_mode == "joblib":
		from joblib import Parallel, delayed

		n_jobs = n_workers
		batch_indices = [
			batch_idx
			for batch_idx in range(n_batches)
			if not (output_dir / f"batch_{batch_offset + batch_idx:04d}.tfrecord").exists()
		]
		Parallel(n_jobs=n_jobs, backend="loky")(
			delayed(_write_batch)(
				ao_instru=ao_instru,
				ref_ao_instru=ref_ao_instru,
				batch_idx=batch_offset + batch_idx,
				output_dir=output_dir,
				n_ex_per_batch=n_ex_per_batch,
				seed=batch_seeds[batch_idx],
				random_phase_config=random_phase_config,
				random_sky_config=random_sky_config,
				random_noise_config=random_noise_config,
			)
			for batch_idx in batch_indices
		)
		return

	if parallel and parallel_mode == "queue":
		for batch_idx in range(n_batches):
			_write_batch_queue(
				ao_instru=ao_instru,
				ref_ao_instru=ref_ao_instru,
				batch_idx=batch_offset + batch_idx,
				output_dir=output_dir,
				n_ex_per_batch=n_ex_per_batch,
				seed=batch_seeds[batch_idx],
				n_workers=n_workers,
				random_phase_config=random_phase_config,
				random_sky_config=random_sky_config,
				random_noise_config=random_noise_config,
			)
		return

	for batch_idx in range(n_batches):
		_write_batch(
			ao_instru=ao_instru,
			ref_ao_instru=ref_ao_instru,
			batch_idx=batch_offset + batch_idx,
			output_dir=output_dir,
			n_ex_per_batch=n_ex_per_batch,
			seed=batch_seeds[batch_idx],
			random_phase_config=random_phase_config,
			random_sky_config=random_sky_config,
			random_noise_config=random_noise_config,
		)


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate TFRecord dataset (sharded)")
	parser.add_argument(
		"--config",
		required=True,
		help="Path to experiment config .py file",
	)
	parser.add_argument(
		"--parallel-mode",
		choices=("joblib", "queue", "none"),
		default="joblib",
		help="Parallelization mode for CPU runs",
	)
	parser.add_argument(
		"--n-workers",
		type=int,
		default=40,
		help="Number of worker processes when parallel",
	)
	parser.add_argument("--output-dir", default=None, help="Root dataset output directory")
	parser.add_argument("--n-batches", type=int, default=None, help="Number of batches for this shard")
	parser.add_argument("--batch-offset", type=int, default=0, help="Batch index offset for this shard")
	parser.add_argument("--val-n-batches", type=int, default=None, help="Number of validation batches")
	parser.add_argument("--val-batch-offset", type=int, default=0, help="Validation batch index offset")
	parser.add_argument("--seed-base", type=int, default=None, help="Base seed for deterministic shards")
	args = parser.parse_args()

	# ---- Resolve config source ----
	cfg = load_experiment_config(args.config)
	ao_config = cfg.INSTRUMENT_CONFIG
	random_phase_config = cfg.PHASE_CONFIG
	random_sky_config = cfg.SKY_CONFIG
	random_noise_config = cfg.RANDOM_NOISE_CONFIG
	ds_gen = cfg.DATASET_GEN_CONFIG
	default_output_dir = ds_gen.get("output_dir", "/tmp/dataset")
	default_n_batches = ds_gen.get("n_batches", 1)
	default_n_ex_per_batch = ds_gen.get("n_ex_per_batch", 100)
	default_seed = ds_gen.get("seed", 1234)

	output_dir = args.output_dir or default_output_dir
	n_batches = args.n_batches if args.n_batches is not None else default_n_batches
	val_n_batches = args.val_n_batches if args.val_n_batches is not None else max(n_batches // 16, 1)
	seed_base = args.seed_base if args.seed_base is not None else default_seed

	output_root = Path(output_dir)
	train_dir = output_root / "train"
	val_dir = output_root / "val"

	config = dict(ao_config)
	ao_instru = get_ao_instrument(**config)
	ref_ao_instru = get_obstructed_circular_ao_instrument(
		diameter_m=VLT_PRIMARY_DIAMETER_M,
		central_obscuration_diameter_m=VLT_CENTRAL_OBSCURATION_DIAMETER_M,
		n_pix=int(config["n_pix"]),
		wavelength=float(config["wavelength"]),
		angular_pixel_scale=float(config["angular_pixel_scale"]),
		max_high_res_pixel_scale=float(config.get("max_high_res_pixel_scale", 2e-3)),
		angles=config.get("angles", None),
		zernike_n_rad=int(config.get("zernike_n_rad", 6)),
		zernike_n_min=int(config.get("zernike_n_min", 1)),
		n_frames=int(config.get("n_frames", 1)),
		use_cupy=bool(config.get("use_cupy", False)),
	)

	parallel = not bool(getattr(ao_instru, "use_cupy", False)) and args.parallel_mode != "none"
	parallel_mode = args.parallel_mode if parallel else "none"
	n_workers = int(args.n_workers)

	write_dataset(
		ao_instru,
		ref_ao_instru,
		train_dir,
		int(n_batches),
		default_n_ex_per_batch,
		parallel=parallel,
		parallel_mode=parallel_mode,
		n_workers=n_workers,
		batch_offset=int(args.batch_offset),
		seed_base=seed_base,
		random_phase_config=random_phase_config,
		random_sky_config=random_sky_config,
		random_noise_config=random_noise_config,
	)
	write_dataset(
		ao_instru,
		ref_ao_instru,
		val_dir,
		int(val_n_batches),
		default_n_ex_per_batch,
		parallel=parallel,
		parallel_mode=parallel_mode,
		n_workers=n_workers,
		batch_offset=int(args.val_batch_offset),
		seed_base=seed_base + 10_000 if seed_base is not None else None,
		random_phase_config=random_phase_config,
		random_sky_config=random_sky_config,
		random_noise_config=random_noise_config,
	)


if __name__ == "__main__":
	main()

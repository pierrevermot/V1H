from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from neural_networks.dataset import list_tfrecord_files
from utils.io import _resolve_first_file


def _bytes_feature(value: bytes) -> tf.train.Feature:
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value: int) -> tf.train.Feature:
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, obs, psf, noise, ref_psf) -> bytes:
	"""Serialize arrays into a TFRecord example."""
	image = np.asarray(image, dtype=np.float32)
	obs = np.asarray(obs, dtype=np.float32)
	psf = np.asarray(psf, dtype=np.float32)
	noise = np.asarray(noise, dtype=np.float32)
	ref_psf = np.asarray(ref_psf, dtype=np.float32)

	feature = {
		"image": _bytes_feature(tf.io.serialize_tensor(image).numpy()),
		"obs": _bytes_feature(tf.io.serialize_tensor(obs).numpy()),
		"psf": _bytes_feature(tf.io.serialize_tensor(psf).numpy()),
		"noise": _bytes_feature(tf.io.serialize_tensor(noise).numpy()),
		"ref_psf": _bytes_feature(tf.io.serialize_tensor(ref_psf).numpy()),
		"n_frames": _int64_feature(int(obs.shape[0]) if obs.ndim == 3 else 1),
		"n_pix": _int64_feature(int(image.shape[-1])),
	}

	proto = tf.train.Example(features=tf.train.Features(feature=feature))
	return proto.SerializeToString()


def _deserialize_tensor_bytes(payload: bytes) -> np.ndarray:
	tensor_proto = tensor_pb2.TensorProto()
	tensor_proto.ParseFromString(payload)
	return np.asarray(tf.make_ndarray(tensor_proto), dtype=np.float32)


def _decode_raw_example(serialized: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	example = tf.train.Example()
	example.ParseFromString(serialized)
	features = example.features.feature
	n_frames = int(features["n_frames"].int64_list.value[0])
	n_pix = int(features["n_pix"].int64_list.value[0])
	image = _deserialize_tensor_bytes(features["image"].bytes_list.value[0])
	obs = _deserialize_tensor_bytes(features["obs"].bytes_list.value[0])
	psf = _deserialize_tensor_bytes(features["psf"].bytes_list.value[0])
	noise_key = "noise" if "noise" in features else "residuals"
	noise = _deserialize_tensor_bytes(features[noise_key].bytes_list.value[0])
	return (
		np.reshape(image, (n_pix, n_pix)).astype(np.float32),
		np.reshape(obs, (n_frames, n_pix, n_pix)).astype(np.float32),
		np.reshape(psf, (n_frames, n_pix, n_pix)).astype(np.float32),
		np.reshape(noise, (n_frames, n_pix, n_pix)).astype(np.float32),
	)


def _load_preview_raw_from_first_tfrecord(data_dir: Path) -> dict[str, np.ndarray]:
	tfrecord_files = list_tfrecord_files(data_dir)
	first_path = Path(tfrecord_files[0])
	serialized = next(iter(tf.data.TFRecordDataset([str(first_path)]).take(1)), None)
	if serialized is not None:
		image, obs, psf, noise = _decode_raw_example(bytes(serialized.numpy()))
		return {
			"image_hh": image[np.newaxis, ...],
			"obs_fhh": obs[np.newaxis, ...],
			"psf_fhh": psf[np.newaxis, ...],
			"noise_fhh": noise[np.newaxis, ...],
		}
	raise RuntimeError(f"No examples decoded from TFRecord: {first_path}")


def _first_example_raw_from_dataset(data_dir: Path) -> dict[str, np.ndarray]:
	files = list_tfrecord_files(data_dir)
	for file_path in files:
		serialized = next(iter(tf.data.TFRecordDataset([str(file_path)]).take(1)), None)
		if serialized is None:
			continue
		image, obs, psf, noise = _decode_raw_example(bytes(serialized.numpy()))
		return {
			"image_hh": image[np.newaxis, ...].astype(np.float32),
			"obs_fhh": obs[np.newaxis, ...].astype(np.float32),
			"psf_fhh": psf[np.newaxis, ...].astype(np.float32),
			"noise_fhh": noise[np.newaxis, ...].astype(np.float32),
		}
	raise RuntimeError(f"No examples decoded from dataset under {data_dir}")


def _parse_example(example_proto: tf.Tensor) -> dict[str, tf.Tensor]:
	feature_spec = {
		"obs": tf.io.FixedLenFeature([], tf.string),
		"psf": tf.io.FixedLenFeature([], tf.string),
		"n_frames": tf.io.FixedLenFeature([], tf.int64),
		"n_pix": tf.io.FixedLenFeature([], tf.int64),
	}
	features = tf.io.parse_single_example(example_proto, feature_spec)
	obs = tf.io.parse_tensor(features["obs"], out_type=tf.float32)
	psf = tf.io.parse_tensor(features["psf"], out_type=tf.float32)
	n_frames = tf.cast(features["n_frames"], tf.int32)
	n_pix = tf.cast(features["n_pix"], tf.int32)
	obs = tf.reshape(obs, (n_frames, n_pix, n_pix))
	psf = tf.reshape(psf, (n_frames, n_pix, n_pix))
	return {
		"obs": obs,
		"psf": psf,
		"n_frames": n_frames,
		"n_pix": n_pix,
	}


def _iter_tfrecord_records(tfrecord_path: Path):
	for serialized in tf.compat.v1.io.tf_record_iterator(str(tfrecord_path)):
		yield serialized


def _load_selected_tfrecord_arrays(
	tfrecord_path: Path,
	*,
	n_examples: int,
	shuffle: bool,
	seed: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
	if n_examples <= 0:
		raise ValueError("n_examples must be > 0")
	rng = np.random.default_rng(seed)
	selected: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
	for idx, serialized in enumerate(_iter_tfrecord_records(tfrecord_path)):
		image, obs, psf, noise = _decode_raw_example(serialized)
		if not shuffle:
			if len(selected) < n_examples:
				selected.append((idx, image, obs, psf, noise))
			else:
				break
			continue
		if len(selected) < n_examples:
			selected.append((idx, image, obs, psf, noise))
			continue
		replace_idx = int(rng.integers(0, idx + 1))
		if replace_idx < n_examples:
			selected[replace_idx] = (idx, image, obs, psf, noise)
	if not selected:
		raise RuntimeError(f"No examples decoded from TFRecord: {tfrecord_path}")
	selected.sort(key=lambda item: item[0])
	indices = np.asarray([item[0] for item in selected], dtype=np.int64)
	return {
		"image_hh": np.stack([item[1] for item in selected], axis=0).astype(np.float32),
		"obs_fhh": np.stack([item[2] for item in selected], axis=0).astype(np.float32),
		"psf_fhh": np.stack([item[3] for item in selected], axis=0).astype(np.float32),
		"noise_fhh": np.stack([item[4] for item in selected], axis=0).astype(np.float32),
	}, indices


def _resolve_tfrecord_path(run_dir: Path, tfrecord_path: Path | None) -> Path | None:
	if tfrecord_path is not None:
		path = tfrecord_path.expanduser().resolve()
		if not path.exists():
			raise FileNotFoundError(path)
		return path
	candidate = _resolve_first_file(run_dir, "*.tfrecord")
	return candidate.resolve() if candidate is not None else None


def _resolve_data_path(run_dir: Path, data_path: Path | None) -> Path | None:
	if data_path is not None:
		path = data_path.expanduser().resolve()
		if not path.exists():
			raise FileNotFoundError(path)
		return path
	candidate = run_dir / "data.npy"
	return candidate.resolve() if candidate.exists() else None


def _resolve_joint_tfrecord_path(run_dir: Path, tfrecord_path: Path | None, dataset_cfg: dict[str, Any]) -> Path | None:
	path = _resolve_tfrecord_path(run_dir, tfrecord_path)
	if path is not None:
		return path
	data_dir = str(dataset_cfg.get("data_dir", "")).strip()
	if not data_dir:
		return None
	dataset_root = Path(data_dir).expanduser().resolve()
	for split in ("val", "train"):
		split_dir = dataset_root / split
		candidate = _resolve_first_file(split_dir, "*.tfrecord") if split_dir.exists() else None
		if candidate is not None:
			return candidate.resolve()
	return None

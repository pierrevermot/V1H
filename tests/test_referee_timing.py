"""Check timing units and trained-head selection without requiring a GPU."""

import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from workflow import benchmark_timing as timing


def test_batch_throughput_is_not_single_image_latency():
    result = timing._latency_statistics([0.064, 0.128], batch_size=64)
    assert result["latency"]["batch_mean_s"] == pytest.approx(0.096)
    assert result["amortized_time_per_image_ms"] == pytest.approx(1.5)
    assert result["single_image_latency_ms"] is None
    assert result["throughput"]["samples_per_second_mean"] == pytest.approx(750)
    assert timing._latency_statistics([0.003], 1)["single_image_latency_ms"] == pytest.approx(3)


@pytest.mark.parametrize("values,batch", [([], 1), ([0], 1), ([np.nan], 1), ([1], 0)])
def test_invalid_timing_samples_are_rejected(values, batch):
    with pytest.raises(ValueError):
        timing._latency_statistics(values, batch)


def test_checkpoint_identity_hashes_exact_bytes(tmp_path):
    checkpoint = tmp_path / "weights.keras"
    checkpoint.write_bytes(b"trained-checkpoint")
    result = timing._checkpoint_identity(checkpoint)
    assert result == {"path": str(checkpoint), "size_bytes": 18,
                      "sha256": hashlib.sha256(b"trained-checkpoint").hexdigest()}


def test_object_only_selection_does_not_time_other_heads(tmp_path, monkeypatch):
    checkpoint = tmp_path / "joint.keras"
    checkpoint.write_bytes(b"joint")
    initial_checkpoint = tmp_path / "object.keras"
    initial_checkpoint.write_bytes(b"object")
    cfg = SimpleNamespace(N_PIX=160, DATASET_LOAD_CONFIG={"half_n_pix_crop": 16},
                          INSTRUMENT_CONFIG={"n_frames": 1})
    image_head = SimpleNamespace(input_shape=(None, 128, 128, 1))
    initial_head = SimpleNamespace(input_shape=(None, 128, 128, 1))
    joint = SimpleNamespace(image_model=image_head)
    backend = SimpleNamespace(model=joint, model_path=checkpoint)
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(
        float32=np.float32, zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype)))
    monkeypatch.setitem(sys.modules, "testing.galsim_evaluation", SimpleNamespace(
        JointPinnBackend=SimpleNamespace(from_run=lambda **kwargs: backend)))
    monkeypatch.setitem(sys.modules, "utils.io", SimpleNamespace(
        _load_joint_run_config=lambda root: {"source_models": {"image": {"model_path": str(initial_checkpoint)}}}))
    monkeypatch.setitem(sys.modules, "utils.model_io", SimpleNamespace(
        _load_independent_head_model=lambda *args, **kwargs: (initial_head, {}),
        _resolve_model_paths=lambda root: pytest.fail("Recorded source checkpoint should be used")))
    options = dict(cfg=cfg, run_dir=tmp_path, mode="independent", head_target="im", model_label="best_model")
    model, shape, name, identity = timing._load_checkpoint_model(**options, object_source="finetuned")
    assert model is image_head and model is not joint
    assert shape == (128, 128, 1) and name == "finetuned_object"
    assert identity["component"] == "image_model"
    model, _, name, identity = timing._load_checkpoint_model(**options, object_source="pretrained")
    assert model is initial_head and name == "pretrained_im"
    assert identity["path"] == str(initial_checkpoint)


def test_architecture_psf_uncertainty_has_one_output_channel():
    pytest.importorskip("tensorflow")
    from configs import config_naco_M
    model, shape = timing._build_joint_model(config_naco_M)
    assert shape == (128, 128, 1)
    assert tuple(model.psf_unc_model.output_shape) == (None, 128, 128, 1)
    assert sum(head.count_params() for head in (
        model.image_model, model.noise_model, model.psf_mean_model, model.psf_unc_model,
    )) == 59_179_876

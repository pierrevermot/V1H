"""Exercise the actual Keras checkpoint/rebuild path with tiny real networks."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def saved_four_head_run(tmp_path, monkeypatch):
    tf = pytest.importorskip("tensorflow")
    from neural_networks.gpkh import build_gpkh
    from neural_networks.unet import build_unet
    from workflow.joint_pinn_fourhead_training import FourHeadJointPinnModel

    # Model I/O extracts weights under SCRATCH. Keep every test artifact local
    # to this fixture, including when running on a workstation without Jean Zay.
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    tf.keras.utils.set_random_seed(1234)
    side = 16
    shape = (side, side, 1)
    common = dict(layers_per_block=1, base_filters=2, normalization="none",
                  group_norm_groups=8, weight_decay=0,
                  inner_activation_function="relu", output_activation_function="linear")
    psf_arch = dict(common, latent_dim=8, inner_activation_function="softplus",
                    normalize_output_sum=True, normalize_with_first=True,
                    normalize_first_only=False, normalize_by_mean=True)
    unc_arch = dict(psf_arch, normalize_output_sum=False,
                    normalize_with_first=False, normalize_by_mean=False)
    image = build_unet(shape, (side, side, 2), **common)
    noise = build_unet(shape, (side, side, 2), **common)
    psf_mean = build_gpkh(shape, shape, **psf_arch)
    psf_unc = build_gpkh((side, side, 2), shape, **unc_arch)
    dataset = dict(norm_psf="npix2", norm_noise=None, half_n_pix_crop=0)
    models = {}
    for label, target, model, name, arch, nll in (
        ("image", "im", image, "unet", common, True),
        ("noise", "noise", noise, "unet", common, True),
        ("psf_mean", "psf", psf_mean, "gpkh", psf_arch, False),
        ("psf_unc", "psf", psf_unc, "gpkh", unc_arch, False),
    ):
        directory = tmp_path / label
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(parents=True)
        path = checkpoints / f"{name}_best.keras"
        model.save(path)
        config = dict(dataset=dataset, head_target=target, model=dict(name=name),
                      nll=nll, output_activation_function="linear",
                      loss=dict(loss="nll" if nll else "var_normalized_mse"),
                      **{name: arch})
        if label == "psf_unc":
            config["model"].update(input_shape=[side, side, 2],
                                   output_shape=[side, side, 1], output_channels=1)
        (directory / "training_config.json").write_text(json.dumps(config))
        models[label] = dict(model_path=str(path), run_dir=str(directory),
                             model_label="best_model", trainable=label == "image")
    for model in (noise, psf_mean, psf_unc):
        model.trainable = False
    joint = FourHeadJointPinnModel(
        image_model=image, noise_model=noise, psf_mean_model=psf_mean,
        psf_unc_model=psf_unc, pinn_weight=1, im_weight=1, psf_weight=1,
        noise_weight=1, log_sigma=False, log_min=-6, log_max=20,
        sigma2_eps=1e-12, psf_mean_source_norm_psf="npix2",
        psf_unc_input_norm_psf="npix2", norm_psf="npix2", norm_noise=None,
        reconstruction_crop=2, name="joint_pinn_fourhead_model",
    )
    observation = tf.reshape(tf.linspace(0.1, 1.0, side * side), (1,) + shape)
    pretrained = joint(observation, training=False).numpy()
    # Change only the object mean. Independent checkpoints above remain the
    # original weights, while the full joint checkpoint contains this update.
    final_conv = [layer for layer in image.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
    final_conv.bias.assign_add(tf.constant([0.25, 0], dtype=final_conv.bias.dtype))
    finetuned = joint(observation, training=False).numpy()
    run = tmp_path / "joint_pinn_fourhead"
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/joint_pinn_fourhead_best.keras"
    joint.save(checkpoint)
    training_config = dict(
        dataset=dataset,
        loss=dict(log_sigma=False, log_min=-6, log_max=20, sigma2_eps=1e-12,
                  reconstruction_crop=2, weights=dict(pinn=1, im=1, psf=1, noise=1)),
        training=dict(checkpoint_path=str(checkpoint)), source_models=models,
    )
    (run / "training_config.json").write_text(json.dumps(training_config))
    cfg = SimpleNamespace(
        OUTPUT_BASE_DIR=str(tmp_path), N_PIX=side, DATASET_LOAD_CONFIG=dataset,
        INSTRUMENT_CONFIG=dict(n_frames=1),
        IMAGE_HEAD_CONFIG=dict(run_name="image"),
        NOISE_HEAD_CONFIG=dict(run_name="noise"),
        PSF_HEAD_CONFIG=dict(run_name="psf_mean"),
        PSF_UNC_CONFIG=dict(run_name="psf_unc"),
        JOINT_PINN_CONFIG=dict(run_name="joint_pinn_fourhead"),
        LOSS_CONFIG=dict(log_sigma=False, log_min=-6, log_max=20, sigma2_eps=1e-12),
    )
    return SimpleNamespace(cfg=cfg, run=run, observation=observation,
                           pretrained=pretrained, finetuned=finetuned,
                           checkpoint=checkpoint, source_models=models)


def test_saved_independent_and_joint_checkpoints_roundtrip(saved_four_head_run):
    from testing.galsim_evaluation import BACKEND_REGISTRY

    case = saved_four_head_run
    before = BACKEND_REGISTRY["pretrained"].from_run(
        cfg=case.cfg, run_dir=case.run, model_label="best_model", preview_obs=case.observation)
    after = BACKEND_REGISTRY["joint_pinn"].from_run(
        cfg=case.cfg, run_dir=case.run, model_label="best_model", preview_obs=case.observation)
    before_prediction = before.predict_batch(case.observation).numpy()
    after_prediction = after.predict_batch(case.observation).numpy()
    np.testing.assert_allclose(before_prediction, case.pretrained, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(after_prediction, case.finetuned, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(after_prediction[..., 0] - before_prediction[..., 0], 0.25, atol=2e-6)
    # Noise, PSF mean, and all three uncertainty parameters remain identical.
    np.testing.assert_allclose(before_prediction[..., 1:], after_prediction[..., 1:], rtol=2e-5, atol=2e-6)
    assert before.describe()["joint_checkpoint_loaded"] is False
    assert after.describe()["joint_checkpoint_loaded"] is True


def test_pretrained_baseline_does_not_need_a_joint_checkpoint(saved_four_head_run):
    from testing.galsim_evaluation import BACKEND_REGISTRY

    case = saved_four_head_run
    case.checkpoint.unlink()
    # The real campaign evaluates the baseline once pretraining finishes,
    # potentially before joint training has produced any checkpoint/config.
    (case.run / "training_config.json").unlink()
    backend = BACKEND_REGISTRY["pretrained"].from_run(
        cfg=case.cfg, run_dir=case.run, model_label="best_model", preview_obs=case.observation)
    np.testing.assert_allclose(backend.predict_batch(case.observation), case.pretrained, rtol=2e-5, atol=2e-6)


def test_object_only_timing_loads_the_requested_saved_component(saved_four_head_run):
    from workflow.benchmark_timing import _load_checkpoint_model

    case = saved_four_head_run
    before, shape, description, identity = _load_checkpoint_model(
        case.cfg, run_dir=case.run, mode="independent", head_target="im",
        object_source="pretrained", model_label="best_model")
    after, shape_after, description_after, identity_after = _load_checkpoint_model(
        case.cfg, run_dir=case.run, mode="independent", head_target="im",
        object_source="finetuned", model_label="best_model")
    np.testing.assert_allclose(before(case.observation).numpy()[..., 0], case.pretrained[..., 0], atol=2e-6)
    np.testing.assert_allclose(after(case.observation).numpy()[..., 0], case.finetuned[..., 0], atol=2e-6)
    assert shape == shape_after == (16, 16, 1)
    assert description == "pretrained_im" and description_after == "finetuned_object"
    assert identity["path"] == case.source_models["image"]["model_path"]
    assert identity_after["path"] == str(case.checkpoint)
    assert identity_after["component"] == "image_model"


def test_serial_tfrecord_reads_have_identical_cross_job_record_order(tmp_path):
    tf = pytest.importorskip("tensorflow")
    from neural_networks.dataset import make_dataset
    from utils.tfrecord_io import serialize_example

    for shard in range(3):
        with tf.io.TFRecordWriter(str(tmp_path / f"batch_{shard:04d}.tfrecord")) as writer:
            for offset in range(2):
                identifier = 10 * shard + offset
                image = np.full((8, 8), identifier, dtype=np.float32)
                psf = np.zeros((1, 8, 8), dtype=np.float32)
                psf[0, 4, 4] = 1
                writer.write(serialize_example(image, image[None], psf,
                                               np.zeros_like(psf), psf[0]))
    def observations():
        dataset = make_dataset(tmp_path, batch_size=2, shuffle=False, repeat=False,
                               half_n_pix_crop=0, norm_psf=None, norm_noise=None,
                               num_parallel_reads=1, num_parallel_calls=1, prefetch=False)
        return np.concatenate([obs[:, 0, 0, 0].numpy() for obs, _ in dataset])
    expected = [0, 1, 10, 11, 20, 21]
    np.testing.assert_array_equal(observations(), expected)
    np.testing.assert_array_equal(observations(), expected)

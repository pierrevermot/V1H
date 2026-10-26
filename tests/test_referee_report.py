"""Numerical/report checks independent of TensorFlow or trained weights."""

import json
from pathlib import Path

import numpy as np
import pytest

from workflow import referee_report as report


def _campaign(tmp_path):
    (tmp_path / "campaign.json").write_text(json.dumps(dict(pixel_scale_arcsec=0.027,
        expected_evaluations=["pretrained", "joint", "lambda_0", "rl_10", "rl_30"])))
    truth = np.zeros((2, 8, 8, 3), dtype=np.float32)
    truth[:, 2:6, 2:6, 0] = 1
    for label, algorithm, val, test in (
        ("pretrained", "pretrained", 1, 1), ("joint", "joint_pinn", 2, 2),
        ("lambda_0", "joint_pinn", 3, 3),
        ("rl_10", "richardson_lucy", 0.4, 0.9),
        ("rl_30", "richardson_lucy", 0.5, 0.1),
    ):
        folder = tmp_path / "referee_evaluation" / label
        folder.mkdir(parents=True)
        manifest = dict(label=label, algorithm=algorithm, datasets={}, crop_border=1)
        if label.startswith("rl_"):
            manifest["rl_iterations"] = int(label.split("_")[1])
        for dataset, error in (("val", val), ("galsim", test)):
            metrics = dict(r2_im=np.full(4, error), psnr_im=np.arange(4) + error,
                           ssim_im=np.arange(4) * 0.1 + error, r2_pinn=np.full(4, error * 0.01))
            if algorithm in ("pretrained", "joint_pinn"):
                for component in ("im", "psf", "noise", "obs"):
                    metrics[f"coverage_1sigma_{component}"] = np.asarray([0.6, 0.7, 0.8, 0.9])
                    metrics[f"coverage_2sigma_{component}"] = np.asarray([0.9, 0.92, 0.96, 0.98])
            np.savez(folder / f"metrics_{dataset}.npz", **metrics)
            pred = np.concatenate((truth, np.ones_like(truth)), axis=-1) if algorithm in ("pretrained", "joint_pinn") else truth
            np.savez(folder / f"samples_{dataset}.npz", obs=truth[..., :1], y_true=truth,
                     y_pred=pred, indices=np.asarray([0, 3]))
            manifest["datasets"][dataset] = dict(n_examples=4, metrics=f"metrics_{dataset}.npz",
                                                 samples=f"samples_{dataset}.npz")
        (folder / "manifest.json").write_text(json.dumps(manifest))
    generation = tmp_path / "dataset/galsim_test"
    generation.mkdir(parents=True)
    (generation / "generation_parameter_log.json").write_text(json.dumps(dict(
        per_example=[dict(example_index=i, noise_sigma=i // 2) for i in range(4)])))


def test_validation_selects_rl_without_test_leakage_and_pairs_differences(tmp_path, monkeypatch):
    _campaign(tmp_path)
    monkeypatch.setattr(report, "_make_figures", lambda *args: None)
    result = report.aggregate(tmp_path)
    assert result["complete"] and result["selected_rl"] == "rl_10"
    # RL30 wins on the test set, but must not be selected using that fact.
    sensitivity = [row for row in result["rl_sensitivity"] if row["dataset"] == "galsim" and row["metric"] == "r2_im"]
    assert next(r for r in sensitivity if r["label"] == "rl_30")["median"] == pytest.approx(0.1)
    assert next(r for r in sensitivity if r["label"] == "rl_10")["selected"]
    paired = next(r for r in result["paired"] if r["dataset"] == "val" and r["label"] == "joint" and r["metric"] == "r2_im")
    assert paired["median"] == pytest.approx(1) and paired["fraction_improved"] == 0
    assert paired["n_examples"] == 4
    coverage = [r for r in result["coverage_by_noise"] if r["label"] == "joint" and r["metric"] == "coverage_1sigma_im"]
    assert [r["mean"] for r in coverage] == pytest.approx([0.65, 0.85])
    assert [r["n_examples"] for r in coverage] == [2, 2]
    saved = json.loads((tmp_path / "referee_report/summary.json").read_text())
    assert saved["selected_rl"] == "rl_10"
    assert "ground-truth noise" in saved["classical_comparison_notes"][1]


def test_missing_expected_evaluation_is_explicit_and_partial_report_survives(tmp_path, monkeypatch):
    _campaign(tmp_path)
    monkeypatch.setattr(report, "_make_figures", lambda *args: None)
    (tmp_path / "referee_evaluation/lambda_0/manifest.json").unlink()
    with pytest.raises(RuntimeError, match="lambda_0"):
        report.aggregate(tmp_path)
    result = report.aggregate(tmp_path, allow_partial=True)
    assert not result["complete"]
    assert any("lambda_0" in problem for problem in result["problems"])
    assert (tmp_path / "referee_report/report.md").is_file()


def test_pairing_rejects_different_ground_truths(tmp_path, monkeypatch):
    _campaign(tmp_path)
    monkeypatch.setattr(report, "_make_figures", lambda *args: None)
    sample_file = tmp_path / "referee_evaluation/joint/samples_val.npz"
    samples = report._read_npz(sample_file)
    samples["y_true"] = samples["y_true"] + 1
    np.savez(sample_file, **samples)
    result = report.aggregate(tmp_path, allow_partial=True)
    assert not result["complete"]
    assert any("different ground truths" in problem for problem in result["problems"])
    assert not any(row["label"] == "joint" and row["dataset"] == "val" for row in result["paired"])


def test_perfect_psnr_remains_visible_in_json_summary():
    result = report.distribution([np.inf, np.inf])
    assert result["median"] == "Infinity"
    assert result["n_valid"] == 2 and result["n_finite"] == 0
    json.dumps(result, allow_nan=False)


def test_requested_auxiliary_products_are_required_but_missing_mistral_is_not(tmp_path, monkeypatch):
    _campaign(tmp_path)
    monkeypatch.setattr(report, "_make_figures", lambda *args: None)
    config = json.loads((tmp_path / "campaign.json").read_text())
    config.update(expected_timing_cases=["joint_bs1"], inputs={"naco": {"path": "input.npy"}}, source_commit="abc123")
    (tmp_path / "campaign.json").write_text(json.dumps(config))
    result = report.aggregate(tmp_path, allow_partial=True)
    assert not result["complete"] and result["source_revision"] == "abc123"
    assert any("timing" in problem for problem in result["problems"])
    assert any("NaCo" in problem for problem in result["problems"])
    folder = tmp_path / "benchmark_timing/results"
    folder.mkdir(parents=True)
    (folder / "joint_bs1.json").write_text(json.dumps(dict(weights="checkpoint", checkpoint={"path": "joint.keras"},
        batch_size=1, n_params=59_179_876, timing=dict(scope="network_only", includes_preprocessing=False,
            latency={"batch_mean_s": 0.003}, throughput={"per_sample_mean_ms": 3, "samples_per_second_mean": 333.33}))))
    folder = tmp_path / "naco_comparison"
    folder.mkdir()
    np.savez(folder / "naco_predictions.npz", observed_mean=np.ones((8, 8)),
        pretrained_pred_im=np.ones((8, 8)), joint_pinn_pred_im=np.ones((8, 8)))
    (folder / "naco_provenance.json").write_text(json.dumps(dict(models={"pretrained": {}, "joint_pinn": {}})))
    result = report.aggregate(tmp_path)
    assert result["complete"] and result["timing"][0]["batch_mean_ms"] == pytest.approx(3)
    assert not any("mistral" in problem.lower() for problem in result["problems"])


def test_figure_generation_uses_saved_samples_and_common_units(tmp_path):
    pytest.importorskip("matplotlib")
    _campaign(tmp_path)
    result = report.aggregate(tmp_path)
    assert result["complete"]
    figures = list((tmp_path / "referee_report/figures").glob("*.pdf"))
    assert len(figures) == 8
    assert all(path.stat().st_size > 1000 for path in figures)

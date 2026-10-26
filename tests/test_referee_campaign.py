"""Exercise campaign scheduling without a Slurm installation or TensorFlow."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from workflow.archive_run import archive_run
from workflow.referee_campaign import prepare, submit


@pytest.fixture
def prepared(tmp_path):
    out = tmp_path / "paper_run"
    dataset = tmp_path / "dataset"
    original = tmp_path / "original_data"
    store = tmp_path / "store"
    for split in ("train", "val"):
        (original / split).mkdir(parents=True)
    store.mkdir()
    observation = tmp_path / "observation.npy"
    np.save(observation, np.ones((128, 128)))
    campaign = prepare(out, dataset, store, reuse_dataset=original,
                       dataset_job="1580098", naco_data=observation)
    return out, campaign


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    """Record accepted submissions and support an interrupted submit/resume."""
    directory = tmp_path / "bin"
    directory.mkdir()
    ledger = tmp_path / "submitted.json"
    script = directory / "sbatch"
    script.write_text('''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if '--test-only' in sys.argv:
    sys.exit(0)
ledger=Path(os.environ['V1H_FAKE_LEDGER'])
entries=json.loads(ledger.read_text()) if ledger.exists() else []
fail_after=os.environ.get('V1H_FAKE_FAIL_AFTER')
if fail_after and len(entries)>=int(fail_after):
    sys.exit(7)
identifier=str(1001+len(entries))
entries.append(dict(job_id=identifier,script=sys.argv[-1]))
ledger.write_text(json.dumps(entries))
print(identifier+';test_cluster')
''')
    script.chmod(0o755)
    monkeypatch.setenv("PATH", str(directory) + ":" + __import__("os").environ["PATH"])
    monkeypatch.setenv("V1H_FAKE_LEDGER", str(ledger))
    return ledger


def _script(out, name, mode=None):
    if mode:
        candidate = out / "logs" / mode / (name + ".slurm")
        if candidate.exists():
            return candidate
    return out / "logs" / (name + ".slurm")


def _dependency(script):
    lines = script.read_text().splitlines()
    directive = next(line for line in lines if line.startswith("#SBATCH --dependency="))
    policy, *identifiers = directive.split("=", 1)[1].split(":")
    return policy, set(identifiers)


def test_dry_run_has_complete_dag_and_archive_waits_for_every_writer(prepared, fake_slurm):
    out, campaign = prepared
    jobs = submit(out, dry_run=True)
    assert not fake_slurm.exists()
    assert not (out / "campaign_submissions.jsonl").exists()
    assert jobs["dataset_source"] == "1580098"
    assert _dependency(_script(out, "dataset_check", "dry_run")) == ("afterok", {"1580098"})
    for head in ("image", "noise", "psf_mean"):
        assert _dependency(_script(out, head, "dry_run")) == ("afterok", {jobs["dataset_check"]})
    ready = {jobs["image"], jobs["noise"], jobs["psf_uncertainty"]}
    for arm in ("joint", "lambda_0", "lambda_0p1", "lambda_1"):
        assert _dependency(_script(out, arm, "dry_run")) == ("afterok", ready)
    archive = _script(out, "archive", "dry_run")
    assert _dependency(archive) == ("afterany", {value for name, value in jobs.items() if name != "archive"})
    contents = archive.read_text()
    assert "#SBATCH --partition=archive" in contents
    assert "#SBATCH --account=nab@cpu" in contents
    assert "#SBATCH --qos=qos_cpu-t3" in contents
    assert "--run-dir" in contents and str(out) in contents and campaign["store"] in contents
    for label in campaign["expected_evaluations"]:
        assert "eval_" + label in jobs
    assert _dependency(_script(out, "report", "dry_run"))[1] >= {
        jobs["eval_" + label] for label in campaign["expected_evaluations"]
    } | {jobs["naco"], jobs["timing"], jobs["training_plots"]}
    for name in jobs:
        if name != "dataset_source":
            subprocess.run(["bash", "-n", str(_script(out, name, "dry_run"))], check=True)


def test_ablation_configs_share_pretraining_data_order_and_budget(prepared):
    out, campaign = prepared
    main = runpy.run_path(str(out / "configs/main.py"))
    for arm in campaign["ablations"]:
        config = runpy.run_path(arm["config"])
        assert config["OUTPUT_BASE_DIR"] == main["OUTPUT_BASE_DIR"]
        for section in ("IMAGE_HEAD_CONFIG", "NOISE_HEAD_CONFIG", "PSF_HEAD_CONFIG", "PSF_UNC_CONFIG", "DATASET_LOAD_CONFIG"):
            assert config[section] == main[section]
        joint = config["JOINT_PINN_CONFIG"]
        assert joint["pinn_weight"] == arm["weight"]
        assert joint["n_epochs"] == 1
        assert joint["seed"] == main["JOINT_PINN_CONFIG"]["seed"] == 1234
        assert config["DATASET_LOAD_CONFIG"]["seed"] == 1234
        assert joint["train_image_head"] is True
        assert not any(joint[name] for name in ("train_noise_head", "train_psf_mean_head", "train_psf_unc_head"))
        assert joint["head_model_label"] == "best_model"
        assert joint["lr_0"] == main["JOINT_PINN_CONFIG"]["lr_0"]
        assert joint["n_steps_per_epoch"] == main["JOINT_PINN_CONFIG"]["n_steps_per_epoch"]
    assert main["DATASET_GEN_CONFIG"]["n_batches"] * main["DATASET_GEN_CONFIG"]["n_ex_per_batch"] == 8388608
    assert main["DATASET_GEN_CONFIG"]["val_n_batches"] * main["DATASET_GEN_CONFIG"]["n_ex_per_batch"] == 524288
    assert main["TEST_ON_GALSIM_CONFIG"]["first_batch_only"] is False


def test_resume_submits_only_missing_jobs_and_repeating_is_idempotent(prepared, fake_slurm, monkeypatch):
    out, _ = prepared
    monkeypatch.setenv("V1H_FAKE_FAIL_AFTER", "3")
    with pytest.raises(subprocess.CalledProcessError):
        submit(out)
    first = [json.loads(line) for line in (out / "campaign_submissions.jsonl").read_text().splitlines()]
    assert len(first) == 3
    monkeypatch.delenv("V1H_FAKE_FAIL_AFTER")
    jobs = submit(out)
    entries = json.loads(fake_slurm.read_text())
    assert len(entries) == len(jobs) - 1  # The dataset source is an external job.
    assert len({entry["script"] for entry in entries}) == len(entries)
    for entry in first:
        assert jobs[entry["name"]] == entry["job_id"]
    before = fake_slurm.read_bytes()
    journal_before = (out / "campaign_submissions.jsonl").read_bytes()
    assert submit(out) == jobs
    assert fake_slurm.read_bytes() == before
    assert (out / "campaign_submissions.jsonl").read_bytes() == journal_before


def test_preview_after_submission_preserves_production_scripts(prepared, fake_slurm):
    out, _ = prepared
    jobs = submit(out)
    originals = {name: _script(out, name).read_bytes() for name in jobs if name != "dataset_source"}
    submitted = fake_slurm.read_bytes()
    submit(out, dry_run=True)
    submit(out, test_only=True)
    assert fake_slurm.read_bytes() == submitted
    for name, contents in originals.items():
        assert _script(out, name).read_bytes() == contents


def test_archive_omits_raw_data_and_cache_but_keeps_models_and_provenance(prepared):
    out, campaign = prepared
    dataset = Path(campaign["dataset_dir"])
    (dataset / "psf_uncertainty_stage2_cache" / "raw.cache").write_bytes(b"large cached training features")
    original = Path(campaign["reuse_dataset"])
    (original / "train/raw.tfrecord").write_bytes(b"raw training records")
    checkpoint = out / "image_only/checkpoints/model.keras"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"trained weights")
    receipt = archive_run(out, Path(campaign["store"]))
    with tarfile.open(receipt["archive"]) as archive:
        names = archive.getnames()
        assert out.name + "/dataset" not in names
        assert not any("raw.tfrecord" in name or "raw.cache" in name for name in names)
        assert archive.getmember(out.name + "/psf_uncertainty_stage2/cache").issym()
        assert archive.extractfile(out.name + "/image_only/checkpoints/model.keras").read() == b"trained weights"
        assert out.name + "/provenance/source.json" in names


def test_prepare_rejects_dataset_nested_under_run(tmp_path):
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="separate"):
        prepare(out, out / "bulk_data", tmp_path / "store")


def test_modified_arm_config_is_rejected_before_submission(prepared, fake_slurm):
    out, campaign = prepared
    config = Path(campaign["ablations"][0]["config"])
    config.write_text(config.read_text() + "\nJOINT_PINN_CONFIG['seed'] = 99\n")
    with pytest.raises(ValueError, match="config|Config"):
        submit(out)
    assert not fake_slurm.exists()


def test_dataset_inventory_rejects_missing_and_truncated_shards(prepared, monkeypatch):
    from workflow.referee_evaluate import verify_dataset

    out, campaign = prepared
    config = out / "configs/inventory_test.py"
    config.write_text((out / "configs/main.py").read_text() + "\nDATASET_GEN_CONFIG = dict(DATASET_GEN_CONFIG, n_batches=3, val_n_batches=2, n_ex_per_batch=4)\n")
    # Supply a record-reader stand-in so inventory logic runs without a TF
    # installation; the numerical integration suite exercises real TFRecords.
    records = lambda path: range(int(Path(path).read_text()))
    monkeypatch.setitem(sys.modules, "tensorflow", SimpleNamespace(data=SimpleNamespace(TFRecordDataset=records)))
    original = Path(campaign["reuse_dataset"])
    for split, count in (("train", 3), ("val", 2)):
        for index in range(count):
            (original / split / f"batch_{index:04d}.tfrecord").write_text("4")
    result = verify_dataset(config, out / "inventory.json")
    assert result["splits"]["train"]["examples"] == 12
    assert result["splits"]["val"]["examples"] == 8
    last = original / "train/batch_0002.tfrecord"
    last.write_text("3")
    with pytest.raises(ValueError, match="Incomplete shard"):
        verify_dataset(config, out / "inventory.json")
    last.unlink()
    with pytest.raises(ValueError, match="expected 3 shards"):
        verify_dataset(config, out / "inventory.json")

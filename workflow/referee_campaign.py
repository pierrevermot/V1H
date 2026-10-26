"""Prepare and submit the paper's training, ablations and referee evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(out: Path, dataset_dir: Path, store: Path, reuse_dataset: Path | None = None,
            dataset_job: str | None = None, naco_data: Path | None = None,
            mistral: Path | None = None, reference_psf: Path | None = None) -> dict:
    out, dataset_dir, store = (p.expanduser().resolve() for p in (out, dataset_dir, store))
    if dataset_job is not None and (not dataset_job.isdigit() or reuse_dataset is None):
        raise ValueError("A dataset producer job requires a reused dataset and numeric job ID")
    if store == out or out in store.parents or dataset_dir == out or out in dataset_dir.parents:
        raise ValueError("Dataset and STORE paths must be separate from the run directory")
    out.mkdir(parents=True, exist_ok=False)
    dataset_dir.mkdir(parents=True, exist_ok=False)
    (out / "dataset").symlink_to(dataset_dir, target_is_directory=True)
    if reuse_dataset is not None:
        reuse_dataset = reuse_dataset.expanduser().resolve()
        for split in ("train", "val"):
            (dataset_dir / split).symlink_to(reuse_dataset / split, target_is_directory=True)
    cache = dataset_dir / "psf_uncertainty_stage2_cache"
    cache.mkdir()
    (out / "psf_uncertainty_stage2").mkdir()
    (out / "psf_uncertainty_stage2/cache").symlink_to("../dataset/psf_uncertainty_stage2_cache", target_is_directory=True)
    for name in ("logs", "configs", "provenance", "inputs", "referee_evaluation", "benchmark_timing/results"):
        (out / name).mkdir(parents=True, exist_ok=True)
    inputs = {}
    for name, path in (("naco", naco_data), ("mistral", mistral), ("reference_psf", reference_psf)):
        if path is not None:
            target = out / "inputs" / (name + path.suffix)
            shutil.copyfile(path, target)
            inputs[name] = dict(path=str(target), original=str(path),
                               sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    base = (ROOT / "configs/config_naco_M.py").read_text()
    overrides = f'''
# Reproducible referee campaign: retain the paper's model and training settings.
OUTPUT_BASE_DIR = {str(out)!r}
DATASET_GEN_CONFIG = dict(DATASET_GEN_CONFIG, output_dir={str(out / 'dataset')!r},
                          n_batches=8192, val_n_batches=512, n_ex_per_batch=1024)
DATASET_LOAD_CONFIG = dict(DATASET_LOAD_CONFIG, data_dir={str(out / 'dataset')!r}, seed=1234, num_parallel_reads=1)
GALSIM_TEST_CONFIG = dict(GALSIM_TEST_CONFIG, output_dir={str(out / 'dataset/galsim_test')!r})
TEST_ON_GALSIM_CONFIG = dict(TEST_ON_GALSIM_CONFIG, first_batch_only=False,
    eval_batch_size=32, output_dir={str(out / 'joint_pinn_fourhead/test_on_galsim')!r})
JOINT_PINN_CONFIG = dict(JOINT_PINN_CONFIG, seed=1234)
TRAINING_SEED = 1234
'''
    (out / "configs/main.py").write_text(base + overrides)
    arms = []
    for weight, label in ((0.0, "lambda_0"), (0.1, "lambda_0p1"), (1.0, "lambda_1")):
        path = out / "configs" / (label + ".py")
        path.write_text(base + overrides + f"\nJOINT_PINN_CONFIG = dict(JOINT_PINN_CONFIG, pinn_weight={weight}, n_epochs=1, run_name={label!r})\n")
        arms.append(dict(label=label, weight=weight, config=str(path)))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with (out / "provenance/source.tar.gz").open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar.gz", "HEAD"], cwd=ROOT, stdout=handle, check=True)
    write_json(out / "provenance/source.json", dict(checkout=str(ROOT), commit=revision,
        archive_sha256=hashlib.sha256((out / "provenance/source.tar.gz").read_bytes()).hexdigest()))
    campaign = dict(run_dir=str(out), dataset_dir=str(dataset_dir), store=str(store),
        reuse_dataset=str(reuse_dataset) if reuse_dataset else None, dataset_job=dataset_job,
        source_dir=str(ROOT), source_commit=revision, pixel_scale_arcsec=0.027,
        validation_examples=1024, eval_batch_size=32, rl_iterations=[10, 30, 100],
        expected_dataset_examples={"val": 1024, "galsim": 1210},
        expected_timing_cases=[f"{label}_bs{batch}.json" for label in ("joint", "pretrained", "finetuned") for batch in (1, 64)],
        ablations=arms, inputs=inputs,
        expected_evaluations=["pretrained", "joint", "lambda_0", "lambda_0p1", "lambda_1", "rl_10", "rl_30", "rl_100", "wiener"],
        created_utc=datetime.now(timezone.utc).isoformat(),
        uncertainty_assumptions="pixelwise simulation-conditioned predictive variances; independent component/pixel propagation, including the product-variance term; no epistemic or real-data calibration guarantee",
        missing_inputs=[name for name in ("naco", "mistral", "reference_psf") if name not in inputs])
    campaign["config_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (out / "configs").glob("*.py")}
    sources = [p for folder in ("configs", "instruments", "neural_networks", "noises", "phases", "psfs", "skies", "testing", "utils", "workflow")
               for p in (ROOT / folder).rglob("*") if p.is_file() and p.suffix in (".py", ".sh")]
    campaign["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    write_json(out / "campaign.json", campaign)
    print(json.dumps(campaign, indent=2))
    return campaign


def submit(out: Path, dry_run: bool = False, test_only: bool = False) -> dict:
    out = out.resolve()
    campaign = json.loads((out / "campaign.json").read_text())
    source = Path(campaign["source_dir"])
    if source != ROOT:
        raise ValueError("Submit using the checkout recorded by prepare")
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if current != campaign["source_commit"]:
        raise ValueError("Checkout revision changed since campaign preparation")
    for name, digest in campaign["source_sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Source changed since preparation: {name}")
    for path, digest in campaign["config_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Configuration changed since preparation: {path}")
    for item in campaign["inputs"].values():
        if hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Input changed since preparation: {item['path']}")
    main_config = out / "configs/main.py"
    journal = out / "campaign_submissions.jsonl"
    previous = {}
    if journal.exists():
        previous = {item["name"]: item for item in map(json.loads, journal.read_text().splitlines())}
    jobs = {}

    def command(script: str, *args) -> str:
        return shlex.join(["python3", "-u", str(source / "workflow" / script), *map(str, args)])

    def schedule(name: str, body: str, dependencies=(), *, gpu=False, cpus=None,
                 walltime=None, array=None, archive=False, afterany=False) -> str:
        dependencies = list(dict.fromkeys(str(x) for x in dependencies if x))
        cpus = cpus or (24 if gpu else 8)
        partition = "archive" if archive else ("gpu_p6" if gpu else "cpu_p1")
        walltime = walltime or ("100:00:00" if gpu else "04:00:00")
        directives = ["#!/usr/bin/env bash", f"#SBATCH --job-name=v1h_{name}",
            f"#SBATCH --output={out}/logs/{name}_%A_%a.out", f"#SBATCH --error={out}/logs/{name}_%A_%a.err",
            f"#SBATCH --account=nab@{'h100' if gpu else 'cpu'}", f"#SBATCH --partition={partition}",
            f"#SBATCH --cpus-per-task={cpus}", f"#SBATCH --time={walltime}", "#SBATCH --hint=nomultithread"]
        if gpu:
            directives += ["#SBATCH --qos=qos_gpu_h100-t4", "#SBATCH --gres=gpu:1", "#SBATCH -C h100"]
        elif archive:
            directives += ["#SBATCH --qos=qos_cpu-t3", "#SBATCH --nodes=1", "#SBATCH --ntasks=1"]
        if array:
            directives += [f"#SBATCH --array={array}"]
        if dependencies:
            directives += [f"#SBATCH --dependency={'afterany' if afterany else 'afterok'}:" + ":".join(dependencies)]
            if not afterany:
                directives += ["#SBATCH --kill-on-invalid-dep=yes"]
        setup = ["set -eo pipefail", "export PYTHONDONTWRITEBYTECODE=1", "export MPLBACKEND=Agg",
            'export TF_CPP_MIN_LOG_LEVEL=2', f"export PYTHONPATH={shlex.quote(str(source))}",
            f"cd {shlex.quote(str(source))}"]
        if not archive:
            setup += ["module purge"]
            if gpu:
                setup += ["module load arch/h100", "module load tensorflow-gpu/py3/2.17.0"]
            else:
                setup += ["module load tensorflow-gpu/py3/2.16.1"]
        contents = "\n".join(directives + [""] + setup + [body, ""])
        digest = hashlib.sha256(contents.encode()).hexdigest()
        script_dir = out / "logs"
        if dry_run or test_only:
            script_dir = script_dir / ("test_only" if test_only else "dry_run")
            script_dir.mkdir(exist_ok=True)
        path = script_dir / (name + ".slurm")
        if name in previous and not dry_run and not test_only:
            if previous[name]["script_sha256"] != digest:
                raise ValueError(f"Refusing to change an already submitted job: {name}")
            job = previous[name]["job_id"]
        else:
            path.write_text(contents)
            subprocess.run(["bash", "-n", str(path)], check=True)
            if test_only:
                check = script_dir / (name + ".test.slurm")
                check.write_text("\n".join(line for line in contents.splitlines() if not line.startswith("#SBATCH --dependency=")) + "\n")
                subprocess.run(["sbatch", "--test-only", str(check)], check=True)
                check.unlink()
                job = "TEST_" + name
            elif dry_run:
                job = "DRY_" + name
            else:
                job = subprocess.check_output(["sbatch", "--parsable", str(path)], text=True).strip().split(";")[0]
                if not job.isdigit():
                    raise ValueError(f"Unexpected sbatch response: {job}")
                with journal.open("a") as handle:
                    handle.write(json.dumps(dict(name=name, job_id=job, script_sha256=digest)) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
        jobs[name] = job
        print(f"{name}: {job}", flush=True)
        if not dry_run and not test_only:
            write_json(out / "campaign_jobs.json", jobs)
        return job

    if campaign["reuse_dataset"]:
        producer = campaign["dataset_job"]
        if not producer:
            raise ValueError("Reusing data requires its producer job ID to prevent reading incomplete shards")
        jobs["dataset_source"] = producer
    else:
        body = '\n'.join([
            'OFFSET=$((SLURM_ARRAY_TASK_ID * 820))',
            'COUNT=$((8192 - OFFSET)); if (( COUNT > 820 )); then COUNT=820; fi',
            command("create_dataset.py", "--config", main_config, "--split", "train", "--n-workers", 40,
                    "--parallel-mode", "joblib") + ' --batch-offset "$OFFSET" --n-batches "$COUNT"',
            'if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then',
            command("create_dataset.py", "--config", main_config, "--split", "val", "--val-n-batches", 512,
                    "--n-workers", 40, "--parallel-mode", "joblib"), 'fi'])
        producer = schedule("dataset", body, cpus=40, walltime="10:00:00", array="0-9")
    checked = schedule("dataset_check", command("referee_evaluate.py", "verify-dataset", "--config", main_config,
                       "--output", out / "dataset_inventory.json"), [producer], cpus=4, walltime="01:00:00")
    heads = {}
    for label, target in (("image", "im"), ("noise", "noise"), ("psf_mean", "psf")):
        heads[label] = schedule(label, command("independent_training.py", "--config", main_config, "--head-target", target), [checked], gpu=True)
    uncertainty = schedule("psf_uncertainty", command("psf_uncertainty_stage2_training.py", "--config", main_config), [heads["psf_mean"]], gpu=True)
    pretrained_ready = [heads["image"], heads["noise"], uncertainty]
    joint = schedule("joint", command("joint_pinn_fourhead_training.py", "--config", main_config), pretrained_ready, gpu=True)
    plot = schedule("training_plots", command("plot_results.py", "--run-dir", out / "joint_pinn_fourhead",
                    "--skip-data-inference", "--pixel-scale-arcsec", campaign["pixel_scale_arcsec"]), [joint], cpus=8, walltime="02:00:00")
    arms = {}
    for arm in campaign["ablations"]:
        arms[arm["label"]] = schedule(arm["label"], command("joint_pinn_fourhead_training.py", "--config", arm["config"]), pretrained_ready, gpu=True)
    galsim = schedule("galsim", command("galsim_ao_grid.py", "--config", main_config), cpus=4, walltime="02:00:00")
    evaluations = []

    def evaluate(label, algorithm, deps, run_name="joint_pinn_fourhead", iterations=None):
        args = ["evaluate", "--config", main_config, "--output", out / "referee_evaluation" / label,
                "--label", label, "--algorithm", algorithm, "--run-dir", out / run_name,
                "--validation-examples", campaign["validation_examples"], "--batch-size", campaign["eval_batch_size"]]
        if iterations is not None:
            args += ["--rl-iterations", iterations]
        evaluations.append(schedule("eval_" + label, command("referee_evaluate.py", *args),
                           [galsim, checked, *deps], gpu=algorithm in ("joint_pinn", "pretrained"), walltime="04:00:00"))

    evaluate("pretrained", "pretrained", pretrained_ready)
    evaluate("joint", "joint_pinn", [joint])
    for label, identifier in arms.items():
        evaluate(label, "joint_pinn", [identifier], run_name=label)
    for iterations in campaign["rl_iterations"]:
        evaluate(f"rl_{iterations}", "richardson_lucy", [], iterations=iterations)
    evaluate("wiener", "wiener", [])
    timing_body = []
    for batch in (1, 64):
        for mode, object_source in (("joint", None), ("independent", "pretrained"), ("independent", "finetuned")):
            label = object_source or "joint"
            args = ["--config", main_config, "--run-dir", out / "joint_pinn_fourhead", "--mode", mode,
                    "--batch-size", batch, "--n-warmup", 10, "--n-repeats", 50, "--device", "gpu", "--end-to-end"]
            if object_source:
                args += ["--head-target", "im", "--object-source", object_source]
            target = out / "benchmark_timing/results" / f"{label}_bs{batch}.json"
            timing_body.append(command("benchmark_timing.py", *args) + " > " + shlex.quote(str(target)))
    timing = schedule("timing", "\n".join(timing_body), [joint], gpu=True, walltime="02:00:00")
    naco = None
    if "naco" in campaign["inputs"]:
        args = ["--config", main_config, "--out", out / "naco_comparison", "--data-path", campaign["inputs"]["naco"]["path"],
                "--pixel-scale-arcsec", campaign["pixel_scale_arcsec"]]
        for key, flag in (("mistral", "--mistral"), ("reference_psf", "--reference-psf")):
            if key in campaign["inputs"]:
                args += [flag, campaign["inputs"][key]["path"]]
        naco = schedule("naco", command("referee_naco.py", *args), [joint], gpu=True, walltime="02:00:00")
    report_deps = [*evaluations, timing, plot] + ([naco] if naco else [])
    report = schedule("report", command("referee_report.py", "--campaign-dir", out), report_deps, cpus=8, walltime="02:00:00", afterany=True)
    # afterany on every writer preserves failed runs too; impossible afterok jobs
    # are cancelled by Slurm above rather than blocking this archive forever.
    writers = list(jobs.values())
    archive_body = command("archive_run.py", "--run-dir", out, "--store", campaign["store"])
    schedule("archive", archive_body, writers, archive=True, afterany=True, cpus=2, walltime="20:00:00")
    if dry_run or test_only:
        write_json(out / ("campaign_jobs_test.json" if test_only else "campaign_jobs_dry_run.json"), jobs)
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("out", "dataset-dir", "store"):
        prep.add_argument("--" + name, type=Path, required=True)
    for name in ("reuse-dataset", "naco-data", "mistral", "reference-psf"):
        prep.add_argument("--" + name, type=Path)
    prep.add_argument("--dataset-job")
    launch = sub.add_parser("submit")
    launch.add_argument("--out", type=Path, required=True)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument("--test-only", action="store_true")
    args = vars(parser.parse_args())
    action = args.pop("command")
    (prepare if action == "prepare" else submit)(**args)


if __name__ == "__main__":
    main()

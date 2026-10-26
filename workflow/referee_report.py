"""Aggregate the referee campaign without importing TensorFlow or loading models.

Each evaluation directory supplies a manifest, per-image metrics, and a small
sample of full-size predictions. RL stopping is selected using validation only.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np


GAUSSIAN_COVERAGE = {"1sigma": 0.682689492, "2sigma": 0.954499736}
OBJECT_METRICS = ("r2_im", "psnr_im", "ssim_im", "r2_pinn")
METRIC_LABELS = {"r2_im": "Object vnMSE", "psnr_im": "Object PSNR [dB]",
                 "ssim_im": "Object SSIM", "r2_pinn": "Observation vnMSE"}


def _json_scalar(value):
    if value is None or np.isfinite(value):
        return value
    return "Infinity" if value > 0 else "-Infinity" if value < 0 else None


def distribution(values) -> dict:
    """Describe image-level values, retaining infinite PSNR for exact matches."""
    values = np.asarray(values, dtype=float).reshape(-1)
    valid = values[~np.isnan(values)]
    if not valid.size:
        return dict(n_examples=int(values.size), n_valid=0, n_finite=0,
                    median=None, q25=None, q75=None)
    # Nearest quantiles also remain defined for arrays containing +infinity.
    quantiles = np.quantile(valid, [0.25, 0.5, 0.75], method="nearest")
    return dict(n_examples=int(values.size), n_valid=int(valid.size),
                n_finite=int(np.isfinite(values).sum()),
                median=_json_scalar(float(np.median(valid))),
                q25=_json_scalar(float(quantiles[0])), q75=_json_scalar(float(quantiles[2])))


def _write_csv(path: Path, rows: list[dict], fields=None) -> None:
    fields = fields or sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def load_evaluations(root: Path) -> tuple[dict, list[str]]:
    evaluations, problems = {}, []
    for path in sorted((root / "referee_evaluation").glob("*/manifest.json")):
        try:
            manifest = json.loads(path.read_text())
            label = str(manifest["label"])
            if label in evaluations:
                raise ValueError(f"Duplicate evaluation label: {label}")
            record = dict(manifest=manifest, path=path.parent, datasets={})
            for dataset, info in manifest["datasets"].items():
                try:
                    metrics = _read_npz(path.parent / info["metrics"])
                    count = int(info["n_examples"])
                    if not metrics or any(np.asarray(a).shape != (count,) for a in metrics.values()):
                        raise ValueError("Metric arrays do not match the manifest example count")
                    samples_path = path.parent / info["samples"] if info.get("samples") else None
                    samples = _read_npz(samples_path) if samples_path and samples_path.is_file() else None
                    record["datasets"][dataset] = dict(info=info, metrics=metrics, samples=samples)
                except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
                    problems.append(f"{label}/{dataset}: {exc}")
            evaluations[label] = record
        except (OSError, ValueError, KeyError, EOFError, BadZipFile) as exc:
            problems.append(f"{path}: {exc}")
    return evaluations, problems


def select_rl(evaluations: dict) -> str | None:
    """Choose one global iteration count using validation median object vnMSE."""
    candidates = []
    for label, record in evaluations.items():
        if record["manifest"]["algorithm"] != "richardson_lucy":
            continue
        metric = record["datasets"].get("val", {}).get("metrics", {}).get("r2_im")
        if metric is None or not np.all(np.isfinite(metric)):
            continue
        candidates.append((float(np.median(metric)), int(record["manifest"].get("rl_iterations", 0)), label))
    return min(candidates)[2] if candidates else None


def _main_labels(evaluations: dict) -> tuple[str | None, str | None]:
    pretrained = next((name for name, r in evaluations.items() if r["manifest"]["algorithm"] == "pretrained"), None)
    candidates = [name for name, r in evaluations.items() if r["manifest"]["algorithm"] == "joint_pinn"]
    joint = next((name for name in ("joint", "joint_pinn") if name in candidates), None)
    if joint is None:
        joint = next((name for name in candidates if "lambda" not in name and "weight" not in name), None)
    return pretrained, joint


def _noise_levels(root: Path) -> np.ndarray | None:
    path = root / "dataset/galsim_test/generation_parameter_log.json"
    if not path.is_file():
        return None
    log = json.loads(path.read_text())
    entries = log.get("per_example", [])
    if not entries:
        return None
    indices = [int(e.get("example_index", index)) for index, e in enumerate(entries)]
    if indices != list(range(len(entries))):
        return None
    return np.asarray([entry["noise_sigma"] for entry in entries], dtype=float)


def _sample_pair(reference: dict, candidate: dict) -> tuple[dict, dict] | None:
    """Check common saved sample IDs and truth rather than pairing by position."""
    if reference is None or candidate is None:
        return None
    r_indices = np.asarray(reference["indices"], dtype=int)
    c_indices = np.asarray(candidate["indices"], dtype=int)
    ids, ri, ci = np.intersect1d(r_indices, c_indices, return_indices=True)
    if not len(ids):
        return None
    r = {key: np.asarray(value)[ri] for key, value in reference.items()}
    c = {key: np.asarray(value)[ci] for key, value in candidate.items()}
    if not np.array_equal(r["y_true"], c["y_true"]):
        raise ValueError("Paired samples have different ground truths")
    return r, c


def _auxiliary_products(root: Path, campaign: dict, problems: list[str]) -> list[dict]:
    """Check the campaign's explicitly requested timing and real-data products."""
    rows = []
    for case in campaign.get("expected_timing_cases", []):
        name = case.get("name", case.get("label")) if isinstance(case, dict) else case
        path = root / "benchmark_timing/results" / (str(name).removesuffix(".json") + ".json")
        try:
            result = json.loads(path.read_text())
            if result.get("weights") != "checkpoint" or not result.get("checkpoint"):
                raise ValueError("Timing must identify a trained checkpoint")
            for key in ("timing", "timing_with_transfers"):
                if key not in result:
                    continue
                timed = result[key]
                rows.append(dict(case=str(name), scope=timed["scope"], batch_size=result["batch_size"],
                    n_params=result["n_params"], checkpoint=result["checkpoint"]["path"],
                    batch_mean_ms=1000 * timed["latency"]["batch_mean_s"],
                    amortized_ms_per_image=timed["throughput"]["per_sample_mean_ms"],
                    samples_per_second=timed["throughput"]["samples_per_second_mean"],
                    includes_preprocessing=timed["includes_preprocessing"]))
            if "timing" not in result:
                raise ValueError("Missing network-only timing")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            problems.append(f"Missing or invalid timing product {name}: {exc}")
    if campaign.get("inputs", {}).get("naco"):
        try:
            arrays = _read_npz(root / "naco_comparison/naco_predictions.npz")
            provenance = json.loads((root / "naco_comparison/naco_provenance.json").read_text())
            observed = arrays["observed_mean"]
            if any(arrays[key].shape != observed.shape for key in ("pretrained_pred_im", "joint_pinn_pred_im")):
                raise ValueError("NaCo comparison images have mismatched shapes")
            if any(not np.isfinite(arrays[key]).all() for key in ("observed_mean", "pretrained_pred_im", "joint_pinn_pred_im")):
                raise ValueError("NaCo comparison contains nonfinite values")
            if not {"pretrained", "joint_pinn"}.issubset(provenance["models"]):
                raise ValueError("NaCo provenance must identify both compared models")
        except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile) as exc:
            problems.append(f"Missing or invalid NaCo comparison: {exc}")
    return rows


def aggregate(root: Path, *, allow_partial: bool = False) -> dict:
    root = root.expanduser().resolve()
    output = root / "referee_report"
    output.mkdir(exist_ok=True)
    config_path = root / "campaign.json"
    campaign = json.loads(config_path.read_text()) if config_path.is_file() else {}
    evaluations, problems = load_evaluations(root)
    pretrained, joint = _main_labels(evaluations)
    selected_rl = select_rl(evaluations)
    for role, label in (("pretrained", pretrained), ("joint", joint), ("validation-selected RL", selected_rl)):
        if label is None:
            problems.append(f"Missing required evaluation: {role}")
    expected = campaign.get("expected_evaluations", [])
    for name in expected:
        name = name.get("label") if isinstance(name, dict) else name
        if name not in evaluations:
            problems.append(f"Missing expected evaluation: {name}")
    required_labels = set(filter(None, (pretrained, joint, selected_rl)))
    required_labels.update(name.get("label") if isinstance(name, dict) else name for name in expected)
    for name in sorted(required_labels):
        for dataset in ("val", "galsim"):
            if dataset not in evaluations.get(name, {}).get("datasets", {}):
                problems.append(f"Missing evaluation data: {name}/{dataset}")
                continue
            data = evaluations[name]["datasets"][dataset]
            expected_count = campaign.get("expected_dataset_examples", {}).get(dataset)
            if expected_count is not None and data["info"]["n_examples"] != int(expected_count):
                problems.append(f"Wrong example count: {name}/{dataset}: {data['info']['n_examples']} != {expected_count}")
            for metric in ("r2_im", "psnr_im", "ssim_im"):
                if metric not in data["metrics"]:
                    problems.append(f"Missing metric: {name}/{dataset}/{metric}")
                elif np.isnan(data["metrics"][metric]).any():
                    problems.append(f"Undefined metric values: {name}/{dataset}/{metric}")
            if evaluations[name]["manifest"]["algorithm"] in ("pretrained", "joint_pinn"):
                for component in ("im", "psf", "noise"):
                    for interval in ("1sigma", "2sigma"):
                        key = f"coverage_{interval}_{component}"
                        if key not in data["metrics"]:
                            problems.append(f"Missing coverage metric: {name}/{dataset}/{key}")

    timings = _auxiliary_products(root, campaign, problems)

    summary, coverage, paired, sensitivity, noise_coverage = [], [], [], [], []
    noise_levels = _noise_levels(root)
    for label, record in evaluations.items():
        for dataset, data in record["datasets"].items():
            algorithm = record["manifest"]["algorithm"]
            for metric, values in data["metrics"].items():
                row = dict(dataset=dataset, label=label, algorithm=algorithm, metric=metric, **distribution(values))
                summary.append(row)
                if algorithm == "richardson_lucy" and metric in OBJECT_METRICS:
                    sensitivity.append(dict(row, iterations=record["manifest"].get("rl_iterations"), selected=label == selected_rl))
                match = re.fullmatch(r"coverage_([12]sigma)_(.+)", metric)
                if not match:
                    continue
                nominal = GAUSSIAN_COVERAGE[match.group(1)]
                coverage.append(dict(row, nominal=nominal, mean=float(np.mean(values)),
                                     mean_minus_nominal=float(np.mean(values) - nominal)))
                if dataset == "galsim" and noise_levels is not None and len(noise_levels) == len(values):
                    for level in np.unique(noise_levels):
                        selected = values[noise_levels == level]
                        noise_coverage.append(dict(dataset=dataset, label=label, metric=metric,
                            noise_sigma=float(level), nominal=nominal, mean=float(np.mean(selected)),
                            **distribution(selected)))

    if pretrained:
        base = evaluations[pretrained]
        for label, record in evaluations.items():
            if record["manifest"]["algorithm"] != "joint_pinn":
                continue
            for dataset in ("val", "galsim"):
                reference, candidate = base["datasets"].get(dataset), record["datasets"].get(dataset)
                if reference is None or candidate is None:
                    continue
                if reference["info"]["n_examples"] != candidate["info"]["n_examples"]:
                    problems.append(f"Cannot pair different example counts: {pretrained}/{label}/{dataset}")
                    continue
                try:
                    _sample_pair(reference["samples"], candidate["samples"])
                except ValueError as exc:
                    problems.append(f"Cannot pair {pretrained}/{label}/{dataset}: {exc}")
                    continue
                for metric in OBJECT_METRICS:
                    if metric not in reference["metrics"] or metric not in candidate["metrics"]:
                        continue
                    a, b = reference["metrics"][metric], candidate["metrics"][metric]
                    with np.errstate(invalid="ignore"):
                        delta = b - a
                    # Identical perfect PSNR values have zero paired difference.
                    delta = np.where(np.isinf(a) & (a == b), 0.0, delta)
                    improved = b > a if metric in ("psnr_im", "ssim_im") else b < a
                    paired.append(dict(dataset=dataset, reference=pretrained, label=label, metric=metric,
                        delta_definition="candidate minus pretrained", fraction_improved=float(np.mean(improved)),
                        pairing="identical dataset ordering; saved sample identities checked", **distribution(delta)))

    result = dict(complete=not problems, problems=problems, selected_rl=selected_rl,
        rl_selection="minimum validation median object vnMSE; ties prefer fewer iterations",
        main_labels=dict(pretrained=pretrained, joint=joint),
        coverage_interpretation="Marginal predictive intervals conditional on the simulator; correlated pixels are not independent trials.",
        backend_details={label: record["manifest"].get("backend", {}) for label, record in evaluations.items()},
        classical_comparison_notes=[
            "RL receives the true PSF and clips negative observation pixels to zero before inference; output clipping is a separate setting.",
            "The current Wiener baseline is an oracle: it receives the true PSF and estimates its noise variance from the ground-truth noise map.",
        ],
        source_revision=campaign.get("source_commit", campaign.get("source_revision", campaign.get("git_commit"))),
        timing=timings,
        summary=summary, paired=paired, coverage=coverage, coverage_by_noise=noise_coverage,
        rl_sensitivity=sensitivity)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for filename, rows in (("summary", summary), ("paired", paired), ("coverage", coverage),
                           ("coverage_by_noise", noise_coverage), ("rl_sensitivity", sensitivity), ("timing", timings)):
        _write_csv(output / f"{filename}.csv", rows)
    (output / "selected_rl.json").write_text(json.dumps(dict(label=selected_rl,
        selection_dataset="val", selection_metric="median r2_im"), indent=2) + "\n")
    _make_figures(output, evaluations, pretrained, joint, selected_rl, campaign, problems)
    # Figure pairing failures are material and must also appear in the journal.
    result["complete"] = not problems
    _write_markdown(output, result)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if problems and not allow_partial:
        raise RuntimeError("Referee report is incomplete: " + "; ".join(problems))
    return result


def _write_markdown(output: Path, result: dict) -> None:
    lines = ["# Referee evaluation", "", "Status: " + ("complete" if result["complete"] else "partial"), "",
        "All tables summarize individual images. IQR is reported in the CSV/JSON files.",
        "RL is selected once on validation data; GalSim results do not affect its stopping criterion.", ""]
    if result["problems"]:
        lines += ["Missing or inconsistent products:", ""] + ["- " + p for p in result["problems"]] + [""]
    for dataset in ("val", "galsim"):
        lines += ["## " + dataset, "", "| Method | Examples | Object vnMSE | PSNR (dB) | SSIM | Observation vnMSE |",
                  "|---|---:|---:|---:|---:|---:|"]
        rows = [row for row in result["summary"] if row["dataset"] == dataset]
        for label in sorted({row["label"] for row in rows}):
            lookup = {row["metric"]: row for row in rows if row["label"] == label}
            count = max(row["n_examples"] for row in rows if row["label"] == label)
            values = [lookup.get(metric, {}).get("median") for metric in OBJECT_METRICS]
            text = [f"{value:.5g}" if isinstance(value, (int, float)) else str(value or "—") for value in values]
            marker = " (validation-selected RL)" if label == result["selected_rl"] else ""
            lines.append("| " + " | ".join([label + marker, str(count), *text]) + " |")
        lines.append("")
    lines += ["Coverage targets are 68.2689% and 95.4499% for Gaussian 1σ and 2σ marginal intervals.",
              "Coverage alone does not validate real-data or model-parameter uncertainty.", "",
              *result["classical_comparison_notes"], ""]
    if result["timing"]:
        lines += ["## Inference cost", "",
            "Batch latency and amortized time per image are distinct. Image decoding and scientific preprocessing are excluded.", "",
            "| Case | Scope | Batch latency (ms) | Amortized ms/image | Images/s | Parameters |",
            "|---|---|---:|---:|---:|---:|"]
        for row in result["timing"]:
            lines.append(f"| {row['case']} | {row['scope']} | {row['batch_mean_ms']:.4g} | {row['amortized_ms_per_image']:.4g} | {row['samples_per_second']:.5g} | {row['n_params']} |")
        lines.append("")
    (output / "report.md").write_text("\n".join(lines))


def _make_figures(output, evaluations, pretrained, joint, selected_rl, campaign, problems):
    if not pretrained or not joint:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    scale = campaign.get("pixel_scale_arcsec")
    for dataset in ("val", "galsim"):
        base = evaluations[pretrained]["datasets"].get(dataset)
        fine = evaluations[joint]["datasets"].get(dataset)
        if base is None or fine is None:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, metric in zip(axes, ("psnr_im", "ssim_im")):
            a, b = base["metrics"].get(metric), fine["metrics"].get(metric)
            if a is None or b is None or len(a) != len(b):
                continue
            finite = np.isfinite(a) & np.isfinite(b)
            ax.scatter(a[finite], b[finite], s=12, alpha=0.4)
            if finite.any():
                low, high = min(a[finite].min(), b[finite].min()), max(a[finite].max(), b[finite].max())
                ax.plot([low, high], [low, high], "k--", linewidth=1)
            ax.set_xlabel("Pretrained " + METRIC_LABELS[metric])
            ax.set_ylabel("Fine-tuned " + METRIC_LABELS[metric])
            ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_paired_metrics.pdf")
        plt.close(fig)
        try:
            pair = _sample_pair(base["samples"], fine["samples"])
        except ValueError as exc:
            problems.append(f"Cannot plot {dataset}: {exc}")
            continue
        if pair is None:
            continue
        a, b = pair
        crop = int(evaluations[pretrained]["manifest"].get("crop_border", 0))
        def object_pixels(array):
            array = array[..., 0]
            if crop:
                array = array[:, crop:-crop, crop:-crop]
            return array.reshape(-1)
        truth = object_pixels(a["y_true"])
        index = np.random.default_rng(1234).choice(len(truth), min(len(truth), 20_000), replace=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        for ax, sample, title in zip(axes, (a, b), ("Pretrained object", "Fine-tuned object")):
            estimate = object_pixels(sample["y_pred"])[index]
            target = truth[index]
            finite = np.isfinite(target) & np.isfinite(estimate)
            ax.hexbin(target[finite], estimate[finite], gridsize=50, bins="log", mincnt=1)
            if finite.any():
                low, high = min(target[finite].min(), estimate[finite].min()), max(target[finite].max(), estimate[finite].max())
                ax.plot([low, high], [low, high], "r--", linewidth=1)
            ax.set(xlabel="Ground truth [model intensity units]", ylabel="Prediction [model intensity units]", title=title)
        fig.tight_layout()
        fig.savefig(figures / f"{dataset}_object_truth_hexbin.pdf")
        plt.close(fig)
        rl = evaluations.get(selected_rl, {}).get("datasets", {}).get(dataset, {}).get("samples")
        rl_lookup = {} if rl is None else {int(v): i for i, v in enumerate(rl["indices"])}
        for index in range(min(3, len(a["indices"]))):
            example_id = int(a["indices"][index])
            panels = [("Observed image", a["obs"][index, ..., 0]),
                      ("Ground-truth object", a["y_true"][index, ..., 0]),
                      ("Pretrained object", a["y_pred"][index, ..., 0]),
                      ("Fine-tuned object", b["y_pred"][index, ..., 0])]
            if example_id in rl_lookup:
                panels.append(("RL (validation-selected)", rl["y_pred"][rl_lookup[example_id], ..., 0]))
            truth_image = panels[1][1]
            limit = max(float(np.max(np.abs(truth_image))), 1e-12)
            minimum = min(float(np.min(image)) for _, image in panels)
            norm = SymLogNorm(linthresh=limit * 0.01, vmin=min(0, minimum), vmax=limit)
            side = truth_image.shape[0]
            extent = (-side * scale / 2, side * scale / 2, -side * scale / 2, side * scale / 2) if scale else None
            fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
            for i, (ax, (title, image)) in enumerate(zip(axes[0], panels)):
                im = ax.imshow(image, origin="lower", norm=norm, extent=extent)
                ax.set_title(f"({chr(97 + i)}) {title}")
                ax.set_xlabel("Angular offset [arcsec]" if scale else "Pixel")
                if crop:
                    left = extent[0] + crop * scale if scale else crop - 0.5
                    width = (side - 2 * crop) * scale if scale else side - 2 * crop
                    ax.add_patch(Rectangle((left, left), width, width, fill=False,
                                           edgecolor="white", linewidth=0.8, linestyle="--"))
            axes[0, 0].set_ylabel("Angular offset [arcsec]" if scale else "Pixel")
            fig.colorbar(im, ax=list(axes[0]), label="Model intensity units", fraction=0.02, pad=0.02)
            fig.savefig(figures / f"{dataset}_comparison_{example_id:04d}.pdf", bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = aggregate(args.campaign_dir, allow_partial=args.allow_partial)
    print(json.dumps(dict(complete=result["complete"], selected_rl=result["selected_rl"], problems=result["problems"])))


if __name__ == "__main__":
    main()

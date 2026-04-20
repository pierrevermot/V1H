"""Collect benchmark JSON results into a summary CSV.

Usage
-----
    python benchmark_collect_results.py --results-dir /path/to/results --output summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def collect_results(results_dir: Path) -> list[dict]:
    """Read all JSON result files and return a list of flat dicts."""
    rows = []
    for json_path in sorted(results_dir.glob("*.json")):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    print(f"[WARNING] Empty file: {json_path.name}", file=sys.stderr)
                    continue
                data = json.loads(content)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[WARNING] Could not parse {json_path.name}: {exc}", file=sys.stderr)
            continue

        timing = data.get("timing", {})
        latency = timing.get("latency", timing)
        throughput = data.get("throughput", timing.get("throughput", {}))
        if not throughput and latency:
            mean_latency = latency.get("batch_mean_s", latency.get("mean_s", ""))
            batch_size = data.get("batch_size", "")
            if mean_latency and batch_size:
                throughput = {
                    "samples_per_second_mean": float(batch_size) / float(mean_latency),
                    "per_sample_mean_ms": float(mean_latency) / float(batch_size) * 1000,
                }
        device_info = data.get("device_info", {})
        precision = data.get("precision", {})

        row = {
            "file": json_path.name,
            "mode": data.get("mode", ""),
            "head_target": data.get("head_target", ""),
            "model": data.get("model", ""),
            "batch_size": data.get("batch_size", ""),
            "device": data.get("device", ""),
            "dtype": data.get("dtype", precision.get("dtype", "")),
            "mixed_precision": data.get("mixed_precision", precision.get("mixed_precision_policy", "")),
            "jit_compile": data.get("jit_compile", timing.get("jit_compile", "")),
            "n_params": data.get("n_params", ""),
            "input_shape": str(data.get("input_shape", "")),
            "n_warmup": data.get("n_warmup", ""),
            "n_repeats": data.get("n_repeats", ""),
            # Latency / throughput
            "benchmark_execution_mode": timing.get("execution_mode", ""),
            "synchronization_mode": timing.get("synchronization_mode", ""),
            "batch_latency_mean_s": latency.get("batch_mean_s", latency.get("mean_s", "")),
            "batch_latency_std_s": latency.get("batch_std_s", latency.get("std_s", "")),
            "batch_latency_median_s": latency.get("batch_median_s", latency.get("median_s", "")),
            "batch_latency_min_s": latency.get("batch_min_s", latency.get("min_s", "")),
            "batch_latency_max_s": latency.get("batch_max_s", latency.get("max_s", "")),
            "samples_per_second_mean": throughput.get("samples_per_second_mean", ""),
            "samples_per_second_std": throughput.get("samples_per_second_std", ""),
            "samples_per_second_median": throughput.get("samples_per_second_median", ""),
            "samples_per_second_min": throughput.get("samples_per_second_min", ""),
            "samples_per_second_max": throughput.get("samples_per_second_max", ""),
            "per_sample_mean_ms": throughput.get("per_sample_mean_ms", ""),
            # Device info
            "n_cpu_devices_tf": device_info.get("n_cpu_devices_tf", ""),
            "n_gpus_tf": device_info.get("n_gpus_tf", ""),
            "gpu_names": str(device_info.get("gpu_names", "")),
            "cpu_count_os": device_info.get("cpu_count_os", ""),
            "cpu_affinity_count": device_info.get("cpu_affinity_count", ""),
            "slurm_cpus_per_task": device_info.get("slurm_cpus_per_task", ""),
            "tf_num_interop_threads": device_info.get("tf_num_interop_threads", ""),
            "tf_num_intraop_threads": device_info.get("tf_num_intraop_threads", ""),
            "omp_num_threads": device_info.get("omp_num_threads", ""),
            "slurm_job_id": device_info.get("slurm_job_id", ""),
            "slurm_partition": device_info.get("slurm_partition", ""),
            "slurm_nodelist": device_info.get("slurm_nodelist", ""),
        }
        rows.append(row)

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    """Write collected rows to a CSV file."""
    if not rows:
        print("[WARNING] No results to write.", file=sys.stderr)
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    """Print a human-readable summary to stdout."""
    if not rows:
        print("No results collected.")
        return

    # Group by device / model / precision / XLA mode.
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["device"], row["model"], row["dtype"], row["mixed_precision"], row["jit_compile"])
        groups.setdefault(key, []).append(row)

    print(f"\n{'='*120}")
    print(f"{'Device':<8} {'Model':<24} {'DType':<8} {'Mixed':<16} {'XLA':<5} {'BS':<6} {'Lat(s)':<10} {'Std(s)':<10} {'Samp/s':<12} {'ms/sample':<10}")
    print(f"{'='*120}")

    for key in sorted(groups.keys()):
        device, model, dtype_name, mixed_precision, jit_compile = key
        group = sorted(groups[key], key=lambda r: int(r["batch_size"]) if r["batch_size"] else 0)
        for row in group:
            mean_s = row["batch_latency_mean_s"]
            std_s = row["batch_latency_std_s"]
            throughput = row["samples_per_second_mean"]
            ms_sample = row["per_sample_mean_ms"]
            mean_str = f"{float(mean_s):.4f}" if mean_s else "N/A"
            std_str = f"{float(std_s):.4f}" if std_s else "N/A"
            throughput_str = f"{float(throughput):.2f}" if throughput else "N/A"
            ms_str = f"{float(ms_sample):.3f}" if ms_sample else "N/A"
            print(f"{device:<8} {model:<24} {dtype_name:<8} {mixed_precision:<16} {str(jit_compile):<5} {row['batch_size']:<6} {mean_str:<10} {std_str:<10} {throughput_str:<12} {ms_str:<10}")
        print(f"{'-'*120}")

    print(f"\nTotal results: {len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect benchmark timing results.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Directory with JSON result files")
    parser.add_argument("--output", type=Path, default=None, help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Skip printing summary to stdout")
    args = parser.parse_args()

    rows = collect_results(args.results_dir)

    if args.output:
        write_csv(rows, args.output)
        print(f"Wrote {len(rows)} results to {args.output}")

    if not args.quiet:
        print_summary(rows)


if __name__ == "__main__":
    main()

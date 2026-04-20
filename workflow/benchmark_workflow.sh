#!/usr/bin/env bash
# ===========================================================================
# benchmark_workflow.sh — SLURM orchestrator for computational timing benchmarks
#
# Launches inference timing benchmarks for independent heads and joint model
# across multiple hardware architectures and batch sizes, collects results.
#
# Usage:
#   ./benchmark_workflow.sh <config.py> [--dry-run] [--n-warmup 5] [--n-repeats 50]
#       [--output-dir /path/to/results]
#
# Architectures tested:
#   GPU:
#     - V100 32GB, 1 GPU  (v100-32g, gpu default partition)
#     - V100 32GB, 4 GPUs (v100-32g, gpu default partition)
#     - H100 80GB, 1 GPU  (gpu_p6 partition)
#     - H100 80GB, 4 GPUs (gpu_p6 partition)
#   CPU (on V100 nodes): Intel Xeon Gold 6248, 40 cores/node
#     - 1 CPU core  (cpu_p1 partition, constraint v100)
#     - 40 CPU cores (cpu_p1 partition, constraint v100)
#   CPU (on H100 nodes): Intel Xeon Platinum 8468, 96 cores/node
#     - 1 CPU core  (gpu_p6 partition, CPU-only)
#     - 96 CPU cores (gpu_p6 partition, CPU-only)
#   CPU (scalar partition): Intel Xeon Gold 6248, 40 cores/node
#     - 1 CPU core  (cpu_p1 partition)
#     - 40 CPU cores (cpu_p1 partition)
#
# Batch sizes: 2^n for n in 0..9 (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
#
# Models:
#   - independent im  (image head)
#   - independent psf (PSF head)
#   - independent noise (noise head)
#   - joint (four-head PINN model)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG=""
DRY_RUN=false
N_WARMUP=5
N_REPEATS=50
OUTPUT_DIR=""
CPU_MODULE_TF="tensorflow-gpu/py3/2.16.1"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--dry-run)    DRY_RUN=true; shift ;;
		--n-warmup)   N_WARMUP="$2"; shift 2 ;;
		--n-repeats)  N_REPEATS="$2"; shift 2 ;;
		--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
		-*)           echo "Unknown option: $1" >&2; exit 1 ;;
		*)            CONFIG="$1"; shift ;;
	esac
done

if [[ -z "$CONFIG" ]]; then
	echo "Usage: $0 <config.py> [--dry-run] [--n-warmup N] [--n-repeats N] [--output-dir DIR]" >&2
	exit 1
fi
CONFIG="$(realpath "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
	echo "Config file not found: $CONFIG" >&2
	exit 1
fi

# ---------------------------------------------------------------------------
# Ensure we can read the config from the launcher
# ---------------------------------------------------------------------------
ensure_launcher_python() {
	if python3 -c "import numpy" >/dev/null 2>&1; then
		return 0
	fi
	if command -v module >/dev/null 2>&1; then
		module purge
		module load "$CPU_MODULE_TF"
		python3 -c "import numpy" >/dev/null 2>&1 && return 0
	fi
	echo "Unable to import numpy: activate an environment or load $CPU_MODULE_TF before running this script." >&2
	exit 1
}

ensure_launcher_python

# ---------------------------------------------------------------------------
# Read key parameters from config
# ---------------------------------------------------------------------------
read_config() {
	python3 -c "
import sys, os
sys.path.insert(0, '$ROOT_DIR')
from configs.load_config import load_experiment_config
cfg = load_experiment_config('$CONFIG')
$1
" || return 1
}

OUTPUT_BASE_DIR=$(read_config 'print(cfg.OUTPUT_BASE_DIR)') || exit 1
SLURM_ACCOUNT=$(read_config 'print(cfg.SLURM_CONFIG.get("account", "nab"))') || exit 1
SLURM_CPU_ACCOUNT=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; print(cfg.SLURM_CONFIG.get("cpu_account", f"{base}@cpu"))') || exit 1
SLURM_GPU_ACCOUNT_V100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; common = cfg.SLURM_CONFIG.get("gpu_account", None); print(cfg.SLURM_CONFIG.get("v100_account", common if common is not None else f"{base}@v100"))') || exit 1
SLURM_GPU_ACCOUNT_H100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; common = cfg.SLURM_CONFIG.get("gpu_account", None); print(cfg.SLURM_CONFIG.get("h100_account", common if common is not None else f"{base}@h100"))') || exit 1
SLURM_EXCLUDE=$(read_config 'print(cfg.SLURM_CONFIG.get("exclude_nodes", ""))') || exit 1

if [[ -z "$OUTPUT_DIR" ]]; then
	OUTPUT_DIR="$OUTPUT_BASE_DIR/benchmark_timing"
fi

LOG_DIR="$OUTPUT_DIR/slurm_logs"
RESULTS_DIR="$OUTPUT_DIR/results"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

echo "=========================================="
echo "Benchmark timing workflow"
echo "=========================================="
echo "Config            : $CONFIG"
echo "Output dir        : $OUTPUT_DIR"
echo "Log dir           : $LOG_DIR"
echo "Results dir       : $RESULTS_DIR"
echo "CPU account       : $SLURM_CPU_ACCOUNT"
echo "V100 GPU account  : $SLURM_GPU_ACCOUNT_V100"
echo "H100 GPU account  : $SLURM_GPU_ACCOUNT_H100"
echo "N warmup          : $N_WARMUP"
echo "N repeats         : $N_REPEATS"
echo "Dry run           : $DRY_RUN"
echo "=========================================="

# ---------------------------------------------------------------------------
# Common SLURM options
# ---------------------------------------------------------------------------
EXCLUDE_OPT=""
if [[ -n "$SLURM_EXCLUDE" ]]; then
	EXCLUDE_OPT="#SBATCH --exclude=$SLURM_EXCLUDE"
fi

# ---------------------------------------------------------------------------
# Helper: submit or print
# ---------------------------------------------------------------------------
submit_job() {
	local job_script="$1"
	if $DRY_RUN; then
		echo "[DRY-RUN] Would submit: $job_script"
		echo "DRY_RUN_$(basename "$job_script" .sh)"
	else
		sbatch "$job_script" | awk '{print $NF}'
	fi
}

# ---------------------------------------------------------------------------
# Batch sizes: 2^n for n in [0, 9]
# ---------------------------------------------------------------------------
BATCH_SIZES=(1 2 4 8 16 32 64 128 256 512)

# ---------------------------------------------------------------------------
# Model modes to benchmark
# ---------------------------------------------------------------------------
INDEPENDENT_HEADS=("im" "psf" "noise")

# Track all submitted job IDs for the final collect step
ALL_JOB_IDS=()
JOB_COUNT=0

# ===========================================================================
# Architecture definitions
# ===========================================================================
# Each architecture is defined by:
#   ARCH_TAG        - short identifier for file naming
#   ARCH_DESC       - human-readable description
#   SLURM_ACCOUNT   - --account
#   SLURM_PARTITION - --partition (or empty)
#   SLURM_QOS       - --qos
#   SLURM_GRES      - --gres (e.g. gpu:1, gpu:4, or empty for CPU)
#   SLURM_CPUS      - --cpus-per-task
#   SLURM_CONSTRAINT- -C constraint (or empty)
#   ARCH_PREMODULE  - module to load before TF module (e.g. arch/h100)
#   MODULE_TF       - TensorFlow module name
#   DEVICE_FLAG     - --device argument for benchmark script (cpu or gpu)
#   N_GPUS_FLAG     - --n-gpus argument for benchmark script
#   TIME_LIMIT      - --time for the SLURM job

declare -a ARCH_TAGS ARCH_DESCS ARCH_ACCOUNTS ARCH_PARTITIONS ARCH_QOSS ARCH_GRESS
declare -a ARCH_CPUSS ARCH_CONSTRAINTS ARCH_PREMODULES ARCH_MODULES ARCH_DEVICES
declare -a ARCH_NGPUS ARCH_TIMES

idx=0

# --- V100 1 GPU ---
ARCH_TAGS[$idx]="v100_1gpu"
ARCH_DESCS[$idx]="V100-32G 1 GPU (Intel Xeon Gold 6248)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_V100"
ARCH_PARTITIONS[$idx]=""
ARCH_QOSS[$idx]="qos_gpu-t3"
ARCH_GRESS[$idx]="gpu:1"
ARCH_CPUSS[$idx]="10"
ARCH_CONSTRAINTS[$idx]="v100-32g"
ARCH_PREMODULES[$idx]=""
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.16.1"
ARCH_DEVICES[$idx]="gpu"
ARCH_NGPUS[$idx]="1"
ARCH_TIMES[$idx]="04:00:00"
((idx++))

# --- V100 4 GPUs ---
ARCH_TAGS[$idx]="v100_4gpu"
ARCH_DESCS[$idx]="V100-32G 4 GPUs (Intel Xeon Gold 6248)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_V100"
ARCH_PARTITIONS[$idx]=""
ARCH_QOSS[$idx]="qos_gpu-t3"
ARCH_GRESS[$idx]="gpu:4"
ARCH_CPUSS[$idx]="40"
ARCH_CONSTRAINTS[$idx]="v100-32g"
ARCH_PREMODULES[$idx]=""
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.16.1"
ARCH_DEVICES[$idx]="gpu"
ARCH_NGPUS[$idx]="4"
ARCH_TIMES[$idx]="04:00:00"
((idx++))

# --- H100 1 GPU ---
ARCH_TAGS[$idx]="h100_1gpu"
ARCH_DESCS[$idx]="H100-80G 1 GPU (Intel Xeon Platinum 8468)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_H100"
ARCH_PARTITIONS[$idx]="gpu_p6"
ARCH_QOSS[$idx]="qos_gpu_h100-t3"
ARCH_GRESS[$idx]="gpu:1"
ARCH_CPUSS[$idx]="24"
ARCH_CONSTRAINTS[$idx]="h100"
ARCH_PREMODULES[$idx]="module load arch/h100"
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.17.0"
ARCH_DEVICES[$idx]="gpu"
ARCH_NGPUS[$idx]="1"
ARCH_TIMES[$idx]="04:00:00"
((idx++))

# --- H100 4 GPUs ---
ARCH_TAGS[$idx]="h100_4gpu"
ARCH_DESCS[$idx]="H100-80G 4 GPUs (Intel Xeon Platinum 8468)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_H100"
ARCH_PARTITIONS[$idx]="gpu_p6"
ARCH_QOSS[$idx]="qos_gpu_h100-t3"
ARCH_GRESS[$idx]="gpu:4"
ARCH_CPUSS[$idx]="96"
ARCH_CONSTRAINTS[$idx]="h100"
ARCH_PREMODULES[$idx]="module load arch/h100"
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.17.0"
ARCH_DEVICES[$idx]="gpu"
ARCH_NGPUS[$idx]="4"
ARCH_TIMES[$idx]="04:00:00"
((idx++))

# --- CPU on V100 node: 1 core ---
ARCH_TAGS[$idx]="cpu_v100node_1core"
ARCH_DESCS[$idx]="CPU 1 core (Intel Xeon Gold 6248, V100 node)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_V100"
ARCH_PARTITIONS[$idx]=""
ARCH_QOSS[$idx]="qos_gpu-t3"
ARCH_GRESS[$idx]="gpu:1"
ARCH_CPUSS[$idx]="1"
ARCH_CONSTRAINTS[$idx]="v100-32g"
ARCH_PREMODULES[$idx]=""
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.16.1"
ARCH_DEVICES[$idx]="cpu"
ARCH_NGPUS[$idx]="0"
ARCH_TIMES[$idx]="10:00:00"
((idx++))

# --- CPU on V100 node: 40 cores ---
ARCH_TAGS[$idx]="cpu_v100node_40core"
ARCH_DESCS[$idx]="CPU 40 cores (Intel Xeon Gold 6248, V100 node)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_V100"
ARCH_PARTITIONS[$idx]=""
ARCH_QOSS[$idx]="qos_gpu-t3"
ARCH_GRESS[$idx]="gpu:1"
ARCH_CPUSS[$idx]="40"
ARCH_CONSTRAINTS[$idx]="v100-32g"
ARCH_PREMODULES[$idx]=""
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.16.1"
ARCH_DEVICES[$idx]="cpu"
ARCH_NGPUS[$idx]="0"
ARCH_TIMES[$idx]="10:00:00"
((idx++))

# --- CPU on H100 node: 1 core ---
ARCH_TAGS[$idx]="cpu_h100node_1core"
ARCH_DESCS[$idx]="CPU 1 core (Intel Xeon Platinum 8468, H100 node)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_H100"
ARCH_PARTITIONS[$idx]="gpu_p6"
ARCH_QOSS[$idx]="qos_gpu_h100-t3"
ARCH_GRESS[$idx]="gpu:1"
ARCH_CPUSS[$idx]="1"
ARCH_CONSTRAINTS[$idx]="h100"
ARCH_PREMODULES[$idx]="module load arch/h100"
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.17.0"
ARCH_DEVICES[$idx]="cpu"
ARCH_NGPUS[$idx]="0"
ARCH_TIMES[$idx]="10:00:00"
((idx++))

# --- CPU on H100 node: 96 cores ---
ARCH_TAGS[$idx]="cpu_h100node_96core"
ARCH_DESCS[$idx]="CPU 96 cores (Intel Xeon Platinum 8468, H100 node)"
ARCH_ACCOUNTS[$idx]="$SLURM_GPU_ACCOUNT_H100"
ARCH_PARTITIONS[$idx]="gpu_p6"
ARCH_QOSS[$idx]="qos_gpu_h100-t3"
ARCH_GRESS[$idx]="gpu:1"
ARCH_CPUSS[$idx]="96"
ARCH_CONSTRAINTS[$idx]="h100"
ARCH_PREMODULES[$idx]="module load arch/h100"
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.17.0"
ARCH_DEVICES[$idx]="cpu"
ARCH_NGPUS[$idx]="0"
ARCH_TIMES[$idx]="10:00:00"
((idx++))

# --- CPU partition (scalar): 1 core ---
ARCH_TAGS[$idx]="cpu_scalar_1core"
ARCH_DESCS[$idx]="CPU 1 core (Intel Xeon Gold 6248, scalar partition cpu_p1)"
ARCH_ACCOUNTS[$idx]="$SLURM_CPU_ACCOUNT"
ARCH_PARTITIONS[$idx]="cpu_p1"
ARCH_QOSS[$idx]="qos_cpu-t3"
ARCH_GRESS[$idx]=""
ARCH_CPUSS[$idx]="1"
ARCH_CONSTRAINTS[$idx]=""
ARCH_PREMODULES[$idx]=""
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.16.1"
ARCH_DEVICES[$idx]="cpu"
ARCH_NGPUS[$idx]="0"
ARCH_TIMES[$idx]="20:00:00"
((idx++))

# --- CPU partition (scalar): 40 cores ---
ARCH_TAGS[$idx]="cpu_scalar_40core"
ARCH_DESCS[$idx]="CPU 40 cores (Intel Xeon Gold 6248, scalar partition cpu_p1)"
ARCH_ACCOUNTS[$idx]="$SLURM_CPU_ACCOUNT"
ARCH_PARTITIONS[$idx]="cpu_p1"
ARCH_QOSS[$idx]="qos_cpu-t3"
ARCH_GRESS[$idx]=""
ARCH_CPUSS[$idx]="40"
ARCH_CONSTRAINTS[$idx]=""
ARCH_PREMODULES[$idx]=""
ARCH_MODULES[$idx]="tensorflow-gpu/py3/2.16.1"
ARCH_DEVICES[$idx]="cpu"
ARCH_NGPUS[$idx]="0"
ARCH_TIMES[$idx]="20:00:00"
((idx++))

N_ARCHS=$idx

# ===========================================================================
# Generate and submit benchmark jobs
# ===========================================================================
echo ""
echo "Submitting benchmark jobs..."
echo "Architectures: $N_ARCHS | Batch sizes: ${#BATCH_SIZES[@]} | Models: ${#INDEPENDENT_HEADS[@]} independent + 1 joint"
echo ""

for (( a=0; a<N_ARCHS; a++ )); do
	ATAG="${ARCH_TAGS[$a]}"
	ADESC="${ARCH_DESCS[$a]}"
	AACCOUNT="${ARCH_ACCOUNTS[$a]}"
	APARTITION="${ARCH_PARTITIONS[$a]}"
	AQOS="${ARCH_QOSS[$a]}"
	AGRES="${ARCH_GRESS[$a]}"
	ACPUS="${ARCH_CPUSS[$a]}"
	ACONSTRAINT="${ARCH_CONSTRAINTS[$a]}"
	APREMODULE="${ARCH_PREMODULES[$a]}"
	AMODULE="${ARCH_MODULES[$a]}"
	ADEVICE="${ARCH_DEVICES[$a]}"
	ANGPUS="${ARCH_NGPUS[$a]}"
	ATIME="${ARCH_TIMES[$a]}"

	# Build optional SBATCH directives
	PARTITION_OPT=""
	if [[ -n "$APARTITION" ]]; then
		PARTITION_OPT="#SBATCH --partition=$APARTITION"
	fi
	GRES_OPT=""
	if [[ -n "$AGRES" ]]; then
		GRES_OPT="#SBATCH --gres=$AGRES"
	fi
	CONSTRAINT_OPT=""
	if [[ -n "$ACONSTRAINT" ]]; then
		CONSTRAINT_OPT="#SBATCH -C $ACONSTRAINT"
	fi
	PREMODULE_CMD=""
	if [[ -n "$APREMODULE" ]]; then
		PREMODULE_CMD="$APREMODULE"
	fi

	# TF threading config for CPU benchmarks
	TF_THREADING=""
	if [[ "$ADEVICE" == "cpu" ]]; then
		TF_THREADING="export TF_NUM_INTEROP_THREADS=$ACPUS
export TF_NUM_INTRAOP_THREADS=$ACPUS
export OMP_NUM_THREADS=$ACPUS"
	fi

	echo "--- Architecture: $ADESC ---"

	for BS in "${BATCH_SIZES[@]}"; do
		# ---- Independent heads ----
		for HEAD in "${INDEPENDENT_HEADS[@]}"; do
			JOB_TAG="bench_${ATAG}_indep_${HEAD}_bs${BS}"
			JOB_SCRIPT="$LOG_DIR/${JOB_TAG}.sh"
			RESULT_FILE="$RESULTS_DIR/${JOB_TAG}.json"

			cat > "$JOB_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_TAG
#SBATCH --output=$LOG_DIR/${JOB_TAG}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_TAG}_%j.err
#SBATCH --account=$AACCOUNT
#SBATCH --qos=$AQOS
$PARTITION_OPT
$GRES_OPT
#SBATCH --cpus-per-task=$ACPUS
#SBATCH --time=$ATIME
#SBATCH --hint=nomultithread
$CONSTRAINT_OPT
$EXCLUDE_OPT

module purge
$PREMODULE_CMD
module load $AMODULE
export PYTHONPATH="$ROOT_DIR":"\${PYTHONPATH:-}"
cd "$ROOT_DIR"
$TF_THREADING

python3 -u workflow/benchmark_timing.py \\
	--config "$CONFIG" \\
	--mode independent \\
	--head-target $HEAD \\
	--batch-size $BS \\
	--n-warmup $N_WARMUP \\
	--n-repeats $N_REPEATS \\
	--device $ADEVICE \\
	--n-gpus $ANGPUS \\
	> "$RESULT_FILE"
SLURM_EOF
			chmod +x "$JOB_SCRIPT"
			JOB_ID=$(submit_job "$JOB_SCRIPT")
			ALL_JOB_IDS+=("$JOB_ID")
			((JOB_COUNT++))
		done

		# ---- Joint model ----
		JOB_TAG="bench_${ATAG}_joint_bs${BS}"
		JOB_SCRIPT="$LOG_DIR/${JOB_TAG}.sh"
		RESULT_FILE="$RESULTS_DIR/${JOB_TAG}.json"

		cat > "$JOB_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=$JOB_TAG
#SBATCH --output=$LOG_DIR/${JOB_TAG}_%j.out
#SBATCH --error=$LOG_DIR/${JOB_TAG}_%j.err
#SBATCH --account=$AACCOUNT
#SBATCH --qos=$AQOS
$PARTITION_OPT
$GRES_OPT
#SBATCH --cpus-per-task=$ACPUS
#SBATCH --time=$ATIME
#SBATCH --hint=nomultithread
$CONSTRAINT_OPT
$EXCLUDE_OPT

module purge
$PREMODULE_CMD
module load $AMODULE
export PYTHONPATH="$ROOT_DIR":"\${PYTHONPATH:-}"
cd "$ROOT_DIR"
$TF_THREADING

python3 -u workflow/benchmark_timing.py \\
	--config "$CONFIG" \\
	--mode joint \\
	--batch-size $BS \\
	--n-warmup $N_WARMUP \\
	--n-repeats $N_REPEATS \\
	--device $ADEVICE \\
	--n-gpus $ANGPUS \\
	> "$RESULT_FILE"
SLURM_EOF
		chmod +x "$JOB_SCRIPT"
		JOB_ID=$(submit_job "$JOB_SCRIPT")
		ALL_JOB_IDS+=("$JOB_ID")
		((JOB_COUNT++))
	done
	echo ""
done

echo "=========================================="
echo "Total jobs submitted: $JOB_COUNT"
echo "=========================================="

# ===========================================================================
# Submit a collection job that depends on all benchmark jobs
# ===========================================================================
if ! $DRY_RUN && [[ ${#ALL_JOB_IDS[@]} -gt 0 ]]; then
	# Build dependency string
	DEP_STR=$(IFS=:; echo "${ALL_JOB_IDS[*]}")

	COLLECT_SCRIPT="$LOG_DIR/collect_results.sh"
	cat > "$COLLECT_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=bench_collect
#SBATCH --output=$LOG_DIR/collect_%j.out
#SBATCH --error=$LOG_DIR/collect_%j.err
#SBATCH --account=$SLURM_CPU_ACCOUNT
#SBATCH --qos=qos_cpu-t3
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --hint=nomultithread
#SBATCH --dependency=afterany:$DEP_STR
$EXCLUDE_OPT

module purge
module load $CPU_MODULE_TF
export PYTHONPATH="$ROOT_DIR":"\${PYTHONPATH:-}"
cd "$ROOT_DIR"

python3 -u workflow/benchmark_collect_results.py \\
	--results-dir "$RESULTS_DIR" \\
	--output "$OUTPUT_DIR/benchmark_summary.csv"
SLURM_EOF
	chmod +x "$COLLECT_SCRIPT"
	COLLECT_JOB=$(submit_job "$COLLECT_SCRIPT")
	echo ""
	echo "Collection job (runs after all benchmarks): JobID=$COLLECT_JOB"
fi

echo ""
echo "=========================================="
echo "When all jobs complete, results will be in:"
echo "  Individual JSONs : $RESULTS_DIR/"
echo "  Summary CSV      : $OUTPUT_DIR/benchmark_summary.csv"
echo ""
echo "To manually collect results:"
echo "  python3 workflow/benchmark_collect_results.py \\"
echo "    --results-dir $RESULTS_DIR \\"
echo "    --output $OUTPUT_DIR/benchmark_summary.csv"
echo "=========================================="

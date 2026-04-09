#!/usr/bin/env bash
# ===========================================================================
# testing_workflow.sh — SLURM orchestrator for testing workflows
#
# Usage:
#   ./testing_workflow.sh <config.py> [--gpu v100|h100] [--dry-run]
#
# Steps:
#   1. Generate GalSim testing dataset (CPU)
#   2a. Run joint PINN inference on val + GalSim and save artifacts (GPU)
#   2b. Run Richardson-Lucy inference on val + GalSim and save artifacts (CPU)
#   3. Compute statistics and losses from saved inference artifacts for all algorithms (GPU)
#   4. Generate comparison figures and histograms from saved artifacts (CPU)
# ===========================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG=""
GPU_TYPE="h100"
DRY_RUN=false
CPU_MODULE_TF="tensorflow-gpu/py3/2.16.1"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--gpu) GPU_TYPE="$2"; shift 2 ;;
		--dry-run) DRY_RUN=true; shift ;;
		-*) echo "Unknown option: $1" >&2; exit 1 ;;
		*) CONFIG="$1"; shift ;;
	esac
done

if [[ -z "$CONFIG" ]]; then
	echo "Usage: $0 <config.py> [--gpu v100|h100] [--dry-run]" >&2
	exit 1
fi
CONFIG="$(realpath "$CONFIG")"

if [[ ! -f "$CONFIG" ]]; then
	echo "Config file not found: $CONFIG" >&2
	exit 1
fi

ensure_launcher_python() {
	if python3 -c "import numpy" >/dev/null 2>&1; then
		return 0
	fi
	if command -v module >/dev/null 2>&1; then
		module purge
		module load "$CPU_MODULE_TF"
		python3 -c "import numpy" >/dev/null 2>&1 && return 0
	fi
	echo "Unable to import numpy while loading config: activate an environment or load $CPU_MODULE_TF before running this script." >&2
	exit 1
}

ensure_launcher_python

# ---------------------------------------------------------------------------
# Read key parameters from config using Python
# ---------------------------------------------------------------------------
read_config() {
	python3 -c "
import sys
sys.path.insert(0, '$ROOT_DIR')
from configs.load_config import load_experiment_config
cfg = load_experiment_config('$CONFIG')
$1
" || return 1
}

OUTPUT_BASE_DIR=$(read_config 'print(cfg.OUTPUT_BASE_DIR)') || exit 1
DATASET_OUTPUT_DIR=$(read_config 'print(cfg.DATASET_GEN_CONFIG.get("output_dir", f"{cfg.OUTPUT_BASE_DIR}/dataset"))') || exit 1
SLURM_CPU_ACCOUNT=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; print(cfg.SLURM_CONFIG.get("cpu_account", f"{base}@cpu"))') || exit 1
SLURM_GPU_ACCOUNT_V100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; common = cfg.SLURM_CONFIG.get("gpu_account", None); print(cfg.SLURM_CONFIG.get("v100_account", common if common is not None else f"{base}@v100"))') || exit 1
SLURM_GPU_ACCOUNT_H100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; common = cfg.SLURM_CONFIG.get("gpu_account", None); print(cfg.SLURM_CONFIG.get("h100_account", common if common is not None else f"{base}@h100"))') || exit 1
SLURM_EXCLUDE=$(read_config 'print(cfg.SLURM_CONFIG.get("exclude_nodes", ""))') || exit 1
GALSIM_TEST_CPUS=$(read_config 'print(getattr(cfg, "GALSIM_TEST_CONFIG", {}).get("slurm_cpus_per_task", 4))') || exit 1
GALSIM_TEST_TIME=$(read_config 'print(getattr(cfg, "GALSIM_TEST_CONFIG", {}).get("slurm_time_limit", "02:00:00"))') || exit 1
GALSIM_TEST_OUTPUT_DIR=$(read_config 'print(getattr(cfg, "GALSIM_TEST_CONFIG", {}).get("output_dir", ""))') || exit 1
TEST_EVAL_CPUS=$(read_config 'print(getattr(cfg, "TEST_ON_GALSIM_CONFIG", {}).get("slurm_cpus_per_task", cfg.SLURM_CONFIG.get("cpus_per_task", 24)))') || exit 1
TEST_EVAL_TIME=$(read_config 'print(getattr(cfg, "TEST_ON_GALSIM_CONFIG", {}).get("slurm_time_limit", "04:00:00"))') || exit 1
TEST_EVAL_OUTPUT_DIR=$(read_config 'print(getattr(cfg, "TEST_ON_GALSIM_CONFIG", {}).get("output_dir", ""))') || exit 1
if [[ -z "$GALSIM_TEST_OUTPUT_DIR" ]]; then
	GALSIM_TEST_OUTPUT_DIR="$DATASET_OUTPUT_DIR/galsim_test"
fi

case "$GPU_TYPE" in
	v100)
		SLURM_GPU_ACCOUNT="$SLURM_GPU_ACCOUNT_V100"
		QOS="qos_gpu-t4"
		GPU_CONSTRAINT=""
		ARCH_PREMODULE=""
		MODULE_TF="tensorflow-gpu/py3/2.16.1"
		;;
	h100)
		SLURM_GPU_ACCOUNT="$SLURM_GPU_ACCOUNT_H100"
		QOS="qos_gpu_h100-t4"
		GPU_CONSTRAINT="#SBATCH -C h100"
		ARCH_PREMODULE="module load arch/h100"
		MODULE_TF="tensorflow-gpu/py3/2.17.0"
		;;
	*)
		echo "Unknown GPU type: $GPU_TYPE (must be v100 or h100)" >&2
		exit 1
		;;
esac

LOG_DIR="$OUTPUT_BASE_DIR/slurm_logs_testing"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Testing config    : $CONFIG"
echo "Output base dir   : $OUTPUT_BASE_DIR"
echo "CPU account       : $SLURM_CPU_ACCOUNT"
echo "GPU account       : $SLURM_GPU_ACCOUNT"
echo "GPU type          : $GPU_TYPE"
echo "GalSim test dir   : $GALSIM_TEST_OUTPUT_DIR"
echo "Eval output dir   : $TEST_EVAL_OUTPUT_DIR"
echo "Dry run           : $DRY_RUN"
echo "=========================================="

EXCLUDE_OPT=""
if [[ -n "$SLURM_EXCLUDE" ]]; then
	EXCLUDE_OPT="#SBATCH --exclude=$SLURM_EXCLUDE"
fi

join_dependency_opt() {
	local dep_ids=()
	for dep in "$@"; do
		if [[ -n "$dep" && "$dep" != "SKIPPED" ]]; then
			dep_ids+=("$dep")
		fi
	done
	if [[ ${#dep_ids[@]} -eq 0 ]]; then
		printf '%s' ""
		return
	fi
	local joined
	joined=$(IFS=:; echo "${dep_ids[*]}")
	printf '#SBATCH --dependency=afterok:%s' "$joined"
}

submit_job() {
	local job_script="$1"
	if $DRY_RUN; then
		echo "[DRY-RUN] Would submit: $job_script"
		echo "DRY_RUN_JOB_$(basename "$job_script" .sh)"
	else
		sbatch "$job_script" | awk '{print $NF}'
	fi
}

# ===========================================================================
# Step 1: Generate GalSim testing dataset (CPU)
# ===========================================================================
STEP1_SCRIPT="$LOG_DIR/testing_step1_galsim_dataset.sh"
cat > "$STEP1_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=galsim_test_dataset
#SBATCH --output=$LOG_DIR/testing_step1_%j.out
#SBATCH --error=$LOG_DIR/testing_step1_%j.err
#SBATCH --account=$SLURM_CPU_ACCOUNT
#SBATCH --cpus-per-task=$GALSIM_TEST_CPUS
#SBATCH --time=$GALSIM_TEST_TIME
#SBATCH --hint=nomultithread
$EXCLUDE_OPT

module purge
module load $CPU_MODULE_TF
export PYTHONPATH="$ROOT_DIR":"${PYTHONPATH:-}"
cd "$ROOT_DIR"

python3 -u workflow/galsim_ao_grid.py \
	--config "$CONFIG"
SLURM_EOF
chmod +x "$STEP1_SCRIPT"

JOB1=$(submit_job "$STEP1_SCRIPT")
echo "Step 1 (GalSim test dataset) : JobID=$JOB1"

STEP2_DEPENDENCY_OPT=$(join_dependency_opt "$JOB1")

# ===========================================================================
# Step 2a: Run inference on val + GalSim (GPU)
# ===========================================================================
STEP2A_SCRIPT="$LOG_DIR/testing_step2a_inference_joint_on_galsim.sh"
cat > "$STEP2A_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=test_on_galsim_infer
#SBATCH --output=$LOG_DIR/testing_step2a_%j.out
#SBATCH --error=$LOG_DIR/testing_step2a_%j.err
#SBATCH --account=$SLURM_GPU_ACCOUNT
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$TEST_EVAL_CPUS
#SBATCH --time=$TEST_EVAL_TIME
#SBATCH --hint=nomultithread
$GPU_CONSTRAINT
$STEP2_DEPENDENCY_OPT
$EXCLUDE_OPT

module purge
$ARCH_PREMODULE
module load $MODULE_TF
export PYTHONPATH="$ROOT_DIR":"${PYTHONPATH:-}"
cd "$ROOT_DIR"

python3 -u workflow/test_on_galsim_step2a_inference.py \
	--config "$CONFIG"
SLURM_EOF
chmod +x "$STEP2A_SCRIPT"

JOB2A=$(submit_job "$STEP2A_SCRIPT")
echo "Step 2a (save val + GalSim inference): JobID=$JOB2A"

# ===========================================================================
# Step 2b: Run Richardson-Lucy inference on val + GalSim (CPU)
# ===========================================================================
STEP2B_SCRIPT="$LOG_DIR/testing_step2b_inference_richardson_lucy.sh"
cat > "$STEP2B_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=test_on_galsim_rl_infer
#SBATCH --output=$LOG_DIR/testing_step2b_%j.out
#SBATCH --error=$LOG_DIR/testing_step2b_%j.err
#SBATCH --account=$SLURM_CPU_ACCOUNT
#SBATCH --cpus-per-task=$TEST_EVAL_CPUS
#SBATCH --time=$TEST_EVAL_TIME
#SBATCH --hint=nomultithread
$STEP2_DEPENDENCY_OPT
$EXCLUDE_OPT

module purge
module load $CPU_MODULE_TF
export PYTHONPATH="$ROOT_DIR":"${PYTHONPATH:-}"
cd "$ROOT_DIR"

python3 -u workflow/test_on_galsim_step2b_richardson_lucy.py \
	--config "$CONFIG"
SLURM_EOF
chmod +x "$STEP2B_SCRIPT"

JOB2B=$(submit_job "$STEP2B_SCRIPT")
echo "Step 2b (save Richardson-Lucy inference): JobID=$JOB2B"

STEP3_DEPENDENCY_OPT=$(join_dependency_opt "$JOB2A" "$JOB2B")

# ===========================================================================
# Step 3: Compute statistics and losses from saved inference (CPU)
# ===========================================================================
STEP3_SCRIPT="$LOG_DIR/testing_step3_analyze_joint_on_galsim.sh"
cat > "$STEP3_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=test_on_galsim_analyze
#SBATCH --output=$LOG_DIR/testing_step3_%j.out
#SBATCH --error=$LOG_DIR/testing_step3_%j.err
#SBATCH --account=$SLURM_CPU_ACCOUNT
#SBATCH --cpus-per-task=$TEST_EVAL_CPUS
#SBATCH --time=$TEST_EVAL_TIME
#SBATCH --hint=nomultithread
$STEP3_DEPENDENCY_OPT
$EXCLUDE_OPT

module purge
module load $CPU_MODULE_TF
export PYTHONPATH="$ROOT_DIR":"${PYTHONPATH:-}"
cd "$ROOT_DIR"

python3 -u workflow/test_on_galsim_step3_analysis.py \
	--config "$CONFIG" \
	--algorithm joint_pinn

python3 -u workflow/test_on_galsim_step3_analysis.py \
	--config "$CONFIG" \
	--algorithm richardson_lucy
SLURM_EOF
chmod +x "$STEP3_SCRIPT"

JOB3=$(submit_job "$STEP3_SCRIPT")
echo "Step 3 (analyze saved inference for all algorithms): JobID=$JOB3"

STEP4_DEPENDENCY_OPT=$(join_dependency_opt "$JOB3")

# ===========================================================================
# Step 4: Generate plots from saved inference + metrics (CPU)
# ===========================================================================
STEP4_SCRIPT="$LOG_DIR/testing_step4_plot_on_galsim.sh"
cat > "$STEP4_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=test_on_galsim_plot
#SBATCH --output=$LOG_DIR/testing_step4_%j.out
#SBATCH --error=$LOG_DIR/testing_step4_%j.err
#SBATCH --account=$SLURM_CPU_ACCOUNT
#SBATCH --cpus-per-task=$TEST_EVAL_CPUS
#SBATCH --time=$TEST_EVAL_TIME
#SBATCH --hint=nomultithread
$STEP4_DEPENDENCY_OPT
$EXCLUDE_OPT

module purge
module load $CPU_MODULE_TF
export PYTHONPATH="$ROOT_DIR":"${PYTHONPATH:-}"
cd "$ROOT_DIR"

python3 -u workflow/test_on_galsim_step4_plot.py \
	--config "$CONFIG"
SLURM_EOF
chmod +x "$STEP4_SCRIPT"

JOB4=$(submit_job "$STEP4_SCRIPT")
echo "Step 4 (generate plots and histograms): JobID=$JOB4"

echo ""
echo "=========================================="
echo "All jobs submitted. Testing pipeline:"
echo "  Step 1 ($JOB1) -> GalSim test dataset generation"
echo "  Step 1 ($JOB1) -> Step 2a ($JOB2A) save val + GalSim inference"
echo "  Step 1 ($JOB1) -> Step 2b ($JOB2B) save Richardson-Lucy inference"
echo "  Step 2a ($JOB2A), Step 2b ($JOB2B) -> Step 3 ($JOB3) compute statistics and losses"
echo "  Step 3 ($JOB3) -> Step 4 ($JOB4) generate figures and histograms"
echo "=========================================="
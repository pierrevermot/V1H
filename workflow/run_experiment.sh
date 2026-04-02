#!/usr/bin/env bash
# ===========================================================================
# run_experiment.sh — Unified SLURM orchestrator for the full V1H pipeline
#
# Usage:
#   ./run_experiment.sh <config.py> [--gpu v100|h100] [--dry-run]
#
# Steps (with SLURM job dependencies):
#   1. Create TFRecord dataset            (CPU)
#   2. Train independent heads             (GPU, depends on 1)
#      2a. Image head  (unet, NLL)
#      2b. Noise head  (unet, NLL)
#      2c. PSF head    (gpkh, R2)
#   3. PSF uncertainty stage 2             (GPU, depends on 2c)
#   4. Joint PINN four-head fine-tuning    (GPU, depends on 2a+2b+3)
#   5. Plot results                        (CPU, depends on 4)
# ===========================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG=""
GPU_TYPE="h100"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
	case "$1" in
		--gpu)    GPU_TYPE="$2"; shift 2 ;;
		--dry-run) DRY_RUN=true; shift ;;
		-*)       echo "Unknown option: $1" >&2; exit 1 ;;
		*)        CONFIG="$1"; shift ;;
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

# ---------------------------------------------------------------------------
# Read key parameters from config using Python
# ---------------------------------------------------------------------------
read_config() {
	python3 -c "
import sys, os
sys.path.insert(0, '$ROOT_DIR')
from configs.load_config import load_experiment_config
cfg = load_experiment_config('$CONFIG')
$1
"
}

OUTPUT_BASE_DIR=$(read_config 'print(cfg.OUTPUT_BASE_DIR)')
SLURM_ACCOUNT=$(read_config 'print(cfg.SLURM_CONFIG.get("account", "nab"))')
SLURM_CPU_ACCOUNT=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; print(cfg.SLURM_CONFIG.get("cpu_account", f"{base}@cpu"))')
SLURM_GPU_ACCOUNT_V100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; common = cfg.SLURM_CONFIG.get("gpu_account", None); print(cfg.SLURM_CONFIG.get("v100_account", common if common is not None else f"{base}@v100"))')
SLURM_GPU_ACCOUNT_H100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; common = cfg.SLURM_CONFIG.get("gpu_account", None); print(cfg.SLURM_CONFIG.get("h100_account", common if common is not None else f"{base}@h100"))')
SLURM_TIME=$(read_config 'print(cfg.SLURM_CONFIG.get("time_limit", "100:00:00"))')
SLURM_CPUS=$(read_config 'print(cfg.SLURM_CONFIG.get("cpus_per_task", 24))')
SLURM_EXCLUDE=$(read_config 'print(cfg.SLURM_CONFIG.get("exclude_nodes", ""))')
DATASET_N_ARRAY=$(read_config 'print(cfg.SLURM_CONFIG.get("dataset_n_array_jobs", 10))')
DATASET_CPUS=$(read_config 'print(cfg.SLURM_CONFIG.get("dataset_cpus_per_task", 40))')
DATASET_TIME=$(read_config 'print(cfg.SLURM_CONFIG.get("dataset_time_limit", "10:00:00"))')
SLURM_PARTITION_V100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; print(cfg.SLURM_CONFIG.get("v100_partition", f"{base}@v100"))')
SLURM_PARTITION_H100=$(read_config 'base = str(cfg.SLURM_CONFIG.get("account", "nab")).split("@", 1)[0]; print(cfg.SLURM_CONFIG.get("h100_partition", f"{base}@h100"))')

# GPU-type specific SLURM settings
case "$GPU_TYPE" in
	v100)
		SLURM_GPU_ACCOUNT="$SLURM_GPU_ACCOUNT_V100"
		PARTITION="$SLURM_PARTITION_V100"
		QOS="qos_gpu-t4"
		MODULE_TF="tensorflow-gpu/py3/2.16.1"
		;;
	h100)
		SLURM_GPU_ACCOUNT="$SLURM_GPU_ACCOUNT_H100"
		PARTITION="$SLURM_PARTITION_H100"
		QOS="qos_gpu_h100-t4"
		MODULE_TF="tensorflow-gpu/py3/2.17.0"
		;;
	*)
		echo "Unknown GPU type: $GPU_TYPE (must be v100 or h100)" >&2
		exit 1
		;;
esac

LOG_DIR="$OUTPUT_BASE_DIR/slurm_logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Experiment config : $CONFIG"
echo "Output base dir   : $OUTPUT_BASE_DIR"
echo "CPU account       : $SLURM_CPU_ACCOUNT"
echo "GPU account       : $SLURM_GPU_ACCOUNT"
echo "GPU type          : $GPU_TYPE"
echo "Partition         : $PARTITION"
echo "QOS               : $QOS"
echo "=========================================="

# ---------------------------------------------------------------------------
# Common sbatch options
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
		echo "DRY_RUN_JOB_$(basename "$job_script" .sh)"
	else
		sbatch "$job_script" | awk '{print $NF}'
	fi
}

# ===========================================================================
# Step 1: Create dataset (CPU array job)
# ===========================================================================
STEP1_SCRIPT="$LOG_DIR/step1_create_dataset.sh"
cat > "$STEP1_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=create_dataset
#SBATCH --output=$LOG_DIR/step1_%A_%a.out
#SBATCH --error=$LOG_DIR/step1_%A_%a.err
#SBATCH --account=$SLURM_CPU_ACCOUNT
#SBATCH --cpus-per-task=$DATASET_CPUS
#SBATCH --time=$DATASET_TIME
#SBATCH --array=0-$((DATASET_N_ARRAY - 1))
#SBATCH --hint=nomultithread
$EXCLUDE_OPT

set -euo pipefail
cd "$ROOT_DIR"

TOTAL_BATCHES=\$(python3 -c "
import sys; sys.path.insert(0, '$ROOT_DIR')
from configs.load_config import load_experiment_config
cfg = load_experiment_config('$CONFIG')
print(cfg.DATASET_GEN_CONFIG.get('n_batches', 1024))
")

BATCHES_PER_JOB=\$(( (TOTAL_BATCHES + $DATASET_N_ARRAY - 1) / $DATASET_N_ARRAY ))
BATCH_OFFSET=\$(( SLURM_ARRAY_TASK_ID * BATCHES_PER_JOB ))

python3 -u workflow/create_dataset.py \\
	--config "$CONFIG" \\
	--n-batches "\$BATCHES_PER_JOB" \\
	--batch-offset "\$BATCH_OFFSET" \\
	--n-workers "$DATASET_CPUS" \\
	--parallel-mode joblib
SLURM_EOF
chmod +x "$STEP1_SCRIPT"
JOB1=$(submit_job "$STEP1_SCRIPT")
echo "Step 1 (create dataset)      : JobID=$JOB1"

# ===========================================================================
# Step 2a: Train image head
# ===========================================================================
STEP2A_SCRIPT="$LOG_DIR/step2a_image_head.sh"
cat > "$STEP2A_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=train_image_head
#SBATCH --output=$LOG_DIR/step2a_%j.out
#SBATCH --error=$LOG_DIR/step2a_%j.err
#SBATCH --account=$SLURM_GPU_ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --hint=nomultithread
#SBATCH --dependency=afterok:$JOB1
$EXCLUDE_OPT

set -euo pipefail
module purge
module load $MODULE_TF
cd "$ROOT_DIR"

python3 -u workflow/independent_training.py \\
	--config "$CONFIG" --head-target im
SLURM_EOF
chmod +x "$STEP2A_SCRIPT"
JOB2A=$(submit_job "$STEP2A_SCRIPT")
echo "Step 2a (image head)         : JobID=$JOB2A"

# ===========================================================================
# Step 2b: Train noise head
# ===========================================================================
STEP2B_SCRIPT="$LOG_DIR/step2b_noise_head.sh"
cat > "$STEP2B_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=train_noise_head
#SBATCH --output=$LOG_DIR/step2b_%j.out
#SBATCH --error=$LOG_DIR/step2b_%j.err
#SBATCH --account=$SLURM_GPU_ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --hint=nomultithread
#SBATCH --dependency=afterok:$JOB1
$EXCLUDE_OPT

set -euo pipefail
module purge
module load $MODULE_TF
cd "$ROOT_DIR"

python3 -u workflow/independent_training.py \\
	--config "$CONFIG" --head-target noise
SLURM_EOF
chmod +x "$STEP2B_SCRIPT"
JOB2B=$(submit_job "$STEP2B_SCRIPT")
echo "Step 2b (noise head)         : JobID=$JOB2B"

# ===========================================================================
# Step 2c: Train PSF head
# ===========================================================================
STEP2C_SCRIPT="$LOG_DIR/step2c_psf_head.sh"
cat > "$STEP2C_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=train_psf_head
#SBATCH --output=$LOG_DIR/step2c_%j.out
#SBATCH --error=$LOG_DIR/step2c_%j.err
#SBATCH --account=$SLURM_GPU_ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --hint=nomultithread
#SBATCH --dependency=afterok:$JOB1
$EXCLUDE_OPT

set -euo pipefail
module purge
module load $MODULE_TF
cd "$ROOT_DIR"

python3 -u workflow/independent_training.py \\
	--config "$CONFIG" --head-target psf
SLURM_EOF
chmod +x "$STEP2C_SCRIPT"
JOB2C=$(submit_job "$STEP2C_SCRIPT")
echo "Step 2c (PSF head)           : JobID=$JOB2C"

# ===========================================================================
# Step 3: PSF uncertainty (stage 2) — depends on PSF head
# ===========================================================================
STEP3_SCRIPT="$LOG_DIR/step3_psf_uncertainty.sh"
cat > "$STEP3_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=psf_unc_stage2
#SBATCH --output=$LOG_DIR/step3_%j.out
#SBATCH --error=$LOG_DIR/step3_%j.err
#SBATCH --account=$SLURM_GPU_ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --hint=nomultithread
#SBATCH --dependency=afterok:$JOB2C
$EXCLUDE_OPT

set -euo pipefail
module purge
module load $MODULE_TF
cd "$ROOT_DIR"

python3 -u workflow/psf_uncertainty_stage2_training.py \\
	--config "$CONFIG"
SLURM_EOF
chmod +x "$STEP3_SCRIPT"
JOB3=$(submit_job "$STEP3_SCRIPT")
echo "Step 3 (PSF uncertainty)     : JobID=$JOB3"

# ===========================================================================
# Step 4: Joint PINN four-head — depends on im, noise, and PSF unc
# ===========================================================================
STEP4_SCRIPT="$LOG_DIR/step4_joint_pinn.sh"
cat > "$STEP4_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=joint_pinn_fourhead
#SBATCH --output=$LOG_DIR/step4_%j.out
#SBATCH --error=$LOG_DIR/step4_%j.err
#SBATCH --account=$SLURM_GPU_ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --qos=$QOS
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=$SLURM_CPUS
#SBATCH --time=$SLURM_TIME
#SBATCH --hint=nomultithread
#SBATCH --dependency=afterok:$JOB2A:$JOB2B:$JOB3
$EXCLUDE_OPT

set -euo pipefail
module purge
module load $MODULE_TF
cd "$ROOT_DIR"

python3 -u workflow/joint_pinn_fourhead_training.py \\
	--config "$CONFIG"
SLURM_EOF
chmod +x "$STEP4_SCRIPT"
JOB4=$(submit_job "$STEP4_SCRIPT")
echo "Step 4 (joint PINN)          : JobID=$JOB4"

# ===========================================================================
# Step 5: Plot results — depends on joint PINN
# ===========================================================================
STEP5_SCRIPT="$LOG_DIR/step5_plot_results.sh"
cat > "$STEP5_SCRIPT" <<SLURM_EOF
#!/usr/bin/env bash
#SBATCH --job-name=plot_results
#SBATCH --output=$LOG_DIR/step5_%j.out
#SBATCH --error=$LOG_DIR/step5_%j.err
#SBATCH --account=$SLURM_CPU_ACCOUNT
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --dependency=afterok:$JOB4
$EXCLUDE_OPT

set -euo pipefail
module purge
module load $MODULE_TF
cd "$ROOT_DIR"

JOINT_RUN_NAME=\$(python3 -c "
import sys; sys.path.insert(0, '$ROOT_DIR')
from configs.load_config import load_experiment_config
cfg = load_experiment_config('$CONFIG')
print(cfg.JOINT_PINN_CONFIG['run_name'])
")

python3 -u workflow/plot_results.py \\
	--run-dir "$OUTPUT_BASE_DIR/\$JOINT_RUN_NAME"
SLURM_EOF
chmod +x "$STEP5_SCRIPT"
JOB5=$(submit_job "$STEP5_SCRIPT")
echo "Step 5 (plot results)        : JobID=$JOB5"

echo ""
echo "=========================================="
echo "All jobs submitted. Pipeline:"
echo "  Step 1 ($JOB1) -> Steps 2a,2b,2c ($JOB2A,$JOB2B,$JOB2C)"
echo "  Step 2c ($JOB2C) -> Step 3 ($JOB3)"
echo "  Steps 2a+2b+3 ($JOB2A,$JOB2B,$JOB3) -> Step 4 ($JOB4)"
echo "  Step 4 ($JOB4) -> Step 5 ($JOB5)"
echo "=========================================="

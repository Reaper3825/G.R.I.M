#!/bin/bash
#SBATCH -p gpu-shared
#SBATCH --gpus=1
#SBATCH -t 24:00:00
#SBATCH -N 1
#
# GRIM-text training on SDSC Expanse.
# Submit through scripts/run_train_on_expanse.sh so the account, partition,
# logs, GRIM_EXPANSE_DIR, and optional GRIM_TRAIN_CONFIG are passed consistently.
#
# Example:
#   GRIM_EXPANSE_ACCOUNT=<project> ./scripts/run_train_on_expanse.sh --build --sbatch
#

set -e

if [[ -z "${GRIM_EXPANSE_DIR:-}" ]]; then
  echo "ERROR: GRIM_EXPANSE_DIR is not set. Submit with scripts/run_train_on_expanse.sh."
  exit 1
fi

cd "$GRIM_EXPANSE_DIR"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load gpu 2>/dev/null || true
module load slurm 2>/dev/null || true
module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda/12.0 2>/dev/null || true

export GRIM_PROJECT_DIR="$GRIM_EXPANSE_DIR"
source "$GRIM_EXPANSE_DIR/scripts/ensure_cuda12_for_training.sh" 2>/dev/null || true
export PATH="${GRIM_CUDA_ROOT:-}/bin:$PATH"
export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT:-}/lib64:${LD_LIBRARY_PATH:-}"

CONFIG="${GRIM_TRAIN_CONFIG:-$GRIM_EXPANSE_DIR/ai_config.json}"
exec "$GRIM_EXPANSE_DIR/resources/models/GRIM-text/training/TrainingLoop/build/train_gpu" --config "$CONFIG"

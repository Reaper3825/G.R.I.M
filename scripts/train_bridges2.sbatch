#!/bin/bash
#SBATCH --job-name=grim-train
#SBATCH -p GPU-shared
#SBATCH -A cis210058p
#SBATCH --gpus=h100-80:1
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_bridges2_%j.out
#SBATCH --error=logs/train_bridges2_%j.err
#
# GRIM-text training on PSC Bridges-2.
# Defaults match scripts/run_train_on_bridges2.sh (override with sbatch -p/-A/--gpus/-t).
# Walltime 24h; if your partition QOS caps lower, submit with e.g. sbatch -t 12:00:00 ....
#
# Submit from repo root on Bridges-2:
#   cd /ocean/projects/<alloc>/<user>/G.R.I.M
#   mkdir -p logs
#   export GRIM_BRIDGES2_DIR=$PWD   # optional if you use the default path below
#   sbatch scripts/train_bridges2.sbatch
#
# From your machine (syncs + submits with your allocation):
#   ./scripts/run_train_on_bridges2.sh --sbatch
#
# Optional env (same idea as scripts/train_anvil.sbatch):
#   GRIM_TRAIN_CONFIG   Config path relative to repo root (default: ai_config.json)

set -e
GRIM_ROOT="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis210058p/uwadkins/G.R.I.M}"
cd "$GRIM_ROOT"
mkdir -p logs

TRAINING_DIR="resources/models/GRIM-text/training"
BUILD_DIR="$TRAINING_DIR/TrainingLoop/build"
EXE="$GRIM_ROOT/$BUILD_DIR/train_gpu"
CONFIG_REL="${GRIM_TRAIN_CONFIG:-ai_config.json}"
if [[ "$CONFIG_REL" = /* ]]; then
  CONFIG_PATH="$CONFIG_REL"
else
  CONFIG_PATH="$GRIM_ROOT/$CONFIG_REL"
fi

# Same runtime setup as interactive srun in run_train_on_bridges2.sh (modules + CUDA 12 for CUTLASS)
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda/12.0 2>/dev/null || true
module load gcc 2>/dev/null || true

export GRIM_PROJECT_DIR="$GRIM_ROOT"
source "$GRIM_ROOT/scripts/ensure_cuda12_for_training.sh"
export PATH="${GRIM_CUDA_ROOT:-}/bin:$PATH"
export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT:-}/lib64:${LD_LIBRARY_PATH:-}"

echo "Running train_gpu on $(hostname) at $(date)"
echo "  GRIM_ROOT=$GRIM_ROOT"
echo "  config=$CONFIG_PATH"
"$EXE" --config "$CONFIG_PATH"
echo "Done at $(date)"

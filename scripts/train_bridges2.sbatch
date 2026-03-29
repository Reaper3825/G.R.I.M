#!/bin/bash
#SBATCH --job-name=grim-train
#SBATCH --partition=GPU-shared
#SBATCH --account=cis210058p
#SBATCH --gpus=h100-80:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/train_bridges2_%j.out
#SBATCH --error=logs/train_bridges2_%j.err
#
# GRIM-text training on PSC Bridges-2 (batch). Runtime matches the interactive
# srun wrapper in scripts/run_train_on_bridges2.sh (BRIDGES2_RUN_WRAPPER).
# Bridges-2: sbatch uses --gpus=type:n; srun uses --gres=gpu:type:n.
#
# Defaults match run_train_on_bridges2.sh (override at submit: sbatch -p/-A/--gpus/-t).
# If your partition QOS caps walltime lower, submit with e.g. sbatch -t 12:00:00 ....
#
# Submit from repo root on Bridges-2:
#   cd /ocean/projects/<alloc>/<user>/G.R.I.M
#   mkdir -p logs
#   export GRIM_BRIDGES2_DIR=$PWD   # optional if you use the default path below
#   sbatch scripts/train_bridges2.sbatch
#
# From your machine (syncs + submits a real batch job via sbatch on Bridges-2):
#   ./scripts/run_train_on_bridges2.sh --sbatch
# Build train_gpu on the cluster first (SSH; not Slurm batch):
#   ./scripts/run_train_on_bridges2.sh --build
#
# Do not run this .sbatch file with ./ on a laptop — it only runs on a compute node
# after sbatch (SLURM_JOB_ID will be set).
#
# Optional env (same idea as scripts/train_anvil.sbatch):
#   GRIM_TRAIN_CONFIG   Config path relative to repo root, or absolute (default: ai_config.json at repo root)

set -e
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This script runs only on a Bridges-2 compute node after sbatch (SLURM_JOB_ID is unset here)." >&2
  echo "From your Mac — submit a Slurm batch training job:" >&2
  echo "  ./scripts/run_train_on_bridges2.sh --sbatch" >&2
  echo "Build on the cluster first, then batch (or both in one go):" >&2
  echo "  ./scripts/run_train_on_bridges2.sh --build" >&2
  echo "  ./scripts/run_train_on_bridges2.sh --build --sbatch" >&2
  echo "On a Bridges-2 login node: sbatch scripts/train_bridges2.sbatch" >&2
  exit 1
fi

GRIM_ROOT="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis210058p/uwadkins/G.R.I.M}"
cd "$GRIM_ROOT" || {
  echo "ERROR: cannot cd to GRIM_ROOT=$GRIM_ROOT (set GRIM_BRIDGES2_DIR if your clone lives elsewhere)." >&2
  exit 1
}
mkdir -p logs

TRAINING_DIR="resources/models/GRIM-text/training"
BUILD_DIR="$TRAINING_DIR/TrainingLoop/build"
EXE="$GRIM_ROOT/$BUILD_DIR/train_gpu"

# Same default as interactive: $GRIM_ROOT/ai_config.json. run_train_on_bridges2.sh --sbatch --config X sets GRIM_TRAIN_CONFIG.
if [[ -n "${GRIM_TRAIN_CONFIG:-}" ]]; then
  if [[ "$GRIM_TRAIN_CONFIG" = /* ]]; then
    CONFIG_PATH="$GRIM_TRAIN_CONFIG"
  else
    CONFIG_PATH="$GRIM_ROOT/$GRIM_TRAIN_CONFIG"
  fi
else
  CONFIG_PATH="$GRIM_ROOT/ai_config.json"
fi

# Mirror BRIDGES2_RUN_WRAPPER in scripts/run_train_on_bridges2.sh
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda 2>/dev/null || true
export GRIM_PROJECT_DIR="$GRIM_ROOT"
source "$GRIM_ROOT/scripts/ensure_cuda12_for_training.sh" 2>/dev/null || true
export PATH="${GRIM_CUDA_ROOT:-}/bin:$PATH"
export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT:-}/lib64:${LD_LIBRARY_PATH:-}"

echo "Running train_gpu on $(hostname) at $(date)"
echo "  GRIM_ROOT=$GRIM_ROOT"
echo "  EXE=$EXE"
echo "  config=$CONFIG_PATH"
"$EXE" --config "$CONFIG_PATH"
echo "Done at $(date)"

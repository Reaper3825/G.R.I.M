#!/bin/bash
#SBATCH --job-name=grim-train
#SBATCH --account=x-cis210085
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=64G
#SBATCH --output=logs/train_anvil_%j.out
#SBATCH --error=logs/train_anvil_%j.err
# mem-per-gpu overrides partition DefMemPerGPU=120000 so we request 64G and schedule faster
#
# Run GRIM-text training on Purdue Anvil (Anvil-G: 4× A100).
# Partitions: gpu, gpu-debug. See: showpartitions or sinfo -s
#
# Submit from repo root on Anvil (project path; set GRIM_ANVIL_DIR if different):
#   cd /anvil/projects/x-cis210085/GRIM/G.R.I.M
#   mkdir -p logs
#   sbatch scripts/train_anvil.sbatch
#
# Or from your Mac: ./scripts/run_train_on_anvil.sh --sbatch

set -e
GRIM_ROOT="${GRIM_ANVIL_DIR:-/anvil/projects/x-cis210085/GRIM/G.R.I.M}"
cd "$GRIM_ROOT"
mkdir -p logs
TRAINING_DIR="resources/models/GRIM-text/training"
BUILD_DIR="$TRAINING_DIR/TrainingLoop/build"
CONFIG="${GRIM_TRAIN_CONFIG:-model_config.json}"

# Load Anvil modules (uncomment if your login doesn't auto-load)
# module load cuda gcc cmake

mkdir -p "$TRAINING_DIR/logs"
# Run from repo root so ai_config paths (e.g. resources/models/.../data/training_data.grmt) resolve correctly
echo "Running train_gpu on $(hostname) at $(date)"
"$GRIM_ROOT/$BUILD_DIR/train_gpu" --config "$TRAINING_DIR/$CONFIG"
echo "Done at $(date)"

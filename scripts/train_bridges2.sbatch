#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -A YOUR_ALLOCATION_ID
#SBATCH --gpus=h100-80:1
#SBATCH -t 4:00:00
#SBATCH -N 1
#
# GRIM-text training on PSC Bridges-2.
# Set YOUR_ALLOCATION_ID to your ACCESS allocation (e.g. abc1234p).
# Override -p or --gpus via sbatch command line when submitting.
#
# Before first run: clone repo to /ocean/projects/<alloc_id>/G.R.I.M on Bridges-2,
# then use run_train_on_bridges2.sh --build to build, or build manually.
#
set -e
cd "${GRIM_BRIDGES2_DIR:-$HOME/G.R.I.M}"
# Default to project CUDA 12 when default module is 11.x (CUTLASS requires 12+)
GRIM_PROJECT_DIR="${GRIM_BRIDGES2_DIR:-$HOME/G.R.I.M}"
source "${GRIM_PROJECT_DIR}/scripts/ensure_cuda12_for_training.sh" 2>/dev/null || true
export PATH="${GRIM_CUDA_ROOT:-}/bin:$PATH"
export LD_LIBRARY_PATH="${GRIM_CUDA_ROOT:-}/lib64:${LD_LIBRARY_PATH:-}"
./resources/models/GRIM-text/training/TrainingLoop/build/train_gpu --config ai_config.json

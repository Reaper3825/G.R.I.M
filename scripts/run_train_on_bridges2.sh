#!/usr/bin/env bash
# Run GRIM-text training on PSC Bridges-2 via SSH.
# Usage: ./scripts/run_train_on_bridges2.sh [--build] [--config CONFIG] [--sbatch]
#
# Prerequisites:
#   - SSH: ssh uwadkins@bridges2.psc.edu (or add to ~/.ssh/config as Host bridges2)
#   - Allocation: Set GRIM_BRIDGES2_ACCOUNT to your ACCESS allocation ID (e.g. abc1234p)
#   - Path: Set GRIM_BRIDGES2_DIR to your repo path, e.g. /ocean/projects/<alloc_id>/<username>/G.R.I.M (default: cis210058p/uwadkins)
#   - Data: Pushes merged_verified_cache.jsonl and (if present) concept_blocks.jsonl to
#     resources/models/GRIM-text/training/data/ on Bridges-2 (same layout as local).
#   - Submodules: flash-attention (script runs git submodule update --init before build)
#   - vcpkg: Script clones to GRIM_BRIDGES2_DIR/vcpkg if missing
#   - CUDA 12+ for training (flash-attention). Bridges-2: module load cuda (check with module avail cuda)
#
# Bridges-2 GPU partitions: GPU-shared (1-4 GPUs, faster queue) or GPU (full node).
# GPU types: h100-80, v100-32, v100-16, l40s-48. Default: h100-80.
#
# Options:
#   --build          Build train_gpu before running.
#   --config X       Config (default: ai_config.json).
#   --sbatch         Submit batch job (scripts/train_bridges2.sbatch).
#   --partition P    GPU-shared (default) or GPU.
#   --gpu-type T     h100-80 (default), v100-32, v100-16, or l40s-48.
#   --account A      Override GRIM_BRIDGES2_ACCOUNT.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="resources/models/GRIM-text/training"
GRIM_DIR="resources/models/GRIM-text/GRIM"
BUILD_DIR="$TRAINING_DIR/TrainingLoop/build"
EXE="$BUILD_DIR/train_gpu"
CONFIG="${CONFIG:-../../../../ai_config.json}"
TRAINING_DATA_DIR="$REPO_ROOT/resources/models/GRIM-text/training/data"
CACHE_PATH="$TRAINING_DATA_DIR/merged_verified_cache.jsonl"
CONCEPT_BLOCKS_PATH="$TRAINING_DATA_DIR/concept_blocks.jsonl"

# Bridges-2 path: /ocean/projects/<alloc_id>/<username>/G.R.I.M (override with GRIM_BRIDGES2_DIR)
BRIDGES2_DIR="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis210058p/uwadkins/G.R.I.M}"
ACCOUNT="${GRIM_BRIDGES2_ACCOUNT:-cis210058p}"
PARTITION="${PARTITION:-GPU-shared}"
GPU_TYPE="${GPU_TYPE:-h100-80}"
DO_BUILD=false
USE_SBATCH=false
DO_INCREMENTAL=false
DO_CLEAN_BUILD=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)          DO_BUILD=true; shift ;;
    --incremental)    DO_INCREMENTAL=true; shift ;;
    --clean)          DO_CLEAN_BUILD=true; shift ;;
    --config)         CONFIG="$2"; shift 2 ;;
    --sbatch)         USE_SBATCH=true; shift ;;
    --partition)      PARTITION="$2"; shift 2 ;;
    --gpu-type)       GPU_TYPE="$2"; shift 2 ;;
    --account)        ACCOUNT="$2"; shift 2 ;;
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Validate (path/account have defaults; override with env if needed)
if [[ -z "$BRIDGES2_DIR" ]]; then
  echo "ERROR: Set GRIM_BRIDGES2_DIR to your Bridges-2 repo path."
  echo "  Example: export GRIM_BRIDGES2_DIR=/ocean/projects/cis210058p/uwadkins/G.R.I.M"
  echo "  Your dir is under the allocation: /ocean/projects/<alloc_id>/<username>/"
  exit 1
fi
if [[ -z "$ACCOUNT" ]]; then
  echo "ERROR: Set GRIM_BRIDGES2_ACCOUNT to your ACCESS allocation ID (e.g. cis210058p)."
  echo "  Find it in your ACCESS allocation summary."
  exit 1
fi

REMOTE_TRAINING="$BRIDGES2_DIR/$TRAINING_DIR"
REMOTE_EXE="$BRIDGES2_DIR/$EXE"
REMOTE_DATA="$REMOTE_TRAINING/data"
REMOTE_CACHE="$REMOTE_DATA/merged_verified_cache.jsonl"
REMOTE_CONCEPT_BLOCKS="$REMOTE_DATA/concept_blocks.jsonl"
CACHE_PATH_EXPANDED="${CACHE_PATH/#\~/$HOME}"
CONCEPT_BLOCKS_PATH_EXPANDED="${CONCEPT_BLOCKS_PATH/#\~/$HOME}"

# SSH target: bridges2 or bridges2.psc.edu
BRIDGES2_SSH="${GRIM_BRIDGES2_SSH:-bridges2}"
if [[ "$BRIDGES2_SSH" == "bridges2" ]] && ! grep -q "Host bridges2" ~/.ssh/config 2>/dev/null; then
  BRIDGES2_SSH="uwadkins@bridges2.psc.edu"
fi

# SLURM
SLURM_ACCOUNT_ARGS="-A $ACCOUNT"
GRIM_SLURM_MAIL="${GRIM_SLURM_MAIL:-}"
[[ -n "$GRIM_SLURM_MAIL" ]] && SLURM_MAIL_ARGS="--mail-type=BEGIN,END,FAIL --mail-user=$GRIM_SLURM_MAIL" || SLURM_MAIL_ARGS=""

# One long-lived SSH using a script-unique socket in /tmp (avoids ~/.ssh permission issues)
BRIDGES2_CTRL="/tmp/cm-grim-$$"
if ! ssh -f -N -M -S "$BRIDGES2_CTRL" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$BRIDGES2_SSH"; then
  echo "SSH to Bridges-2 failed. Try: ssh bridges2"
  exit 1
fi
trap 'ssh -S "$BRIDGES2_CTRL" -O exit "$BRIDGES2_SSH" 2>/dev/null; rm -f "$BRIDGES2_CTRL"' EXIT
BRIDGES2_SSH_OPTS="-S $BRIDGES2_CTRL -o ControlMaster=no"

# Sync repo
BRIDGES2_SYNCED=false
if [[ -z "${GRIM_BRIDGES2_SKIP_PULL:-}" ]]; then
  echo "Syncing Bridges-2 repo..."
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && git fetch origin && git reset --hard origin/\$(git rev-parse --abbrev-ref HEAD)"
  echo "  Done."
  BRIDGES2_SYNCED=true
fi

if [[ "$DO_BUILD" == true ]] && [[ "$BRIDGES2_SYNCED" == true ]] && [[ "$DO_CLEAN_BUILD" != true ]]; then
  DO_INCREMENTAL=true
fi

# Bridges-2 modules: cuda, gcc, cmake. CUDA 12+ required.
BRIDGES2_MODULES="source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda/12.0 2>/dev/null || true; module load gcc 2>/dev/null || true; module load cmake 2>/dev/null || true"
# Default to project CUDA 12 when 11.x is detected (CUTLASS requires 12+). Uses GRIM_PROJECT_DIR=$BRIDGES2_DIR to find cuda-12.0.
BRIDGES2_ENSURE_CUDA12="export GRIM_PROJECT_DIR=\$BRIDGES2_DIR; source \"\$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true"
BRIDGES2_CUDA_ROOT='NVCC=$(which nvcc 2>/dev/null); if [ -z "$NVCC" ]; then echo "ERROR: nvcc not found. module load cuda" >&2; exit 1; fi; CUDAToolkit_ROOT=$(dirname "$(dirname "$NVCC")"); echo "[Bridges-2] $(nvcc --version | grep release | head -1)"'

# vcpkg
BRIDGES2_VCPKG="${GRIM_VCPKG_ROOT:-$BRIDGES2_DIR/vcpkg}"
VCPKG_TOOLCHAIN="$BRIDGES2_VCPKG/scripts/buildsystems/vcpkg.cmake"
TRAINING_VCPKG_JSON='{"name":"grim-training","version-string":"0.1.0","dependencies":["nlohmann-json","flatbuffers"]}'
BRIDGES2_VCPKG_ENSURE="true"
if [[ -z "${GRIM_VCPKG_ROOT:-}" ]]; then
  BRIDGES2_VCPKG_ENSURE="if [ ! -f \"$BRIDGES2_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake\" ]; then (cd \"$BRIDGES2_DIR\" && git clone https://github.com/Microsoft/vcpkg.git vcpkg && cd vcpkg && ./bootstrap-vcpkg.sh) || exit 1; fi"
fi
BRIDGES2_MANIFEST_ENSURE="mkdir -p $BRIDGES2_DIR/$TRAINING_DIR && [ -f $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json ] || printf '%s' '$TRAINING_VCPKG_JSON' > $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json"

# Submodule + flash-attention patch (same as Anvil)
CUTLASS_PIN="bbe579a9e3beb6ea6626d9227ec32d0dae119a49"
if [[ "${GRIM_USE_LATEST_CUTLASS:-}" == "1" ]]; then
  BRIDGES2_SUBMODULE="(git submodule deinit -f external/vcpkg 2>/dev/null || true) && git submodule update --init --force external/flash-attention && (cd external/flash-attention && git submodule update --init csrc/cutlass)"
  BRIDGES2_CLEAN=""
else
  BRIDGES2_SUBMODULE="(git submodule deinit -f external/vcpkg 2>/dev/null || true) && git submodule update --init --force external/flash-attention && (cd external/flash-attention && git submodule update --init --force csrc/cutlass) && (cd external/flash-attention/csrc/cutlass && git fetch origin && git checkout -f $CUTLASS_PIN) && (cd external/flash-attention/csrc/cutlass && git apply -p1 < ../../../../scripts/patches/cutlass-stride-nvcc-fix.patch 2>/dev/null || true) && (cd external/flash-attention && (git apply -p1 < ../../scripts/patches/flash-attention-bwd-template-fix.patch 2>/dev/null || (sed -i.bak -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, true>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, true, true>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, false>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, true, false>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, false>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, false, false>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, true>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, false, true>/g' csrc/flash_attn/src/flash_bwd_kernel.h))) || true"
  [[ "$DO_INCREMENTAL" == true ]] && [[ "$DO_CLEAN_BUILD" != true ]] && BRIDGES2_CLEAN="" || BRIDGES2_CLEAN="rm -rf $BRIDGES2_DIR/$BUILD_DIR && "
fi

# CUDA arch: sm_90 for H100, sm_80 for V100
if [[ "$GPU_TYPE" == "h100-80" ]]; then
  BRIDGES2_CUDA_ARCH="export GRIM_CUDA_ARCH=90; "
else
  BRIDGES2_CUDA_ARCH="export GRIM_CUDA_ARCH=80; "
fi

BRIDGES2_CMAKE_OPTS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN -DVCPKG_TARGET_TRIPLET=x64-linux -DVCPKG_MANIFEST_DIR=$BRIDGES2_DIR/$TRAINING_DIR"

# --build
if [[ "$DO_BUILD" == true ]]; then
  echo "Building train_gpu on Bridges-2 ($BRIDGES2_DIR/$BUILD_DIR)..."
  echo "  GPU type: $GPU_TYPE, CUDA arch: $([ "$GPU_TYPE" == "h100-80" ] && echo sm_90 || echo sm_80)"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "BRIDGES2_DIR=$BRIDGES2_DIR; $BRIDGES2_CUDA_ARCH cd \$BRIDGES2_DIR && $BRIDGES2_SUBMODULE && $BRIDGES2_VCPKG_ENSURE && $BRIDGES2_MANIFEST_ENSURE && cd \$BRIDGES2_DIR/$TRAINING_DIR/TrainingLoop && ${BRIDGES2_CLEAN}mkdir -p build && cd build && $BRIDGES2_MODULES && $BRIDGES2_ENSURE_CUDA12 && $BRIDGES2_CUDA_ROOT && cmake .. $BRIDGES2_CMAKE_OPTS -DCUDAToolkit_ROOT=\$CUDAToolkit_ROOT && make -j \$(nproc) train_gpu"
fi

# Transfer data
if [[ ! -f "$CACHE_PATH_EXPANDED" ]]; then
  echo "ERROR: merged_verified_cache.jsonl not found at $CACHE_PATH_EXPANDED"
  exit 1
fi
echo "Transferring merged_verified_cache.jsonl..."
ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "mkdir -p $REMOTE_DATA"
ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $REMOTE_CACHE" < "$CACHE_PATH_EXPANDED"
echo "  -> $REMOTE_CACHE"

if [[ -f "$CONCEPT_BLOCKS_PATH_EXPANDED" ]]; then
  echo "Transferring concept_blocks.jsonl..."
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $REMOTE_CONCEPT_BLOCKS" < "$CONCEPT_BLOCKS_PATH_EXPANDED"
  echo "  -> $REMOTE_CONCEPT_BLOCKS"
else
  echo "Skipping concept_blocks.jsonl (not found at $CONCEPT_BLOCKS_PATH_EXPANDED)."
  echo "  DataLoader will use cache-only curriculum; add the file locally to ship UltraChat/stem blocks."
fi

# Transfer ai_config.json
if [[ -f "$REPO_ROOT/ai_config.json" ]]; then
  echo "Transferring ai_config.json..."
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $BRIDGES2_DIR/ai_config.json" < "$REPO_ROOT/ai_config.json"
fi

# Batch job
if [[ "$USE_SBATCH" == true ]]; then
  SBATCH_PATH="$REPO_ROOT/scripts/train_bridges2.sbatch"
  if [[ ! -f "$SBATCH_PATH" ]]; then
    echo "ERROR: scripts/train_bridges2.sbatch not found."
    echo "  Create it with: #SBATCH -p $PARTITION, #SBATCH -A $ACCOUNT, #SBATCH --gpus=$GPU_TYPE:1"
    exit 1
  fi
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "mkdir -p $BRIDGES2_DIR/scripts $BRIDGES2_DIR/logs"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $BRIDGES2_DIR/scripts/train_bridges2.sbatch" < "$SBATCH_PATH"
  echo "Submitting batch job (partition=$PARTITION, gpu=$GPU_TYPE)..."
  SUBMIT_OUT=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && sbatch --export=ALL,GRIM_BRIDGES2_DIR=$BRIDGES2_DIR --output=$BRIDGES2_DIR/logs/train_%j.out --error=$BRIDGES2_DIR/logs/train_%j.err $SLURM_MAIL_ARGS -p $PARTITION $SLURM_ACCOUNT_ARGS --gpus=$GPU_TYPE:1 -t 4:00:00 scripts/train_bridges2.sbatch")
  echo "$SUBMIT_OUT"
  exit 0
fi

# Interactive: srun (load cuda module + set LD_LIBRARY_PATH so compute node finds libcudart)
BRIDGES2_RUN_WRAPPER="bash -c 'source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || true; export GRIM_PROJECT_DIR=\"$BRIDGES2_DIR\"; source \"$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; exec \"$REMOTE_EXE\" --config \"$BRIDGES2_DIR/ai_config.json\"'"
echo "Running train_gpu on Bridges-2 (partition=$PARTITION, gpu=$GPU_TYPE)..."
SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS --gres=gpu:$GPU_TYPE:1 -t 4:00:00 --pty"
if [[ -t 0 ]]; then
  ssh -t $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $SRUN_ARGS $BRIDGES2_RUN_WRAPPER"
else
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $SRUN_ARGS $BRIDGES2_RUN_WRAPPER"
fi

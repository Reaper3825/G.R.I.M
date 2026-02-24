#!/usr/bin/env bash
# Run GRIM-text training on Purdue Anvil via SSH.
# Usage: ./scripts/run_train_on_anvil.sh [--build] [--config CONFIG] [--sbatch]
#
# Prerequisites:
#   - SSH: ssh anvil  (anvil.rcac.purdue.edu; key in ~/.ssh/id_ed25519)
#   - Submodules: flash-attention (script runs git submodule update --init before build)
#   - Repo on Anvil at GRIM_ANVIL_DIR. Default: /anvil/projects/x-cis210085/GRIM/G.R.I.M
#     Override if you use home or scratch: export GRIM_ANVIL_DIR=\$HOME/G.R.I.M
#   - On Anvil, build once (see first-time setup in docs or below).
#   - vcpkg manifest mode (vcpkg.json at repo root). On Anvil we use the vcpkg toolchain (x64-linux).
#     If vcpkg is not found at GRIM_ANVIL_DIR/vcpkg, the script clones and bootstraps it there before building.
#     Override location with GRIM_VCPKG_ROOT (script will not auto-download in that case).
#   - CMake 3.22+ for TrainingLoop. If default module is older, on Anvil run "module avail cmake"
#     then export ANVIL_CMAKE_MODULE=cmake/X.XX (e.g. cmake/3.26) before this script.
#
# Note: To use Cursor Run: in a terminal run "ssh-add ~/.ssh/id_ed25519" (enter passphrase once).
# Then Run works because the key is in ssh-agent. Or run this script from the terminal.
#
# Anvil partitions (Anvil-G = 4× A100): gpu, gpu-debug. Check: ssh anvil showpartitions
#
# Options:
#   --build          Same as --build-training (build train_gpu before running).
#   --build-training Build train_gpu (TrainingLoop) on Anvil before running.
#   --build-grim     Build grim_text_server (GRIM-text inference server) on Anvil.
#   --build-grim-exe Build GRIM (main host / adaptive controller for future agent) from repo root.
#   --config X    Config file in training/ (default: model_config.json).
#   --sbatch      Submit batch job (scripts/train_anvil.sbatch) instead of interactive.
#   --partition P  SLURM partition (default: gpu). Use gpu-debug for short test runs.
#   --account A    SLURM account (default: cis210085-gpu). Override with GRIM_SLURM_ACCOUNT.
#   --qos Q       SLURM QOS (optional; set if your allocation requires it).

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Path on Anvil (project space; override with GRIM_ANVIL_DIR if you use $HOME or $SCRATCH)
ANVIL_DIR="${GRIM_ANVIL_DIR:-/anvil/projects/x-cis210085/GRIM/G.R.I.M}"
TRAINING_DIR="resources/models/GRIM-text/training"
GRIM_DIR="resources/models/GRIM-text/GRIM"
BUILD_DIR="$TRAINING_DIR/TrainingLoop/build"
EXE="$BUILD_DIR/train_gpu"
CONFIG="${CONFIG:-model_config.json}"
DO_BUILD=false
DO_BUILD_GRIM=false
DO_BUILD_GRIM_EXE=false
USE_SBATCH=false
PARTITION="${PARTITION:-gpu}"
# SLURM account: use project allocation for gpu partition (override with GRIM_SLURM_ACCOUNT)
ACCOUNT="${GRIM_SLURM_ACCOUNT:-cis210085-gpu}"
QOS="${GRIM_SLURM_QOS:-}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)          DO_BUILD=true; shift ;;
    --build-training) DO_BUILD=true; shift ;;
    --build-grim)     DO_BUILD_GRIM=true; shift ;;
    --build-grim-exe) DO_BUILD_GRIM_EXE=true; shift ;;
    --config)         CONFIG="$2"; shift 2 ;;
    --sbatch)         USE_SBATCH=true; shift ;;
    --partition)      PARTITION="$2"; shift 2 ;;
    --account)        ACCOUNT="$2"; shift 2 ;;
    --qos)            QOS="$2"; shift 2 ;;
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Build srun/sbatch SLURM args (account required; qos optional)
SLURM_ACCOUNT_ARGS="--account=$ACCOUNT"
[[ -n "$QOS" ]] && SLURM_ACCOUNT_ARGS="$SLURM_ACCOUNT_ARGS --qos=$QOS"

# When Run from Cursor/IDE there is no prompt for SSH passphrase. Require key in agent.
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 anvil true 2>/dev/null; then
  echo "SSH to Anvil failed: key not loaded (no way to enter passphrase when Run from Cursor)."
  echo ""
  echo "In a terminal, run once:"
  echo "  eval \"\$(ssh-agent -s)\""
  echo "  ssh-add ~/.ssh/id_ed25519"
  echo "Enter your passphrase there. Then Run this script again from Cursor."
  echo ""
  echo "Or run this script from the terminal instead of Cursor Run."
  exit 1
fi

REMOTE_TRAINING="$ANVIL_DIR/$TRAINING_DIR"
REMOTE_EXE="$ANVIL_DIR/$EXE"
REMOTE_CFG="$REMOTE_TRAINING/$CONFIG"

# Anvil: init module system (non-interactive SSH doesn't load profile), then load GPU stack.
# RCAC recommends "modtree/gpu" (provides cuda/11.2.2, gcc, openmpi). CMake 3.22+ required for TrainingLoop.
# Try newer cmake first (module name may be cmake/3.26 or cmake/3.22 on Anvil); override with ANVIL_CMAKE_MODULE.
ANVIL_CMAKE_MODULE="${ANVIL_CMAKE_MODULE:-}"
ANVIL_MODULES='source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true; module load modtree/gpu 2>/dev/null || module load cuda gcc 2>/dev/null || true; if [ -n "$ANVIL_CMAKE_MODULE" ]; then module load $ANVIL_CMAKE_MODULE 2>/dev/null || true; else module load cmake/3.26 2>/dev/null || module load cmake/3.22 2>/dev/null || module load cmake 2>/dev/null || true; fi'
# Set CUDAToolkit_ROOT only when nvcc is found (avoid passing "." when module load failed)
ANVIL_CUDA_ROOT='NVCC=$(which nvcc 2>/dev/null); if [ -z "$NVCC" ]; then echo "ERROR: nvcc not found. On Anvil run: module load modtree/gpu (or module load cuda); then re-run this script." >&2; exit 1; fi; CUDAToolkit_ROOT=$(dirname "$(dirname "$NVCC")")'
# vcpkg: use toolchain with x64-linux. TrainingLoop uses minimal manifest (training/vcpkg.json) so only
# nlohmann-json and flatbuffers are installed (avoids full repo manifest deps that need Python 3.7+ / meson).
ANVIL_VCPKG="${GRIM_VCPKG_ROOT:-$ANVIL_DIR/vcpkg}"
VCPKG_TOOLCHAIN="$ANVIL_VCPKG/scripts/buildsystems/vcpkg.cmake"
ANVIL_CMAKE_OPTS='-DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=$CUDAToolkit_ROOT'
ANVIL_CMAKE_OPTS="$ANVIL_CMAKE_OPTS -DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN -DVCPKG_TARGET_TRIPLET=x64-linux -DVCPKG_MANIFEST_DIR=$ANVIL_DIR/$TRAINING_DIR"

# If using default vcpkg location (not GRIM_VCPKG_ROOT), ensure vcpkg exists on Anvil: clone + bootstrap if missing.
ANVIL_VCPKG_ENSURE="true"
if [[ -z "${GRIM_VCPKG_ROOT:-}" ]]; then
  ANVIL_VCPKG_ENSURE="if [ ! -f \"$ANVIL_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake\" ]; then echo \"vcpkg not found at $ANVIL_DIR/vcpkg, cloning and bootstrapping ...\"; (cd \"$ANVIL_DIR\" && git clone https://github.com/Microsoft/vcpkg.git vcpkg && cd vcpkg && ./bootstrap-vcpkg.sh) || exit 1; fi"
fi

# Ensure training/vcpkg.json exists on Anvil (minimal manifest for TrainingLoop). Create if missing so build works before git pull.
TRAINING_VCPKG_JSON='{"name":"grim-training","version-string":"0.1.0","dependencies":["nlohmann-json","flatbuffers"]}'
ANVIL_TRAINING_MANIFEST_ENSURE="mkdir -p $ANVIL_DIR/$TRAINING_DIR && [ -f $ANVIL_DIR/$TRAINING_DIR/vcpkg.json ] || printf '%s' '$TRAINING_VCPKG_JSON' > $ANVIL_DIR/$TRAINING_DIR/vcpkg.json"

# Init flash-attention submodule only (avoids "No url for external/vcpkg" if vcpkg was ever a submodule).
# Also init flash-attention's submodules (cutlass) - required for cute/tensor.hpp
ANVIL_SUBMODULE_INIT="(git submodule deinit -f external/vcpkg 2>/dev/null || true) && git submodule update --init external/flash-attention && (cd external/flash-attention && git submodule update --init csrc/cutlass)"

# --build / --build-training: GRIM-text/training/TrainingLoop CMake → train_gpu
if [[ "$DO_BUILD" == true ]]; then
  echo "Building train_gpu (training loop) on Anvil in $ANVIL_DIR/$BUILD_DIR ..."
  ssh anvil "cd $ANVIL_DIR && $ANVIL_SUBMODULE_INIT && $ANVIL_VCPKG_ENSURE && $ANVIL_TRAINING_MANIFEST_ENSURE && cd $ANVIL_DIR/$TRAINING_DIR/TrainingLoop && mkdir -p build && cd build && $ANVIL_MODULES && $ANVIL_CUDA_ROOT && cmake .. $ANVIL_CMAKE_OPTS && make -j \$(nproc) train_gpu"
fi

# --build-grim: GRIM-text/GRIM CMake → grim_text_server (inference)
if [[ "$DO_BUILD_GRIM" == true ]]; then
  echo "Building grim_text_server (GRIM-text inference) on Anvil in $ANVIL_DIR/$GRIM_DIR/build ..."
  ssh anvil "cd $ANVIL_DIR && $ANVIL_SUBMODULE_INIT && $ANVIL_VCPKG_ENSURE && cd $ANVIL_DIR/$GRIM_DIR && mkdir -p build && cd build && $ANVIL_MODULES && $ANVIL_CUDA_ROOT && cmake .. $ANVIL_CMAKE_OPTS && make -j \$(nproc) grim_text_server"
fi

# --build-grim-exe: repo root CMake → GRIM (main host / adaptive controller for future agent)
if [[ "$DO_BUILD_GRIM_EXE" == true ]]; then
  echo "Building GRIM (main host / grim.exe) on Anvil in $ANVIL_DIR/build ..."
  ssh anvil "cd $ANVIL_DIR && $ANVIL_SUBMODULE_INIT && $ANVIL_VCPKG_ENSURE && mkdir -p build && cd build && $ANVIL_MODULES && $ANVIL_CUDA_ROOT && cmake .. $ANVIL_CMAKE_OPTS && make -j \$(nproc) GRIM"
fi

if [[ "$USE_SBATCH" == true ]]; then
  echo "Submitting batch job on Anvil (partition=$PARTITION, account=$ACCOUNT) ..."
  ssh anvil "cd $ANVIL_DIR && GRIM_SLURM_ACCOUNT=$ACCOUNT GRIM_SLURM_QOS=$QOS sbatch --partition=$PARTITION $SLURM_ACCOUNT_ARGS scripts/train_anvil.sbatch"
  echo "Job submitted. Check with: ssh anvil squeue -u \$USER"
  exit 0
fi

# Interactive run: allocate one GPU and run train_gpu (streams output)
# Use -t only when stdin is a TTY (real terminal); avoids "Pseudo-terminal will not be allocated" in IDE/Code Runner
# --account required on Anvil/ACCESS (fixes "Invalid qos specification")
echo "Running train_gpu on Anvil (partition=$PARTITION, account=$ACCOUNT) ..."
SRUN_ARGS="--partition=$PARTITION $SLURM_ACCOUNT_ARGS --gres=gpu:1 --time=04:00:00 --pty"
if [[ -t 0 ]]; then
  ssh -t anvil "cd $REMOTE_TRAINING && srun $SRUN_ARGS $REMOTE_EXE --config $CONFIG"
else
  ssh anvil "cd $REMOTE_TRAINING && srun $SRUN_ARGS $REMOTE_EXE --config $CONFIG"
fi

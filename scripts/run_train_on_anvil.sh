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
#   - CUDA 12+ for training build (CUTLASS/flash-attention). Anvil default is CUDA 11.2 (modtree/gpu).
#     Script defaults to project CUDA 12 when 11.x is detected (looks for $ANVIL_PROJECT_PARENT/cuda-12.0).
#     Install once to project space if missing; then no need to set GRIM_CUDA_ROOT.
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
# Parent of repo (where cuda-12.0 is typically installed); used by ensure_cuda12_for_training.sh
ANVIL_PROJECT_PARENT="${ANVIL_DIR%/*}"
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

# Anvil: init module system, load GPU stack. CUTLASS (flash-attention) requires CUDA 12+.
# Anvil has only CUDA 11.2 (modtree/gpu). To use a user-installed CUDA 12:
#   1. Download runfile: open https://developer.nvidia.com/cuda-12-0-1-download-archive
#      Select Linux > x86_64 > CentOS > runfile (local), get the wget/curl command.
#      Or: ssh anvil "cd /anvil/projects/x-cis210085/GRIM && wget -q --show-progress https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda_12.0.1_535.104.05_linux.run"
#   2. Verify file size ~2.5GB before installing. If tiny, the URL failed.
#   3. ssh anvil "cd /anvil/projects/x-cis210085/GRIM && chmod +x cuda_12*.run && ./cuda_12*.run --silent --toolkit --override --no-man-page --toolkitpath=/anvil/projects/x-cis210085/GRIM/cuda-12.0"
#   4. GRIM_CUDA_ROOT=/anvil/projects/x-cis210085/GRIM/cuda-12.0 ./scripts/run_train_on_anvil.sh --build
ANVIL_CMAKE_MODULE="${ANVIL_CMAKE_MODULE:-}"
ANVIL_CUDA_MODULE="${ANVIL_CUDA_MODULE:-}"
# If GRIM_CUDA_ROOT set: skip CUDA modules, use that path. Else: try module (cuda/12.0.1, modtree/gpu, etc.)
ANVIL_MODULES="source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true; if [ -z \"\$GRIM_CUDA_ROOT\" ]; then if [ -n \"${ANVIL_CUDA_MODULE}\" ]; then module load ${ANVIL_CUDA_MODULE} 2>/dev/null || true; fi; if ! which nvcc >/dev/null 2>&1; then module load cuda/12.0.1 2>/dev/null || module load cuda/11.4.2 2>/dev/null || module load modtree/gpu 2>/dev/null || module load cuda 2>/dev/null || true; fi; fi; module load gcc 2>/dev/null || true; if [ -n \"${ANVIL_CMAKE_MODULE}\" ]; then module load ${ANVIL_CMAKE_MODULE} 2>/dev/null || true; else module load cmake/3.26 2>/dev/null || module load cmake/3.22 2>/dev/null || module load cmake 2>/dev/null || true; fi"
# Default to project CUDA 12 when 11.x is detected (CUTLASS requires 12+). Inlined so remote does not need ensure_cuda12_for_training.sh.
ANVIL_ENSURE_CUDA12='_p='"$ANVIL_PROJECT_PARENT"'; if [ -z "$GRIM_CUDA_ROOT" ] && NVCC=$(which nvcc 2>/dev/null) && [ -n "$NVCC" ]; then _v=$(nvcc --version 2>/dev/null | sed -n "s/.*release \([0-9]*\)\..*/\1/p" | head -1); if [ -n "$_v" ] && [ "$_v" -lt 12 ] && [ -x "${_p}/cuda-12.0/bin/nvcc" ]; then export GRIM_CUDA_ROOT="${_p}/cuda-12.0"; export PATH="$GRIM_CUDA_ROOT/bin:$PATH"; export LD_LIBRARY_PATH="$GRIM_CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"; echo "[Anvil] Using project CUDA 12: $GRIM_CUDA_ROOT" >&2; fi; fi; unset _p _v; true'
# Set CUDAToolkit_ROOT and require CUDA 12+. If GRIM_CUDA_ROOT set (by user or ensure script), use that toolkit.
ANVIL_CUDA_ROOT='if [ -n "$GRIM_CUDA_ROOT" ]; then if [ ! -x "$GRIM_CUDA_ROOT/bin/nvcc" ]; then echo "ERROR: GRIM_CUDA_ROOT=$GRIM_CUDA_ROOT but bin/nvcc not found or not executable" >&2; exit 1; fi; export PATH="$GRIM_CUDA_ROOT/bin:$PATH"; CUDAToolkit_ROOT="$GRIM_CUDA_ROOT"; echo "[Anvil] Using GRIM_CUDA_ROOT=$GRIM_CUDA_ROOT"; else NVCC=$(which nvcc 2>/dev/null); if [ -z "$NVCC" ]; then echo "ERROR: nvcc not found. Set GRIM_CUDA_ROOT=/path/to/cuda-12 or use a CUDA 12 module." >&2; exit 1; fi; CUDAToolkit_ROOT=$(dirname "$(dirname "$NVCC")"); fi; NVCC_RELEASE=$("$CUDAToolkit_ROOT/bin/nvcc" --version 2>/dev/null | grep -E "release [0-9]" | head -1); echo "[Anvil] $NVCC_RELEASE"; CUDA_MAJOR=$(echo "$NVCC_RELEASE" | sed -n "s/.*release \([0-9]*\)\..*/\1/p"); if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -lt 12 ]; then echo "ERROR: CUDA $CUDA_MAJOR.x detected. CUTLASS (flash-attention) requires CUDA 12+." >&2; echo "  Install CUDA 12 to project space, then:" >&2; echo "  GRIM_CUDA_ROOT=/anvil/projects/x-cis210085/GRIM/cuda-12.0 ./scripts/run_train_on_anvil.sh --build" >&2; exit 1; fi'
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
# Pin cutlass to CUTLASS 3.4.1 (bbe579a) for flash-attention. CUDA 12+ nvcc required regardless—
# CUDA 11.x triggers stride.hpp "unexpected function type" template error.
# Set GRIM_USE_LATEST_CUTLASS=1 to skip pin (still requires CUDA 12+).
CUTLASS_PIN="bbe579a9e3beb6ea6626d9227ec32d0dae119a49"
if [[ "${GRIM_USE_LATEST_CUTLASS:-}" == "1" ]]; then
  ANVIL_SUBMODULE_INIT="(git submodule deinit -f external/vcpkg 2>/dev/null || true) && git submodule update --init external/flash-attention && (cd external/flash-attention && git submodule update --init csrc/cutlass)"
  ANVIL_CLEAN_BEFORE_BUILD=""
else
  ANVIL_SUBMODULE_INIT="(git submodule deinit -f external/vcpkg 2>/dev/null || true) && git submodule update --init external/flash-attention && (cd external/flash-attention && git submodule update --init csrc/cutlass) && (echo 'Pinning cutlass to 3.4.1 (bbe579a)...' && cd external/flash-attention/csrc/cutlass && git fetch origin && git checkout $CUTLASS_PIN) && (echo 'Applying flash-attention bwd template fix...' && cd external/flash-attention && (git apply -p1 < ../../scripts/patches/flash-attention-bwd-template-fix.patch || (echo '  Patch failed, trying sed fallback...' && sed -i.bak -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, true>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, true, true>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, true, false>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, true, false>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, false>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, false, false>/g' -e 's/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, Has_alibi, Is_even_M, Is_even_K, false, true>/compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_causal, false, Has_alibi, Is_even_M, Is_even_K, false, false, true>/g' csrc/flash_attn/src/flash_bwd_kernel.h && rm -f csrc/flash_attn/src/flash_bwd_kernel.h.bak))) || { echo 'Flash-attention patch failed. Build requires the bwd template fix.'; exit 1; }"
  # Force clean rebuild when cutlass is pinned (headers changed; Make would otherwise use cached objects)
  ANVIL_CLEAN_BEFORE_BUILD="rm -rf $ANVIL_DIR/$BUILD_DIR && "
fi

# Pass GRIM_CUDA_ROOT to remote if set (user-installed CUDA 12 for Anvil)
ANVIL_EXPORT_CUDA="${GRIM_CUDA_ROOT:+export GRIM_CUDA_ROOT='$GRIM_CUDA_ROOT'; }"

# Anvil-G has A100 (sm_80) only. Default to sm_80; override with GRIM_CUDA_ARCH if needed (e.g. 80;86 for A100+RTX).
ANVIL_CUDA_ARCH="export GRIM_CUDA_ARCH=${GRIM_CUDA_ARCH:-80}; "

# --build / --build-training: GRIM-text/training/TrainingLoop CMake → train_gpu
if [[ "$DO_BUILD" == true ]]; then
  echo "Building train_gpu (training loop) on Anvil in $ANVIL_DIR/$BUILD_DIR ..."
  [[ -n "$GRIM_CUDA_ROOT" ]] && echo "  Using GRIM_CUDA_ROOT=$GRIM_CUDA_ROOT (user-installed CUDA 12)"
  ssh anvil "$ANVIL_EXPORT_CUDA $ANVIL_CUDA_ARCH cd $ANVIL_DIR && $ANVIL_SUBMODULE_INIT && $ANVIL_VCPKG_ENSURE && $ANVIL_TRAINING_MANIFEST_ENSURE && cd $ANVIL_DIR/$TRAINING_DIR/TrainingLoop && ${ANVIL_CLEAN_BEFORE_BUILD}mkdir -p build && cd build && $ANVIL_MODULES && $ANVIL_ENSURE_CUDA12 && $ANVIL_CUDA_ROOT && cmake .. $ANVIL_CMAKE_OPTS && make -j \$(nproc) train_gpu"
fi

# --build-grim: GRIM-text/GRIM CMake → grim_text_server (inference)
if [[ "$DO_BUILD_GRIM" == true ]]; then
  echo "Building grim_text_server (GRIM-text inference) on Anvil in $ANVIL_DIR/$GRIM_DIR/build ..."
  [[ -n "$GRIM_CUDA_ROOT" ]] && echo "  Using GRIM_CUDA_ROOT=$GRIM_CUDA_ROOT"
  ssh anvil "$ANVIL_EXPORT_CUDA $ANVIL_CUDA_ARCH cd $ANVIL_DIR && $ANVIL_SUBMODULE_INIT && $ANVIL_VCPKG_ENSURE && cd $ANVIL_DIR/$GRIM_DIR && mkdir -p build && cd build && $ANVIL_MODULES && $ANVIL_ENSURE_CUDA12 && $ANVIL_CUDA_ROOT && cmake .. $ANVIL_CMAKE_OPTS && make -j \$(nproc) grim_text_server"
fi

# --build-grim-exe: repo root CMake → GRIM (main host / adaptive controller for future agent)
if [[ "$DO_BUILD_GRIM_EXE" == true ]]; then
  echo "Building GRIM (main host / grim.exe) on Anvil in $ANVIL_DIR/build ..."
  [[ -n "$GRIM_CUDA_ROOT" ]] && echo "  Using GRIM_CUDA_ROOT=$GRIM_CUDA_ROOT"
  ssh anvil "$ANVIL_EXPORT_CUDA $ANVIL_CUDA_ARCH cd $ANVIL_DIR && $ANVIL_SUBMODULE_INIT && $ANVIL_VCPKG_ENSURE && mkdir -p build && cd build && $ANVIL_MODULES && $ANVIL_ENSURE_CUDA12 && $ANVIL_CUDA_ROOT && cmake .. $ANVIL_CMAKE_OPTS && make -j \$(nproc) GRIM"
fi

if [[ "$USE_SBATCH" == true ]]; then
  echo "Submitting batch job on Anvil (partition=$PARTITION, account=$ACCOUNT) ..."
  # Inlined: default to project CUDA 12 on submit node, then pass env to job via --export=ALL (no remote script needed)
  ANVIL_SBATCH_ENSURE_CUDA12='_p='"$ANVIL_PROJECT_PARENT"'; if [ -z "$GRIM_CUDA_ROOT" ] && NVCC=$(which nvcc 2>/dev/null) && [ -n "$NVCC" ]; then _v=$(nvcc --version 2>/dev/null | sed -n "s/.*release \([0-9]*\)\..*/\1/p" | head -1); if [ -n "$_v" ] && [ "$_v" -lt 12 ] && [ -x "${_p}/cuda-12.0/bin/nvcc" ]; then export GRIM_CUDA_ROOT="${_p}/cuda-12.0"; export PATH="$GRIM_CUDA_ROOT/bin:$PATH"; export LD_LIBRARY_PATH="$GRIM_CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"; fi; fi; unset _p _v'
  ssh anvil "cd $ANVIL_DIR && $ANVIL_SBATCH_ENSURE_CUDA12 && GRIM_SLURM_ACCOUNT=$ACCOUNT GRIM_SLURM_QOS=$QOS sbatch --export=ALL --partition=$PARTITION $SLURM_ACCOUNT_ARGS scripts/train_anvil.sbatch"
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

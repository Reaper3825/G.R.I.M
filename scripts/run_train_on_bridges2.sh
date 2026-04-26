#!/usr/bin/env bash
# Run GRIM-text training on PSC Bridges-2 via SSH.
# Usage: ./scripts/run_train_on_bridges2.sh [--build] [--jobs N] [--config CONFIG] [--sbatch] [--sync TARGET...] [--sync-all|--sync-mcs|--sync-cbs|--sync-crs|--sync-fas]
#
# Prerequisites:
#   - SSH: ssh uwadkins@bridges2.psc.edu (or add to ~/.ssh/config as Host bridges2)
#   - Allocation: Set GRIM_BRIDGES2_ACCOUNT to your ACCESS allocation ID (e.g. abc1234p)
#   - Path: Set GRIM_BRIDGES2_DIR to your repo path, e.g. /ocean/projects/<alloc_id>/<username>/G.R.I.M (default: cis210058p/uwadkins)
#   - Remote git: Each run still git fetch + reset on Bridges-2 unless GRIM_BRIDGES2_SKIP_PULL is set (unrelated to MCS/CBS/FAS).
#   - Data: By default does NOT push merged_verified_cache.jsonl, concept_blocks.jsonl, or curriculum_registry.json,
#     and does NOT run the flash-attention submodule step on --build. Opt in with --sync-all or
#     --sync-mcs|--sync-cbs|--sync-crs|--sync-fas, or env GRIM_BRIDGES2_SYNC_ALL=1 / GRIM_BRIDGES2_SYNC_MCS|CBS|CRS|FAS=1.
#   - Submodules: With --sync-fas / SYNC_FAS, flash-attention is refreshed via bridges2_ensure_flash_attention.sh
#     (no forced submodule update when the expected commits are already checked out on the remote).
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
#   --sync-all       Enable MCS + CBS + CRS + FAS (push caches + flash-attention submodule on --build).
#   --sync-mcs       Push merged_verified_cache.jsonl.
#   --sync-cbs       Push concept_blocks.jsonl (if present locally).
#   --sync-crs       Push curriculum_registry.json (if present locally).
#   --sync-fas       On --build, run scripts/bridges2_ensure_flash_attention.sh (skips forced git pull if FA
#                    gitlink + pinned Cutlass SHA already match remote; still applies patches).
#   --jobs N         make -j N for train_gpu (default 100; override with GRIM_BRIDGES2_MAKE_JOBS).
#   --TD             Run grmt_vocab_metrics_test instead of full training (no GPU needed, uses RM-shared).
#   --UT             Run unigrambyte_self_test instead of full training (needs GPU for GPU decode test).
#   --TT             Run train_tokenizer: full tokenizer training on entire corpus (vocab.bin + .grmt).
#                    Pass --force to rebuild even if files exist: --TT --force

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
CURRICULUM_REGISTRY_PATH="$TRAINING_DATA_DIR/curriculum_registry.json"

# Bridges-2 path: /ocean/projects/<alloc_id>/<username>/G.R.I.M (override with GRIM_BRIDGES2_DIR)
BRIDGES2_DIR="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis210058p/uwadkins/G.R.I.M}"
ACCOUNT="${GRIM_BRIDGES2_ACCOUNT:-cis210058p}"
PARTITION="${PARTITION:-GPU-shared}"
GPU_TYPE="${GPU_TYPE:-h100-80}"
BRIDGES2_MAKE_JOBS="${GRIM_BRIDGES2_MAKE_JOBS:-100}"
DO_BUILD=false
USE_SBATCH=false
DO_INCREMENTAL=false
DO_CLEAN_BUILD=false
FLAG_SYNC_ALL=false
FLAG_SYNC_MCS=false
FLAG_SYNC_CBS=false
FLAG_SYNC_FAS=false
FLAG_SYNC_CRS=false
DO_TD=false
DO_UT=false
DO_TT=false
TT_FORCE=false

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
    --sync-all)       FLAG_SYNC_ALL=true; shift ;;
    --sync-mcs)       FLAG_SYNC_MCS=true; shift ;;
    --sync-cbs)       FLAG_SYNC_CBS=true; shift ;;
    --sync-crs)       FLAG_SYNC_CRS=true; shift ;;
    --sync-fas)       FLAG_SYNC_FAS=true; shift ;;
    --sync)
      shift
      while [[ $# -gt 0 ]] && [[ "$1" != --* ]]; do
        case "$1" in
          all) FLAG_SYNC_ALL=true ;;
          mcs) FLAG_SYNC_MCS=true ;;
          cbs) FLAG_SYNC_CBS=true ;;
          crs) FLAG_SYNC_CRS=true ;;
          fas) FLAG_SYNC_FAS=true ;;
          *)   echo "ERROR: Unknown sync target: $1 (valid: all mcs cbs crs fas)"; exit 1 ;;
        esac
        shift
      done
      ;;
    --TD)             DO_TD=true; shift ;;
    --UT)             DO_UT=true; shift ;;
    --TT)             DO_TT=true; shift ;;
    --force)          TT_FORCE=true; shift ;;
    --jobs)
      [[ $# -lt 2 ]] && { echo "ERROR: --jobs requires a positive integer"; exit 1; }
      [[ "$2" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: --jobs must be a positive integer"; exit 1; }
      BRIDGES2_MAKE_JOBS="$2"
      shift 2
      ;;
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

# MCS/CBS/CRS/FAS = merged cache / concept blocks / curriculum registry / flash-attention (build). Default: skip all; opt in via flags or SYNC_* env.
SKIP_MCS=1
SKIP_CBS=1
SKIP_CRS=1
SKIP_FAS=1
if [[ "$FLAG_SYNC_ALL" == true ]] || [[ "${GRIM_BRIDGES2_SYNC_ALL:-0}" == "1" ]]; then
  SKIP_MCS=0
  SKIP_CBS=0
  SKIP_CRS=0
  SKIP_FAS=0
fi
[[ "$FLAG_SYNC_MCS" == true ]] || [[ "${GRIM_BRIDGES2_SYNC_MCS:-0}" == "1" ]] && SKIP_MCS=0
[[ "$FLAG_SYNC_CBS" == true ]] || [[ "${GRIM_BRIDGES2_SYNC_CBS:-0}" == "1" ]] && SKIP_CBS=0
[[ "$FLAG_SYNC_CRS" == true ]] || [[ "${GRIM_BRIDGES2_SYNC_CRS:-0}" == "1" ]] && SKIP_CRS=0
[[ "$FLAG_SYNC_FAS" == true ]] || [[ "${GRIM_BRIDGES2_SYNC_FAS:-0}" == "1" ]] && SKIP_FAS=0

_assets_mcs=$([[ "$SKIP_MCS" == 0 ]] && echo sync || echo off)
_assets_cbs=$([[ "$SKIP_CBS" == 0 ]] && echo sync || echo off)
_assets_crs=$([[ "$SKIP_CRS" == 0 ]] && echo sync || echo off)
_assets_fas=$([[ "$SKIP_FAS" == 0 ]] && echo sync || echo off)
echo "[Bridges-2] training assets: MCS=$_assets_mcs  CBS=$_assets_cbs  CRS=$_assets_crs  FAS=$_assets_fas  (default off — use --sync-all or --sync-{mcs,cbs,crs,fas})"

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
REMOTE_CURRICULUM_REGISTRY="$REMOTE_DATA/curriculum_registry.json"
CACHE_PATH_EXPANDED="${CACHE_PATH/#\~/$HOME}"
CONCEPT_BLOCKS_PATH_EXPANDED="${CONCEPT_BLOCKS_PATH/#\~/$HOME}"
CURRICULUM_REGISTRY_PATH_EXPANDED="${CURRICULUM_REGISTRY_PATH/#\~/$HOME}"

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
# CUDAToolkit_ROOT for cmake comes from GRIM_CUDA_ROOT (set by ensure_cuda12_for_training.sh). Do not inline
# nvcc discovery here — unescaped \" and \$( in a var would break the local ssh \"...\" string (parse errors like SH_OPTS).

# vcpkg
BRIDGES2_VCPKG="${GRIM_VCPKG_ROOT:-$BRIDGES2_DIR/vcpkg}"
VCPKG_TOOLCHAIN="$BRIDGES2_VCPKG/scripts/buildsystems/vcpkg.cmake"
TRAINING_VCPKG_JSON='{"name":"grim-training","version-string":"0.1.0","dependencies":["nlohmann-json","flatbuffers"]}'
BRIDGES2_VCPKG_ENSURE="true"
if [[ -z "${GRIM_VCPKG_ROOT:-}" ]]; then
  BRIDGES2_VCPKG_ENSURE="if [ ! -f \"$BRIDGES2_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake\" ]; then (cd \"$BRIDGES2_DIR\" && git clone https://github.com/Microsoft/vcpkg.git vcpkg && cd vcpkg && ./bootstrap-vcpkg.sh) || exit 1; fi"
fi
BRIDGES2_MANIFEST_ENSURE="mkdir -p $BRIDGES2_DIR/$TRAINING_DIR && [ -f $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json ] || printf '%s' '$TRAINING_VCPKG_JSON' > $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json"

# Flash-attention + Cutlass: remote script skips forced submodule pull when gitlink + pin already match.
CUTLASS_PIN="bbe579a9e3beb6ea6626d9227ec32d0dae119a49"
if [[ "$SKIP_FAS" == "1" ]]; then
  BRIDGES2_SUBMODULE="true"
  BRIDGES2_CLEAN=""
else
  BRIDGES2_SUBMODULE="export CUTLASS_PIN='${CUTLASS_PIN}' GRIM_USE_LATEST_CUTLASS='${GRIM_USE_LATEST_CUTLASS:-0}'; bash scripts/bridges2_ensure_flash_attention.sh ."
  if [[ "${GRIM_USE_LATEST_CUTLASS:-}" == "1" ]]; then
    BRIDGES2_CLEAN=""
  else
    [[ "$DO_INCREMENTAL" == true ]] && [[ "$DO_CLEAN_BUILD" != true ]] && BRIDGES2_CLEAN="" || BRIDGES2_CLEAN="rm -rf $BRIDGES2_DIR/$BUILD_DIR && "
  fi
fi

# CUDA arch: sm_90 for H100, sm_80 for V100
if [[ "$GPU_TYPE" == "h100-80" ]]; then
  BRIDGES2_CUDA_ARCH="export GRIM_CUDA_ARCH=90; "
else
  BRIDGES2_CUDA_ARCH="export GRIM_CUDA_ARCH=80; "
fi

BRIDGES2_CMAKE_OPTS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN -DVCPKG_TARGET_TRIPLET=x64-linux -DVCPKG_MANIFEST_DIR=$BRIDGES2_DIR/$TRAINING_DIR"

# --build
# Default to building train_gpu PLUS train_tokenizer because train_gpu spawns
# train_tokenizer as a subprocess at runtime (see Subprocess/tokenizer_subprocess.cpp);
# rebuilding train_gpu alone leaves a stale train_tokenizer that's pinned to whatever
# IPC contract was current the last time it was built. The two binaries share an
# IPC schema (--status-file / --config flags + status JSON envelope), so any time
# train_gpu is rebuilt train_tokenizer MUST be rebuilt too or the parent will fail
# with "subprocess exited but did not write a status file" the first time it spawns
# the child. Sub-target builds (--TD/--UT/--TT) keep their single-target footprint.
BUILD_TARGET="train_gpu train_tokenizer"
if [[ "$DO_TD" == true ]]; then
  BUILD_TARGET="grmt_vocab_metrics_test"
elif [[ "$DO_UT" == true ]]; then
  BUILD_TARGET="unigrambyte_self_test"
elif [[ "$DO_TT" == true ]]; then
  BUILD_TARGET="train_tokenizer"
fi

if [[ "$DO_BUILD" == true ]]; then
  echo "Building $BUILD_TARGET on Bridges-2 ($BRIDGES2_DIR/$BUILD_DIR)..."
  echo "  GPU type: $GPU_TYPE, CUDA arch: $([ "$GPU_TYPE" == "h100-80" ] && echo sm_90 || echo sm_80), make -j $BRIDGES2_MAKE_JOBS"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "BRIDGES2_DIR=$BRIDGES2_DIR; $BRIDGES2_CUDA_ARCH cd \$BRIDGES2_DIR && $BRIDGES2_SUBMODULE && $BRIDGES2_VCPKG_ENSURE && $BRIDGES2_MANIFEST_ENSURE && cd \$BRIDGES2_DIR/$TRAINING_DIR/TrainingLoop && ${BRIDGES2_CLEAN}mkdir -p build && cd build && $BRIDGES2_MODULES && $BRIDGES2_ENSURE_CUDA12 && cmake .. $BRIDGES2_CMAKE_OPTS -DCUDAToolkit_ROOT=\$GRIM_CUDA_ROOT && make -j $BRIDGES2_MAKE_JOBS $BUILD_TARGET"
fi

: # Transfer data
if [[ "$SKIP_MCS" == "1" ]]; then
  :
else
  if [[ ! -f "$CACHE_PATH_EXPANDED" ]]; then
    echo "ERROR: merged_verified_cache.jsonl not found at $CACHE_PATH_EXPANDED"
    exit 1
  fi
  echo "Transferring merged_verified_cache.jsonl..."
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "mkdir -p $REMOTE_DATA"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $REMOTE_CACHE" < "$CACHE_PATH_EXPANDED"
  echo "  -> $REMOTE_CACHE"
fi

if [[ "$SKIP_CBS" == "1" ]]; then
  :
elif [[ -f "$CONCEPT_BLOCKS_PATH_EXPANDED" ]]; then
  echo "Transferring concept_blocks.jsonl..."
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "mkdir -p $REMOTE_DATA"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $REMOTE_CONCEPT_BLOCKS" < "$CONCEPT_BLOCKS_PATH_EXPANDED"
  echo "  -> $REMOTE_CONCEPT_BLOCKS"
else
  echo "Skipping concept_blocks.jsonl (not found at $CONCEPT_BLOCKS_PATH_EXPANDED)."
  echo "  DataLoader will use cache-only curriculum; add the file locally to ship UltraChat/stem blocks."
fi

if [[ "$SKIP_CRS" == "1" ]]; then
  :
elif [[ -f "$CURRICULUM_REGISTRY_PATH_EXPANDED" ]]; then
  echo "Transferring curriculum_registry.json..."
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "mkdir -p $REMOTE_DATA"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $REMOTE_CURRICULUM_REGISTRY" < "$CURRICULUM_REGISTRY_PATH_EXPANDED"
  echo "  -> $REMOTE_CURRICULUM_REGISTRY"
else
  echo "Skipping curriculum_registry.json (not found at $CURRICULUM_REGISTRY_PATH_EXPANDED)."
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
  # Batch script defaults to ai_config.json at repo root (same as transferred file). Pass through --config when set.
  SBATCH_EXPORT="ALL,GRIM_BRIDGES2_DIR=$BRIDGES2_DIR"
  if [[ "$CONFIG" != "../../../../ai_config.json" ]] && [[ "$CONFIG" != "ai_config.json" ]]; then
    SBATCH_EXPORT="$SBATCH_EXPORT,GRIM_TRAIN_CONFIG=$CONFIG"
  fi
  SUBMIT_OUT=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && sbatch --export=$SBATCH_EXPORT --output=$BRIDGES2_DIR/logs/train_%j.out --error=$BRIDGES2_DIR/logs/train_%j.err $SLURM_MAIL_ARGS -p $PARTITION $SLURM_ACCOUNT_ARGS --gpus=$GPU_TYPE:1 -t 24:00:00 scripts/train_bridges2.sbatch")
  echo "$SUBMIT_OUT"
  exit 0
fi

# Interactive run
if [[ "$DO_UT" == true ]]; then
  # --UT: run unigrambyte_self_test (needs GPU for GPU decode test)
  REMOTE_UT_EXE="$BRIDGES2_DIR/$BUILD_DIR/unigrambyte_self_test"
  UT_RUN_WRAPPER="bash -c 'source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || true; export GRIM_PROJECT_DIR=\"$BRIDGES2_DIR\"; source \"$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; exec \"$REMOTE_UT_EXE\"'"
  echo "Running unigrambyte_self_test on Bridges-2 (partition=$PARTITION, gpu=$GPU_TYPE)..."
  UT_SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS --gres=gpu:$GPU_TYPE:1 -t 0:10:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $UT_SRUN_ARGS $UT_RUN_WRAPPER"
  else
    ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $UT_SRUN_ARGS $UT_RUN_WRAPPER"
  fi
elif [[ "$DO_TT" == true ]]; then
  # --TT: run train_tokenizer (full tokenizer training on entire corpus)
  REMOTE_TT_EXE="$BRIDGES2_DIR/$BUILD_DIR/train_tokenizer"
  TT_FORCE_ARG=""
  if [[ "$TT_FORCE" == true ]]; then
    TT_FORCE_ARG="--force"
  fi
  TT_RUN_WRAPPER="bash -c 'source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || true; export GRIM_PROJECT_DIR=\"$BRIDGES2_DIR\"; source \"$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; cd \"$BRIDGES2_DIR\" && exec \"$REMOTE_TT_EXE\" $TT_FORCE_ARG'"
  echo "Running train_tokenizer on Bridges-2 (partition=$PARTITION, gpu=$GPU_TYPE)..."
  if [[ "$TT_FORCE" == true ]]; then
    echo "  Mode: FORCE REBUILD"
  fi
  TT_SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS --gres=gpu:$GPU_TYPE:1 -t 2:00:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $TT_SRUN_ARGS $TT_RUN_WRAPPER"
  else
    ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $TT_SRUN_ARGS $TT_RUN_WRAPPER"
  fi
elif [[ "$DO_TD" == true ]]; then
  # --TD: run grmt_vocab_metrics_test on RM-shared (no GPU needed)
  REMOTE_TD_EXE="$BRIDGES2_DIR/$BUILD_DIR/grmt_vocab_metrics_test"
  REMOTE_VOCAB="$BRIDGES2_DIR/resources/models/GRIM-text/training/data/vocab.bin"
  REMOTE_GRMT="$BRIDGES2_DIR/resources/models/GRIM-text/training/data/training_data.grmt"
  TD_RUN_WRAPPER="bash -c 'source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || true; export GRIM_PROJECT_DIR=\"$BRIDGES2_DIR\"; source \"$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; exec \"$REMOTE_TD_EXE\" --vocab \"$REMOTE_VOCAB\" --grmt \"$REMOTE_GRMT\"'"
  echo "Running grmt_vocab_metrics_test on Bridges-2 (partition=RM-shared, no GPU)..."
  TD_SRUN_ARGS="-p RM-shared $SLURM_ACCOUNT_ARGS --ntasks=1 --cpus-per-task=4 --mem-per-cpu=2000M -t 0:30:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $TD_SRUN_ARGS $TD_RUN_WRAPPER"
  else
    ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $TD_SRUN_ARGS $TD_RUN_WRAPPER"
  fi
else
  # Normal: srun train_gpu (load cuda module + set LD_LIBRARY_PATH so compute node finds libcudart)
  BRIDGES2_RUN_WRAPPER="bash -c 'source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || true; export GRIM_PROJECT_DIR=\"$BRIDGES2_DIR\"; source \"$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; exec \"$REMOTE_EXE\" --config \"$BRIDGES2_DIR/ai_config.json\"'"
  echo "Running train_gpu on Bridges-2 (partition=$PARTITION, gpu=$GPU_TYPE)..."
  SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS --gres=gpu:$GPU_TYPE:1 -t 24:00:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $SRUN_ARGS $BRIDGES2_RUN_WRAPPER"
  else
    ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $SRUN_ARGS $BRIDGES2_RUN_WRAPPER"
  fi
fi

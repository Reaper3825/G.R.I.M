#!/usr/bin/env bash
# Run GRIM-text training on SDSC Expanse via SSH.
# Usage: ./scripts/run_train_on_expanse.sh [--build] [--jobs N] [--config CONFIG] [--sbatch] [--sync TARGET...] [--sync-all|--sync-mcs|--sync-cbs|--sync-crs|--sync-fas]
#
# Prerequisites:
#   - SSH: ssh uwadkins@login.expanse.sdsc.edu (or add to ~/.ssh/config as Host expanse)
#   - Allocation: Set GRIM_EXPANSE_ACCOUNT to your Expanse GPU project/account (check with `module load sdsc && expanse-client user -r expanse_gpu`)
#   - Path: Set GRIM_EXPANSE_DIR to your repo path, e.g. /expanse/lustre/projects/<project>/<username>/G.R.I.M
#     If GRIM_EXPANSE_DIR is unset, the script uses /expanse/lustre/projects/$GRIM_EXPANSE_ACCOUNT/$GRIM_EXPANSE_USER/G.R.I.M.
#   - Remote git: Each run still git fetch + reset on Expanse unless GRIM_EXPANSE_SKIP_PULL is set (unrelated to MCS/CBS/FAS).
#   - Data: By default does NOT push merged_verified_cache.jsonl, concept_blocks.jsonl, or curriculum_registry.json,
#     and does NOT run the flash-attention submodule step on --build. Opt in with --sync-all or
#     --sync-mcs|--sync-cbs|--sync-crs|--sync-fas, or env GRIM_EXPANSE_SYNC_ALL=1 / GRIM_EXPANSE_SYNC_MCS|CBS|CRS|FAS=1.
#   - Submodules: With --sync-fas / SYNC_FAS, flash-attention is refreshed via bridges2_ensure_flash_attention.sh
#     (no forced submodule update when the expected commits are already checked out on the remote).
#   - vcpkg: Script clones to GRIM_EXPANSE_DIR/vcpkg if missing
#   - CUDA 12+ for training (flash-attention). Expanse GPU nodes require `module load gpu` before CUDA modules are visible.
#
# Expanse GPU partitions: gpu-shared (1-3 GPUs, faster queue) or gpu (full GPU node).
# GPU type: NVIDIA V100 32GB. Default CUDA arch: sm_70.
#
# Options:
#   --build          Build train_gpu before running.
#   --config X       Config (default: ai_config.json).
#   --sbatch         Submit batch job (scripts/train_expanse.sbatch).
#   --partition P    gpu-shared (default) or gpu.
#   --gpu-type T     v100-32 (default). Used for CUDA arch selection only; Expanse Slurm uses --gpus=1.
#   --account A      Override GRIM_EXPANSE_ACCOUNT.
#   --sync-all       Enable MCS + CBS + CRS + FAS (push caches + flash-attention submodule on --build).
#   --sync-mcs       Push merged_verified_cache.jsonl.
#   --sync-cbs       Push concept_blocks.jsonl (if present locally).
#   --sync-crs       Push curriculum_registry.json (if present locally).
#   --sync-fas       On --build, run scripts/bridges2_ensure_flash_attention.sh (skips forced git pull if FA
#                    gitlink + pinned Cutlass SHA already match remote; still applies patches).
#   --jobs N         make -j N for train_gpu (default 40; override with GRIM_EXPANSE_MAKE_JOBS).
#   --TD             Run grmt_vocab_metrics_test instead of full training (no GPU needed, uses shared).
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

EXPANSE_USER="${GRIM_EXPANSE_USER:-uwadkins}"
ACCOUNT="${GRIM_EXPANSE_ACCOUNT:-}"
PARTITION="${PARTITION:-gpu-shared}"
GPU_TYPE="${GPU_TYPE:-v100-32}"
EXPANSE_MAKE_JOBS="${GRIM_EXPANSE_MAKE_JOBS:-40}"
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
      EXPANSE_MAKE_JOBS="$2"
      shift 2
      ;;
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -n "${GRIM_EXPANSE_DIR:-}" ]]; then
  EXPANSE_DIR="$GRIM_EXPANSE_DIR"
elif [[ -n "$ACCOUNT" ]]; then
  EXPANSE_DIR="/expanse/lustre/projects/$ACCOUNT/$EXPANSE_USER/G.R.I.M"
else
  EXPANSE_DIR=""
fi

# MCS/CBS/CRS/FAS = merged cache / concept blocks / curriculum registry / flash-attention (build). Default: skip all; opt in via flags or SYNC_* env.
SKIP_MCS=1
SKIP_CBS=1
SKIP_CRS=1
SKIP_FAS=1
if [[ "$FLAG_SYNC_ALL" == true ]] || [[ "${GRIM_EXPANSE_SYNC_ALL:-0}" == "1" ]]; then
  SKIP_MCS=0
  SKIP_CBS=0
  SKIP_CRS=0
  SKIP_FAS=0
fi
[[ "$FLAG_SYNC_MCS" == true ]] || [[ "${GRIM_EXPANSE_SYNC_MCS:-0}" == "1" ]] && SKIP_MCS=0
[[ "$FLAG_SYNC_CBS" == true ]] || [[ "${GRIM_EXPANSE_SYNC_CBS:-0}" == "1" ]] && SKIP_CBS=0
[[ "$FLAG_SYNC_CRS" == true ]] || [[ "${GRIM_EXPANSE_SYNC_CRS:-0}" == "1" ]] && SKIP_CRS=0
[[ "$FLAG_SYNC_FAS" == true ]] || [[ "${GRIM_EXPANSE_SYNC_FAS:-0}" == "1" ]] && SKIP_FAS=0

_assets_mcs=$([[ "$SKIP_MCS" == 0 ]] && echo sync || echo off)
_assets_cbs=$([[ "$SKIP_CBS" == 0 ]] && echo sync || echo off)
_assets_crs=$([[ "$SKIP_CRS" == 0 ]] && echo sync || echo off)
_assets_fas=$([[ "$SKIP_FAS" == 0 ]] && echo sync || echo off)
echo "[Expanse] training assets: MCS=$_assets_mcs  CBS=$_assets_cbs  CRS=$_assets_crs  FAS=$_assets_fas  (default off — use --sync-all or --sync-{mcs,cbs,crs,fas})"

# Validate. Expanse projects vary, so the account is intentionally not hard-coded.
if [[ -z "$ACCOUNT" ]]; then
  echo "ERROR: Set GRIM_EXPANSE_ACCOUNT to your Expanse GPU project/account, or pass --account."
  echo "  Check accounts on Expanse with: module load sdsc && expanse-client user -r expanse_gpu"
  exit 1
fi
if [[ -z "$EXPANSE_DIR" ]]; then
  echo "ERROR: Set GRIM_EXPANSE_DIR to your Expanse repo path."
  echo "  Example: export GRIM_EXPANSE_DIR=/expanse/lustre/projects/$ACCOUNT/$EXPANSE_USER/G.R.I.M"
  exit 1
fi

REMOTE_TRAINING="$EXPANSE_DIR/$TRAINING_DIR"
REMOTE_EXE="$EXPANSE_DIR/$EXE"
REMOTE_DATA="$REMOTE_TRAINING/data"
REMOTE_CACHE="$REMOTE_DATA/merged_verified_cache.jsonl"
REMOTE_CONCEPT_BLOCKS="$REMOTE_DATA/concept_blocks.jsonl"
REMOTE_CURRICULUM_REGISTRY="$REMOTE_DATA/curriculum_registry.json"
CACHE_PATH_EXPANDED="${CACHE_PATH/#\~/$HOME}"
CONCEPT_BLOCKS_PATH_EXPANDED="${CONCEPT_BLOCKS_PATH/#\~/$HOME}"
CURRICULUM_REGISTRY_PATH_EXPANDED="${CURRICULUM_REGISTRY_PATH/#\~/$HOME}"

# SSH target: expanse or login.expanse.sdsc.edu
EXPANSE_SSH="${GRIM_EXPANSE_SSH:-expanse}"
if [[ "$EXPANSE_SSH" == "expanse" ]] && ! grep -q "Host expanse" ~/.ssh/config 2>/dev/null; then
  EXPANSE_SSH="$EXPANSE_USER@login.expanse.sdsc.edu"
fi

# SLURM
SLURM_ACCOUNT_ARGS="-A $ACCOUNT"
GRIM_SLURM_MAIL="${GRIM_SLURM_MAIL:-}"
[[ -n "$GRIM_SLURM_MAIL" ]] && SLURM_MAIL_ARGS="--mail-type=BEGIN,END,FAIL --mail-user=$GRIM_SLURM_MAIL" || SLURM_MAIL_ARGS=""
GPU_REQUEST="${GRIM_EXPANSE_GPU_REQUEST:---gpus=1}"

# One long-lived SSH using a script-unique socket in /tmp (avoids ~/.ssh permission issues)
EXPANSE_CTRL="/tmp/cm-grim-expanse-$$"
if ! ssh -f -N -M -S "$EXPANSE_CTRL" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$EXPANSE_SSH"; then
  echo "SSH to Expanse failed. Try: ssh $EXPANSE_USER@login.expanse.sdsc.edu"
  exit 1
fi
trap 'ssh -S "$EXPANSE_CTRL" -O exit "$EXPANSE_SSH" 2>/dev/null; rm -f "$EXPANSE_CTRL"' EXIT
EXPANSE_SSH_OPTS="-S $EXPANSE_CTRL -o ControlMaster=no"

# Sync repo
EXPANSE_SYNCED=false
if [[ -z "${GRIM_EXPANSE_SKIP_PULL:-}" ]]; then
  echo "Syncing Expanse repo..."
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && git fetch origin && git reset --hard origin/\$(git rev-parse --abbrev-ref HEAD)"
  echo "  Done."
  EXPANSE_SYNCED=true
fi

if [[ "$DO_BUILD" == true ]] && [[ "$EXPANSE_SYNCED" == true ]] && [[ "$DO_CLEAN_BUILD" != true ]]; then
  DO_INCREMENTAL=true
fi

# Expanse modules: GPU node module paths are enabled by module load gpu.
EXPANSE_MODULES="source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; module load gpu 2>/dev/null || true; module load slurm 2>/dev/null || true; module load cmake 2>/dev/null || true; module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda/12.0 2>/dev/null || true"
EXPANSE_CPU_MODULES="source /etc/profile.d/modules.sh 2>/dev/null || true; module purge 2>/dev/null || true; module load cpu 2>/dev/null || true; module load slurm 2>/dev/null || true; module load cuda 2>/dev/null || true"
# Default to project CUDA 12 when 11.x is detected (CUTLASS requires 12+). Uses GRIM_PROJECT_DIR=$EXPANSE_DIR to find cuda-12.0.
EXPANSE_ENSURE_CUDA12="export GRIM_PROJECT_DIR=\"$EXPANSE_DIR\"; source \"$EXPANSE_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true"

# vcpkg
EXPANSE_VCPKG="${GRIM_VCPKG_ROOT:-$EXPANSE_DIR/vcpkg}"
VCPKG_TOOLCHAIN="$EXPANSE_VCPKG/scripts/buildsystems/vcpkg.cmake"
TRAINING_VCPKG_JSON='{"name":"grim-training","version-string":"0.1.0","dependencies":["nlohmann-json","flatbuffers"]}'
EXPANSE_VCPKG_ENSURE="true"
if [[ -z "${GRIM_VCPKG_ROOT:-}" ]]; then
  EXPANSE_VCPKG_ENSURE="if [ ! -f \"$EXPANSE_DIR/vcpkg/scripts/buildsystems/vcpkg.cmake\" ]; then (cd \"$EXPANSE_DIR\" && git clone https://github.com/Microsoft/vcpkg.git vcpkg && cd vcpkg && ./bootstrap-vcpkg.sh) || exit 1; fi"
fi
EXPANSE_MANIFEST_ENSURE="mkdir -p $EXPANSE_DIR/$TRAINING_DIR && [ -f $EXPANSE_DIR/$TRAINING_DIR/vcpkg.json ] || printf '%s' '$TRAINING_VCPKG_JSON' > $EXPANSE_DIR/$TRAINING_DIR/vcpkg.json"

# Flash-attention + Cutlass: remote script skips forced submodule pull when gitlink + pin already match.
CUTLASS_PIN="bbe579a9e3beb6ea6626d9227ec32d0dae119a49"
if [[ "$SKIP_FAS" == "1" ]]; then
  EXPANSE_SUBMODULE="true"
  EXPANSE_CLEAN=""
else
  EXPANSE_SUBMODULE="export CUTLASS_PIN='${CUTLASS_PIN}' GRIM_USE_LATEST_CUTLASS='${GRIM_USE_LATEST_CUTLASS:-0}'; bash scripts/bridges2_ensure_flash_attention.sh ."
  if [[ "${GRIM_USE_LATEST_CUTLASS:-}" == "1" ]]; then
    EXPANSE_CLEAN=""
  else
    [[ "$DO_INCREMENTAL" == true ]] && [[ "$DO_CLEAN_BUILD" != true ]] && EXPANSE_CLEAN="" || EXPANSE_CLEAN="rm -rf $EXPANSE_DIR/$BUILD_DIR && "
  fi
fi

# CUDA arch: sm_70 for Expanse V100s. Keep a small override table for testing builds elsewhere.
case "$GPU_TYPE" in
  h100*|H100*) EXPANSE_CUDA_ARCH="export GRIM_CUDA_ARCH=90; "; CUDA_ARCH_LABEL="sm_90" ;;
  a100*|A100*) EXPANSE_CUDA_ARCH="export GRIM_CUDA_ARCH=80; "; CUDA_ARCH_LABEL="sm_80" ;;
  v100*|V100*) EXPANSE_CUDA_ARCH="export GRIM_CUDA_ARCH=70; "; CUDA_ARCH_LABEL="sm_70" ;;
  *)           EXPANSE_CUDA_ARCH="export GRIM_CUDA_ARCH=70; "; CUDA_ARCH_LABEL="sm_70" ;;
esac

EXPANSE_CMAKE_OPTS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$VCPKG_TOOLCHAIN -DVCPKG_TARGET_TRIPLET=x64-linux -DVCPKG_MANIFEST_DIR=$EXPANSE_DIR/$TRAINING_DIR"

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
  echo "Building $BUILD_TARGET on Expanse ($EXPANSE_DIR/$BUILD_DIR)..."
  echo "  GPU type: $GPU_TYPE, CUDA arch: $CUDA_ARCH_LABEL, make -j $EXPANSE_MAKE_JOBS"
  BUILD_SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS $GPU_REQUEST -t 2:00:00"
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "EXPANSE_DIR=$EXPANSE_DIR; $EXPANSE_CUDA_ARCH cd \$EXPANSE_DIR && $EXPANSE_SUBMODULE && $EXPANSE_VCPKG_ENSURE && $EXPANSE_MANIFEST_ENSURE && cd \$EXPANSE_DIR/$TRAINING_DIR/TrainingLoop && ${EXPANSE_CLEAN}mkdir -p build && cd build && srun $BUILD_SRUN_ARGS bash -lc '$EXPANSE_MODULES && $EXPANSE_ENSURE_CUDA12 && cmake .. $EXPANSE_CMAKE_OPTS -DCUDAToolkit_ROOT=\$GRIM_CUDA_ROOT && make -j $EXPANSE_MAKE_JOBS $BUILD_TARGET'"
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
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "mkdir -p $REMOTE_DATA"
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cat > $REMOTE_CACHE" < "$CACHE_PATH_EXPANDED"
  echo "  -> $REMOTE_CACHE"
fi

if [[ "$SKIP_CBS" == "1" ]]; then
  :
elif [[ -f "$CONCEPT_BLOCKS_PATH_EXPANDED" ]]; then
  echo "Transferring concept_blocks.jsonl..."
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "mkdir -p $REMOTE_DATA"
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cat > $REMOTE_CONCEPT_BLOCKS" < "$CONCEPT_BLOCKS_PATH_EXPANDED"
  echo "  -> $REMOTE_CONCEPT_BLOCKS"
else
  echo "Skipping concept_blocks.jsonl (not found at $CONCEPT_BLOCKS_PATH_EXPANDED)."
  echo "  DataLoader will use cache-only curriculum; add the file locally to ship UltraChat/stem blocks."
fi

if [[ "$SKIP_CRS" == "1" ]]; then
  :
elif [[ -f "$CURRICULUM_REGISTRY_PATH_EXPANDED" ]]; then
  echo "Transferring curriculum_registry.json..."
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "mkdir -p $REMOTE_DATA"
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cat > $REMOTE_CURRICULUM_REGISTRY" < "$CURRICULUM_REGISTRY_PATH_EXPANDED"
  echo "  -> $REMOTE_CURRICULUM_REGISTRY"
else
  echo "Skipping curriculum_registry.json (not found at $CURRICULUM_REGISTRY_PATH_EXPANDED)."
fi

# Transfer ai_config.json
if [[ -f "$REPO_ROOT/ai_config.json" ]]; then
  echo "Transferring ai_config.json..."
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cat > $EXPANSE_DIR/ai_config.json" < "$REPO_ROOT/ai_config.json"
fi

# Batch job
if [[ "$USE_SBATCH" == true ]]; then
  SBATCH_PATH="$REPO_ROOT/scripts/train_expanse.sbatch"
  if [[ ! -f "$SBATCH_PATH" ]]; then
    echo "ERROR: scripts/train_expanse.sbatch not found."
    echo "  Create it with: #SBATCH -p $PARTITION, #SBATCH --gpus=1, and submit with -A $ACCOUNT"
    exit 1
  fi
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "mkdir -p $EXPANSE_DIR/scripts $EXPANSE_DIR/logs"
  ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cat > $EXPANSE_DIR/scripts/train_expanse.sbatch" < "$SBATCH_PATH"
  echo "Submitting batch job (partition=$PARTITION, gpu_request=$GPU_REQUEST)..."
  # Batch script defaults to ai_config.json at repo root (same as transferred file). Pass through --config when set.
  SBATCH_EXPORT="ALL,GRIM_EXPANSE_DIR=$EXPANSE_DIR"
  if [[ "$CONFIG" != "../../../../ai_config.json" ]] && [[ "$CONFIG" != "ai_config.json" ]]; then
    SBATCH_EXPORT="$SBATCH_EXPORT,GRIM_TRAIN_CONFIG=$CONFIG"
  fi
  SUBMIT_OUT=$(ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && sbatch --export=$SBATCH_EXPORT --output=$EXPANSE_DIR/logs/train_%j.out --error=$EXPANSE_DIR/logs/train_%j.err $SLURM_MAIL_ARGS -p $PARTITION $SLURM_ACCOUNT_ARGS $GPU_REQUEST -t 24:00:00 scripts/train_expanse.sbatch")
  echo "$SUBMIT_OUT"
  exit 0
fi

# Interactive run
if [[ "$DO_UT" == true ]]; then
  # --UT: run unigrambyte_self_test (needs GPU for GPU decode test)
  REMOTE_UT_EXE="$EXPANSE_DIR/$BUILD_DIR/unigrambyte_self_test"
  UT_RUN_WRAPPER="bash -c '$EXPANSE_MODULES; $EXPANSE_ENSURE_CUDA12; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; exec \"$REMOTE_UT_EXE\"'"
  echo "Running unigrambyte_self_test on Expanse (partition=$PARTITION, gpu_request=$GPU_REQUEST)..."
  UT_SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS $GPU_REQUEST -t 0:10:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $UT_SRUN_ARGS $UT_RUN_WRAPPER"
  else
    ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $UT_SRUN_ARGS $UT_RUN_WRAPPER"
  fi
elif [[ "$DO_TT" == true ]]; then
  # --TT: run train_tokenizer (full tokenizer training on entire corpus)
  REMOTE_TT_EXE="$EXPANSE_DIR/$BUILD_DIR/train_tokenizer"
  TT_FORCE_ARG=""
  if [[ "$TT_FORCE" == true ]]; then
    TT_FORCE_ARG="--force"
  fi
  TT_RUN_WRAPPER="bash -c '$EXPANSE_MODULES; $EXPANSE_ENSURE_CUDA12; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; cd \"$EXPANSE_DIR\" && exec \"$REMOTE_TT_EXE\" $TT_FORCE_ARG'"
  echo "Running train_tokenizer on Expanse (partition=$PARTITION, gpu_request=$GPU_REQUEST)..."
  if [[ "$TT_FORCE" == true ]]; then
    echo "  Mode: FORCE REBUILD"
  fi
  TT_SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS $GPU_REQUEST -t 2:00:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $TT_SRUN_ARGS $TT_RUN_WRAPPER"
  else
    ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $TT_SRUN_ARGS $TT_RUN_WRAPPER"
  fi
elif [[ "$DO_TD" == true ]]; then
  # --TD: run grmt_vocab_metrics_test on shared (no GPU needed)
  REMOTE_TD_EXE="$EXPANSE_DIR/$BUILD_DIR/grmt_vocab_metrics_test"
  REMOTE_VOCAB="$EXPANSE_DIR/resources/models/GRIM-text/training/data/vocab.bin"
  REMOTE_GRMT="$EXPANSE_DIR/resources/models/GRIM-text/training/data/training_data.grmt"
  TD_RUN_WRAPPER="bash -c '$EXPANSE_CPU_MODULES; $EXPANSE_ENSURE_CUDA12; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; exec \"$REMOTE_TD_EXE\" --vocab \"$REMOTE_VOCAB\" --grmt \"$REMOTE_GRMT\"'"
  echo "Running grmt_vocab_metrics_test on Expanse (partition=shared, no GPU)..."
  TD_SRUN_ARGS="-p shared $SLURM_ACCOUNT_ARGS --ntasks=1 --cpus-per-task=4 --mem=8G -t 0:30:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $TD_SRUN_ARGS $TD_RUN_WRAPPER"
  else
    ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $TD_SRUN_ARGS $TD_RUN_WRAPPER"
  fi
else
  # Normal: srun train_gpu (load Expanse GPU module path + set LD_LIBRARY_PATH so compute node finds libcudart)
  EXPANSE_RUN_WRAPPER="bash -c '$EXPANSE_MODULES; $EXPANSE_ENSURE_CUDA12; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; exec \"$REMOTE_EXE\" --config \"$EXPANSE_DIR/ai_config.json\"'"
  echo "Running train_gpu on Expanse (partition=$PARTITION, gpu_request=$GPU_REQUEST)..."
  SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS $GPU_REQUEST -t 24:00:00 --pty"
  if [[ -t 0 ]]; then
    ssh -t $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $SRUN_ARGS $EXPANSE_RUN_WRAPPER"
  else
    ssh $EXPANSE_SSH_OPTS "$EXPANSE_SSH" "cd $EXPANSE_DIR && srun $SRUN_ARGS $EXPANSE_RUN_WRAPPER"
  fi
fi

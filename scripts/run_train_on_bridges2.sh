#!/usr/bin/env bash
# Run GRIM-text training on PSC Bridges-2 via SSH.
# Usage: ./scripts/run_train_on_bridges2.sh [--build] [--jobs N] [--sbatch] [--sync TARGET...] [--sync-all|--sync-mcs|--sync-cbs|--sync-crs|--sync-fas] [--pull-vocab] [--pull-logs] [--allow-vcpkg-tool-downloads]
#
# Prerequisites:
#   - SSH: ssh uwadkins@bridges2.psc.edu (or add to ~/.ssh/config as Host bridges2)
#   - Allocation: Set GRIM_BRIDGES2_ACCOUNT to your ACCESS allocation ID (e.g. abc1234p)
#   - Path: Set GRIM_BRIDGES2_DIR to your repo path, e.g. /ocean/projects/<alloc_id>/<username>/G.R.I.M (default: cis210058p/uwadkins)
#   - Remote git: Each run still git fetch + reset on Bridges-2 unless GRIM_BRIDGES2_SKIP_PULL is set (unrelated to MCS/CBS/FAS).
#   - Data: By default does NOT push merged_verified_cache.jsonl, concept_blocks.jsonl, or curriculum_registry.json,
#     and does NOT run the flash-attention submodule step on --build. Opt in with --sync-all or
#     --sync-mcs|--sync-cbs|--sync-crs|--sync-fas, or env GRIM_BRIDGES2_SYNC_ALL=1 / GRIM_BRIDGES2_SYNC_MCS|CBS|CRS|FAS=1.
#   - Large transfers: default auto mode uses rsync --compress when available on both ends, then zstd, then gzip.
#     Override with GRIM_BRIDGES2_TRANSFER_METHOD=auto|rsync|zstd|gzip|raw. gzip level defaults to 1 via GRIM_BRIDGES2_GZIP_LEVEL.
#   - Submodules: With --sync-fas / SYNC_FAS, flash-attention is refreshed via bridges2_ensure_flash_attention.sh
#     (no forced submodule update when the expected commits are already checked out on the remote).
#   - TrainingLoop deps: Launcher uses the repo's TrainingLoop vcpkg manifest on Bridges-2 by default
#     (resources/models/GRIM-text/training/vcpkg.json) so the Linux build matches local manifest-mode builds.
#     It uses the repo's pinned external/vcpkg checkout and installs into training/vcpkg_installed/x64-linux.
#     The old manual-header staging path is intentionally not used by this launcher anymore.
#   - Tool downloads: In vcpkg mode, launcher prefers Bridges-2 system cmake+ninja when they satisfy the pinned helper
#     minimums. It exports VCPKG_DOWNLOADS (default: external/vcpkg/downloads; override with
#     GRIM_BRIDGES2_VCPKG_DOWNLOADS) so helper archives stay cached between runs. Use --allow-vcpkg-tool-downloads or
#     env GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1 only to permit helper downloads when system cmake or ninja is missing, or
#     when the cluster cmake is older than the pinned helper requirement; the launcher then prefetches the helper
#     CMake archive with aria2c/wget/curl when possible so vcpkg reuses the cached file instead of waiting on its own
#     slow download.
#   - Reusing deps: When the remote TrainingLoop `vcpkg_installed/x64-linux` tree already contains
#     nlohmann-json, flatbuffers, and cpp-httplib, the launcher disables manifest auto-install for that configure and
#     reuses the existing tree. Set GRIM_BRIDGES2_FORCE_VCPKG_INSTALL=1 to force a fresh vcpkg install anyway.
#   - CUDA 12+ for training (flash-attention). Bridges-2: module load cuda (check with module avail cuda)
#
# Bridges-2 GPU partitions: GPU-shared (1-4 GPUs, faster queue) or GPU (full node).
# GPU types: h100-80, v100-32, v100-16, l40s-48. Default: h100-80.
#
# Options:
#   --build          Build train_gpu before running.
#   ai_config.json   Canonical config at the repo root. Edit that file before launching.
#   --pull-vocab     Download Bridges-2 training/data/vocab.bin, vocab.txt, and training_data.grmt into the local training data dir, then exit.
#   --pull-logs      Download the latest Bridges-2 *_tape.log, training_<session>.log, and telemetry_<session>.csv from training/logs into the local logs dir, then exit.
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
#   --allow-vcpkg-tool-downloads
#                    Permit vcpkg to download helper tools when Bridges-2 system cmake/ninja is missing or when the
#                    cluster cmake is older than the pinned helper requirement. Compatible system tools still win.
#   --jobs N         make -j N for train_gpu (default 100; override with GRIM_BRIDGES2_MAKE_JOBS).
#   --time T         SLURM wall-clock time limit for train_gpu (--sbatch and interactive srun).
#                    Format: HH:MM:SS or D-HH:MM:SS. Default: omitted (SLURM uses partition maximum).
#                    Examples: --time 1:00:00  --time 0:30:00  --time 2-12:00:00
#   --TD             Run grmt_vocab_metrics_test instead of full training (no GPU needed, uses RM-shared).
#   --UT             Run unigrambyte_self_test instead of full training (needs GPU for GPU decode test).
#   --TT             Run train_tokenizer: full tokenizer training on entire corpus (vocab.bin + .grmt).
#                    Pass --force to rebuild even if files exist: --TT --force

set -e
set -o pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="resources/models/GRIM-text/training"
GRIM_DIR="resources/models/GRIM-text/GRIM"
BUILD_DIR="$TRAINING_DIR/TrainingLoop/build"
EXE="$BUILD_DIR/train_gpu"
TRAINING_DATA_DIR="$REPO_ROOT/resources/models/GRIM-text/training/data"
TRAINING_LOGS_DIR="$REPO_ROOT/resources/models/GRIM-text/training/logs"
CACHE_PATH="$TRAINING_DATA_DIR/merged_verified_cache.jsonl"
CONCEPT_BLOCKS_PATH="$TRAINING_DATA_DIR/concept_blocks.jsonl"
CURRICULUM_REGISTRY_PATH="$TRAINING_DATA_DIR/curriculum_registry.json"
LOCAL_VOCAB_PATH="$TRAINING_DATA_DIR/vocab.bin"
LOCAL_VOCAB_TXT_PATH="$TRAINING_DATA_DIR/vocab.txt"
LOCAL_GRMT_PATH="$TRAINING_DATA_DIR/training_data.grmt"

# Bridges-2 path: /ocean/projects/<alloc_id>/<username>/G.R.I.M (override with GRIM_BRIDGES2_DIR)
BRIDGES2_DIR="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis210058p/uwadkins/G.R.I.M}"
ACCOUNT="${GRIM_BRIDGES2_ACCOUNT:-cis210058p}"
PARTITION="${PARTITION:-GPU-shared}"
GPU_TYPE="${GPU_TYPE:-h100-80}"
BRIDGES2_MAKE_JOBS="${GRIM_BRIDGES2_MAKE_JOBS:-100}"
BRIDGES2_TIME_LIMIT_EXPLICIT=false
if [[ -n "${GRIM_BRIDGES2_TIME_LIMIT:-}" ]]; then
  BRIDGES2_TIME_LIMIT="$GRIM_BRIDGES2_TIME_LIMIT"
  BRIDGES2_TIME_LIMIT_EXPLICIT=true
else
  BRIDGES2_TIME_LIMIT=""
fi
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
DO_PULL_VOCAB=false
DO_PULL_LOGS=false
TT_FORCE=false
ALLOW_VCPKG_TOOL_DOWNLOADS=false
BRIDGES2_SESSION_CLEANED=0
ACTIVE_REMOTE_STATE_FILE=""
ACTIVE_REMOTE_LABEL=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)          DO_BUILD=true; shift ;;
    --incremental)    DO_INCREMENTAL=true; shift ;;
    --clean)          DO_CLEAN_BUILD=true; shift ;;
    --config)
      echo "ERROR: --config is no longer supported by scripts/run_train_on_bridges2.sh." >&2
      echo "  train_gpu now loads the canonical ai_config.json from the repo root on Bridges-2." >&2
      echo "  Edit ./ai_config.json locally, then rerun this launcher so it transfers that file." >&2
      exit 1
      ;;
    --pull-vocab)     DO_PULL_VOCAB=true; shift ;;
    --pull-logs)      DO_PULL_LOGS=true; shift ;;
    --sbatch)         USE_SBATCH=true; shift ;;
    --partition)      PARTITION="$2"; shift 2 ;;
    --gpu-type)       GPU_TYPE="$2"; shift 2 ;;
    --account)        ACCOUNT="$2"; shift 2 ;;
    --sync-all)       FLAG_SYNC_ALL=true; shift ;;
    --sync-mcs)       FLAG_SYNC_MCS=true; shift ;;
    --sync-cbs)       FLAG_SYNC_CBS=true; shift ;;
    --sync-crs)       FLAG_SYNC_CRS=true; shift ;;
    --sync-fas)       FLAG_SYNC_FAS=true; shift ;;
    --allow-vcpkg-tool-downloads) ALLOW_VCPKG_TOOL_DOWNLOADS=true; shift ;;
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
    --time)
      [[ $# -lt 2 ]] && { echo "ERROR: --time requires a time argument (e.g. 1:00:00)"; exit 1; }
      BRIDGES2_TIME_LIMIT="$2"
      BRIDGES2_TIME_LIMIT_EXPLICIT=true
      shift 2
      ;;
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$ALLOW_VCPKG_TOOL_DOWNLOADS" == true ]]; then
  export GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1
fi

if [[ "${GRIM_BRIDGES2_USE_MANUAL_DEPS:-0}" == "1" ]]; then
  echo "ERROR: GRIM_BRIDGES2_USE_MANUAL_DEPS=1 is no longer supported by scripts/run_train_on_bridges2.sh." >&2
  echo "  Bridges-2 now builds TrainingLoop through resources/models/GRIM-text/training/vcpkg.json" >&2
  echo "  using the repo's pinned external/vcpkg checkout, matching local manifest-mode builds." >&2
  echo "  Remove GRIM_BRIDGES2_USE_MANUAL_DEPS from your environment and rerun." >&2
  exit 1
fi

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
REMOTE_TRAINING_LOGS="$REMOTE_TRAINING/logs"
REMOTE_CACHE="$REMOTE_DATA/merged_verified_cache.jsonl"
REMOTE_CONCEPT_BLOCKS="$REMOTE_DATA/concept_blocks.jsonl"
REMOTE_CURRICULUM_REGISTRY="$REMOTE_DATA/curriculum_registry.json"
REMOTE_VOCAB="$REMOTE_DATA/vocab.bin"
REMOTE_VOCAB_TXT="$REMOTE_DATA/vocab.txt"
REMOTE_GRMT="$REMOTE_DATA/training_data.grmt"
CACHE_PATH_EXPANDED="${CACHE_PATH/#\~/$HOME}"
CONCEPT_BLOCKS_PATH_EXPANDED="${CONCEPT_BLOCKS_PATH/#\~/$HOME}"
CURRICULUM_REGISTRY_PATH_EXPANDED="${CURRICULUM_REGISTRY_PATH/#\~/$HOME}"
LOCAL_VOCAB_PATH_EXPANDED="${LOCAL_VOCAB_PATH/#\~/$HOME}"
LOCAL_VOCAB_TXT_PATH_EXPANDED="${LOCAL_VOCAB_TXT_PATH/#\~/$HOME}"
LOCAL_GRMT_PATH_EXPANDED="${LOCAL_GRMT_PATH/#\~/$HOME}"
TRAINING_LOGS_DIR_EXPANDED="${TRAINING_LOGS_DIR/#\~/$HOME}"

# SSH target: bridges2 or bridges2.psc.edu
BRIDGES2_SSH="${GRIM_BRIDGES2_SSH:-bridges2}"
if [[ "$BRIDGES2_SSH" == "bridges2" ]] && ! grep -q "Host bridges2" ~/.ssh/config 2>/dev/null; then
  BRIDGES2_SSH="uwadkins@bridges2.psc.edu"
fi

# SLURM
SLURM_ACCOUNT_ARGS="-A $ACCOUNT"
GRIM_SLURM_MAIL="${GRIM_SLURM_MAIL:-}"
[[ -n "$GRIM_SLURM_MAIL" ]] && SLURM_MAIL_ARGS="--mail-type=BEGIN,END,FAIL --mail-user=$GRIM_SLURM_MAIL" || SLURM_MAIL_ARGS=""
# Omit -t entirely when no explicit limit is set so SLURM uses the partition maximum.
[[ "$BRIDGES2_TIME_LIMIT_EXPLICIT" == true ]] && SLURM_TIME_ARGS="-t $BRIDGES2_TIME_LIMIT" || SLURM_TIME_ARGS=""

# One long-lived SSH using a script-unique socket in /tmp (avoids ~/.ssh permission issues)
BRIDGES2_CTRL="/tmp/cm-grim-$$"
if ! ssh -f -N -M -S "$BRIDGES2_CTRL" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$BRIDGES2_SSH"; then
  echo "SSH to Bridges-2 failed. Try: ssh bridges2"
  exit 1
fi
REMOTE_SNAPSHOT_PATHS=()
trap 'cleanup_bridges2_session' EXIT
trap 'handle_bridges2_signal INT' INT
trap 'handle_bridges2_signal TERM' TERM
trap 'handle_bridges2_signal HUP' HUP
BRIDGES2_SSH_OPTS="-S $BRIDGES2_CTRL -o ControlMaster=no"

remote_quote() {
  printf '%q' "$1"
}

print_bridges2_time_limit() {
  if [[ "$BRIDGES2_TIME_LIMIT_EXPLICIT" == true ]]; then
    echo "[Bridges-2] SLURM time limit: $BRIDGES2_TIME_LIMIT (set via --time or GRIM_BRIDGES2_TIME_LIMIT)"
  else
    echo "[Bridges-2] SLURM time limit: partition max (no -t flag; override with --time T or GRIM_BRIDGES2_TIME_LIMIT)"
  fi
}

cancel_remote_activity() {
  local q_state_file
  local q_label

  [[ -n "$ACTIVE_REMOTE_STATE_FILE" ]] || return 0

  q_state_file="$(remote_quote "$ACTIVE_REMOTE_STATE_FILE")"
  q_label="$(remote_quote "${ACTIVE_REMOTE_LABEL:-remote activity}")"

  echo
  echo "Canceling ${ACTIVE_REMOTE_LABEL:-remote activity} on Bridges-2 and sweeping stale locks..."
  if ! ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "STATE_FILE=$q_state_file LABEL=$q_label bash -se" <<'EOF'
set -e
state_file="$STATE_FILE"
label="$LABEL"

if [ ! -f "$state_file" ]; then
  exit 0
fi

. "$state_file"

echo "  remote cleanup: terminating $label (pid=${pid:-?}, pgid=${pgid:-?})"
if [ -n "${pgid:-}" ]; then
  kill -TERM -- "-$pgid" 2>/dev/null || true
fi
if [ -n "${pid:-}" ]; then
  kill -TERM "$pid" 2>/dev/null || true
fi

for _ in 1 2 3 4 5; do
  if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
    sleep 1
  else
    break
  fi
done

if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
  if [ -n "${pgid:-}" ]; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  fi
  kill -KILL "$pid" 2>/dev/null || true
fi

lock_in_use=0
if [ -n "${lock_path:-}" ] && [ -e "$lock_path" ]; then
  if command -v lsof >/dev/null 2>&1; then
    lsof "$lock_path" >/dev/null 2>&1 && lock_in_use=1 || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser "$lock_path" >/dev/null 2>&1 && lock_in_use=1 || true
  elif [ -n "${vcpkg_root:-}" ]; then
    active_vcpkg=$(ps -eo pid=,args= | grep -F -- "$vcpkg_root" | grep -E '/vcpkg([[:space:]]|$)|/downloads/tools/.*/cmake([[:space:]]|$)|/downloads/tools/.*/ninja([[:space:]]|$)' | grep -v grep || true)
    if [ -n "$active_vcpkg" ]; then
      lock_in_use=1
    fi
  fi

  if [ "$lock_in_use" -eq 0 ]; then
    echo "  remote cleanup: ensuring vcpkg root marker exists at $lock_path"
    touch "$lock_path"
  else
    echo "  remote cleanup: vcpkg activity still appears active; leaving root marker $lock_path in place"
  fi
fi

if [ "${kind:-}" = "build" ]; then
  active_build=""
  if [ -n "${build_dir:-}" ]; then
    active_build=$(ps -eo pid=,args= | grep -F -- "$build_dir" | grep -v grep || true)
    if [ -z "$active_build" ] && [ -d "$build_dir" ]; then
      echo "  remote cleanup: removing interrupted build dir $build_dir"
      rm -rf "$build_dir"
    fi
  fi

  if [ "$lock_in_use" -eq 0 ] && [ -n "${training_vcpkg_triplet_dir:-}" ] && [ -d "$training_vcpkg_triplet_dir" ]; then
    active_triplet=$(ps -eo pid=,args= | grep -F -- "$training_vcpkg_triplet_dir" | grep -v grep || true)
    if [ -z "$active_triplet" ]; then
      echo "  remote cleanup: removing interrupted TrainingLoop install tree $training_vcpkg_triplet_dir"
      rm -rf "$training_vcpkg_triplet_dir"
    fi
  fi
fi

rm -f "$state_file"
EOF
  then
    echo "WARNING: remote cancellation cleanup could not complete automatically." >&2
  fi
}

handle_bridges2_signal() {
  local signal_name="$1"
  local signal_exit=130

  case "$signal_name" in
    TERM) signal_exit=143 ;;
    HUP)  signal_exit=129 ;;
  esac

  trap - INT TERM HUP
  echo
  echo "Caught $signal_name; canceling active Bridges-2 work before exiting..."
  cancel_remote_activity || true
  ACTIVE_REMOTE_STATE_FILE=""
  ACTIVE_REMOTE_LABEL=""
  cleanup_bridges2_session
  exit "$signal_exit"
}

cleanup_bridges2_session() {
  local exit_code=$?
  local snapshot_path
  local q_snapshot_path

  if [[ "$BRIDGES2_SESSION_CLEANED" == "1" ]]; then
    return "$exit_code"
  fi
  BRIDGES2_SESSION_CLEANED=1

  for snapshot_path in "${REMOTE_SNAPSHOT_PATHS[@]}"; do
    [[ -n "$snapshot_path" ]] || continue
    q_snapshot_path="$(remote_quote "$snapshot_path")"
    if ! ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "rm -f $q_snapshot_path" 2>/dev/null; then
      echo "ERROR: failed to remove remote snapshot: $snapshot_path" >&2
    fi
  done

  ssh -S "$BRIDGES2_CTRL" -O exit "$BRIDGES2_SSH" 2>/dev/null
  rm -f "$BRIDGES2_CTRL"
  return "$exit_code"
}

verify_remote_size() {
  local label="$1"
  local remote_path="$2"
  local expected_bytes="$3"
  local q_remote_path
  local actual_bytes

  q_remote_path="$(remote_quote "$remote_path")"
  actual_bytes=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "stat -c %s $q_remote_path 2>/dev/null || wc -c < $q_remote_path" | tr -d '[:space:]')
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "ERROR: $label transfer size mismatch: local=$expected_bytes remote=$actual_bytes"
    exit 1
  fi
  echo "  verified: $actual_bytes bytes"
}

verify_local_size() {
  local label="$1"
  local local_path="$2"
  local expected_bytes="$3"
  local actual_bytes

  actual_bytes=$(wc -c < "$local_path" | tr -d '[:space:]')
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "ERROR: $label transfer size mismatch: remote=$expected_bytes local=$actual_bytes"
    exit 1
  fi
  echo "  verified: $actual_bytes bytes"
}

compute_file_sha512() {
  local file_path="$1"

  if command -v sha512sum >/dev/null 2>&1; then
    sha512sum "$file_path" | awk '{print tolower($1)}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 512 "$file_path" | awk '{print tolower($1)}'
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha512 "$file_path" | sed 's/^.*= //' | tr '[:upper:]' '[:lower:]'
  elif command -v powershell.exe >/dev/null 2>&1; then
    FILE_PATH_FOR_HASH="$file_path" powershell.exe -NoProfile -Command '$p=$env:FILE_PATH_FOR_HASH; (Get-FileHash -Algorithm SHA512 -LiteralPath $p).Hash.ToLowerInvariant()' | tr -d '\r'
  elif command -v pwsh >/dev/null 2>&1; then
    FILE_PATH_FOR_HASH="$file_path" pwsh -NoProfile -Command '$p=$env:FILE_PATH_FOR_HASH; (Get-FileHash -Algorithm SHA512 -LiteralPath $p).Hash.ToLowerInvariant()' | tr -d '\r'
  else
    echo "ERROR: cannot compute SHA512 for $file_path (need sha512sum, shasum, openssl, powershell.exe, or pwsh)." >&2
    return 1
  fi
}

remote_compute_file_sha512() {
  local remote_path="$1"
  local q_remote_path

  q_remote_path="$(remote_quote "$remote_path")"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "REMOTE_HASH_PATH=$q_remote_path bash -se" <<'EOF'
set -e
remote_hash_path="$REMOTE_HASH_PATH"

if command -v sha512sum >/dev/null 2>&1; then
  sha512sum "$remote_hash_path" | awk '{print tolower($1)}'
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 512 "$remote_hash_path" | awk '{print tolower($1)}'
elif command -v openssl >/dev/null 2>&1; then
  openssl dgst -sha512 "$remote_hash_path" | sed 's/^.*= //' | tr '[:upper:]' '[:lower:]'
else
  echo "ERROR: cannot compute SHA512 for $remote_hash_path on Bridges-2 (need sha512sum, shasum, or openssl)." >&2
  exit 1
fi
EOF
}

remote_file_size() {
  local remote_path="$1"
  local q_remote_path

  q_remote_path="$(remote_quote "$remote_path")"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "stat -c %s $q_remote_path 2>/dev/null || wc -c < $q_remote_path" | tr -d '[:space:]'
}

remote_has_command() {
  local command_name="$1"
  local q_command_name

  q_command_name="$(remote_quote "$command_name")"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "command -v $q_command_name >/dev/null 2>&1"
}

remote_latest_log_matching() {
  local remote_dir="$1"
  local name_glob="$2"
  local missing_message="$3"
  local q_remote_dir
  local q_name_glob
  local latest

  q_remote_dir="$(remote_quote "$remote_dir")"
  q_name_glob="$(remote_quote "$name_glob")"
  latest=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; if [ ! -d $q_remote_dir ]; then echo 'ERROR: remote training log directory not found: $remote_dir' >&2; exit 1; fi; find $q_remote_dir -maxdepth 1 -type f -name $q_name_glob -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-")
  if [[ -z "$latest" ]]; then
    echo "$missing_message" >&2
    exit 1
  fi
  printf '%s\n' "$latest"
}

remote_latest_training_log() {
  local remote_dir="$1"
  local q_remote_dir
  local latest

  q_remote_dir="$(remote_quote "$remote_dir")"
  latest=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; if [ ! -d $q_remote_dir ]; then echo 'ERROR: remote training log directory not found: $remote_dir' >&2; exit 1; fi; find $q_remote_dir -maxdepth 1 -type f -name 'training_*.log' ! -name '*_tape.log' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-")
  if [[ -z "$latest" ]]; then
    echo "ERROR: no non-tape training_*.log files found in $remote_dir" >&2
    exit 1
  fi
  printf '%s\n' "$latest"
}

remote_latest_tape_log() {
  local remote_dir="$1"
  remote_latest_log_matching "$remote_dir" '*_tape.log' "ERROR: no *_tape.log files found in $remote_dir"
}

remote_latest_telemetry_csv() {
  local remote_dir="$1"
  local q_remote_dir
  local latest

  q_remote_dir="$(remote_quote "$remote_dir")"
  latest=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; if [ ! -d $q_remote_dir ]; then echo 'ERROR: remote training log directory not found: $remote_dir' >&2; exit 1; fi; find $q_remote_dir -maxdepth 1 -type f -name 'telemetry_*.csv' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-")
  if [[ -z "$latest" ]]; then
    echo "ERROR: no telemetry_*.csv files found in $remote_dir" >&2
    exit 1
  fi
  printf '%s\n' "$latest"
}

remote_snapshot_file() {
  local label="$1"
  local remote_path="$2"
  local remote_dir
  local snapshot_path
  local q_remote_path
  local q_snapshot_path
  local size_bytes

  remote_dir="$(dirname "$remote_path")"
  snapshot_path="$remote_dir/.grim_pull_snapshot_$(basename "$remote_path").$$"
  q_remote_path="$(remote_quote "$remote_path")"
  q_snapshot_path="$(remote_quote "$snapshot_path")"

  echo "Snapshotting $label on Bridges-2 so active writes cannot change the transfer size..." >&2
  size_bytes=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; if [ ! -f $q_remote_path ]; then echo 'ERROR: remote file not found: $remote_path' >&2; exit 1; fi; initial_size=\$(stat -c %s $q_remote_path); rm -f $q_snapshot_path; head -c \"\$initial_size\" $q_remote_path > $q_snapshot_path; actual_size=\$(stat -c %s $q_snapshot_path); if [ \"\$actual_size\" != \"\$initial_size\" ]; then echo \"ERROR: snapshot size mismatch for $remote_path: source_at_start=\$initial_size snapshot=\$actual_size\" >&2; rm -f $q_snapshot_path; exit 1; fi; printf '%s\n' \"\$actual_size\"")
  echo "  snapshot: $snapshot_path ($size_bytes bytes)" >&2
  printf '%s\n' "$snapshot_path"
}

remove_remote_file() {
  local remote_path="$1"
  local q_remote_path

  q_remote_path="$(remote_quote "$remote_path")"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "rm -f $q_remote_path"
}

download_remote_snapshot_file() {
  local label="$1"
  local remote_path="$2"
  local local_path="$3"
  local snapshot_path

  snapshot_path="$(remote_snapshot_file "$label" "$remote_path")"
  REMOTE_SNAPSHOT_PATHS+=("$snapshot_path")
  download_training_file "$label" "$snapshot_path" "$local_path"
  remove_remote_file "$snapshot_path"
}

stream_file_with_progress() {
  local local_path="$1"
  local size_bytes="$2"

  if command -v pv >/dev/null 2>&1; then
    pv -s "$size_bytes" "$local_path"
  elif dd if=/dev/null of=/dev/null bs=1 count=0 status=progress >/dev/null 2>&1; then
    echo "  progress: using dd status=progress (bytes read before compression)" >&2
    dd if="$local_path" bs=16M status=progress
  else
    echo "  progress: install pv for live transfer progress; streaming without progress meter" >&2
    cat "$local_path"
  fi
}

transfer_training_file() {
  local label="$1"
  local local_path="$2"
  local remote_path="$3"
  local remote_dir
  local q_remote_dir
  local q_remote_path
  local q_tmp_path
  local size_bytes
  local method
  local gzip_level
  local compressor_name=""
  local tmp_path

  if [[ ! -f "$local_path" ]]; then
    echo "ERROR: $label not found at $local_path"
    exit 1
  fi

  remote_dir="$(dirname "$remote_path")"
  q_remote_dir="$(remote_quote "$remote_dir")"
  q_remote_path="$(remote_quote "$remote_path")"
  tmp_path="$remote_path.transfer.$$"
  q_tmp_path="$(remote_quote "$tmp_path")"
  size_bytes=$(wc -c < "$local_path" | tr -d '[:space:]')
  method="${GRIM_BRIDGES2_TRANSFER_METHOD:-auto}"
  gzip_level="${GRIM_BRIDGES2_GZIP_LEVEL:-1}"

  case "$method" in
    auto|rsync|zstd|gzip|raw) ;;
    *) echo "ERROR: GRIM_BRIDGES2_TRANSFER_METHOD must be auto, rsync, zstd, gzip, or raw (got: $method)"; exit 1 ;;
  esac
  [[ "$gzip_level" =~ ^[1-9]$ ]] || { echo "ERROR: GRIM_BRIDGES2_GZIP_LEVEL must be 1..9 (got: $gzip_level)"; exit 1; }

  echo "Transferring $label ($size_bytes bytes)..."
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "mkdir -p $q_remote_dir"

  if [[ "$method" == "auto" || "$method" == "rsync" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      if remote_has_command rsync; then
        echo "  method: rsync --compress --partial --progress"
        if rsync -a --compress --partial --progress -e "ssh $BRIDGES2_SSH_OPTS" "$local_path" "$BRIDGES2_SSH:$remote_path"; then
          verify_remote_size "$label" "$remote_path" "$size_bytes"
          echo "  -> $remote_path"
          return 0
        fi
        if [[ "$method" == "rsync" ]]; then
          echo "ERROR: rsync transfer failed for $label"
          exit 1
        fi
        echo "  rsync failed; trying gzip stream."
      elif [[ "$method" == "rsync" ]]; then
        echo "ERROR: rsync requested but not found on Bridges-2. Set GRIM_BRIDGES2_TRANSFER_METHOD=gzip or install rsync remotely."
        exit 1
      else
        echo "  rsync not found on Bridges-2; trying gzip stream."
      fi
    elif [[ "$method" == "rsync" ]]; then
      echo "ERROR: rsync requested but not found locally. Install rsync or set GRIM_BRIDGES2_TRANSFER_METHOD=gzip."
      exit 1
    else
      echo "  rsync not found locally; trying gzip stream."
    fi
  fi

  if [[ "$method" == "auto" || "$method" == "zstd" ]]; then
    if command -v zstd >/dev/null 2>&1 && remote_has_command zstd; then
      echo "  method: zstd -1 -T0 | ssh zstd -dc"
      stream_file_with_progress "$local_path" "$size_bytes" | zstd -1 -T0 -c | ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; zstd -dc > $q_tmp_path; mv -f $q_tmp_path $q_remote_path"
      verify_remote_size "$label" "$remote_path" "$size_bytes"
      echo "  -> $remote_path"
      return 0
    fi

    if [[ "$method" == "zstd" ]]; then
      echo "ERROR: zstd requested, but local zstd or remote Bridges-2 zstd is unavailable. Set GRIM_BRIDGES2_TRANSFER_METHOD=gzip."
      exit 1
    fi
    echo "  zstd unavailable on one side; trying gzip stream."
  fi

  if [[ "$method" == "auto" || "$method" == "gzip" ]]; then
    if command -v pigz >/dev/null 2>&1; then
      compressor_name="pigz"
    elif command -v gzip >/dev/null 2>&1; then
      compressor_name="gzip"
    fi

    if [[ -n "$compressor_name" ]] && remote_has_command gzip; then
      echo "  method: $compressor_name -$gzip_level | ssh gzip -dc"
      stream_file_with_progress "$local_path" "$size_bytes" | "$compressor_name" -"$gzip_level" -c | ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; gzip -dc > $q_tmp_path; mv -f $q_tmp_path $q_remote_path"
      verify_remote_size "$label" "$remote_path" "$size_bytes"
      echo "  -> $remote_path"
      return 0
    fi

    if [[ "$method" == "gzip" ]]; then
      echo "ERROR: gzip transfer requested, but local compressor or remote gzip is unavailable."
      exit 1
    fi
    echo "  gzip stream unavailable; using raw ssh transfer."
  fi

  echo "  method: raw ssh cat (slow fallback)"
  stream_file_with_progress "$local_path" "$size_bytes" | ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; cat > $q_tmp_path; mv -f $q_tmp_path $q_remote_path"
  verify_remote_size "$label" "$remote_path" "$size_bytes"
  echo "  -> $remote_path"
}

download_local_helper_file() {
  local label="$1"
  local url="$2"
  local local_path="$3"
  local expected_sha512="${4,,}"
  local local_dir
  local tmp_path
  local archive_name
  local actual_sha512

  if [[ -f "$local_path" ]]; then
    if [[ -n "$expected_sha512" ]]; then
      actual_sha512="$(compute_file_sha512 "$local_path")" || return 1
      if [[ "$actual_sha512" == "$expected_sha512" ]]; then
        echo "  local helper cache: using cached $label at $local_path"
        return 0
      fi

      echo "  local helper cache: deleting stale $label because SHA512 mismatched"
      echo "    expected: $expected_sha512"
      echo "    actual:   $actual_sha512"
      rm -f "$local_path" "$local_path.partial"
    else
      echo "  local helper cache: using cached $label at $local_path"
      return 0
    fi
  fi

  local_dir="$(dirname "$local_path")"
  tmp_path="$local_path.partial"
  archive_name="$(basename "$local_path")"
  mkdir -p "$local_dir"
  rm -f "$tmp_path"

  echo "  local helper cache: downloading $label to $local_path"

  if command -v aria2c >/dev/null 2>&1; then
    aria2c --allow-overwrite=true --auto-file-renaming=false --continue=true --dir="$local_dir" --out="${archive_name}.partial" -x 8 -s 8 -k 1M "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --continue-at - --output "$tmp_path" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$tmp_path" "$url"
  else
    echo "WARNING: local helper cache: aria2c/curl/wget unavailable; cannot prefetch $label locally." >&2
    rm -f "$tmp_path"
    return 1
  fi

  if [[ ! -f "$tmp_path" ]]; then
    echo "WARNING: local helper cache: download did not produce $tmp_path" >&2
    return 1
  fi

  mv -f "$tmp_path" "$local_path"
  if [[ -n "$expected_sha512" ]]; then
    actual_sha512="$(compute_file_sha512 "$local_path")" || return 1
    if [[ "$actual_sha512" != "$expected_sha512" ]]; then
      echo "WARNING: local helper cache: downloaded $label with the wrong SHA512; deleting it so Bridges-2 can fetch a clean copy." >&2
      echo "  expected: $expected_sha512" >&2
      echo "  actual:   $actual_sha512" >&2
      rm -f "$local_path"
      return 1
    fi
  fi
  echo "  local helper cache: cached $label"
}

seed_remote_vcpkg_helper_archive() {
  local local_archive_path="$1"
  local remote_archive_path="$2"
  local archive_label="$3"
  local helper_url="$4"
  local expected_sha512="${5,,}"
  local q_remote_archive_path
  local q_remote_downloads_dir
  local remote_actual_sha512

  q_remote_archive_path="$(remote_quote "$remote_archive_path")"
  q_remote_downloads_dir="$(remote_quote "$(dirname "$remote_archive_path")")"

  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "mkdir -p $q_remote_downloads_dir"
  if ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "test -f $q_remote_archive_path"; then
    if [[ -n "$expected_sha512" ]]; then
      remote_actual_sha512="$(remote_compute_file_sha512 "$remote_archive_path")" || return 1
      if [[ "$remote_actual_sha512" == "$expected_sha512" ]]; then
        echo "  vcpkg helper seed: remote cache already has a valid $(basename "$remote_archive_path")"
        return 0
      fi

      echo "  vcpkg helper seed: replacing stale remote $(basename "$remote_archive_path") because SHA512 mismatched"
      echo "    expected: $expected_sha512"
      echo "    actual:   $remote_actual_sha512"
      ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "rm -f $q_remote_archive_path ${q_remote_archive_path}.partial"
    else
      echo "  vcpkg helper seed: remote cache already has $(basename "$remote_archive_path")"
      return 0
    fi
  fi

  if [[ ! -f "$local_archive_path" ]]; then
    if [[ "${GRIM_BRIDGES2_PREFETCH_HELPER_LOCALLY:-1}" == "1" ]]; then
      if ! download_local_helper_file "$archive_label" "$helper_url" "$local_archive_path" "$expected_sha512"; then
        echo "WARNING: vcpkg helper seed: local prefetch failed; remote will download $archive_label if required." >&2
        return 0
      fi
    else
      echo "  vcpkg helper seed: local cache miss for $local_archive_path; remote will download $archive_label if required"
      return 0
    fi
  fi

  transfer_training_file "$archive_label" "$local_archive_path" "$remote_archive_path"
}

download_training_file() {
  local label="$1"
  local remote_path="$2"
  local local_path="$3"
  local local_dir
  local q_remote_path
  local size_bytes
  local method
  local gzip_level
  local decompressor_name=""
  local tmp_path

  local_dir="$(dirname "$local_path")"
  q_remote_path="$(remote_quote "$remote_path")"
  size_bytes="$(remote_file_size "$remote_path")"
  method="${GRIM_BRIDGES2_TRANSFER_METHOD:-auto}"
  gzip_level="${GRIM_BRIDGES2_GZIP_LEVEL:-1}"
  tmp_path="$local_path.transfer.$$"

  case "$method" in
    auto|rsync|zstd|gzip|raw) ;;
    *) echo "ERROR: GRIM_BRIDGES2_TRANSFER_METHOD must be auto, rsync, zstd, gzip, or raw (got: $method)"; exit 1 ;;
  esac
  [[ "$gzip_level" =~ ^[1-9]$ ]] || { echo "ERROR: GRIM_BRIDGES2_GZIP_LEVEL must be 1..9 (got: $gzip_level)"; exit 1; }

  mkdir -p "$local_dir"
  rm -f "$tmp_path"

  echo "Downloading $label ($size_bytes bytes)..."

  if [[ "$method" == "auto" || "$method" == "rsync" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      if remote_has_command rsync; then
        echo "  method: rsync --compress --partial --progress"
        if rsync -a --compress --partial --progress -e "ssh $BRIDGES2_SSH_OPTS" "$BRIDGES2_SSH:$remote_path" "$local_path"; then
          verify_local_size "$label" "$local_path" "$size_bytes"
          echo "  -> $local_path"
          return 0
        fi
        if [[ "$method" == "rsync" ]]; then
          echo "ERROR: rsync transfer failed for $label"
          exit 1
        fi
        echo "  rsync failed; trying gzip stream."
      elif [[ "$method" == "rsync" ]]; then
        echo "ERROR: rsync requested but not found on Bridges-2. Set GRIM_BRIDGES2_TRANSFER_METHOD=gzip or install rsync remotely."
        exit 1
      else
        echo "  rsync not found on Bridges-2; trying gzip stream."
      fi
    elif [[ "$method" == "rsync" ]]; then
      echo "ERROR: rsync requested but not found locally. Install rsync or set GRIM_BRIDGES2_TRANSFER_METHOD=gzip."
      exit 1
    else
      echo "  rsync not found locally; trying gzip stream."
    fi
  fi

  if [[ "$method" == "auto" || "$method" == "zstd" ]]; then
    if command -v zstd >/dev/null 2>&1 && remote_has_command zstd; then
      echo "  method: ssh zstd -1 -T0 -c | local zstd -dc"
      ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; zstd -1 -T0 -c $q_remote_path" | zstd -dc > "$tmp_path"
      verify_local_size "$label" "$tmp_path" "$size_bytes"
      mv -f "$tmp_path" "$local_path"
      echo "  -> $local_path"
      return 0
    fi

    if [[ "$method" == "zstd" ]]; then
      echo "ERROR: zstd requested, but local zstd or remote Bridges-2 zstd is unavailable. Set GRIM_BRIDGES2_TRANSFER_METHOD=gzip."
      exit 1
    fi
    echo "  zstd unavailable on one side; trying gzip stream."
  fi

  if [[ "$method" == "auto" || "$method" == "gzip" ]]; then
    if command -v pigz >/dev/null 2>&1; then
      decompressor_name="pigz"
    elif command -v gzip >/dev/null 2>&1; then
      decompressor_name="gzip"
    fi

    if [[ -n "$decompressor_name" ]] && remote_has_command gzip; then
      echo "  method: ssh gzip -$gzip_level -c | local $decompressor_name -dc"
      ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; gzip -$gzip_level -c $q_remote_path" | "$decompressor_name" -dc > "$tmp_path"
      verify_local_size "$label" "$tmp_path" "$size_bytes"
      mv -f "$tmp_path" "$local_path"
      echo "  -> $local_path"
      return 0
    fi

    if [[ "$method" == "gzip" ]]; then
      echo "ERROR: gzip transfer requested, but local decompressor or remote gzip is unavailable."
      exit 1
    fi
    echo "  gzip stream unavailable; using raw ssh transfer."
  fi

  echo "  method: raw ssh cat (slow fallback)"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "set -e; cat $q_remote_path" > "$tmp_path"
  verify_local_size "$label" "$tmp_path" "$size_bytes"
  mv -f "$tmp_path" "$local_path"
  echo "  -> $local_path"
}

if [[ "$DO_PULL_VOCAB" == true ]]; then
  download_training_file "vocab.bin" "$REMOTE_VOCAB" "$LOCAL_VOCAB_PATH_EXPANDED"
  download_training_file "vocab.txt" "$REMOTE_VOCAB_TXT" "$LOCAL_VOCAB_TXT_PATH_EXPANDED"
  download_training_file "training_data.grmt" "$REMOTE_GRMT" "$LOCAL_GRMT_PATH_EXPANDED"
  exit 0
fi

if [[ "$DO_PULL_LOGS" == true ]]; then
  latest_training_log="$(remote_latest_training_log "$REMOTE_TRAINING_LOGS")"
  latest_tape_log="$(remote_latest_tape_log "$REMOTE_TRAINING_LOGS")"
  latest_telemetry_csv="$(remote_latest_telemetry_csv "$REMOTE_TRAINING_LOGS")"
  local_training_log_path="$TRAINING_LOGS_DIR_EXPANDED/$(basename "$latest_training_log")"
  local_tape_log_path="$TRAINING_LOGS_DIR_EXPANDED/$(basename "$latest_tape_log")"
  local_telemetry_csv_path="$TRAINING_LOGS_DIR_EXPANDED/$(basename "$latest_telemetry_csv")"
  echo "Latest Bridges-2 training log: $latest_training_log"
  download_remote_snapshot_file "$(basename "$latest_training_log")" "$latest_training_log" "$local_training_log_path"
  echo "Latest Bridges-2 tape log: $latest_tape_log"
  download_remote_snapshot_file "$(basename "$latest_tape_log")" "$latest_tape_log" "$local_tape_log_path"
  echo "Latest Bridges-2 telemetry CSV: $latest_telemetry_csv"
  download_remote_snapshot_file "$(basename "$latest_telemetry_csv")" "$latest_telemetry_csv" "$local_telemetry_csv_path"
  exit 0
fi

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

# Bridges-2 modules: cuda, gcc, cmake, ninja. CUDA 12+ required.
# TrainingLoop itself requires CMake 3.20; prefer the cluster's default cmake module instead of pinning 3.30.x here.
BRIDGES2_MODULES="source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || module load cuda/12 2>/dev/null || module load cuda/12.0 2>/dev/null || true; module load gcc 2>/dev/null || true; module load cmake 2>/dev/null || true; module load ninja 2>/dev/null || module load ninja/1.11 2>/dev/null || module load ninja/1.10 2>/dev/null || true"
# Default to project CUDA 12 when 11.x is detected (CUTLASS requires 12+). Uses GRIM_PROJECT_DIR=$BRIDGES2_DIR to find cuda-12.0.
BRIDGES2_ENSURE_CUDA12="export GRIM_PROJECT_DIR=\$BRIDGES2_DIR; source \"\$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true"
# CUDAToolkit_ROOT for cmake comes from GRIM_CUDA_ROOT (set by ensure_cuda12_for_training.sh). Do not inline
# nvcc discovery here — unescaped \" and \$( in a var would break the local ssh \"...\" string (parse errors like SH_OPTS).

# vcpkg: default to the repo's external/vcpkg checkout so Bridges-2 matches local TrainingLoop builds.
# If GRIM_VCPKG_ROOT is set, use that checkout instead.
BRIDGES2_VCPKG="${GRIM_VCPKG_ROOT:-$BRIDGES2_DIR/external/vcpkg}"
VCPKG_TOOLCHAIN="$BRIDGES2_VCPKG/scripts/buildsystems/vcpkg.cmake"
BRIDGES2_VCPKG_DOWNLOADS="${GRIM_BRIDGES2_VCPKG_DOWNLOADS:-$BRIDGES2_VCPKG/downloads}"
LOCAL_VCPKG_DOWNLOADS="${GRIM_LOCAL_VCPKG_DOWNLOADS:-$REPO_ROOT/external/vcpkg/downloads}"
# Keep this helper CMake metadata aligned with external/vcpkg/scripts/vcpkg-tools.json. Override only when you are
# intentionally testing a different pinned helper archive.
BRIDGES2_VCPKG_CMAKE_VERSION="${GRIM_BRIDGES2_VCPKG_CMAKE_VERSION:-3.30.1}"
BRIDGES2_VCPKG_CMAKE_ARCHIVE="${GRIM_BRIDGES2_VCPKG_CMAKE_ARCHIVE:-cmake-${BRIDGES2_VCPKG_CMAKE_VERSION}-linux-x86_64.tar.gz}"
BRIDGES2_VCPKG_CMAKE_SHA512="${GRIM_BRIDGES2_VCPKG_CMAKE_SHA512:-84ce1333ed696a1736986fba2853c5d8db0e4c9addaf4a4723911248c6d49ecf545adf8bd46091d198fc7bd1e6c896798661463aa1ce3a726a093883aaa19adf}"
BRIDGES2_VCPKG_CMAKE_URL="${GRIM_BRIDGES2_VCPKG_CMAKE_URL:-https://github.com/Kitware/CMake/releases/download/v${BRIDGES2_VCPKG_CMAKE_VERSION}/${BRIDGES2_VCPKG_CMAKE_ARCHIVE}}"
LOCAL_VCPKG_CMAKE_ARCHIVE_PATH="$LOCAL_VCPKG_DOWNLOADS/$BRIDGES2_VCPKG_CMAKE_ARCHIVE"
REMOTE_VCPKG_CMAKE_ARCHIVE_PATH="$BRIDGES2_VCPKG_DOWNLOADS/$BRIDGES2_VCPKG_CMAKE_ARCHIVE"
BRIDGES2_REMOTE_SHA512_FUNC='grim_compute_sha512() {
  local grim_hash_path="$1"
  if command -v sha512sum >/dev/null 2>&1; then
    sha512sum "$grim_hash_path" | awk "{print tolower(\$1)}"
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 512 "$grim_hash_path" | awk "{print tolower(\$1)}"
  elif command -v openssl >/dev/null 2>&1; then
    openssl dgst -sha512 "$grim_hash_path" | sed "s/^.*= //" | tr "[:upper:]" "[:lower:]"
  else
    echo "ERROR: cannot compute SHA512 for $grim_hash_path on Bridges-2 (need sha512sum, shasum, or openssl)." >&2
    return 1
  fi
}'
BRIDGES2_REMOTE_VALIDATE_CMAKE_ARCHIVE_FUNC='grim_validate_cmake_archive() {
  local archive_path="$1"
  local expected_sha512="$2"
  local actual_sha512

  if [ ! -f "$archive_path" ]; then
    return 1
  fi

  actual_sha512=$(grim_compute_sha512 "$archive_path") || return 2
  if [ "$actual_sha512" = "$expected_sha512" ]; then
    return 0
  fi

  echo "  vcpkg helper tools: deleting stale $(basename "$archive_path") because SHA512 mismatched"
  echo "    expected: $expected_sha512"
  echo "    actual:   $actual_sha512"
  rm -f "$archive_path" "$archive_path.partial"
  return 1
}'
BRIDGES2_VCPKG_TOOL_POLICY="$BRIDGES2_REMOTE_SHA512_FUNC; $BRIDGES2_REMOTE_VALIDATE_CMAKE_ARCHIVE_FUNC; grim_required_cmake_version=\"$BRIDGES2_VCPKG_CMAKE_VERSION\"; grim_expected_cmake_sha512=\"$BRIDGES2_VCPKG_CMAKE_SHA512\"; grim_system_cmake_version=\"\"; grim_has_compatible_cmake=0; grim_need_helper_downloads=0; if command -v cmake >/dev/null 2>&1; then grim_system_cmake_version=\$(cmake --version | awk 'NR==1 {print \$3}'); fi; if [ -n \"\$grim_system_cmake_version\" ] && [ \"\$(printf '%s\\n%s\\n' \"$BRIDGES2_VCPKG_CMAKE_VERSION\" \"\$grim_system_cmake_version\" | sort -V | head -n1)\" = \"$BRIDGES2_VCPKG_CMAKE_VERSION\" ]; then grim_has_compatible_cmake=1; fi; command -v ninja >/dev/null 2>&1 || grim_need_helper_downloads=1; [ \"\$grim_has_compatible_cmake\" = \"1\" ] || grim_need_helper_downloads=1; if [ \"\$grim_need_helper_downloads\" = \"0\" ]; then echo '  vcpkg helper tools: using Bridges-2 system cmake+ninja'; echo \"    system cmake: \$grim_system_cmake_version (required >= $BRIDGES2_VCPKG_CMAKE_VERSION)\"; export VCPKG_FORCE_SYSTEM_BINARIES=1; elif [ \"${GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS:-0}\" = \"1\" ]; then if [ -z \"\$grim_system_cmake_version\" ]; then echo '  vcpkg helper tools: system cmake missing; allowing helper-tool downloads'; elif [ \"\$grim_has_compatible_cmake\" != \"1\" ]; then echo \"  vcpkg helper tools: system cmake \$grim_system_cmake_version is older than required $BRIDGES2_VCPKG_CMAKE_VERSION; allowing helper-tool downloads\"; else echo '  vcpkg helper tools: system cmake is fine; helper-tool downloads remain enabled because ninja is missing'; fi; if [ \"\$grim_has_compatible_cmake\" != \"1\" ]; then grim_cmake_archive=\"$BRIDGES2_VCPKG_CMAKE_ARCHIVE\"; grim_cmake_url=\"$BRIDGES2_VCPKG_CMAKE_URL\"; grim_cmake_cache_path=\"$BRIDGES2_VCPKG_DOWNLOADS/\$grim_cmake_archive\"; if grim_validate_cmake_archive \"\$grim_cmake_cache_path\" \"\$grim_expected_cmake_sha512\"; then echo \"  vcpkg helper tools: using cached \$grim_cmake_archive from $BRIDGES2_VCPKG_DOWNLOADS\"; else echo \"  vcpkg helper tools: prefetching \$grim_cmake_archive into $BRIDGES2_VCPKG_DOWNLOADS\"; grim_tmp_download=\"\$grim_cmake_cache_path.partial\"; rm -f \"\$grim_tmp_download\"; if command -v aria2c >/dev/null 2>&1; then aria2c --allow-overwrite=true --auto-file-renaming=false --continue=true --dir=\"$BRIDGES2_VCPKG_DOWNLOADS\" --out=\"\$grim_cmake_archive.partial\" -x 8 -s 8 -k 1M \"\$grim_cmake_url\"; mv -f \"\$grim_tmp_download\" \"\$grim_cmake_cache_path\"; elif command -v wget >/dev/null 2>&1; then wget -O \"\$grim_tmp_download\" \"\$grim_cmake_url\"; mv -f \"\$grim_tmp_download\" \"\$grim_cmake_cache_path\"; elif command -v curl >/dev/null 2>&1; then curl -L --fail --retry 5 --continue-at - --output \"\$grim_tmp_download\" \"\$grim_cmake_url\"; mv -f \"\$grim_tmp_download\" \"\$grim_cmake_cache_path\"; else echo 'ERROR: helper downloads are enabled but aria2c, wget, and curl are all unavailable for CMake prefetch.' >&2; exit 1; fi; if ! grim_validate_cmake_archive \"\$grim_cmake_cache_path\" \"\$grim_expected_cmake_sha512\"; then echo \"ERROR: cached helper archive \$grim_cmake_archive still failed SHA512 validation after download.\" >&2; echo '  Remove the bad archive from the Bridges-2 downloads cache and rerun if this persists.' >&2; exit 1; fi; fi; fi; unset VCPKG_FORCE_SYSTEM_BINARIES; else if [ -z \"\$grim_system_cmake_version\" ]; then echo 'ERROR: Bridges-2 system cmake is missing and GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS is not enabled.' >&2; elif ! command -v ninja >/dev/null 2>&1; then echo 'ERROR: Bridges-2 system ninja is missing and GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS is not enabled.' >&2; else echo \"ERROR: Bridges-2 system cmake \$grim_system_cmake_version is older than required $BRIDGES2_VCPKG_CMAKE_VERSION and helper downloads are disabled.\" >&2; fi; echo '  Load the cluster cmake/ninja modules or re-run with --allow-vcpkg-tool-downloads.' >&2; exit 1; fi"
BRIDGES2_VCPKG_ENV="mkdir -p \"$BRIDGES2_VCPKG_DOWNLOADS\"; export VCPKG_ROOT=\"$BRIDGES2_VCPKG\"; export VCPKG_DOWNLOADS=\"$BRIDGES2_VCPKG_DOWNLOADS\"; unset Z_VCPKG_ROOT_DIR; unset _VCPKG_ROOT_DIR"
if [[ -n "${GRIM_VCPKG_ROOT:-}" ]]; then
  BRIDGES2_VCPKG_ENSURE="(set -e; $BRIDGES2_VCPKG_ENV; if [ ! -f \"$VCPKG_TOOLCHAIN\" ]; then echo \"ERROR: GRIM_VCPKG_ROOT does not contain vcpkg toolchain: $VCPKG_TOOLCHAIN\" >&2; exit 1; fi; if [ ! -x \"$BRIDGES2_VCPKG/vcpkg\" ]; then (cd \"$BRIDGES2_VCPKG\" && ./bootstrap-vcpkg.sh -disableMetrics); fi; touch \"$BRIDGES2_VCPKG/.vcpkg-root\")"
else
  BRIDGES2_VCPKG_ENSURE="(set -e; cd \"$BRIDGES2_DIR\"; $BRIDGES2_VCPKG_ENV; VCPKG_PIN=\$(git ls-tree HEAD external/vcpkg | awk '{print \$3}'); if [ -z \"\$VCPKG_PIN\" ]; then echo \"ERROR: external/vcpkg gitlink not found in repo HEAD\" >&2; exit 1; fi; if ! git -C \"$BRIDGES2_VCPKG\" rev-parse --show-toplevel >/dev/null 2>&1; then echo \"Initializing external/vcpkg to match local repo layout...\"; git submodule update --init external/vcpkg; fi; VCPKG_TOP=\$(git -C \"$BRIDGES2_VCPKG\" rev-parse --show-toplevel 2>/dev/null || true); if [ \"\$VCPKG_TOP\" != \"$BRIDGES2_VCPKG\" ]; then echo \"ERROR: $BRIDGES2_VCPKG exists but is not a vcpkg git checkout; remove it or set GRIM_VCPKG_ROOT\" >&2; exit 1; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_PIN^{commit}\" 2>/dev/null; then git -C \"$BRIDGES2_VCPKG\" fetch --depth 1 origin \"\$VCPKG_PIN\" || git -C \"$BRIDGES2_VCPKG\" fetch origin \"\$VCPKG_PIN\"; fi; VCPKG_HEAD=\$(git -C \"$BRIDGES2_VCPKG\" rev-parse HEAD 2>/dev/null || true); if [ \"\$VCPKG_HEAD\" != \"\$VCPKG_PIN\" ]; then echo \"  vcpkg checkout status: aligning \$VCPKG_HEAD -> \$VCPKG_PIN\"; git -c advice.detachedHead=false -C \"$BRIDGES2_VCPKG\" checkout -f \"\$VCPKG_PIN\"; elif ! git -C \"$BRIDGES2_VCPKG\" diff --quiet --ignore-submodules HEAD -- || ! git -C \"$BRIDGES2_VCPKG\" diff --cached --quiet --ignore-submodules HEAD --; then echo \"  vcpkg checkout status: dirty pinned worktree; resetting to \$VCPKG_PIN\"; git -c advice.detachedHead=false -C \"$BRIDGES2_VCPKG\" checkout -f \"\$VCPKG_PIN\"; else echo \"  vcpkg checkout status: already pinned at \$VCPKG_PIN\"; fi; VCPKG_BASELINE=\$(grep -o '\"builtin-baseline\"[[:space:]]*:[[:space:]]*\"[^\"]*\"' \"$BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json\" | head -n 1 | sed -E 's/.*\"([^\"]*)\"$/\1/'); if [ -z \"\$VCPKG_BASELINE\" ]; then echo \"ERROR: builtin-baseline missing from $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json\" >&2; exit 1; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_BASELINE^{commit}\" 2>/dev/null; then git -C \"$BRIDGES2_VCPKG\" fetch --depth 1 origin \"\$VCPKG_BASELINE\" || git -C \"$BRIDGES2_VCPKG\" fetch origin \"\$VCPKG_BASELINE\"; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_BASELINE:versions/baseline.json\" 2>/dev/null; then git -C \"$BRIDGES2_VCPKG\" fetch origin \"\$VCPKG_BASELINE\" || true; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_BASELINE:versions/baseline.json\" 2>/dev/null; then echo \"ERROR: vcpkg builtin-baseline \$VCPKG_BASELINE is unavailable in $BRIDGES2_VCPKG; remove the checkout or fetch the missing history.\" >&2; exit 1; fi; if [ ! -f \"$VCPKG_TOOLCHAIN\" ]; then echo \"ERROR: pinned vcpkg toolchain not found: $VCPKG_TOOLCHAIN\" >&2; exit 1; fi; if [ ! -x \"$BRIDGES2_VCPKG/vcpkg\" ]; then (cd \"$BRIDGES2_VCPKG\" && ./bootstrap-vcpkg.sh -disableMetrics); fi; touch \"$BRIDGES2_VCPKG/.vcpkg-root\")"
fi
BRIDGES2_MANIFEST_ASSERT="if [ ! -f \"$BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json\" ]; then echo \"ERROR: expected TrainingLoop manifest at $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json after repo sync\" >&2; exit 1; fi"

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

# --clean ALWAYS forces a full build-dir wipe, independent of FAS/Cutlass gating
# above (the SKIP_FAS=1 default path otherwise clears BRIDGES2_CLEAN and silently
# ignores --clean). This removes CMakeCache.txt so cmake reconfigures from source
# defaults instead of reusing stale cached options.
if [[ "$DO_CLEAN_BUILD" == true ]]; then
  BRIDGES2_CLEAN="rm -rf $BRIDGES2_DIR/$BUILD_DIR && "
fi

# CUDA arch: sm_90 for H100, sm_80 for V100
if [[ "$GPU_TYPE" == "h100-80" ]]; then
  BRIDGES2_CUDA_ARCH="export GRIM_CUDA_ARCH=90; "
else
  BRIDGES2_CUDA_ARCH="export GRIM_CUDA_ARCH=80; "
fi

BRIDGES2_TRAINING_VCPKG_INSTALLED="$BRIDGES2_DIR/$TRAINING_DIR/vcpkg_installed"
BRIDGES2_TRAINING_VCPKG_TRIPLET_DIR="$BRIDGES2_TRAINING_VCPKG_INSTALLED/x64-linux"
BRIDGES2_VCPKG_MANIFEST_POLICY="cmake_manifest_install=ON; if [ \"${GRIM_BRIDGES2_FORCE_VCPKG_INSTALL:-0}\" != \"1\" ] && [ -f \"$BRIDGES2_TRAINING_VCPKG_TRIPLET_DIR/include/nlohmann/json.hpp\" ] && [ -f \"$BRIDGES2_TRAINING_VCPKG_TRIPLET_DIR/include/flatbuffers/flatbuffers.h\" ] && [ -f \"$BRIDGES2_TRAINING_VCPKG_TRIPLET_DIR/include/httplib.h\" ] && [ -f \"$BRIDGES2_TRAINING_VCPKG_TRIPLET_DIR/share/nlohmann_json/nlohmann_jsonConfig.cmake\" ] && [ -f \"$BRIDGES2_TRAINING_VCPKG_TRIPLET_DIR/share/flatbuffers/flatbuffers-config.cmake\" ]; then echo \"  vcpkg manifest install: reusing existing TrainingLoop installed tree\"; cmake_manifest_install=OFF; else echo \"  vcpkg manifest install: running because installed tree is missing or incomplete\"; fi"
BRIDGES2_RUN_ID="$$.$RANDOM"
BRIDGES2_REMOTE_STATE_DIR="$BRIDGES2_DIR/.grim_bridges2_runtime"
BRIDGES2_BUILD_STATE_FILE="$BRIDGES2_REMOTE_STATE_DIR/run_train_on_bridges2.build.$BRIDGES2_RUN_ID.state"
BRIDGES2_VCPKG_LOCK_PATH="$BRIDGES2_VCPKG/.vcpkg-root"
BRIDGES2_VCPKG_LOCK_SWEEP="grim_lock_path=\"$BRIDGES2_VCPKG_LOCK_PATH\";
grim_state_dir=\"$BRIDGES2_REMOTE_STATE_DIR\";
grim_vcpkg_ps_regex='/vcpkg([[:space:]]|$)|/downloads/tools/.*/cmake([[:space:]]|$)|/downloads/tools/.*/ninja([[:space:]]|$)';
grim_lock_in_use=0;
grim_holder_pids=\"\";
grim_active_vcpkg=\$(ps -eo pid=,args= | grep -F -- \"$BRIDGES2_VCPKG\" | grep -E \"\$grim_vcpkg_ps_regex\" | grep -v grep || true);
grim_scan_lock_holders() {
  grim_holder_pids=\"\";
  if [ -e \"\$grim_lock_path\" ]; then
    if command -v lsof >/dev/null 2>&1; then
      grim_holder_pids=\$(lsof -t \"\$grim_lock_path\" 2>/dev/null | tr '\\n' ' ');
    elif command -v fuser >/dev/null 2>&1; then
      grim_holder_pids=\$(fuser \"\$grim_lock_path\" 2>/dev/null | tr '\\n' ' ');
    fi;
  fi;
};
grim_detect_lock_state() {
  grim_lock_probe_result=unknown;
  grim_lock_probe_error=\"\";
  grim_lock_probe_rc=0;
  if [ ! -e \"\$grim_lock_path\" ]; then
    touch \"\$grim_lock_path\";
  fi;
  if command -v flock >/dev/null 2>&1; then
    exec 9>>\"\$grim_lock_path\";
    grim_lock_probe_error=\$(flock -n 9 2>&1);
    grim_lock_probe_rc=\$?;
    if [ \"\$grim_lock_probe_rc\" -eq 0 ]; then
      grim_lock_probe_result=free;
      flock -u 9;
    elif [ \"\$grim_lock_probe_rc\" -eq 1 ]; then
      grim_lock_probe_result=busy;
    else
      grim_lock_probe_result=unsupported;
    fi;
    exec 9>&-;
  fi;
};
grim_scan_lock_holders;
[ -n \"\$grim_active_vcpkg\" ] && grim_lock_in_use=1;
[ -n \"\$grim_holder_pids\" ] && grim_lock_in_use=1;
grim_other_active_state=0;
if [ -d \"\$grim_state_dir\" ]; then
  for grim_state_path in \"\$grim_state_dir\"/run_train_on_bridges2.build.*.state; do
    [ -f \"\$grim_state_path\" ] || continue;
    [ \"\$grim_state_path\" = \"\$remote_state_file\" ] && continue;
    grim_state_pgid=\$(grep '^pgid=' \"\$grim_state_path\" | head -n 1 | cut -d= -f2- || true);
    if [ -n \"\$grim_state_pgid\" ] && ps -eo pgid= | tr -d '[:space:]' | grep -Fx \"\$grim_state_pgid\" >/dev/null 2>&1; then
      grim_other_active_state=1;
      break;
    fi;
  done;
fi;
if [ \"\$grim_lock_in_use\" -eq 1 ] && [ \"\$grim_other_active_state\" -eq 0 ] && { [ -n \"\$grim_active_vcpkg\" ] || [ -n \"\$grim_holder_pids\" ]; }; then
  echo \"  vcpkg lock sweep: recovering orphaned holder(s) with no competing launcher state\";
  printf '%s\\n' \"\$grim_active_vcpkg\" | sed 's/^/    /';
  grim_orphan_pids=\$({ printf '%s\\n' \"\$grim_active_vcpkg\" | awk '{print \$1}'; printf '%s\\n' \"\$grim_holder_pids\" | tr ' ' '\\n'; } | sed '/^$/d' | sort -u | tr '\\n' ' ');
  if [ -n \"\$grim_orphan_pids\" ]; then
    kill -TERM \$grim_orphan_pids 2>/dev/null || true;
    sleep 2;
    grim_active_vcpkg=\$(ps -eo pid=,args= | grep -F -- \"$BRIDGES2_VCPKG\" | grep -E \"\$grim_vcpkg_ps_regex\" | grep -v grep || true);
    grim_scan_lock_holders;
    grim_detect_lock_state;
    if [ -n \"\$grim_active_vcpkg\" ] || [ -n \"\$grim_holder_pids\" ] || [ \"\$grim_lock_probe_result\" = busy ]; then
      kill -KILL \$grim_orphan_pids 2>/dev/null || true;
    fi;
    for _ in 1 2 3 4 5; do
      sleep 1;
      grim_active_vcpkg=\$(ps -eo pid=,args= | grep -F -- \"$BRIDGES2_VCPKG\" | grep -E \"\$grim_vcpkg_ps_regex\" | grep -v grep || true);
      grim_scan_lock_holders;
      grim_detect_lock_state;
      if [ -z \"\$grim_active_vcpkg\" ] && [ -z \"\$grim_holder_pids\" ] && [ \"\$grim_lock_probe_result\" != busy ]; then
        break;
      fi;
    done;
  fi;
fi;
grim_active_vcpkg=\$(ps -eo pid=,args= | grep -F -- \"$BRIDGES2_VCPKG\" | grep -E \"\$grim_vcpkg_ps_regex\" | grep -v grep || true);
grim_scan_lock_holders;
grim_detect_lock_state;
grim_lock_in_use=0;
if [ \"\$grim_lock_probe_result\" = busy ]; then
  [ -n \"\$grim_active_vcpkg\" ] && grim_lock_in_use=1;
  [ -n \"\$grim_holder_pids\" ] && grim_lock_in_use=1;
  [ \"\$grim_other_active_state\" -eq 1 ] && grim_lock_in_use=1;
elif [ \"\$grim_lock_probe_result\" != free ]; then
  [ -n \"\$grim_active_vcpkg\" ] && grim_lock_in_use=1;
  [ -n \"\$grim_holder_pids\" ] && grim_lock_in_use=1;
  [ \"\$grim_other_active_state\" -eq 1 ] && grim_lock_in_use=1;
fi;
if [ \"\$grim_lock_in_use\" -eq 0 ]; then
  if [ \"\$grim_lock_probe_result\" = free ]; then
    echo \"  vcpkg lock sweep: filesystem lock is free at \$grim_lock_path\";
  elif [ \"\$grim_lock_probe_result\" = busy ]; then
    echo \"  vcpkg lock sweep: flock reported a busy lock at \$grim_lock_path, but no holder is visible; treating the probe as unreliable on this filesystem and continuing\";
  elif [ \"\$grim_lock_probe_result\" = unsupported ]; then
    echo \"  vcpkg lock sweep: flock probe is unavailable for \$grim_lock_path; falling back to process scan\";
    [ -n \"\$grim_lock_probe_error\" ] && echo \"    flock probe detail: \$grim_lock_probe_error\";
  else
    echo \"  vcpkg lock sweep: no active holder detected for \$grim_lock_path\";
  fi;
else
  echo \"  vcpkg lock sweep: filesystem lock still appears busy at \$grim_lock_path\";
  [ -n \"\$grim_active_vcpkg\" ] && printf '%s\\n' \"\$grim_active_vcpkg\" | sed 's/^/    holder: /';
  [ -n \"\$grim_holder_pids\" ] && echo \"    holder pids: \$grim_holder_pids\";
  [ \"\$grim_other_active_state\" -eq 1 ] && echo \"    holder source: another run_train_on_bridges2 build state is still active\";
  [ \"\$grim_lock_probe_result\" = unsupported ] && [ -n \"\$grim_lock_probe_error\" ] && echo \"    flock probe detail: \$grim_lock_probe_error\";
  echo \"ERROR: refusing to continue while the vcpkg filesystem lock is still busy.\" >&2;
  echo \"  Wait for the holder above to exit, or clear that stuck process on Bridges-2, then rerun.\" >&2;
  exit 1;
fi"
BRIDGES2_REMOTE_BUILD_STATE_SETUP="runtime_dir=\"$BRIDGES2_REMOTE_STATE_DIR\"; mkdir -p \"\$runtime_dir\"; remote_state_file=\"$BRIDGES2_BUILD_STATE_FILE\"; remote_pgid=\$(ps -o pgid= -p \$\$ | tr -d '[:space:]'); printf '%s\\n' \"kind=build\" \"pid=\$\$\" \"pgid=\$remote_pgid\" \"lock_path=$BRIDGES2_VCPKG_LOCK_PATH\" \"vcpkg_root=$BRIDGES2_VCPKG\" \"build_dir=$BRIDGES2_DIR/$BUILD_DIR\" \"training_vcpkg_triplet_dir=$BRIDGES2_TRAINING_VCPKG_TRIPLET_DIR\" > \"\$remote_state_file\"; if command -v setsid >/dev/null 2>&1; then { setsid bash -c 'state_file=\"\$1\"; target_pgid=\"\$2\"; while ps -eo pgid= | tr -d \"[:space:]\" | grep -Fx \"\$target_pgid\" >/dev/null 2>&1; do sleep 1; done; rm -f \"\$state_file\"' _ \"\$remote_state_file\" \"\$remote_pgid\" >/dev/null 2>&1 & }; else { nohup bash -c 'state_file=\"\$1\"; target_pgid=\"\$2\"; while ps -eo pgid= | tr -d \"[:space:]\" | grep -Fx \"\$target_pgid\" >/dev/null 2>&1; do sleep 1; done; rm -f \"\$state_file\"' _ \"\$remote_state_file\" \"\$remote_pgid\" >/dev/null 2>&1 & }; fi"
BRIDGES2_CMAKE_PRESET="${GRIM_BRIDGES2_CMAKE_PRESET:-}"
if [[ -z "$BRIDGES2_CMAKE_PRESET" ]]; then
  if [[ "$GPU_TYPE" == "h100-80" ]]; then
    BRIDGES2_CMAKE_PRESET="bridges2-h100-release"
  else
    BRIDGES2_CMAKE_PRESET="bridges2-v100-release"
  fi
fi
BRIDGES2_CMAKE_FALLBACK_PRESET=""
case "$BRIDGES2_CMAKE_PRESET" in
  bridges2-h100-release) BRIDGES2_CMAKE_FALLBACK_PRESET="bridges2-h100-release-make" ;;
  bridges2-v100-release) BRIDGES2_CMAKE_FALLBACK_PRESET="bridges2-v100-release-make" ;;
esac
BRIDGES2_BUILD_TOOL_DETECT="grim_resolved_preset=\"$BRIDGES2_CMAKE_PRESET\"; grim_cmake_make_program=\"\"; grim_cmake_generator=\"\"; if command -v ninja >/dev/null 2>&1; then grim_cmake_make_program=\$(command -v ninja); grim_cmake_generator='Ninja'; echo \"  CMake generator backend: Ninja (\$grim_cmake_make_program)\"; elif command -v ninja-build >/dev/null 2>&1; then grim_cmake_make_program=\$(command -v ninja-build); grim_cmake_generator='Ninja'; echo \"  CMake generator backend: Ninja via ninja-build (\$grim_cmake_make_program)\"; elif command -v make >/dev/null 2>&1; then grim_cmake_make_program=\$(command -v make); grim_cmake_generator='Unix Makefiles'; if [ -n \"$BRIDGES2_CMAKE_FALLBACK_PRESET\" ]; then grim_resolved_preset=\"$BRIDGES2_CMAKE_FALLBACK_PRESET\"; fi; echo \"  CMake generator backend: Unix Makefiles fallback (\$grim_cmake_make_program)\"; elif command -v gmake >/dev/null 2>&1; then grim_cmake_make_program=\$(command -v gmake); grim_cmake_generator='Unix Makefiles'; if [ -n \"$BRIDGES2_CMAKE_FALLBACK_PRESET\" ]; then grim_resolved_preset=\"$BRIDGES2_CMAKE_FALLBACK_PRESET\"; fi; echo \"  CMake generator backend: Unix Makefiles fallback via gmake (\$grim_cmake_make_program)\"; else echo \"ERROR: no compatible CMake generator backend found on Bridges-2 (need ninja, ninja-build, make, or gmake on PATH).\" >&2; exit 1; fi; export GRIM_BRIDGES2_RESOLVED_PRESET=\"\$grim_resolved_preset\"; export GRIM_BRIDGES2_CMAKE_GENERATOR=\"\$grim_cmake_generator\"; export GRIM_BRIDGES2_CMAKE_MAKE_PROGRAM=\"\$grim_cmake_make_program\""
BRIDGES2_DEP_MODE_LABEL="TrainingLoop vcpkg manifest"
BRIDGES2_DEP_BOOTSTRAP="$BRIDGES2_VCPKG_ENSURE && $BRIDGES2_VCPKG_LOCK_SWEEP && $BRIDGES2_MANIFEST_ASSERT"
BRIDGES2_DEP_PRECONFIG="$BRIDGES2_VCPKG_TOOL_POLICY && $BRIDGES2_VCPKG_ENV && $BRIDGES2_VCPKG_MANIFEST_POLICY"
BRIDGES2_CMAKE_DEP_ARGS="-DCMAKE_TOOLCHAIN_FILE=\"$VCPKG_TOOLCHAIN\" -DVCPKG_MANIFEST_DIR=\"$BRIDGES2_DIR/$TRAINING_DIR\" -DVCPKG_INSTALLED_DIR=\"$BRIDGES2_TRAINING_VCPKG_INSTALLED\" -DVCPKG_TARGET_TRIPLET=x64-linux -DVCPKG_MANIFEST_INSTALL=\$cmake_manifest_install -DGRIM_TRAINING_USE_MANUAL_DEPS=OFF -DGRIM_TRAINING_ENABLE_VCPKG_FALLBACK=ON"
BRIDGES2_PREP_BUILD="grim_need_configure=0; if [ -f \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" ]; then cached_toolchain=\$(grep '^CMAKE_TOOLCHAIN_FILE:FILEPATH=' \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" | cut -d= -f2- || true); cached_installed=\$(grep '^VCPKG_INSTALLED_DIR:PATH=' \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" | cut -d= -f2- || true); cached_manual=\$(grep '^GRIM_TRAINING_USE_MANUAL_DEPS:BOOL=' \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" | cut -d= -f2- || true); if [ -n \"\$cached_toolchain\" ] && [ \"\$cached_toolchain\" != \"$VCPKG_TOOLCHAIN\" ]; then echo \"  stale CMake toolchain cache: \$cached_toolchain -> $VCPKG_TOOLCHAIN\"; echo \"  removing $BRIDGES2_DIR/$BUILD_DIR so CMake can reconfigure with the pinned vcpkg cache\"; rm -rf \"$BRIDGES2_DIR/$BUILD_DIR\"; grim_need_configure=1; elif [ -n \"\$cached_installed\" ] && [ \"\$cached_installed\" != \"$BRIDGES2_TRAINING_VCPKG_INSTALLED\" ]; then echo \"  stale TrainingLoop vcpkg installed cache: \$cached_installed -> $BRIDGES2_TRAINING_VCPKG_INSTALLED\"; echo \"  removing $BRIDGES2_DIR/$BUILD_DIR so CMake can reconfigure the manifest install root\"; rm -rf \"$BRIDGES2_DIR/$BUILD_DIR\"; grim_need_configure=1; elif [ -n \"\$cached_manual\" ] && [ \"\$cached_manual\" != \"OFF\" ]; then echo \"  stale TrainingLoop dependency mode cache: manual deps -> OFF\"; echo \"  removing $BRIDGES2_DIR/$BUILD_DIR so CMake can reconfigure for manifest-mode vcpkg\"; rm -rf \"$BRIDGES2_DIR/$BUILD_DIR\"; grim_need_configure=1; fi; else grim_need_configure=1; fi"
BRIDGES2_PRECONFIGURE_BUILD="{ if [ -f \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" ]; then cached_generator=\$(grep '^CMAKE_GENERATOR:INTERNAL=' \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" | cut -d= -f2- || true); cached_make_program=\$(grep '^CMAKE_MAKE_PROGRAM:FILEPATH=' \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" | cut -d= -f2- || true); if [ -n \"\$cached_generator\" ] && [ \"\$cached_generator\" != \"\$GRIM_BRIDGES2_CMAKE_GENERATOR\" ]; then echo \"  stale CMake generator cache: \$cached_generator -> \$GRIM_BRIDGES2_CMAKE_GENERATOR\"; echo \"  removing $BRIDGES2_DIR/$BUILD_DIR so CMake can reconfigure with the current generator backend\"; rm -rf \"$BRIDGES2_DIR/$BUILD_DIR\"; grim_need_configure=1; elif [ -n \"\$cached_make_program\" ] && [ \"\$cached_make_program\" != \"\$GRIM_BRIDGES2_CMAKE_MAKE_PROGRAM\" ]; then echo \"  stale CMake make-program cache: \$cached_make_program -> \$GRIM_BRIDGES2_CMAKE_MAKE_PROGRAM\"; echo \"  removing $BRIDGES2_DIR/$BUILD_DIR so CMake can reconfigure with the current generator executable\"; rm -rf \"$BRIDGES2_DIR/$BUILD_DIR\"; grim_need_configure=1; fi; fi; if [ ! -f \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" ]; then grim_need_configure=1; fi; }"
BRIDGES2_CONFIGURE_STEP="{ if [ \"\${cmake_manifest_install}\" = \"ON\" ] || [ \"\${grim_need_configure:-1}\" = \"1\" ]; then echo \"  CMake configure: running for $BRIDGES2_DIR/$BUILD_DIR\"; cmake -S . -B \"$BRIDGES2_DIR/$BUILD_DIR\" -G \"\$GRIM_BRIDGES2_CMAKE_GENERATOR\" -DCMAKE_MAKE_PROGRAM=\"\$GRIM_BRIDGES2_CMAKE_MAKE_PROGRAM\" -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT=\$GRIM_CUDA_ROOT $BRIDGES2_CMAKE_DEP_ARGS; else echo \"  CMake configure: reusing existing build tree $BRIDGES2_DIR/$BUILD_DIR\"; fi; }"

# --build
# Default to building train_gpu PLUS train_tokenizer because train_gpu spawns
# train_tokenizer as a subprocess at runtime (see Subprocess/tokenizer_subprocess.cpp);
# rebuilding train_gpu alone leaves a stale train_tokenizer that's pinned to whatever
# IPC contract was current the last time it was built. The two binaries share an
# IPC schema (--status-file + status JSON envelope), so any time
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
  if [[ "${GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS:-0}" == "1" ]]; then
    echo "Seeding Bridges-2 helper-tool cache from local downloads when available..."
    echo "  local helper cache: $LOCAL_VCPKG_DOWNLOADS"
    seed_remote_vcpkg_helper_archive "$LOCAL_VCPKG_CMAKE_ARCHIVE_PATH" "$REMOTE_VCPKG_CMAKE_ARCHIVE_PATH" "$BRIDGES2_VCPKG_CMAKE_ARCHIVE" "$BRIDGES2_VCPKG_CMAKE_URL" "$BRIDGES2_VCPKG_CMAKE_SHA512"
  fi
  echo "Building $BUILD_TARGET on Bridges-2 ($BRIDGES2_DIR/$BUILD_DIR)..."
  echo "  GPU type: $GPU_TYPE, CUDA arch: $([ "$GPU_TYPE" == "h100-80" ] && echo sm_90 || echo sm_80), make -j $BRIDGES2_MAKE_JOBS"
  echo "  dependency mode: $BRIDGES2_DEP_MODE_LABEL"
  echo "  vcpkg checkout path: $BRIDGES2_VCPKG (same path as local builds; pinned to external/vcpkg gitlink)"
  echo "  vcpkg manifest: $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json"
  echo "  vcpkg installed dir: $BRIDGES2_TRAINING_VCPKG_INSTALLED"
  echo "  vcpkg downloads cache: $BRIDGES2_VCPKG_DOWNLOADS"
  echo "  CMake preset: $BRIDGES2_CMAKE_PRESET (auto-falls back to -make when Ninja is unavailable)"
  ACTIVE_REMOTE_STATE_FILE="$BRIDGES2_BUILD_STATE_FILE"
  ACTIVE_REMOTE_LABEL="build"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "BRIDGES2_DIR=$BRIDGES2_DIR; $BRIDGES2_CUDA_ARCH cd \$BRIDGES2_DIR && $BRIDGES2_REMOTE_BUILD_STATE_SETUP && $BRIDGES2_SUBMODULE && $BRIDGES2_DEP_BOOTSTRAP && $BRIDGES2_PREP_BUILD && cd \$BRIDGES2_DIR/$TRAINING_DIR/TrainingLoop && ${BRIDGES2_CLEAN}$BRIDGES2_MODULES && $BRIDGES2_ENSURE_CUDA12 && $BRIDGES2_BUILD_TOOL_DETECT && $BRIDGES2_PRECONFIGURE_BUILD && $BRIDGES2_DEP_PRECONFIG && echo \"  CMake profile: \$GRIM_BRIDGES2_RESOLVED_PRESET via \$GRIM_BRIDGES2_CMAKE_GENERATOR\" && $BRIDGES2_CONFIGURE_STEP && cmake --build \"$BRIDGES2_DIR/$BUILD_DIR\" --target $BUILD_TARGET -j $BRIDGES2_MAKE_JOBS"
  ACTIVE_REMOTE_STATE_FILE=""
  ACTIVE_REMOTE_LABEL=""
fi

: # Transfer data
if [[ "$SKIP_MCS" == "1" ]]; then
  :
else
  transfer_training_file "merged_verified_cache.jsonl" "$CACHE_PATH_EXPANDED" "$REMOTE_CACHE"
fi

if [[ "$SKIP_CBS" == "1" ]]; then
  :
elif [[ -f "$CONCEPT_BLOCKS_PATH_EXPANDED" ]]; then
  transfer_training_file "concept_blocks.jsonl" "$CONCEPT_BLOCKS_PATH_EXPANDED" "$REMOTE_CONCEPT_BLOCKS"
else
  echo "Skipping concept_blocks.jsonl (not found at $CONCEPT_BLOCKS_PATH_EXPANDED)."
  echo "  DataLoader will use cache-only curriculum; add the file locally to ship UltraChat/stem blocks."
fi

if [[ "$SKIP_CRS" == "1" ]]; then
  :
elif [[ -f "$CURRICULUM_REGISTRY_PATH_EXPANDED" ]]; then
  transfer_training_file "curriculum_registry.json" "$CURRICULUM_REGISTRY_PATH_EXPANDED" "$REMOTE_CURRICULUM_REGISTRY"
else
  echo "Skipping curriculum_registry.json (not found at $CURRICULUM_REGISTRY_PATH_EXPANDED)."
fi

# Transfer canonical ai_config.json
if [[ ! -f "$REPO_ROOT/ai_config.json" ]]; then
  echo "ERROR: canonical ai_config.json not found at $REPO_ROOT/ai_config.json" >&2
  echo "  scripts/run_train_on_bridges2.sh requires the repo-root ai_config.json so train_gpu can load it canonically." >&2
  exit 1
fi
echo "Transferring ai_config.json..."
ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cat > $BRIDGES2_DIR/ai_config.json" < "$REPO_ROOT/ai_config.json"

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
  print_bridges2_time_limit
  SBATCH_EXPORT="ALL,GRIM_BRIDGES2_DIR=$BRIDGES2_DIR"
  SUBMIT_OUT=$(ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && sbatch --export=$SBATCH_EXPORT --output=$BRIDGES2_DIR/logs/train_%j.out --error=$BRIDGES2_DIR/logs/train_%j.err $SLURM_MAIL_ARGS -p $PARTITION $SLURM_ACCOUNT_ARGS --gpus=$GPU_TYPE:1 $SLURM_TIME_ARGS scripts/train_bridges2.sbatch")
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
  BRIDGES2_RUN_WRAPPER="bash -c 'source /etc/profile.d/modules.sh 2>/dev/null || true; module load cuda 2>/dev/null || true; export GRIM_PROJECT_DIR=\"$BRIDGES2_DIR\"; source \"$BRIDGES2_DIR/scripts/ensure_cuda12_for_training.sh\" 2>/dev/null || true; export PATH=\"\${GRIM_CUDA_ROOT:-}/bin:\$PATH\"; export LD_LIBRARY_PATH=\"\${GRIM_CUDA_ROOT:-}/lib64:\$LD_LIBRARY_PATH\"; cd \"$BRIDGES2_DIR\" && exec \"$REMOTE_EXE\"'"
  echo "Running train_gpu on Bridges-2 (partition=$PARTITION, gpu=$GPU_TYPE)..."
  print_bridges2_time_limit
  SRUN_ARGS="-p $PARTITION $SLURM_ACCOUNT_ARGS --gres=gpu:$GPU_TYPE:1 $SLURM_TIME_ARGS --pty"
  if [[ -t 0 ]]; then
    ssh -t $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $SRUN_ARGS $BRIDGES2_RUN_WRAPPER"
  else
    ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "cd $BRIDGES2_DIR && srun $SRUN_ARGS $BRIDGES2_RUN_WRAPPER"
  fi
fi

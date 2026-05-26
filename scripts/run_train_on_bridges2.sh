#!/usr/bin/env bash
# Run GRIM-text training on PSC Bridges-2 via SSH.
# Usage: ./scripts/run_train_on_bridges2.sh [--build] [--jobs N] [--config CONFIG] [--sbatch] [--sync TARGET...] [--sync-all|--sync-mcs|--sync-cbs|--sync-crs|--sync-fas] [--pull-vocab] [--pull-logs] [--allow-vcpkg-tool-downloads]
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
#   - vcpkg: Script uses the repo's external/vcpkg checkout on Bridges-2 by default (same path as local builds).
#     Override with GRIM_VCPKG_ROOT only when intentionally using another pinned vcpkg checkout.
#   - Tool downloads: Launcher always prefers Bridges-2 system cmake+ninja when they are available. By default it
#     exports VCPKG_FORCE_SYSTEM_BINARIES=1 and requires those tools on PATH. Use --allow-vcpkg-tool-downloads or
#     env GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1 only to permit vcpkg downloads when system cmake or ninja is missing;
#     the pinned vcpkg toolchain may then pull a newer cmake.
#   - CUDA 12+ for training (flash-attention). Bridges-2: module load cuda (check with module avail cuda)
#
# Bridges-2 GPU partitions: GPU-shared (1-4 GPUs, faster queue) or GPU (full node).
# GPU types: h100-80, v100-32, v100-16, l40s-48. Default: h100-80.
#
# Options:
#   --build          Build train_gpu before running.
#   --config X       Config (default: ai_config.json).
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
#                    Permit vcpkg to download helper tools only when Bridges-2 system cmake or ninja is missing.
#                    System tools still win when present.
#   --jobs N         make -j N for train_gpu (default 100; override with GRIM_BRIDGES2_MAKE_JOBS).
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
CONFIG="${CONFIG:-../../../../ai_config.json}"
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

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)          DO_BUILD=true; shift ;;
    --incremental)    DO_INCREMENTAL=true; shift ;;
    --clean)          DO_CLEAN_BUILD=true; shift ;;
    --config)         CONFIG="$2"; shift 2 ;;
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
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ "$ALLOW_VCPKG_TOOL_DOWNLOADS" == true ]]; then
  export GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1
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

# One long-lived SSH using a script-unique socket in /tmp (avoids ~/.ssh permission issues)
BRIDGES2_CTRL="/tmp/cm-grim-$$"
if ! ssh -f -N -M -S "$BRIDGES2_CTRL" -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new "$BRIDGES2_SSH"; then
  echo "SSH to Bridges-2 failed. Try: ssh bridges2"
  exit 1
fi
REMOTE_SNAPSHOT_PATHS=()
trap 'cleanup_bridges2_session' EXIT
BRIDGES2_SSH_OPTS="-S $BRIDGES2_CTRL -o ControlMaster=no"

remote_quote() {
  printf '%q' "$1"
}

cleanup_bridges2_session() {
  local snapshot_path
  local q_snapshot_path

  for snapshot_path in "${REMOTE_SNAPSHOT_PATHS[@]}"; do
    [[ -n "$snapshot_path" ]] || continue
    q_snapshot_path="$(remote_quote "$snapshot_path")"
    if ! ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "rm -f $q_snapshot_path" 2>/dev/null; then
      echo "ERROR: failed to remove remote snapshot: $snapshot_path" >&2
    fi
  done

  ssh -S "$BRIDGES2_CTRL" -O exit "$BRIDGES2_SSH" 2>/dev/null
  rm -f "$BRIDGES2_CTRL"
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
TRAINING_VCPKG_JSON='{"name":"grim-training","version-string":"0.1.0","dependencies":["nlohmann-json","flatbuffers"]}'
BRIDGES2_VCPKG_TOOL_POLICY="if command -v cmake >/dev/null 2>&1 && command -v ninja >/dev/null 2>&1; then echo '  vcpkg helper tools: using Bridges-2 system cmake+ninja'; export VCPKG_FORCE_SYSTEM_BINARIES=1; elif [ \"${GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS:-0}\" = \"1\" ]; then echo '  vcpkg helper tools: system cmake or ninja missing; allowing vcpkg helper-tool downloads'; unset VCPKG_FORCE_SYSTEM_BINARIES; else echo 'ERROR: Bridges-2 system cmake and ninja are required unless GRIM_ALLOW_VCPKG_TOOL_DOWNLOADS=1.' >&2; echo '  Load the cluster cmake/ninja modules or re-run with --allow-vcpkg-tool-downloads.' >&2; exit 1; fi"
BRIDGES2_VCPKG_ENV="export VCPKG_ROOT=\"$BRIDGES2_VCPKG\"; unset Z_VCPKG_ROOT_DIR; unset _VCPKG_ROOT_DIR"
if [[ -n "${GRIM_VCPKG_ROOT:-}" ]]; then
  BRIDGES2_VCPKG_ENSURE="(set -e; $BRIDGES2_VCPKG_ENV; if [ ! -f \"$VCPKG_TOOLCHAIN\" ]; then echo \"ERROR: GRIM_VCPKG_ROOT does not contain vcpkg toolchain: $VCPKG_TOOLCHAIN\" >&2; exit 1; fi; if [ ! -x \"$BRIDGES2_VCPKG/vcpkg\" ]; then (cd \"$BRIDGES2_VCPKG\" && ./bootstrap-vcpkg.sh -disableMetrics); fi)"
else
  BRIDGES2_VCPKG_ENSURE="(set -e; cd \"$BRIDGES2_DIR\"; $BRIDGES2_VCPKG_ENV; VCPKG_PIN=\$(git ls-tree HEAD external/vcpkg | awk '{print \$3}'); if [ -z \"\$VCPKG_PIN\" ]; then echo \"ERROR: external/vcpkg gitlink not found in repo HEAD\" >&2; exit 1; fi; if ! git -C \"$BRIDGES2_VCPKG\" rev-parse --show-toplevel >/dev/null 2>&1; then echo \"Initializing external/vcpkg to match local repo layout...\"; git submodule update --init external/vcpkg; fi; VCPKG_TOP=\$(git -C \"$BRIDGES2_VCPKG\" rev-parse --show-toplevel 2>/dev/null || true); if [ \"\$VCPKG_TOP\" != \"$BRIDGES2_VCPKG\" ]; then echo \"ERROR: $BRIDGES2_VCPKG exists but is not a vcpkg git checkout; remove it or set GRIM_VCPKG_ROOT\" >&2; exit 1; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_PIN^{commit}\" 2>/dev/null; then git -C \"$BRIDGES2_VCPKG\" fetch --depth 1 origin \"\$VCPKG_PIN\" || git -C \"$BRIDGES2_VCPKG\" fetch origin \"\$VCPKG_PIN\"; fi; git -c advice.detachedHead=false -C \"$BRIDGES2_VCPKG\" checkout -f \"\$VCPKG_PIN\"; VCPKG_BASELINE=\$(grep -o '\"builtin-baseline\"[[:space:]]*:[[:space:]]*\"[^\"]*\"' \"$BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json\" | head -n 1 | sed -E 's/.*\"([^\"]*)\"$/\1/'); if [ -z \"\$VCPKG_BASELINE\" ]; then echo \"ERROR: builtin-baseline missing from $BRIDGES2_DIR/$TRAINING_DIR/vcpkg.json\" >&2; exit 1; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_BASELINE^{commit}\" 2>/dev/null; then git -C \"$BRIDGES2_VCPKG\" fetch --depth 1 origin \"\$VCPKG_BASELINE\" || git -C \"$BRIDGES2_VCPKG\" fetch origin \"\$VCPKG_BASELINE\"; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_BASELINE:versions/baseline.json\" 2>/dev/null; then git -C \"$BRIDGES2_VCPKG\" fetch origin \"\$VCPKG_BASELINE\" || true; fi; if ! git -C \"$BRIDGES2_VCPKG\" cat-file -e \"\$VCPKG_BASELINE:versions/baseline.json\" 2>/dev/null; then echo \"ERROR: vcpkg builtin-baseline \$VCPKG_BASELINE is unavailable in $BRIDGES2_VCPKG; remove the checkout or fetch the missing history.\" >&2; exit 1; fi; if [ ! -f \"$VCPKG_TOOLCHAIN\" ]; then echo \"ERROR: pinned vcpkg toolchain not found: $VCPKG_TOOLCHAIN\" >&2; exit 1; fi; if [ ! -x \"$BRIDGES2_VCPKG/vcpkg\" ]; then (cd \"$BRIDGES2_VCPKG\" && ./bootstrap-vcpkg.sh -disableMetrics); fi)"
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

BRIDGES2_TRAINING_VCPKG_INSTALLED="$BRIDGES2_DIR/$TRAINING_DIR/vcpkg_installed"
BRIDGES2_CMAKE_PRESET="${GRIM_BRIDGES2_CMAKE_PRESET:-}"
if [[ -z "$BRIDGES2_CMAKE_PRESET" ]]; then
  if [[ "$GPU_TYPE" == "h100-80" ]]; then
    BRIDGES2_CMAKE_PRESET="bridges2-h100-release"
  else
    BRIDGES2_CMAKE_PRESET="bridges2-v100-release"
  fi
fi
BRIDGES2_PREP_BUILD="if [ -f \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" ]; then cached_toolchain=\$(grep '^CMAKE_TOOLCHAIN_FILE:FILEPATH=' \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" | cut -d= -f2- || true); cached_installed=\$(grep '^VCPKG_INSTALLED_DIR:PATH=' \"$BRIDGES2_DIR/$BUILD_DIR/CMakeCache.txt\" | cut -d= -f2- || true); if [ -n \"\$cached_toolchain\" ] && [ \"\$cached_toolchain\" != \"$VCPKG_TOOLCHAIN\" ]; then echo \"  stale CMake toolchain cache: \$cached_toolchain -> $VCPKG_TOOLCHAIN\"; echo \"  removing $BRIDGES2_DIR/$BUILD_DIR so CMake can reconfigure with the pinned vcpkg cache\"; rm -rf \"$BRIDGES2_DIR/$BUILD_DIR\"; elif [ -n \"\$cached_installed\" ] && [ \"\$cached_installed\" != \"$BRIDGES2_TRAINING_VCPKG_INSTALLED\" ]; then echo \"  stale TrainingLoop vcpkg installed cache: \$cached_installed -> $BRIDGES2_TRAINING_VCPKG_INSTALLED\"; echo \"  removing $BRIDGES2_DIR/$BUILD_DIR so CMake can reconfigure the manifest install root\"; rm -rf \"$BRIDGES2_DIR/$BUILD_DIR\"; fi; fi"

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
  echo "  vcpkg checkout: $BRIDGES2_VCPKG (same path as local builds; pinned to external/vcpkg gitlink)"
  echo "  CMake preset: $BRIDGES2_CMAKE_PRESET"
  ssh $BRIDGES2_SSH_OPTS "$BRIDGES2_SSH" "BRIDGES2_DIR=$BRIDGES2_DIR; $BRIDGES2_CUDA_ARCH cd \$BRIDGES2_DIR && $BRIDGES2_SUBMODULE && $BRIDGES2_VCPKG_ENSURE && $BRIDGES2_MANIFEST_ENSURE && $BRIDGES2_PREP_BUILD && cd \$BRIDGES2_DIR/$TRAINING_DIR/TrainingLoop && ${BRIDGES2_CLEAN}$BRIDGES2_MODULES && $BRIDGES2_ENSURE_CUDA12 && $BRIDGES2_VCPKG_TOOL_POLICY && $BRIDGES2_VCPKG_ENV && cmake --preset $BRIDGES2_CMAKE_PRESET -DCUDAToolkit_ROOT=\$GRIM_CUDA_ROOT && cmake --build --preset $BRIDGES2_CMAKE_PRESET --target $BUILD_TARGET -j $BRIDGES2_MAKE_JOBS"
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

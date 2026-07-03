#!/usr/bin/env bash
# Sync GRIM-text checkpoints (and optional subtrees) between your machine and PSC Bridges-2.
# Bridges-2 project space lives under /ocean/projects/<alloc>/<user>/ — often called "ocean" paths.
#
# NOTE: Keep this file with LF line endings for bash compatibility on Windows.
#
# Usage:
#   ./scripts/sync_models_bridges2.sh collect [--dry-run] [--skip-checkpoints] [--pull-vocab] [--pull-grmt] [--subpath REL]
#   ./scripts/sync_models_bridges2.sh pull-logs [--dry-run] [--subpath REL]
#   ./scripts/sync_models_bridges2.sh delete [--dry-run] [--yes] [--subpath REL]
#   ./scripts/sync_models_bridges2.sh both   [--dry-run] [--yes] [--pull-vocab] [--pull-grmt] [--subpath REL]
#
# Modes:
#   collect — rsync FROM Bridges-2 → local repo (pull checkpoints off ocean storage)
#   pull-logs — rsync training logs FROM Bridges-2 → local repo
#   delete  — remove the same path ON Bridges-2 only (frees /ocean quota; does not delete local)
#   both    — collect then delete remote (safe order: copy first, then remove from cluster)
#
# Environment (same family as run_train_on_bridges2.sh):
#   GRIM_BRIDGES2_DIR          Remote repo root (default: /ocean/projects/cis210058p/uwadkins/G.R.I.M)
#   GRIM_BRIDGES2_SSH          SSH host or user@host (default: bridges2, else uwadkins@bridges2.psc.edu)
#   GRIM_BRIDGES2_ACCOUNT      Shown in help text only (for documentation)
#   GRIM_BRIDGES2_SYNC_RELATIVE  Path under repo to sync (default: resources/models/GRIM-text/checkpoints)
#   GRIM_BRIDGES2_SSH_MUX      SSH ControlMaster use: auto, on, off (default: auto; pull-logs defaults off)
#
# Examples:
#   ./scripts/sync_models_bridges2.sh collect
#   ./scripts/sync_models_bridges2.sh collect --pull-vocab
#   ./scripts/sync_models_bridges2.sh collect --skip-checkpoints --pull-vocab
#   ./scripts/sync_models_bridges2.sh collect --pull-vocab --pull-grmt
#   ./scripts/sync_models_bridges2.sh collect --dry-run
#   ./scripts/sync_models_bridges2.sh pull-logs
#   ./scripts/sync_models_bridges2.sh delete --yes
#   ./scripts/sync_models_bridges2.sh both --yes --pull-vocab --pull-grmt
#   GRIM_BRIDGES2_SYNC_RELATIVE=resources/models/GRIM-text/training/logs ./scripts/sync_models_bridges2.sh collect

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_CHECKPOINTS_REL="resources/models/GRIM-text/checkpoints"
TRAINING_LOGS_REL="resources/models/GRIM-text/training/logs"

BRIDGES2_DIR="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis210058p/uwadkins/G.R.I.M}"
BRIDGES2_SSH="${GRIM_BRIDGES2_SSH:-bridges2}"
if [[ "$BRIDGES2_SSH" == "bridges2" ]] && ! grep -q "Host bridges2" ~/.ssh/config 2>/dev/null; then
  BRIDGES2_SSH="uwadkins@bridges2.psc.edu"
fi

SYNC_REL="${GRIM_BRIDGES2_SYNC_RELATIVE:-$DEFAULT_CHECKPOINTS_REL}"

MODE=""
DRY_RUN=false
SKIP_CONFIRM=false
SKIP_CHECKPOINTS=false
PULL_VOCAB=false
PULL_GRMT=false
SUBPATH=""
REMOTE_RSYNC_CMD=""
TRANSFER_METHOD="${GRIM_BRIDGES2_TRANSFER_METHOD:-auto}"
GZIP_LEVEL="${GRIM_BRIDGES2_GZIP_LEVEL:-1}"
SSH_MUX="${GRIM_BRIDGES2_SSH_MUX:-auto}"

log_progress() {
  local pct="$1"
  shift
  echo "[sync][$(date '+%Y-%m-%d %H:%M:%S')][${pct}%] $*"
}

progress_stream() {
  local label="${1:-download}"
  local size_bytes="${2:-}"
  if command -v pv >/dev/null 2>&1; then
    if [[ -n "$size_bytes" && "$size_bytes" =~ ^[0-9]+$ ]]; then
      pv -s "$size_bytes" -brt -N "$label"
    else
      pv -brt -N "$label"
    fi
    return 0
  fi

  cat
}

usage() {
  sed -n '1,50p' "$0" | tail -n +2
  exit "${1:-0}"
}

remote_quote() {
  local escaped="${1//\'/\'\\\'\'}"
  printf "'%s'" "$escaped"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    collect|delete|both) MODE="$1"; shift ;;
    pull-logs) MODE="$1"; SYNC_REL="$TRAINING_LOGS_REL"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --yes|-y) SKIP_CONFIRM=true; shift ;;
    --skip-checkpoints) SKIP_CHECKPOINTS=true; shift ;;
    --pull-vocab) PULL_VOCAB=true; shift ;;
    --pull-grmt) PULL_GRMT=true; shift ;;
    --subpath)
      [[ $# -lt 2 ]] && { echo "ERROR: --subpath requires a value"; exit 1; }
      SUBPATH="${2// /}"
      [[ "$SUBPATH" == /* ]] && { echo "ERROR: --subpath must be relative (no leading /)"; exit 1; }
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      usage 1
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "ERROR: specify mode: collect | pull-logs | delete | both"
  usage 1
fi

if [[ "$MODE" == "delete" && ( "$PULL_VOCAB" == true || "$PULL_GRMT" == true ) ]]; then
  echo "ERROR: --pull-vocab/--pull-grmt require collect or both mode"
  exit 1
fi

if [[ "$MODE" == "pull-logs" && ( "$PULL_VOCAB" == true || "$PULL_GRMT" == true ) ]]; then
  echo "ERROR: --pull-vocab/--pull-grmt require collect or both mode"
  exit 1
fi

if [[ "$SKIP_CHECKPOINTS" == true && "$MODE" != "collect" ]]; then
  echo "ERROR: --skip-checkpoints requires collect mode"
  exit 1
fi

if [[ "$SKIP_CHECKPOINTS" == true && "$PULL_VOCAB" != true && "$PULL_GRMT" != true ]]; then
  echo "ERROR: --skip-checkpoints requires --pull-vocab and/or --pull-grmt"
  exit 1
fi

case "$TRANSFER_METHOD" in
  auto|rsync|zstd|gzip|raw) ;;
  *) echo "ERROR: GRIM_BRIDGES2_TRANSFER_METHOD must be auto, rsync, zstd, gzip, or raw (got: $TRANSFER_METHOD)"; exit 1 ;;
esac
[[ "$GZIP_LEVEL" =~ ^[1-9]$ ]] || { echo "ERROR: GRIM_BRIDGES2_GZIP_LEVEL must be 1..9 (got: $GZIP_LEVEL)"; exit 1; }
case "$SSH_MUX" in
  auto|on|off) ;;
  *) echo "ERROR: GRIM_BRIDGES2_SSH_MUX must be auto, on, or off (got: $SSH_MUX)"; exit 1 ;;
esac

if [[ "$MODE" == "pull-logs" && -z "${GRIM_BRIDGES2_SSH_MUX+x}" ]]; then
  SSH_MUX="off"
fi

REMOTE_BASE="$BRIDGES2_DIR/$SYNC_REL"
LOCAL_BASE="$REPO_ROOT/$SYNC_REL"
TRAINING_DATA_REL="resources/models/GRIM-text/training/data"
REMOTE_TRAINING_DATA_BASE="$BRIDGES2_DIR/$TRAINING_DATA_REL"
LOCAL_TRAINING_DATA_BASE="$REPO_ROOT/$TRAINING_DATA_REL"
REMOTE_VOCAB="$REMOTE_TRAINING_DATA_BASE/vocab.bin"
REMOTE_VOCAB_TXT="$REMOTE_TRAINING_DATA_BASE/vocab.txt"
REMOTE_GRMT="$REMOTE_TRAINING_DATA_BASE/training_data.grmt"
LOCAL_VOCAB="$LOCAL_TRAINING_DATA_BASE/vocab.bin"
LOCAL_VOCAB_TXT="$LOCAL_TRAINING_DATA_BASE/vocab.txt"
LOCAL_GRMT="$LOCAL_TRAINING_DATA_BASE/training_data.grmt"

if [[ -n "$SUBPATH" ]]; then
  REMOTE_BASE="$REMOTE_BASE/$SUBPATH"
  LOCAL_BASE="$LOCAL_BASE/$SUBPATH"
fi

# Use gzip compression by default for remote transfer.
# macOS ships an older rsync without --info=progress2 (needs rsync 3.1+).
RSYNC_OPTS=(-a -v -z --progress)
if rsync --help 2>&1 | grep -q 'info=progress2'; then
  RSYNC_OPTS=(-a -v -z --info=progress2)
fi
[[ "$DRY_RUN" == true ]] && RSYNC_OPTS+=(--dry-run)

ssh_master() {
  if [[ "$SSH_MUX" == "off" ]]; then
    log_progress 10 "Using direct SSH sessions to $BRIDGES2_SSH (ControlMaster disabled)"
    BRIDGES2_SSH_OPTS=(-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
    # rsync -e must be one shell-invokable command string
    RSYNC_SSH_CMD="ssh ${BRIDGES2_SSH_OPTS[*]}"
    return 0
  fi

  log_progress 10 "Opening SSH control connection to $BRIDGES2_SSH"
  BRIDGES2_CTRL="/tmp/cm-grim-sync-$$"
  if ! ssh -f -N -M -S "$BRIDGES2_CTRL" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "$BRIDGES2_SSH"; then
    if [[ "$SSH_MUX" == "auto" ]]; then
      log_progress 10 "SSH ControlMaster unavailable; falling back to direct SSH sessions"
      BRIDGES2_SSH_OPTS=(-o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new)
      RSYNC_SSH_CMD="ssh ${BRIDGES2_SSH_OPTS[*]}"
      return 0
    fi
    echo "ERROR: SSH to Bridges-2 failed (try: ssh $BRIDGES2_SSH)"
    exit 1
  fi
  BRIDGES2_SSH_OPTS=(-S "$BRIDGES2_CTRL" -o ControlMaster=no)
  # rsync -e must be one shell-invokable command string
  RSYNC_SSH_CMD="ssh ${BRIDGES2_SSH_OPTS[*]}"
  trap 'ssh -S "$BRIDGES2_CTRL" -O exit "$BRIDGES2_SSH" 2>/dev/null; rm -f "$BRIDGES2_CTRL"' EXIT
}

remote_exists() {
  log_progress 20 "Checking remote path exists: $REMOTE_BASE"
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "test -e \"$REMOTE_BASE\""
}

remote_path_exists() {
  local remote_path="$1"
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "test -e \"$remote_path\""
}

detect_remote_rsync() {
  if [[ -n "$REMOTE_RSYNC_CMD" ]]; then
    return 0
  fi

  REMOTE_RSYNC_CMD="$(ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" '
    if command -v rsync >/dev/null 2>&1; then
      command -v rsync
    elif [ -x /usr/bin/rsync ]; then
      echo /usr/bin/rsync
    elif [ -x /bin/rsync ]; then
      echo /bin/rsync
    fi
  ' | head -n 1 | tr -d "\r")"

  [[ -n "$REMOTE_RSYNC_CMD" ]]
}

remote_has_command() {
  local cmd="$1"
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "command -v $cmd >/dev/null 2>&1"
}

collect_via_tar() {
  log_progress 40 "Remote rsync unavailable; using tar-over-ssh fallback"
  log_progress 50 "Starting gzip-compressed full-copy stream (download bytes will stream below)"
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" \
    "if [ ! -d \"$REMOTE_BASE\" ]; then echo 'ERROR: Remote path is not a directory: $REMOTE_BASE' 1>&2; exit 1; fi; cd \"$REMOTE_BASE\" && tar -czf - ." \
    | progress_stream "collect" \
    | tar -xzpf - -C "$LOCAL_BASE"
  log_progress 75 "Completed gzip tar stream transfer"
}

latest_remote_checkpoint_bin() {
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" \
    "ls -1 \"$REMOTE_BASE\"/checkpoint_epoch_*.bin 2>/dev/null | sort -V | tail -n 1" | tr -d "\r"
}

remote_file_size_bytes() {
  local remote_file="$1"
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" \
    "stat -c%s \"$remote_file\" 2>/dev/null || wc -c < \"$remote_file\" 2>/dev/null" \
    | tr -d "\r" | tail -n 1
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
}

log_latest_checkpoint_download_target() {
  local latest_bin_remote latest_bin_size
  latest_bin_remote="$(latest_remote_checkpoint_bin)"
  if [[ -z "$latest_bin_remote" ]]; then
    log_progress 30 "No checkpoint_epoch_*.bin found on Bridges-2 under $REMOTE_BASE"
    return 0
  fi

  latest_bin_size="$(remote_file_size_bytes "$latest_bin_remote")"
  if [[ -z "$latest_bin_size" ]]; then
    latest_bin_size="unknown"
  fi

  log_progress 30 "Checkpoint target: $(basename "$latest_bin_remote") (${latest_bin_size} bytes on Bridges-2)"
}

pull_single_remote_file() {
  local remote_file="$1"
  local local_file="$2"
  local local_dir q_remote_file size_bytes tmp_path decompressor_name=""
  local prefer_raw_before_gzip=false

  local_dir="$(dirname "$local_file")"
  mkdir -p "$local_dir"
  q_remote_file="$(remote_quote "$remote_file")"
  size_bytes="$(remote_file_size_bytes "$remote_file")"
  tmp_path="$local_file.transfer.$$"
  rm -f "$tmp_path"

  if [[ "$TRANSFER_METHOD" == "auto" ]] && [[ "$remote_file" == *.bin || "$remote_file" == *.bin.mtp || "$remote_file" == *.mtp ]]; then
    prefer_raw_before_gzip=true
  fi

  if [[ "$DRY_RUN" == true ]]; then
    log_progress 92 "[dry-run] Would download $(basename "$remote_file") (${size_bytes:-unknown} bytes)"
    return 0
  fi

  if [[ "$TRANSFER_METHOD" == "auto" || "$TRANSFER_METHOD" == "rsync" ]]; then
    if command -v rsync >/dev/null 2>&1; then
      if detect_remote_rsync; then
        log_progress 92 "method: rsync --compress --partial --progress"
        if rsync -a --compress --partial --progress --rsync-path "$REMOTE_RSYNC_CMD" -e "$RSYNC_SSH_CMD" \
          "$BRIDGES2_SSH:$remote_file" "$local_file"; then
          verify_local_size "$(basename "$remote_file")" "$local_file" "$size_bytes"
          return 0
        fi
        if [[ "$TRANSFER_METHOD" == "rsync" ]]; then
          echo "ERROR: rsync transfer failed for $(basename "$remote_file")"
          exit 1
        fi
        log_progress 92 "rsync failed; trying compressed stream fallback"
      elif [[ "$TRANSFER_METHOD" == "rsync" ]]; then
        echo "ERROR: rsync requested but not found on Bridges-2. Set GRIM_BRIDGES2_TRANSFER_METHOD=gzip or zstd."
        exit 1
      fi
    elif [[ "$TRANSFER_METHOD" == "rsync" ]]; then
      echo "ERROR: rsync requested but not found locally."
      exit 1
    fi
  fi

  if [[ "$TRANSFER_METHOD" == "auto" || "$TRANSFER_METHOD" == "zstd" ]]; then
    if command -v zstd >/dev/null 2>&1 && remote_has_command zstd; then
      log_progress 92 "method: ssh zstd -1 -T0 -c | local zstd -dc"
      ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "set -e; zstd -1 -T0 -c $q_remote_file" \
        | progress_stream "$(basename "$remote_file")" "$size_bytes" \
        | zstd -dc > "$tmp_path"
      verify_local_size "$(basename "$remote_file")" "$tmp_path" "$size_bytes"
      mv -f "$tmp_path" "$local_file"
      return 0
    elif [[ "$TRANSFER_METHOD" == "zstd" ]]; then
      echo "ERROR: zstd requested, but local zstd or remote Bridges-2 zstd is unavailable."
      exit 1
    fi
  fi

  if [[ "$TRANSFER_METHOD" == "auto" && "$prefer_raw_before_gzip" == true ]]; then
    log_progress 92 "method: raw ssh cat (checkpoint heuristic: prefer raw over gzip)"
    ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "set -e; cat $q_remote_file" \
      | progress_stream "$(basename "$remote_file")" "$size_bytes" > "$tmp_path"
    verify_local_size "$(basename "$remote_file")" "$tmp_path" "$size_bytes"
    mv -f "$tmp_path" "$local_file"
    return 0
  fi

  if [[ "$TRANSFER_METHOD" == "auto" || "$TRANSFER_METHOD" == "gzip" ]]; then
    if command -v pigz >/dev/null 2>&1; then
      decompressor_name="pigz"
    elif command -v gzip >/dev/null 2>&1; then
      decompressor_name="gzip"
    fi

    if [[ -n "$decompressor_name" ]] && remote_has_command gzip; then
      log_progress 92 "method: ssh gzip -$GZIP_LEVEL -c | local $decompressor_name -dc"
      ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "set -e; gzip -$GZIP_LEVEL -c $q_remote_file" \
        | progress_stream "$(basename "$remote_file")" "$size_bytes" \
        | "$decompressor_name" -dc > "$tmp_path"
      verify_local_size "$(basename "$remote_file")" "$tmp_path" "$size_bytes"
      mv -f "$tmp_path" "$local_file"
      return 0
    elif [[ "$TRANSFER_METHOD" == "gzip" ]]; then
      echo "ERROR: gzip requested, but local decompressor or remote gzip is unavailable."
      exit 1
    fi
  fi

  log_progress 92 "method: raw ssh cat (fallback)"
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "set -e; cat $q_remote_file" \
    | progress_stream "$(basename "$remote_file")" "$size_bytes" > "$tmp_path"
  verify_local_size "$(basename "$remote_file")" "$tmp_path" "$size_bytes"
  mv -f "$tmp_path" "$local_file"
  return 0
}

ensure_latest_checkpoint_pair() {
  local latest_bin_remote latest_bin_name latest_bin_local latest_mtp_remote latest_mtp_local

  latest_bin_remote="$(latest_remote_checkpoint_bin)"
  if [[ -z "$latest_bin_remote" ]]; then
    log_progress 90 "No checkpoint_epoch_*.bin found under $REMOTE_BASE"
    return 0
  fi

  latest_bin_name="$(basename "$latest_bin_remote")"
  latest_bin_local="$LOCAL_BASE/$latest_bin_name"
  log_progress 90 "Ensuring latest checkpoint exists locally: $latest_bin_name"
  pull_single_remote_file "$latest_bin_remote" "$latest_bin_local"

  latest_mtp_remote="$latest_bin_remote.mtp"
  latest_mtp_local="$latest_bin_local.mtp"
  if ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "test -f \"$latest_mtp_remote\""; then
    log_progress 95 "Ensuring latest MTP checkpoint exists locally: $(basename "$latest_mtp_remote")"
    pull_single_remote_file "$latest_mtp_remote" "$latest_mtp_local"
  else
    log_progress 95 "Latest MTP checkpoint not found for $latest_bin_name"
  fi
}

pull_vocab_artifacts() {
  log_progress 96 "Collecting vocab artifacts (vocab.bin, optional vocab.txt)"
  echo "[collect:vocab] Remote: $BRIDGES2_SSH:$REMOTE_TRAINING_DATA_BASE"
  echo "[collect:vocab] Local:  $LOCAL_TRAINING_DATA_BASE"
  mkdir -p "$LOCAL_TRAINING_DATA_BASE"

  if ! remote_path_exists "$REMOTE_VOCAB"; then
    echo "ERROR: requested --pull-vocab, but remote vocab is missing: $REMOTE_VOCAB"
    exit 1
  fi
  pull_single_remote_file "$REMOTE_VOCAB" "$LOCAL_VOCAB"

  if remote_path_exists "$REMOTE_VOCAB_TXT"; then
    pull_single_remote_file "$REMOTE_VOCAB_TXT" "$LOCAL_VOCAB_TXT"
  else
    log_progress 98 "Optional vocab.txt not found on Bridges-2: $REMOTE_VOCAB_TXT"
  fi
}

pull_grmt_artifact() {
  log_progress 97 "Collecting GRMT artifact (training_data.grmt)"
  echo "[collect:grmt] Remote: $BRIDGES2_SSH:$REMOTE_GRMT"
  echo "[collect:grmt] Local:  $LOCAL_GRMT"
  mkdir -p "$LOCAL_TRAINING_DATA_BASE"

  if ! remote_path_exists "$REMOTE_GRMT"; then
    echo "ERROR: requested --pull-grmt, but remote GRMT is missing: $REMOTE_GRMT"
    exit 1
  fi
  pull_single_remote_file "$REMOTE_GRMT" "$LOCAL_GRMT"
}

do_collect() {
  log_progress 25 "Collect start"
  if [[ "$SKIP_CHECKPOINTS" != true ]]; then
    echo "[collect] Remote: $BRIDGES2_SSH:$REMOTE_BASE"
    echo "[collect] Local:  $LOCAL_BASE"
    mkdir -p "$LOCAL_BASE"
    if ! remote_exists; then
      echo "WARNING: Remote path does not exist; skipping checkpoint pull."
    else
      log_latest_checkpoint_download_target

      if [[ -n "$(latest_remote_checkpoint_bin)" ]]; then
        log_progress 40 "Checkpoint mode: downloading latest checkpoint pair with pull-logs transfer path"
        ensure_latest_checkpoint_pair
      elif detect_remote_rsync; then
        log_progress 40 "No checkpoint pattern found; syncing full tree via rsync"
        rsync "${RSYNC_OPTS[@]}" --rsync-path "$REMOTE_RSYNC_CMD" -e "$RSYNC_SSH_CMD" \
          "$BRIDGES2_SSH:$REMOTE_BASE/" "$LOCAL_BASE/"
        log_progress 75 "Full-tree rsync transfer completed"
      else
        collect_via_tar
      fi
    fi
  else
    log_progress 40 "Skipping checkpoint collection (--skip-checkpoints)"
  fi

  if [[ "$PULL_VOCAB" == true ]]; then
    pull_vocab_artifacts
  fi
  if [[ "$PULL_GRMT" == true ]]; then
    pull_grmt_artifact
  fi
  log_progress 100 "Collect completed"
}

do_pull_logs() {
  log_progress 25 "Pull logs start"
  echo "[pull-logs] Remote: $BRIDGES2_SSH:$REMOTE_BASE"
  echo "[pull-logs] Local:  $LOCAL_BASE"
  mkdir -p "$LOCAL_BASE"

  if ! remote_exists; then
    echo "WARNING: Remote logs path does not exist; skipping log pull."
    return 0
  fi

  if detect_remote_rsync; then
    log_progress 40 "Syncing training logs via rsync"
    rsync "${RSYNC_OPTS[@]}" --rsync-path "$REMOTE_RSYNC_CMD" -e "$RSYNC_SSH_CMD" \
      "$BRIDGES2_SSH:$REMOTE_BASE/" "$LOCAL_BASE/"
    log_progress 75 "Logs rsync transfer completed"
  elif [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] Would stream logs via tar from $REMOTE_BASE to $LOCAL_BASE"
  else
    collect_via_tar
  fi

  log_progress 100 "Pull logs completed"
}

do_delete() {
  log_progress 25 "Delete start"
  echo "[delete] Remote only: $BRIDGES2_SSH:$REMOTE_BASE"
  if ! remote_exists; then
    echo "Remote path already absent; nothing to delete."
    return 0
  fi
  if [[ "$DRY_RUN" == true ]]; then
    echo "[dry-run] Would run on remote: find ... mindepth 1 -delete (contents of $REMOTE_BASE)"
    ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "du -sh \"$REMOTE_BASE\" 2>/dev/null || true; find \"$REMOTE_BASE\" -mindepth 1 -maxdepth 2 2>/dev/null | head -50"
    return 0
  fi
  if [[ "$SKIP_CONFIRM" != true ]]; then
    echo -n "Delete EVERYTHING under the remote path above? [y/N] "
    read -r ans
    [[ "${ans,,}" == "y" ]] || { echo "Aborted."; exit 1; }
  fi
  # Remove contents only; keep the directory if it is the sync root (avoids breaking mkdir expectations).
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" \
    "if [ -d \"$REMOTE_BASE\" ]; then find \"$REMOTE_BASE\" -mindepth 1 -delete; else rm -f \"$REMOTE_BASE\"; fi"
  echo "[delete] Done."
  log_progress 100 "Delete completed"
}

log_progress 0 "Init"
ssh_master
log_progress 15 "Mode selected: $MODE"

case "$MODE" in
  collect) do_collect ;;
  pull-logs) do_pull_logs ;;
  delete)  do_delete ;;
  both)
    do_collect
    do_delete
    ;;
esac

echo "Finished ($MODE)."

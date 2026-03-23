#!/usr/bin/env bash
# Sync GRIM-text checkpoints (and optional subtrees) between your machine and PSC Bridges-2.
# Bridges-2 project space lives under /ocean/projects/<alloc>/<user>/ — often called "ocean" paths.
#
# Usage:
#   ./scripts/sync_models_bridges2.sh collect [--dry-run] [--subpath REL]
#   ./scripts/sync_models_bridges2.sh delete [--dry-run] [--yes] [--subpath REL]
#   ./scripts/sync_models_bridges2.sh both   [--dry-run] [--yes] [--subpath REL]
#
# Modes:
#   collect — rsync FROM Bridges-2 → local repo (pull checkpoints off ocean storage)
#   delete  — remove the same path ON Bridges-2 only (frees /ocean quota; does not delete local)
#   both    — collect then delete remote (safe order: copy first, then remove from cluster)
#
# Environment (same family as run_train_on_bridges2.sh):
#   GRIM_BRIDGES2_DIR          Remote repo root (default: /ocean/projects/cis210058p/uwadkins/G.R.I.M)
#   GRIM_BRIDGES2_SSH          SSH host or user@host (default: bridges2, else uwadkins@bridges2.psc.edu)
#   GRIM_BRIDGES2_ACCOUNT      Shown in help text only (for documentation)
#   GRIM_BRIDGES2_SYNC_RELATIVE  Path under repo to sync (default: resources/models/GRIM-text/checkpoints)
#
# Examples:
#   ./scripts/sync_models_bridges2.sh collect
#   ./scripts/sync_models_bridges2.sh collect --dry-run
#   ./scripts/sync_models_bridges2.sh delete --yes
#   ./scripts/sync_models_bridges2.sh both --yes
#   GRIM_BRIDGES2_SYNC_RELATIVE=resources/models/GRIM-text/training/logs ./scripts/sync_models_bridges2.sh collect

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BRIDGES2_DIR="${GRIM_BRIDGES2_DIR:-/ocean/projects/cis210058p/uwadkins/G.R.I.M}"
BRIDGES2_SSH="${GRIM_BRIDGES2_SSH:-bridges2}"
if [[ "$BRIDGES2_SSH" == "bridges2" ]] && ! grep -q "Host bridges2" ~/.ssh/config 2>/dev/null; then
  BRIDGES2_SSH="uwadkins@bridges2.psc.edu"
fi

SYNC_REL="${GRIM_BRIDGES2_SYNC_RELATIVE:-resources/models/GRIM-text/checkpoints}"

MODE=""
DRY_RUN=false
SKIP_CONFIRM=false
SUBPATH=""

usage() {
  sed -n '1,35p' "$0" | tail -n +2
  exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    collect|delete|both) MODE="$1"; shift ;;
    --dry-run) DRY_RUN=true; shift ;;
    --yes|-y) SKIP_CONFIRM=true; shift ;;
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
  echo "ERROR: specify mode: collect | delete | both"
  usage 1
fi

REMOTE_BASE="$BRIDGES2_DIR/$SYNC_REL"
LOCAL_BASE="$REPO_ROOT/$SYNC_REL"

if [[ -n "$SUBPATH" ]]; then
  REMOTE_BASE="$REMOTE_BASE/$SUBPATH"
  LOCAL_BASE="$LOCAL_BASE/$SUBPATH"
fi

# macOS ships an older rsync without --info=progress2 (needs rsync 3.1+).
RSYNC_OPTS=(-a -v --progress)
if rsync --help 2>&1 | grep -q 'info=progress2'; then
  RSYNC_OPTS=(-a -v --info=progress2)
fi
[[ "$DRY_RUN" == true ]] && RSYNC_OPTS+=(--dry-run)

ssh_master() {
  BRIDGES2_CTRL="/tmp/cm-grim-sync-$$"
  if ! ssh -f -N -M -S "$BRIDGES2_CTRL" -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new "$BRIDGES2_SSH"; then
    echo "ERROR: SSH to Bridges-2 failed (try: ssh $BRIDGES2_SSH)"
    exit 1
  fi
  BRIDGES2_SSH_OPTS=(-S "$BRIDGES2_CTRL" -o ControlMaster=no)
  # rsync -e must be one shell-invokable command string
  RSYNC_SSH_CMD="ssh ${BRIDGES2_SSH_OPTS[*]}"
  trap 'ssh -S "$BRIDGES2_CTRL" -O exit "$BRIDGES2_SSH" 2>/dev/null; rm -f "$BRIDGES2_CTRL"' EXIT
}

remote_exists() {
  ssh "${BRIDGES2_SSH_OPTS[@]}" "$BRIDGES2_SSH" "test -e \"$REMOTE_BASE\""
}

do_collect() {
  echo "[collect] Remote: $BRIDGES2_SSH:$REMOTE_BASE"
  echo "[collect] Local:  $LOCAL_BASE"
  mkdir -p "$LOCAL_BASE"
  if ! remote_exists; then
    echo "WARNING: Remote path does not exist; nothing to pull."
    return 0
  fi
  rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_SSH_CMD" \
    "$BRIDGES2_SSH:$REMOTE_BASE/" "$LOCAL_BASE/"
}

do_delete() {
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
}

ssh_master

case "$MODE" in
  collect) do_collect ;;
  delete)  do_delete ;;
  both)
    do_collect
    do_delete
    ;;
esac

echo "Finished ($MODE)."

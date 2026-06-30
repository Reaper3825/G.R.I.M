#!/usr/bin/env bash
# ============================================================
# GRIM Bridges-2 (PSC /ocean) Git Bash SSH setup
# Matches scripts/run_train_on_bridges2.sh and sync_models_bridges2.sh.
#
# Usage:
#   ./scripts/setup_bridges2.sh
#   BRIDGES2_USER=uwadkins ACCESS_ALLOC=cis210058p ./scripts/setup_bridges2.sh
#   ./scripts/setup_bridges2.sh --generate-key
#   ./scripts/setup_bridges2.sh --install-key
# ============================================================

set -euo pipefail

HOST_ALIAS="${BRIDGES2_HOST_ALIAS:-bridges2}"
HOST_NAME="${BRIDGES2_HOST_NAME:-bridges2.psc.edu}"
BRIDGES2_USER="${BRIDGES2_USER:-}"
ACCESS_ALLOC="${ACCESS_ALLOC:-cis210058p}"
IDENTITY_FILE="${BRIDGES2_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}"
SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
GENERATE_KEY=false
INSTALL_KEY=false
SKIP_TEST=false

usage() {
  sed -n '2,12p' "$0"
  echo ""
  echo "Options:"
  echo "  --generate-key   Create \$IDENTITY_FILE if missing (ed25519)"
  echo "  --install-key    Append local public key to remote authorized_keys"
  echo "  --skip-test      Skip batch-mode ssh probe at the end"
  echo ""
  echo "Environment:"
  echo "  BRIDGES2_USER          PSC username (required on first run)"
  echo "  ACCESS_ALLOC           ACCESS allocation id (default: cis210058p)"
  echo "  BRIDGES2_HOST_ALIAS    SSH config alias (default: bridges2)"
  echo "  BRIDGES2_IDENTITY_FILE Private key path (default: ~/.ssh/id_ed25519)"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate-key) GENERATE_KEY=true; shift ;;
    --install-key)  INSTALL_KEY=true; shift ;;
    --skip-test)    SKIP_TEST=true; shift ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

read_existing_user() {
  if [[ ! -f "$SSH_CONFIG" ]]; then
    return 0
  fi
  awk -v alias="$HOST_ALIAS" '
    $1 == "Host" && $2 == alias { in_host=1; next }
    in_host && $1 == "Host" { exit }
    in_host && $1 == "User" { print $2; exit }
  ' "$SSH_CONFIG" || true
}

if [[ -z "$BRIDGES2_USER" ]]; then
  BRIDGES2_USER="$(read_existing_user)"
fi

if [[ -z "$BRIDGES2_USER" ]]; then
  echo "ERROR: Set BRIDGES2_USER or pass it in the environment." >&2
  echo "  Example: BRIDGES2_USER=uwadkins ./scripts/setup_bridges2.sh --generate-key" >&2
  exit 1
fi

REMOTE_REPO_DIR="${GRIM_BRIDGES2_DIR:-/ocean/projects/${ACCESS_ALLOC}/${BRIDGES2_USER}/G.R.I.M}"

if [[ ! -f "$IDENTITY_FILE" ]]; then
  if [[ "$GENERATE_KEY" == true ]]; then
    echo "Creating ED25519 key at $IDENTITY_FILE"
    ssh-keygen -t ed25519 -f "$IDENTITY_FILE" -C "${BRIDGES2_USER}@bridges2.psc.edu"
  else
    echo "ERROR: Missing private key: $IDENTITY_FILE" >&2
    echo "  Re-run with --generate-key to create one." >&2
    exit 1
  fi
fi

identity_for_config="$IDENTITY_FILE"
if [[ "$identity_for_config" == "$HOME/.ssh/"* ]]; then
  identity_for_config="~/.ssh/${identity_for_config#"$HOME/.ssh/"}"
elif [[ "$identity_for_config" == /*[a-zA-Z]:/* ]]; then
  drive="${identity_for_config:1:1}"
  rest="${identity_for_config:2}"
  identity_for_config="/${drive,,}${rest}"
fi
identity_for_config="${identity_for_config//\\//}"

managed_block() {
  cat <<EOF
# >>> GRIM Bridges-2 (PSC /ocean) >>>
Host ${HOST_ALIAS}
  HostName ${HOST_NAME}
  User ${BRIDGES2_USER}
  IdentityFile ${identity_for_config}
  IdentitiesOnly yes
  PubkeyAuthentication yes
  ServerAliveInterval 60
  ServerAliveCountMax 30
  TCPKeepAlive yes
  StrictHostKeyChecking accept-new
# <<< GRIM Bridges-2 (PSC /ocean) <<<
EOF
}

write_ssh_config() {
  local tmp block existing
  block="$(managed_block)"
  tmp="$(mktemp)"
  if [[ -f "$SSH_CONFIG" ]]; then
    existing="$(cat "$SSH_CONFIG")"
    if grep -q '# >>> GRIM Bridges-2 (PSC /ocean) >>>' <<<"$existing"; then
      awk -v repl="$block" '
        BEGIN { skip=0 }
        /# >>> GRIM Bridges-2 \(PSC \/ocean\) >>>/ { print repl; skip=1; next }
        /# <<< GRIM Bridges-2 \(PSC \/ocean\) <<</ { skip=0; next }
        skip==0 { print }
      ' "$SSH_CONFIG" >"$tmp"
    else
      cp "$SSH_CONFIG" "$tmp"
      printf '\n%s\n' "$block" >>"$tmp"
    fi
  else
    printf '%s\n' "$block" >"$tmp"
  fi
  install -m 600 "$tmp" "$SSH_CONFIG"
  rm -f "$tmp"
}

write_ssh_config
echo "Wrote SSH host alias '$HOST_ALIAS' to $SSH_CONFIG"

if [[ "$INSTALL_KEY" == true ]]; then
  pub="${IDENTITY_FILE}.pub"
  [[ -f "$pub" ]] || { echo "ERROR: Missing public key: $pub" >&2; exit 1; }
  echo "Installing public key on $HOST_ALIAS (password prompt possible)..."
  public_key="$(tr -d '\r\n' <"$pub")"
  escaped_public_key="${public_key//\'/\'\\\'\'}"
  ssh "$HOST_ALIAS" "set -e; mkdir -p ~/.ssh; chmod 700 ~/.ssh; touch ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; if grep -qxF '$escaped_public_key' ~/.ssh/authorized_keys; then echo '__GRIM_KEY_ALREADY_PRESENT__'; else printf '%s\n' '$escaped_public_key' >> ~/.ssh/authorized_keys; echo '__GRIM_KEY_INSTALLED__'; fi"
  echo "Public key installed."
fi

cat <<EOF

Export these before running the bash launchers:

  export GRIM_BRIDGES2_SSH=${HOST_ALIAS}
  export GRIM_BRIDGES2_ACCOUNT=${ACCESS_ALLOC}
  export GRIM_BRIDGES2_DIR=${REMOTE_REPO_DIR}

Then:

  ./scripts/run_train_on_bridges2.sh --build
  ./scripts/sync_models_bridges2.sh collect

EOF

if [[ "$SKIP_TEST" == false ]]; then
  echo "Testing: ssh -o BatchMode=yes -o ConnectTimeout=15 ${HOST_ALIAS} hostname"
  if ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST_ALIAS" hostname; then
    echo "[OK] Bridges-2 accepted key auth."
  else
    echo "[WARN] Batch SSH failed. If this is a new key, run:" >&2
    echo "  ./scripts/setup_bridges2.sh --install-key" >&2
    exit 1
  fi
fi

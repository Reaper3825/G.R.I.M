# Anvil on Windows + VS Code Remote-SSH

This repo already expects a local SSH host alias named `anvil`.

That matters because `scripts/run_train_on_anvil.sh` uses:

- `ssh anvil`
- `~/.ssh/id_ed25519`
- the remote repo path `/anvil/projects/x-cis210085/GRIM/G.R.I.M`

## One-command Windows setup

From the repo root in PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_anvil_windows.ps1
```

What the helper does:

- verifies the Windows OpenSSH client exists
- ensures the `ssh-agent` service is running
- ensures `~/.ssh/id_ed25519` exists
- ensures your `Host anvil` block exists in `~/.ssh/config`
- loads the key into `ssh-agent`
- tests whether Anvil accepts the key in batch mode
- prints the exact VS Code Remote-SSH next steps

## One-command public key install to Anvil

If Anvil is prompting for a password instead of accepting your key yet, push the current public key into `~/.ssh/authorized_keys` on Anvil with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_anvil_public_key.ps1
```

What this helper does:

- reads `~/.ssh/id_ed25519.pub`
- connects to the existing `anvil` host alias
- creates `~/.ssh` and `~/.ssh/authorized_keys` on Anvil if needed
- appends the key only if it is not already present

You may be prompted for your Anvil password one last time while installing the key. After that, `ssh anvil` should stop asking for a password if the account accepts key auth.

If your local SSH setup is good but Anvil rejects the key, the script will tell you that explicitly.
That means the fix is on the Anvil side: add `~/.ssh/id_ed25519.pub` to your RCAC/Anvil account or to `~/.ssh/authorized_keys` on Anvil.

## VS Code: connect to Anvil

1. Open VS Code.
2. Press `Ctrl+Shift+P`.
3. Run `Remote-SSH: Connect to Host...`.
4. Pick `anvil`.
5. Open the remote folder:

```text
/anvil/projects/x-cis210085/GRIM/G.R.I.M
```

## Important workflow split

There are **two** useful Anvil workflows in this repo:

### 1. Local Windows -> Anvil wrapper script

Use this when you are in your local Windows checkout and want the repo helper to SSH in and run training for you:

```bash
./scripts/run_train_on_anvil.sh
```

or

```bash
./scripts/run_train_on_anvil.sh --build
```

### 2. VS Code Remote-SSH session already on Anvil

Use this when you are editing the remote checkout directly on Anvil.

In that case, do **not** run `scripts/run_train_on_anvil.sh` from the remote Anvil terminal, because that script SSHs to `anvil` again.
Instead, run native Anvil commands directly from the remote terminal.

Examples:

```bash
cd /anvil/projects/x-cis210085/GRIM/G.R.I.M
```

Build/run from the remote terminal with the normal Anvil module + CMake flow, or use `srun` / `sbatch` directly from that session.

## Quick troubleshooting

### `Permission denied (publickey,...)`

Your Windows machine offered a key, but Anvil did not accept it.
Upload or register:

```text
C:\Users\Merlinthegrim\.ssh\id_ed25519.pub
```

or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_anvil_public_key.ps1
```

### The helper says the SSH key is missing

Re-run with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_anvil_windows.ps1 -GenerateKeyIfMissing
```

### VS Code can connect but the repo is missing

Clone or sync the repo on Anvil to:

```text
/anvil/projects/x-cis210085/GRIM/G.R.I.M
```

or point `GRIM_ANVIL_DIR` to wherever your Anvil checkout actually lives.

## Why this exists

This repo’s Anvil scripts assume the local alias is exactly `anvil`, so the least-chaotic setup is to make Windows and VS Code use the same alias instead of teaching every script a new name every Tuesday.
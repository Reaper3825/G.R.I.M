# Anvil run: file and run command

## File to cat (see what runs the job and the loop)

```bash
cat scripts/run_train_on_anvil.sh
```

That script is the one that runs the job and starts the training loop on Anvil.

## Run command (what actually runs the job and the loop)

From the repo root:

```bash
./scripts/run_train_on_anvil.sh
```

Or with build-first:

```bash
./scripts/run_train_on_anvil.sh --build
```

**What this does:** SSHs to Anvil → allocates a GPU with `srun` → runs `train_gpu --config <config>` (the training loop).

## In Cursor

- **Run button (Code Runner):** Open `scripts/run_train_on_anvil.sh`, then Run. That executes the script above (ensure `ssh-add` has been run in a terminal first).
- **Run Task:** Terminal → Run Task… → **Run GRIM-text training on Anvil** (same command).

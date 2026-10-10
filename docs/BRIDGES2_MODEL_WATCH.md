# Automatic Bridges-2 model-store downloads

Run this on your PC in a separate Bash terminal; leave it running while
training is submitted normally:

```bash
./scripts/sync_models_bridges2.sh watch-models --prune-remote
```

This Bash entry point launches the Python watcher. It also works directly as
`python scripts/watch_models_bridges2.py --prune-remote`.

The same loop also downloads new or changed `vocab.bin` and `vocab.txt` from
`resources/models/GRIM-text/training/data` into that directory locally. Vocabulary
files are **always retained on the HPC**, including when `--prune-remote` is enabled.
Other training data files, including the GRMT corpus and caches, are excluded.
Missing vocabulary files are skipped until they appear. Vocabulary uses the
same stability checks, atomic local replacement, and retry behavior as model files.

The watcher recursively copies new or changed files from
`/ocean/projects/cis250124p/uwadkins/G.R.I.M/resources/models/model_store`
into this checkout's `resources/models/model_store`. Model subdirectories are
scanned recursively for checkpoints, optimizer sidecars, and other artifacts.
Configuration files are managed by Git and excluded from downloads: `.grimcfg`,
`.json`, `.jsonc`, `.json5`, `.yaml`, `.yml`, `.toml`, `.ini`, `.cfg`, and `.conf`.
This includes `model.grimcfg`, `model_config.json`, and `*_configuration.json`.
**Changed remote artifacts replace matching local files.**
Local files are never deleted. With `--prune-remote`, archived `.grimckpt` and
`.opt` files are removed from the HPC, except the **latest PT and latest SFT
checkpoint in each directory and their matching optimizer sidecars**.
Configuration files, other file types (including LoRA `.grimlorackpt` files),
directories, and files outside the watched model store are preserved.
Omit `--prune-remote` for download-only operation. Temporary/hidden files and
symlinks are skipped. Nothing is built, submitted, or executed in the training runtime.

## SSH setup

Requires existing Python 3.10+ and OpenSSH (`ssh` and `sftp`) locally, and
`python3` on the Bridges-2 **login node**. No extra Python packages or rsync
are needed. The data node restricts executable commands and rejects `python3`;
the watcher uses SSH on `bridges2` for listings, metadata, checksums, and cleanup,
and SFTP on `data.bridges2.psc.edu` for file bytes. Transfers originate on
your PC through PSC's data transfer node, as required by the
[Bridges-2 transfer guide](https://www.psc.edu/resources/bridges-2/user-guide/#transferring-files).
The watcher's login host defaults to the same `bridges2` alias used by training
(or `GRIM_BRIDGES2_SSH` when set).

First test the data-node connection interactively to establish its host key and
confirm your key/agent works:

```bash
ssh bridges2 'python3 --version'
sftp uwadkins@data.bridges2.psc.edu
```

At the SFTP prompt, enter `pwd`, then `bye`. Test the DTN with SFTP, not
`ssh ... true`: arbitrary shell commands are restricted there.

The watcher uses `BatchMode=yes`: it cannot prompt for a password or passphrase
on each poll. Use an SSH key accepted by PSC and load its passphrase into your
SSH agent if needed. If your existing `bridges2` SSH alias specifies a particular
identity, create a separate alias with the same user/identity for the data node:

```sshconfig
Host bridges2-data
    HostName data.bridges2.psc.edu
    User uwadkins
    IdentityFile ~/.ssh/id_ed25519
```

Then test `sftp bridges2-data` and run:

```bash
./scripts/sync_models_bridges2.sh watch-models --host bridges2-data --prune-remote
```

## Options and behavior

- `--interval 30`: poll every 30 seconds (default 60).
- `--prune-remote`: reclaim archived checkpoint storage on the HPC while keeping
  latest PT/SFT pairs in each checkpoint directory. Off unless explicitly passed.
- `--host USER@HOST`: override the data-node target; environment equivalent:
  `GRIM_BRIDGES2_DATA_SSH`. This intentionally does not use `GRIM_BRIDGES2_SSH`,
  which commonly points at the training login node.
- `--control-host bridges2`: override the login-node target for metadata and
  cleanup (environment: `GRIM_BRIDGES2_SSH`; default `bridges2`). For setups
  without an alias, use `--control-host uwadkins@bridges2.psc.edu`.
- `--remote-repo /ocean/projects/.../G.R.I.M`: override the remote repository;
  defaults to `GRIM_BRIDGES2_DIR` when set, otherwise the path above.
- `--local-dir D:/somewhere/model_store`: override the local destination.
- `--vocab-local-dir D:/somewhere/data`: override the vocabulary destination
  (defaults to this checkout's `resources/models/GRIM-text/training/data`,
  independently of `--local-dir`).
- `--no-sync-vocab`: disable vocabulary downloads; enabled by default.
- `--ssh C:/Windows/System32/OpenSSH/ssh.exe`: choose an SSH executable.
- `--sftp C:/Windows/System32/OpenSSH/sftp.exe`: choose an SFTP executable.
- `--once --interval 5`: take two polls, copy stable files, then exit. Returns
  nonzero if a poll/transfer fails, any listed file remains unsynchronized, or
  requested cleanup is deferred/fails.
- Ctrl+C stops the watcher without affecting the training job.

Every successful scan prints a timestamped `Heartbeat: scan OK; N remote files`
line, including idle scans. At the default interval it appears about every
60 seconds while idle. Transfers and checksum verification can extend the time
between scans; failed scans print a warning instead of a success heartbeat.

A file becomes eligible after identical metadata appears in two consecutive
successful polls. New files normally start downloading within 60–120 seconds
at the default interval. Source size, timestamps, and inode are checked before
and after streaming; changed sources are discarded and retried. Downloads go
to temporary sibling files, then replace the local destination atomically.
Disconnects retry automatically. Interrupted downloads restart rather than
resume; completed files are remembered across watcher restarts in
`.cache/bridges2-model-watch`. One watcher per local destination is allowed.
The OS releases that lock when the watcher exits or crashes.

## Remote retention and verification

Checkpoint stage is identified by `checkpoint_PT_*.grimckpt` and
`checkpoint_SFT_*.grimckpt`. "Latest" matches the trainer's ordering: most recent
modification time, then highest trailing epoch number, then filename. Empty
checkpoints do not displace a nonempty retained checkpoint. Retention applies
independently in every directory under the watched store, so one model never
causes another model's only PT/SFT checkpoint to be deleted.

An older checkpoint and its existing `.opt` sidecar must both be stable and
downloaded before either can be removed. The newer retained PT/SFT files in
that directory must also have been downloaded before cleanup can proceed.
SHA-256 is computed from the local copy and checked against the remote file;
all existing members of a pair must pass before removal begins. Checksums read
files on each machine but do not transfer the checkpoint a second time.
Already-downloaded files from an earlier watcher run can be cleaned up without
downloading them again.

Before deletion, the login-node control process scans retention again and checks remote
file identities and metadata. Changed files, checksum mismatches, incomplete
downloads, and newly retained checkpoints defer cleanup. The next poll retries.
Removal targets individual checkpoint/sidecar files, never a whole directory.
Non-PT/SFT `.grimckpt` files and orphan `.opt` files are eligible once archived;
there is no latest-file exemption for other training stages.

Keeping the newest PT/SFT pairs requires enough HPC storage for those pairs,
the next checkpoint being written, and any atomic-write temporary files.
The watcher cannot prevent a write failure if that working set already exceeds
your quota. Startup cleanup also needs time to download and verify older files.

This mirrors files individually, **not checkpoint/optimizer pairs as one
transaction**. The parameter checkpoint writer already uses temporary files
and atomic rename. Optimizer sidecars currently write directly; unchanged
metadata is a settling heuristic for those files, not a strict save-completion
signal. A writer paused longer than a poll interval can appear finished. For
strict pair completion, the trainer would need to publish a manifest/ready
marker after both saves, or publish an immutable checkpoint directory atomically.
Do not load an optimizer/checkpoint pair locally during an ongoing sync.

The watcher does not install itself as a Windows startup task. To start it after
a reboot, run the command again; already completed downloads are skipped.

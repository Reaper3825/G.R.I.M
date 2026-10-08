#!/usr/bin/env python3
"""Continuously pull the Bridges-2 model store using only Python and OpenSSH.

Run on your PC: python scripts/watch_models_bridges2.py
Transfers use the PSC data transfer node, independently of the training job.
SSH keys/agent authentication and a trusted host key must already be configured
for that node (test: ssh uwadkins@data.bridges2.psc.edu true).
Ctrl+C stops the watcher; disconnects retry automatically. --prune-remote removes
archived checkpoints from the HPC, keeping the latest PT and SFT in each directory
and their optimizer sidecars. Local files and remote configurations are retained.
Changed remote files replace their local counterparts.

Files must remain unchanged across two successful polls before download. Remote
metadata is checked before/after streaming, and local replacement is atomic.
This is a file mirror, not a transactional checkpoint/optimizer-pair snapshot;
non-atomic writers should publish through a temporary file and rename for a
strict completion guarantee. A pause in a direct write can look like completion.
Interrupted files restart on the next poll (no rsync dependency or resume).
"""

import argparse
import base64
import hashlib
import inspect
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tempfile
import time


MODEL_STORE = "resources/models/model_store"
DEFAULT_REPO = "/ocean/projects/cis250124p/uwadkins/G.R.I.M"
LOG = logging.getLogger("model-watch")

# Same representation for directory listings and open-file checks. ctime/inode
# detect atomic replacement even when a writer preserves size and mtime.
REMOTE_COMMON = '''
import os, sys, json, stat
from pathlib import Path
def signature(s):
    # Windows Python versions disagree on ctime semantics between path stat
    # and handle fstat. Bridges-2 is Linux, where ctime is a change timestamp.
    return [s.st_size, s.st_mtime_ns, s.st_ctime_ns if os.name != 'nt' else 0, s.st_ino]
def temporary(name):
    return name.startswith('.') or name.endswith(('~', '.tmp', '.partial', '.part', '.lock', '.writing.grimlorackpt')) or '.transfer.' in name
'''
REMOTE_INVENTORY = '''
def inventory(root):
    result = {}
    if not root.exists():
        return result
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError('Remote model store must be a real directory')
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not temporary(d) and not (Path(directory)/d).is_symlink()]
        for name in files:
            if temporary(name):
                continue
            path = Path(directory)/name
            try:
                s = path.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(s.st_mode):
                result[path.relative_to(root).as_posix()] = signature(s)
    return result
'''
REMOTE_LIST = REMOTE_COMMON + REMOTE_INVENTORY + '''
print(json.dumps(inventory(Path(sys.argv[1]))))
'''
REMOTE_READ = REMOTE_COMMON + '''
root = Path(sys.argv[1])
relative = Path(sys.argv[2])
expected = json.loads(sys.argv[3])
if relative.is_absolute() or '..' in relative.parts:
    raise RuntimeError('Invalid remote relative path')
path = root/relative
for candidate in [root, *[root/Path(*relative.parts[:n]) for n in range(1, len(relative.parts)+1)]]:
    if candidate.is_symlink():
        raise RuntimeError('Refusing remote symlink')
with path.open('rb') as source:
    if signature(os.fstat(source.fileno())) != expected:
        raise RuntimeError('Source changed before transfer: expected %r, found %r' % (expected, signature(os.fstat(source.fileno()))))
    remaining = expected[0]
    while remaining:
        chunk = source.read(min(4*1024*1024, remaining))
        if not chunk:
            raise RuntimeError('Source truncated during transfer')
        sys.stdout.buffer.write(chunk)
        remaining -= len(chunk)
    sys.stdout.buffer.flush()
    if signature(os.fstat(source.fileno())) != expected or signature(path.stat()) != expected:
        raise RuntimeError('Source changed during transfer')
'''


def protected_checkpoints(snapshot):
    """Match trainer ranking: newest mtime, then trailing epoch, then name."""
    latest = {}
    for relative, metadata in snapshot.items():
        path = PurePosixPath(relative)
        match = re.fullmatch(r"checkpoint_(PT|SFT)_.+\.grimckpt", path.name)
        if not match or metadata[0] == 0:
            continue
        epoch = re.search(r"(\d+)$", path.stem)
        rank = (metadata[1], int(epoch[1]) if epoch else -1, relative)
        key = (str(path.parent), match[1])
        if key not in latest or rank > latest[key]:
            latest[key] = rank
    protected = set()
    for _, _, relative in latest.values():
        protected.add(relative)
        protected.add(str(PurePosixPath(relative).with_suffix(".opt")))
    return protected


# Same ranking function runs locally and again on the HPC immediately before
# cleanup. The second inventory prevents a stale local retention decision.
REMOTE_PRUNE = REMOTE_COMMON + REMOTE_INVENTORY + '''
import hashlib, re
from pathlib import PurePosixPath
''' + inspect.getsource(protected_checkpoints) + '''
root = Path(sys.argv[1])
if not root.is_absolute() or root.name != 'model_store' or root.is_symlink():
    raise RuntimeError('Cleanup requires an absolute model_store directory')
resolved_root = root.resolve()
request = json.load(sys.stdin)
deleted, skipped = [], []
def safe_path(relative):
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or '..' in rel.parts:
        raise RuntimeError('Unsafe cleanup path')
    path = root/rel
    for candidate in [root, *[root/Path(*rel.parts[:n]) for n in range(1, len(rel.parts)+1)]]:
        if candidate.is_symlink():
            raise RuntimeError('Refusing cleanup through a symlink')
    path.resolve().relative_to(resolved_root)
    if path.suffix not in ('.grimckpt', '.opt'):
        raise RuntimeError('Only checkpoint and optimizer files can be pruned')
    return path
for group in request['groups']:
    try:
        current = inventory(root)
        protected = protected_checkpoints(current)
        if any(name in protected for name in group):
            raise RuntimeError('Checkpoint is now retained as latest PT/SFT')
        # Do not remove an old pair while a sidecar appeared after planning.
        for name in group:
            path = PurePosixPath(name)
            counterpart = str(path.with_suffix('.opt' if path.suffix == '.grimckpt' else '.grimckpt'))
            if counterpart in current and counterpart not in group:
                raise RuntimeError('Checkpoint pair changed after planning')
        for name, expected in request['guards'].items():
            if current.get(name) != expected:
                raise RuntimeError('Retained checkpoint changed after planning')
        paths = {}
        for name, expected in group.items():
            path = safe_path(name)
            if current.get(name) != expected['remote']:
                raise RuntimeError('Source changed before cleanup')
            digest = hashlib.sha256()
            with path.open('rb') as source:
                if signature(os.fstat(source.fileno())) != expected['remote']:
                    raise RuntimeError('Source changed before checksum')
                while True:
                    chunk = source.read(4*1024*1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                if signature(os.fstat(source.fileno())) != expected['remote']:
                    raise RuntimeError('Source changed during checksum')
            if digest.hexdigest() != expected['sha256']:
                raise RuntimeError('Local/remote SHA-256 mismatch')
            paths[name] = path
        # Verify the full pair before deleting either member.
        current = inventory(root)
        if any(name in protected_checkpoints(current) for name in group):
            raise RuntimeError('Retention changed during checksum')
        if any(current.get(name) != expected for name, expected in request['guards'].items()):
            raise RuntimeError('Retained checkpoint changed during checksum')
        for name, expected in group.items():
            if signature(safe_path(name).lstat()) != expected['remote']:
                raise RuntimeError('Source changed during checksum')
        for name, path in paths.items():
            # Non-recursive removal of this exact, verified regular file only.
            if signature(safe_path(name).lstat()) != group[name]['remote']:
                raise RuntimeError('Source changed before unlink')
            path.unlink()
            deleted.append(name)
    except (OSError, ValueError, RuntimeError) as error:
        skipped.append(str(error))
print(json.dumps({'deleted': deleted, 'skipped': skipped}))
'''


def local_path(root, relative):
    """Reject traversal, Windows drive/ADS syntax, and existing symlink parents."""
    parts = PurePosixPath(relative).parts
    if not parts or PurePosixPath(relative).is_absolute() or any(
        p in (".", "..") or "\\" in p or ":" in p for p in parts
    ):
        raise ValueError(f"Unsafe model-store path: {relative!r}")
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink() or (hasattr(candidate, "is_junction") and candidate.is_junction()):
            raise ValueError(f"Refusing local symlink/junction: {candidate}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes local model store: {relative!r}")
    return candidate


def local_signature(path):
    try:
        s = path.stat()
        return [s.st_size, s.st_mtime_ns]
    except FileNotFoundError:
        return None


class Watcher:
    def __init__(self, host, remote_root, local_root, state_file, ssh="ssh", prune_remote=False):
        self.host, self.remote_root = host, remote_root
        self.local_root, self.state_file = local_root, state_file
        self.ssh = ssh
        self.previous = {}
        self.prune_remote = prune_remote
        self.prune_failed = False
        try:
            self.state = json.loads(state_file.read_text(encoding="utf-8"))
            if not isinstance(self.state, dict) or any(not isinstance(v, dict) for v in self.state.values()):
                raise ValueError("Expected state object")
        except (FileNotFoundError, ValueError):
            self.state = {}

    def command(self, source, *args):
        encoded = base64.b64encode(source.encode()).decode()
        code = f"import base64;exec(base64.b64decode('{encoded}'))"
        remote = shlex.join(["python3", "-c", code, self.remote_root, *args])
        return [self.ssh, "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                self.host, remote]

    def inventory(self):
        result = subprocess.run(self.command(REMOTE_LIST), stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=120, check=True)
        return json.loads(result.stdout)

    def download(self, relative, signature):
        target = local_path(self.local_root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
        try:
            # Stream to disk rather than holding multi-GB checkpoints in memory.
            with os.fdopen(fd, "wb") as output:
                process = subprocess.Popen(self.command(REMOTE_READ, relative, json.dumps(signature)), stdout=output)
                try:
                    if process.wait() != 0:
                        raise RuntimeError(f"SSH transfer failed: {relative}")
                except BaseException:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise
                output.flush()
                os.fsync(output.fileno())
            if Path(temporary).stat().st_size != signature[0]:
                raise RuntimeError(f"Transfer size mismatch: {relative}")
            # Check again before replacement in case a parent changed locally.
            local_path(self.local_root, relative)
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        self.state[relative] = {"remote": signature, "local": local_signature(target)}
        self.save_state()

    def save_state(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(temporary, self.state_file)

    def archived(self, relative, signature):
        target = local_path(self.local_root, relative)
        saved = self.state.get(relative, {})
        return (target.is_file() and saved.get("remote") == signature
                and saved.get("local") == local_signature(target))

    def prune(self, current):
        protected = protected_checkpoints(current)
        groups = {}
        for relative in current:
            path = PurePosixPath(relative)
            if path.suffix in (".grimckpt", ".opt") and relative not in protected:
                groups.setdefault(str(path.with_suffix(".grimckpt")), []).append(relative)
        eligible = []
        guards = {name: current[name] for name in protected if name in current}
        for names in groups.values():
            # A newer retained checkpoint must also be stable and archived before
            # older copies can be reclaimed. Restrict guards to this directory.
            directory = PurePosixPath(names[0]).parent
            relevant = [name for name in guards if PurePosixPath(name).parent == directory]
            required = names + relevant
            if not all(self.previous.get(name) == current[name] and self.archived(name, current[name])
                       for name in required):
                self.prune_failed = True
                continue
            group = {}
            for name in names:
                path = local_path(self.local_root, name)
                digest = hashlib.sha256()
                with path.open("rb") as source:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                        digest.update(chunk)
                if not self.archived(name, current[name]):
                    self.prune_failed = True
                    break
                group[name] = {"remote": current[name], "sha256": digest.hexdigest()}
            else:
                eligible.append(group)
        if not eligible:
            return
        LOG.info("Verifying %d archived checkpoint group(s) before remote cleanup", len(eligible))
        result = subprocess.run(self.command(REMOTE_PRUNE), input=json.dumps({"groups": eligible, "guards": guards}).encode(),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        report = json.loads(result.stdout)
        for name in report["deleted"]:
            LOG.info("Removed archived remote file: %s", name)
        for reason in report["skipped"]:
            self.prune_failed = True
            LOG.warning("Remote cleanup deferred: %s", reason)

    def poll(self):
        self.prune_failed = False
        current = self.inventory()
        for relative, signature in sorted(current.items()):
            target = local_path(self.local_root, relative)
            saved = self.state.get(relative, {})
            if saved.get("remote") == signature and saved.get("local") == local_signature(target) and target.is_file():
                continue
            if self.previous.get(relative) != signature:
                continue
            LOG.info("Downloading %s (%.1f MiB)", relative, signature[0] / 1048576)
            try:
                self.download(relative, signature)
            except (OSError, RuntimeError) as error:
                LOG.warning("%s; will retry", error)
                continue
            LOG.info("Saved %s", target)
        if self.prune_remote:
            self.prune(current)
        self.previous = current


class InstanceLock:
    """OS-held lock releases on exit/crash, including on Windows."""
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a+b")
        if self.file.tell() == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError("A watcher for this destination is already running") from None
        return self

    def __exit__(self, *_):
        self.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("GRIM_BRIDGES2_DATA_SSH", "uwadkins@data.bridges2.psc.edu"),
                        help="Data-node SSH target or configured alias (env: GRIM_BRIDGES2_DATA_SSH)")
    parser.add_argument("--remote-repo", default=os.environ.get("GRIM_BRIDGES2_DIR", DEFAULT_REPO))
    parser.add_argument("--local-dir", type=Path, default=Path(__file__).resolve().parent.parent / MODEL_STORE)
    parser.add_argument("--interval", type=float, default=60, help="Seconds between polls (default: 60)")
    parser.add_argument("--ssh", default="ssh", help="OpenSSH executable path")
    parser.add_argument("--prune-remote", action="store_true",
                        help="Delete verified archived .grimckpt/.opt files on HPC; retain latest PT and SFT per directory")
    parser.add_argument("--once", action="store_true", help="Wait for two polls, sync stable files, then exit; failures return nonzero")
    args = parser.parse_args()
    if args.interval <= 0 or not math.isfinite(args.interval):
        parser.error("--interval must be a finite positive number")
    if not args.host or args.host.startswith("-"):
        parser.error("--host must be an SSH target")
    local_root = args.local_dir.absolute()
    if local_root.is_symlink() or (hasattr(local_root, "is_junction") and local_root.is_junction()):
        parser.error("--local-dir must not be a symlink or junction")
    remote_root = args.remote_repo.rstrip("/") + "/" + MODEL_STORE
    # Cache is already ignored by this repository. Keep separate state per source
    # and destination, but serialize all watchers writing the same local store.
    cache = Path(__file__).resolve().parent.parent / ".cache" / "bridges2-model-watch"
    key = hashlib.sha256(f"{args.host}\0{remote_root}\0{local_root}".encode()).hexdigest()[:20]
    lock_key = hashlib.sha256(str(local_root.resolve()).encode()).hexdigest()[:20]
    watcher = Watcher(args.host, remote_root, local_root, cache / f"{key}.json", args.ssh, args.prune_remote)
    logging.basicConfig(level=logging.INFO, format="[model-watch] %(asctime)s %(message)s", datefmt="%H:%M:%S")
    LOG.info("Watching %s:%s -> %s (every %gs; Ctrl+C to stop)", args.host, remote_root, local_root, args.interval)
    if args.prune_remote:
        LOG.info("Remote cleanup enabled: retain latest PT + SFT checkpoint and sidecars per directory")
    try:
        with InstanceLock(cache / f"{lock_key}.lock"):
            polls = 0
            while True:
                try:
                    watcher.poll()
                except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
                    watcher.previous = {}
                    detail = getattr(error, "stderr", b"") or b""
                    LOG.warning("Poll failed: %s %s; retrying in %gs", error, detail.decode(errors="replace").strip(), args.interval)
                    if args.once:
                        return 1
                polls += 1
                if args.once and polls >= 2:
                    if watcher.prune_failed:
                        return 1
                    # A failed/deferred transfer must not report a successful sync.
                    for relative, signature in watcher.previous.items():
                        saved = watcher.state.get(relative, {})
                        if saved.get("remote") != signature or saved.get("local") != local_signature(local_path(local_root, relative)):
                            return 1
                    return 0
                time.sleep(args.interval)
    except KeyboardInterrupt:
        LOG.info("Stopped")
        return 0
    except RuntimeError as error:
        LOG.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())

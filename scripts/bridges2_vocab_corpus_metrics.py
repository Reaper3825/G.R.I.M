#!/usr/bin/env python3
"""
Corpus + vocabulary metrics aligned with Bridges-2 training paths (run_train_on_bridges2.sh).

Requires ``vocab.bin`` and ``training_data.grmt`` (default under
``resources/models/GRIM-text/training/data/``).

One-shot from your laptop (same env vars as ``run_train_on_bridges2.sh``):

  # Pull both files from Bridges-2 via scp, then compute locally
  python3 scripts/bridges2_vocab_corpus_metrics.py --fetch

  # Or run on the cluster (one ssh: stdin -> remote cat > /tmp, then python; uses /ocean/… data)
  python3 scripts/bridges2_vocab_corpus_metrics.py --ssh-run

Computes:
  - Shannon entropy (bits) of the token-ID distribution over the GRMT corpus
  - Bytes per token: UTF-8 bytes of decoded token text / token count
  - Fertility: tokens per whitespace-delimited word in decoded text

Environment (Bridges-2; same defaults as run_train_on_bridges2.sh):
    - GRIM_BRIDGES2_DIR: remote repo root (default: /ocean/projects/cis250124p/uwadkins/G.R.I.M)
  - GRIM_BRIDGES2_SSH: ssh host or Host alias (default: bridges2, else uwadkins@bridges2.psc.edu)
  - GRIM_REPO_ROOT: local repo root (default: parent of ``scripts/``)
  - GRIM_REMOTE_PYTHON: interpreter on Bridges-2 for ``--ssh-run`` (default: python3)

Python: written for 3.6+ (Bridges-2 login nodes often ship an older ``python3``).
"""

import argparse
import array
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Optional

DEFAULT_REMOTE_ROOT = "/ocean/projects/cis250124p/uwadkins/G.R.I.M"
REMOTE_DATA_SUFFIX = "resources/models/GRIM-text/training/data"

# Token layout (must match decode_token_ids.py / C++ UniByte)
NUM_SPECIAL_TOKENS = 4
BYTE_TOKEN_OFFSET = NUM_SPECIAL_TOKENS
BYTE_VOCAB_SIZE = 256
ATOM_TOKEN_START = BYTE_TOKEN_OFFSET + BYTE_VOCAB_SIZE
NUM_ATOM_TYPES = 17
ATOM_TOKEN_END = ATOM_TOKEN_START + NUM_ATOM_TYPES
UNIGRAM_TOKEN_START = ATOM_TOKEN_END

TEXT_FEATURE_DIM = 16


def repo_root_from_script() -> Path:
    """Repo root when running as ``python3 scripts/…py``; cwd when run as ``python3 -`` (ssh stdin)."""
    try:
        fp = __file__
    except NameError:
        return Path.cwd().resolve()
    s = str(fp)
    if s in ("-", "<stdin>"):
        return Path.cwd().resolve()
    return Path(fp).resolve().parent.parent


def bridges2_ssh_host() -> str:
    """Match run_train_on_bridges2.sh: GRIM_BRIDGES2_SSH or Host bridges2 / uwadkins@bridges2.psc.edu."""
    host = os.environ.get("GRIM_BRIDGES2_SSH", "bridges2")
    if host == "bridges2":
        cfg = Path.home() / ".ssh" / "config"
        try:
            if not cfg.is_file() or "Host bridges2" not in cfg.read_text():
                return "uwadkins@bridges2.psc.edu"
        except OSError:
            return "uwadkins@bridges2.psc.edu"
    return host


def remote_bridges2_root(cli_remote: Optional[str]) -> str:
    if cli_remote:
        return cli_remote.rstrip("/")
    return os.environ.get("GRIM_BRIDGES2_DIR", DEFAULT_REMOTE_ROOT).rstrip("/")


def remote_training_data_path(remote_root: str) -> str:
    return f"{remote_root}/{REMOTE_DATA_SUFFIX}"


def fetch_training_assets(
    ssh_host: str,
    remote_root: str,
    vocab_dst: Path,
    grmt_dst: Path,
) -> None:
    """scp vocab.bin and training_data.grmt from Bridges-2 into local paths."""
    if not shutil.which("scp"):
        raise SystemExit("scp not found in PATH (install OpenSSH client).")
    rdir = remote_training_data_path(remote_root)
    for name, dst in (("vocab.bin", vocab_dst), ("training_data.grmt", grmt_dst)):
        src = f"{ssh_host}:{rdir}/{name}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] scp {src} -> {dst}", file=sys.stderr)
        subprocess.run(["scp", src, str(dst)], check=True)


def ssh_run_on_bridges2(
    ssh_host: str,
    remote_root: str,
    slurm_partition: str = "RM-shared",
    slurm_account: str = "cis250124p",
) -> None:
    """
    One SSH session: stream this file to the remote with ``cat > /tmp/…py``, then dispatch
    via ``srun`` onto a Bridges-2 compute node (avoids login-node memory kills on large
    training_data.grmt files).  Does not use scp/SFTP.
    """
    if not shutil.which("ssh"):
        raise SystemExit("ssh must be in PATH for --ssh-run.")

    script_path = Path(__file__).resolve()
    # Must land on the shared Lustre filesystem (/ocean/…) so that the compute
    # node allocated by srun can read it.  Local /tmp/ is node-local and
    # invisible to any node other than the login node that writes it.
    remote_py = "{}/grim_vocab_metrics_{}.py".format(remote_root.rstrip("/"), uuid.uuid4().hex)
    rr_q = shlex.quote(remote_root)
    rpy_q = shlex.quote(remote_py)
    py_parts = shlex.split(os.environ.get("GRIM_REMOTE_PYTHON", "python3"))
    py_cmd = " ".join(shlex.quote(p) for p in py_parts)
    part_q = shlex.quote(slurm_partition)
    acct_q = shlex.quote(slurm_account)

    with open(script_path, "rb") as f:
        payload = f.read()

    # Cat the script to /tmp on the login node, then run via srun on a compute node.
    # srun --pty is skipped intentionally (no TTY needed; output streams back fine).
    remote_script = (
        "cat > {rp} && "
        "srun --partition={part} --account={acct} "
        "--ntasks=1 --cpus-per-task=16 --mem-per-cpu=2000M --time=0:30:00 "
        "bash -c {cmd}; "
        "rm -f {rp}"
    ).format(
        rp=rpy_q,
        part=part_q,
        acct=acct_q,
        cmd=shlex.quote("GRIM_REPO_ROOT={rr} {py} {rp}".format(rr=remote_root, py=py_cmd, rp=remote_py)),
    )
    inner = "/bin/bash -lc {}".format(shlex.quote(remote_script))

    print(
        "[ssh-run] {} ({} bytes) -> {}:{} (via srun {}/{})".format(
            ssh_host, len(payload), ssh_host, remote_py, slurm_partition, slurm_account
        ),
        file=sys.stderr,
    )
    proc = subprocess.run(
        ["ssh", "-T", ssh_host, inner],
        input=payload,
    )
    if proc.returncode != 0:
        subprocess.run(
            ["ssh", "-T", ssh_host, "rm", "-f", remote_py],
            check=False,
        )
        raise SystemExit(proc.returncode)


def load_vocab_bin(path: Path) -> Dict[int, str]:
    """token_id -> piece text (same sequential unigram mapping as C++)."""
    if not path.is_file():
        raise FileNotFoundError(f"vocab.bin not found: {path}")

    id_to_text = {}  # type: Dict[int, str]
    for tid in range(NUM_SPECIAL_TOKENS):
        id_to_text[tid] = ("<unk>", "<pad>", "<s>", "</s>")[tid]
    for b in range(BYTE_VOCAB_SIZE):
        id_to_text[BYTE_TOKEN_OFFSET + b] = bytes([b]).decode("latin-1")
    for i in range(NUM_ATOM_TYPES):
        id_to_text[ATOM_TOKEN_START + i] = f"<ATOM{i}>"

    with open(path, "rb") as f:
        magic = f.read(4)
        if magic not in (b"KTMG", b"GMTK", b"GRIM"):
            raise ValueError(f"Bad vocab.bin magic: {magic!r}")

        if magic in (b"KTMG", b"GMTK"):
            version = struct.unpack("<H", f.read(2))[0]
            f.read(4)  # checksum
            unigram_count = struct.unpack("<I", f.read(4))[0]
            f.read(4)  # max_length
            f.read(3)  # flags
            f.read(4)  # total_vocab

            for i in range(unigram_count):
                piece_len = struct.unpack("<I", f.read(4))[0]
                text = f.read(piece_len).decode("utf-8", errors="replace")
                struct.unpack("<f", f.read(4))[0]  # score
                if version >= 3:
                    struct.unpack("<I", f.read(4))[0]  # stored_id (ignored)
                id_to_text[UNIGRAM_TOKEN_START + i] = text
        else:
            struct.unpack("<I", f.read(4))[0]  # version
            vocab_size = struct.unpack("<I", f.read(4))[0]
            for i in range(vocab_size):
                token_len = struct.unpack("<I", f.read(4))[0]
                text = f.read(token_len).decode("utf-8", errors="replace")
                id_to_text[i] = text

    return id_to_text


def iter_grmt_token_sequences(path: Path) -> Iterator["array.array[int]"]:
    """Yield token arrays for each sequence.  Uses array.array (4 B/element)
    instead of Python list (~28 B/element) and seeks past unused fields to
    avoid allocating throw-away buffers."""
    with open(path, "rb") as f:
        header = f.read(16)
        if len(header) < 16:
            return
        _magic, _version, num_sequences, _vocab_size = struct.unpack("<IIII", header)

        for _idx in range(num_sequences):
            raw = f.read(4)
            if not raw or len(raw) < 4:
                break
            seq_len = struct.unpack("<I", raw)[0]
            token_bytes = f.read(4 * seq_len)
            if len(token_bytes) < 4 * seq_len:
                break
            tokens = array.array("I")
            tokens.frombytes(token_bytes)
            # Skip: targets(4) + numeric_values(4) + numeric_mask(1)
            #       + text_features(2*TEXT_FEATURE_DIM) + text_feature_mask(1)
            skip_bytes = (4 + 4 + 1) * seq_len + 2 * seq_len * TEXT_FEATURE_DIM + seq_len
            f.seek(skip_bytes, 1)
            yield tokens


def shannon_entropy_bits(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


def count_words_whitespace(text: str) -> int:
    return len(re.findall(r"\S+", text))


def run_grmt(grmt_path: Path, vocab: Dict[int, str], vocab_path: Path) -> None:
    token_hist = Counter()
    total_tokens = 0
    total_bytes = 0
    total_words = 0
    n_seq = 0

    for tokens in iter_grmt_token_sequences(grmt_path):
        n_seq += 1
        token_hist.update(tokens)
        total_tokens += len(tokens)
        pieces = []  # type: List[str]
        for tid in tokens:
            if tid in vocab:
                pieces.append(vocab[tid])
            elif BYTE_TOKEN_OFFSET <= tid < ATOM_TOKEN_START:
                pieces.append(chr(tid - BYTE_TOKEN_OFFSET))
            else:
                pieces.append("")
        decoded = "".join(pieces).replace("\u2581", " ")
        total_bytes += len(decoded.encode("utf-8"))
        total_words += count_words_whitespace(decoded)

    print(f"vocab:         {vocab_path}")
    print(f"corpus (grmt): {grmt_path}")
    print(f"sequences:     {n_seq}")
    print(f"total_tokens:  {total_tokens}")
    print(f"|V| observed:  {len(token_hist)} distinct token IDs")
    print()
    print(f"Shannon entropy H(token) [bits]: {shannon_entropy_bits(token_hist):.6f}")
    if total_tokens:
        print(f"Bytes per token (decoded UTF-8 / tokens): {total_bytes / total_tokens:.6f}")
    if total_words:
        print(f"Fertility (tokens / word, decoded text): {total_tokens / total_words:.6f}")
    else:
        print("Fertility: n/a (no words after decode)")


def main() -> None:
    root = Path(os.environ.get("GRIM_REPO_ROOT", repo_root_from_script()))
    data_dir = root / "resources/models/GRIM-text/training/data"
    default_vocab = data_dir / "vocab.bin"
    default_grmt = data_dir / "training_data.grmt"

    p = argparse.ArgumentParser(
        description="GRMT + vocab.bin corpus metrics (Bridges-2 training data layout)."
    )
    p.add_argument("--repo-root", type=Path, default=root, help="Local repository root")
    p.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help=f"path to vocab.bin (default: under repo {default_vocab})",
    )
    p.add_argument(
        "--grmt",
        type=Path,
        default=None,
        help=f"path to training_data.grmt (default: under repo {default_grmt})",
    )
    p.add_argument(
        "--fetch",
        action="store_true",
        help="scp vocab.bin + training_data.grmt from Bridges-2 (GRIM_BRIDGES2_DIR), then compute locally",
    )
    p.add_argument(
        "--ssh-run",
        action="store_true",
        help="one ssh: push script via cat (no scp), run GRIM_REMOTE_PYTHON on cluster; uses vocab+grmt on /ocean",
    )
    p.add_argument(
        "--remote-root",
        type=str,
        default=None,
        metavar="DIR",
        help=f"Bridges-2 repo path (default: GRIM_BRIDGES2_DIR or {DEFAULT_REMOTE_ROOT})",
    )
    p.add_argument(
        "--ssh",
        type=str,
        default=None,
        metavar="HOST",
        help="ssh destination (default: GRIM_BRIDGES2_SSH or bridges2 / uwadkins@bridges2.psc.edu)",
    )
    p.add_argument(
        "--partition",
        type=str,
        default="RM-shared",
        metavar="PART",
        help="SLURM partition for --ssh-run compute node dispatch (default: RM-shared)",
    )
    p.add_argument(
        "--account",
        type=str,
        default="cis250124p",
        metavar="ACCT",
        help="SLURM account for --ssh-run (default: cis250124p)",
    )
    args = p.parse_args()

    if args.fetch and args.ssh_run:
        raise SystemExit("Use only one of --fetch or --ssh-run.")

    ssh_host = args.ssh if args.ssh else bridges2_ssh_host()
    remote_root = remote_bridges2_root(args.remote_root)

    if args.ssh_run:
        ssh_run_on_bridges2(ssh_host, remote_root, args.partition, args.account)
        return

    root = args.repo_root.resolve()
    data_dir = root / "resources/models/GRIM-text/training/data"
    vocab_path = (args.vocab or default_vocab).resolve()
    grmt_path = (args.grmt or default_grmt).resolve()

    if args.fetch:
        fetch_training_assets(ssh_host, remote_root, vocab_path, grmt_path)

    if not vocab_path.is_file():
        raise SystemExit(
            f"vocab.bin not found: {vocab_path}\n"
            "  Try: python3 scripts/bridges2_vocab_corpus_metrics.py --fetch"
        )
    if not grmt_path.is_file():
        raise SystemExit(
            f"training_data.grmt not found: {grmt_path}\n"
            "  Try: python3 scripts/bridges2_vocab_corpus_metrics.py --fetch\n"
            "  Or run on cluster: python3 scripts/bridges2_vocab_corpus_metrics.py --ssh-run"
        )

    vocab = load_vocab_bin(vocab_path)

    b2 = os.environ.get("GRIM_BRIDGES2_DIR", "")
    if b2:
        print(f"GRIM_BRIDGES2_DIR (reference): {b2}")
        print()

    run_grmt(grmt_path, vocab, vocab_path)


if __name__ == "__main__":
    main()

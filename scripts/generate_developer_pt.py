#!/usr/bin/env python3
"""Collect developer documents, count with KTMG v9, and append raw PT rows.

Default: stage only. --append updates corpus, PT course, registry and manifest.
Uses the Python standard library; no model execution or dependency installation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import shutil
import sys
import tempfile
import textwrap
import time
from urllib.request import Request, urlopen
from urllib.parse import quote

from pt_vocab import PTTokenizer, ATOM_OPEN

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "resources/models/GRIM-text/training/data"
DEFAULT_SOURCES = ROOT / "DataCollection/developer_pt_sources.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def fetch(url: str, cache: Path, limit=16 * 1024 * 1024) -> bytes:
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / digest(url.encode())
    if path.exists():
        return path.read_bytes()
    headers = {"User-Agent": "GRIM-developer-PT-curator/1"}
    if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers=headers), timeout=45) as response:
                data = response.read(limit + 1)
            if len(data) > limit:
                raise ValueError(f"Source exceeds download limit: {url}")
            atomic_bytes(path, data)
            return data
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def github_documents(source, cache: Path, max_documents: int):
    repo, ref = source["repository"], source["ref"]
    # Resolve once, then fetch all content and licenses at the same immutable commit.
    commit = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{quote(ref, safe='')}", cache))["sha"]
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Invalid GitHub commit SHA")
    base = f"https://raw.githubusercontent.com/{repo}/{commit}/"
    licenses = []
    for path in source["license_paths"]:
        content = fetch(base + quote(path), cache).decode("utf-8")
        licenses.append({"path": path, "text": content, "sha256": digest(content.encode())})
    tree = json.loads(fetch(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1", cache))
    if tree.get("truncated"):
        raise ValueError(f"Truncated source tree: {repo}; use a smaller/local source")
    available = sorted(entry["path"] for entry in tree["tree"] if entry["type"] == "blob")
    paths = list(dict.fromkeys(path for pattern in source["include"] for path in available
                              if fnmatch.fnmatchcase(path, pattern)))
    if not paths:
        raise ValueError(f"No documents matched source {source['id']}")
    for path in paths[:max_documents]:
        print(f"Reading {source['id']}: {path}", flush=True)
        text = fetch(base + quote(path), cache, limit=4 * 1024 * 1024).decode("utf-8")
        source_hash = digest(text.encode())
        dependencies = []
        def load_include(relative, parent):
            target = posixpath.normpath(posixpath.join(posixpath.dirname(parent), relative))
            if target.startswith(("../", "/")) or ":" in target or target not in available:
                raise ValueError(f"Invalid/unavailable include {relative} in {parent}")
            if len(dependencies) >= 64:
                raise ValueError("Too many includes in document")
            included = fetch(base + quote(target), cache, limit=4 * 1024 * 1024).decode("utf-8")
            dependencies.append({"path": target, "sha256": digest(included.encode())})
            return included, target
        if path.endswith(".rst"):
            text = render_rst(text, path, load_include)
        if source.get("start_marker"):
            start = text.find(source["start_marker"])
            if start < 0:
                raise ValueError(f"Missing configured start marker: {source['id']}")
            text = text[start:]
        yield {"text": text, "title": Path(path).stem, "source_id": source["id"],
               "topic": source["topic"], "revision": commit, "version": ref, "path": path,
               "source_url": f"https://github.com/{repo}/blob/{commit}/{quote(path)}",
               "source_sha256": source_hash, "included_files": dependencies, "licenses": licenses}


def render_rst(text, path, load_include, depth=0):
    """Resolve common Sphinx includes and render code blocks without executing them.

    Unknown include selectors fail closed rather than selecting the wrong code.
    Other RST markup is retained as readable source text.
    """
    if depth > 6:
        raise ValueError("Recursive RST include depth exceeded")
    lines = text.splitlines(keepends=True)
    output, i = [], 0
    while i < len(lines):
        match = re.match(r"^([ \t]*)\.\. (include|literalinclude|code-block|code|sourcecode|raw|toctree|image|figure)::\s*(.*?)\s*$", lines[i])
        if not match:
            # Named anchors and comment-only lines are not training prose.
            if not re.match(r"^\s*\.\. _[^:]+:\s*$", lines[i]):
                output.append(lines[i])
            i += 1
            continue
        indent, kind, argument = match.groups()
        i += 1
        body = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= len(indent):
                break
            body.append(line)
            i += 1
        options, content = {}, []
        before_content = True
        for line in body:
            opt = re.match(r"^\s+:([\w-]+):\s*(.*?)\s*$", line)
            if before_content and opt:
                options[opt[1]] = opt[2]
            elif before_content and not line.strip():
                continue
            else:
                before_content = False
                content.append(line)
        if kind in ("raw", "toctree", "image", "figure"):
            if options.get("alt"):
                output.append(options["alt"] + "\n\n")
            continue
        if kind in ("include", "literalinclude"):
            allowed = {"start-after", "end-before", "start-at", "end-at", "lines", "language",
                       "caption", "name", "dedent", "linenos", "emphasize-lines", "lineno-start"}
            if options.keys() - allowed:
                raise ValueError(f"Unsupported include options in {path}: {options.keys() - allowed}")
            selected, target = load_include(argument, path)
            for option in ("start-after", "start-at"):
                if option in options:
                    where = selected.find(options[option])
                    if where < 0:
                        raise ValueError(f"Include marker missing in {target}: {option}")
                    selected = selected[where + (len(options[option]) if option == "start-after" else 0):]
            for option in ("end-before", "end-at"):
                if option in options:
                    where = selected.find(options[option])
                    if where < 0:
                        raise ValueError(f"Include marker missing in {target}: {option}")
                    selected = selected[:where + (len(options[option]) if option == "end-at" else 0)]
            if "lines" in options:
                source_lines = selected.splitlines(keepends=True)
                indexes = []
                for part in options["lines"].split(","):
                    bounds = part.strip().split("-")
                    first = int(bounds[0] or 1)
                    last = int(bounds[-1] or len(source_lines))
                    if first < 1 or first > last or last > len(source_lines):
                        raise ValueError(f"Invalid include lines in {path}")
                    indexes.extend(range(first - 1, last))
                selected = "".join(source_lines[n] for n in indexes)
            if kind == "include":
                output.append(render_rst(selected, target, load_include, depth + 1))
                continue
        else:
            selected = textwrap.dedent("".join(content))
        if "dedent" in options:
            amount = options["dedent"]
            if amount:
                n = int(amount)
                selected = "\n".join(line[n:] if line[:n].isspace() else line for line in selected.split("\n"))
            else:
                selected = textwrap.dedent(selected)
        language = options.get("language", argument if kind in ("code", "code-block", "sourcecode") else "")
        fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", selected)), default=0))
        output.append("\n" + fence + language + "\n" + selected.strip("\n") + "\n" + fence + "\n\n")
    return "".join(output)


def clean_document(text: str) -> str:
    """Conservative source-text cleanup; preserve indentation and original wording."""
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            text = text[end + 5:]
    text = re.sub(r'<a\s+(?:name|id)="[^"]*"\s*>\s*</a>', "", text)
    # Retain link text while removing standalone internal navigation lists.
    text = re.sub(r"(?m)^\s*[*-] \[[^\]\n]+\]\(#[^)]+\)\s*$", "", text)
    return text.strip("\n")


def chunk_document(text: str, tokenizer, max_tokens: int):
    """Prefer paragraphs and whole fenced blocks; fall back to lines/characters.

    Emit contiguous source slices with no inserted text or overlap. Report any
    oversized fenced block instead of manufacturing a broken code example.
    """
    budget = max_tokens - 2  # BOS + EOS reservation
    units, current, fence = [], "", None
    for line in text.splitlines(keepends=True):
        match = re.match(r"^[ \t]*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                if current:
                    units.append((current, False))
                current, fence = line, marker
                continue
            if marker[0] == fence[0] and len(marker) >= len(fence):
                current += line
                units.append((current, True))
                current, fence = "", None
                continue
        current += line
        if not fence and not line.strip():
            units.append((current, False))
            current = ""
    if current:
        units.append((current, "unclosed" if fence else False))
    pending, offset = "", 0
    for unit, protected in units:
        if protected == "unclosed" or (tokenizer.count(unit) > budget and protected):
            if pending:
                yield pending, offset, None
                offset += len(pending)
                pending = ""
            yield unit, offset, "oversized_or_unclosed_code_fence"
            offset += len(unit)
            continue
        pieces = [unit]
        if tokenizer.count(unit) > budget:
            pieces = unit.splitlines(keepends=True)
        for piece in pieces:
            if pending and tokenizer.count(pending + piece) > budget:
                yield pending, offset, None
                offset += len(pending)
                pending = ""
            while tokenizer.count(piece) > budget:
                # Shrink and re-count. Do not assume token count is monotonic.
                end = max(1, len(piece) // 2)
                while tokenizer.count(piece[:end]) > budget and end > 1:
                    end //= 2
                if tokenizer.count(piece[:end]) > budget:
                    raise ValueError("Token budget cannot fit one character")
                # Prefer a nearby whitespace boundary for long prose lines.
                cut = piece.rfind(" ", max(0, end - 120), end)
                if cut > 0 and tokenizer.count(piece[:cut + 1]) <= budget:
                    end = cut + 1
                yield piece[:end], offset, None
                offset += end
                piece = piece[end:]
            pending += piece
    if pending:
        yield pending, offset, None


def raw_text(row):
    if row.get("raw"):
        return row["raw"]
    parts = []
    for field in ("prompt",):
        if row.get(field):
            parts.append(row[field] + "\n")
    for line in row.get("explanation", row.get("intermediates", [])):
        parts.append(line + "\n")
    for field in ("determine", "define", "execute", "update"):
        if row.get(field):
            parts.append(row[field] + "\n")
    return "".join(parts) + row.get("answer", "")


def text_hash(text):
    # Preserve indentation: different code whitespace can change behavior.
    return digest(text.replace("\r\n", "\n").replace("\r", "\n").strip("\n").encode())


def load_target(data_dir: Path, name: str):
    registry = json.loads((data_dir / "curriculum_registry.json").read_text(encoding="utf-8"))
    matches = [c for c in registry["curriculums"] if c["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one existing curriculum named {name}")
    curriculum = matches[0]
    if curriculum.get("training_stage") != "pt" or curriculum.get("format_as_concept") is not False:
        raise ValueError("Target must be a raw PT curriculum")
    courses = {c["id"]: c for c in registry["courses"]}
    if len(courses) != len(registry["courses"]):
        raise ValueError("Duplicate course IDs")
    flat = list(dict.fromkeys(i for c in curriculum["course_ids"] for i in courses[c]["concept_block_ids"]))
    manifest = json.loads((data_dir / f"{name}.json").read_text(encoding="utf-8"))
    if flat != curriculum["concept_block_ids"] or flat != manifest["concept_block_ids"]:
        raise ValueError("Course/registry/manifest membership differs; repair before appending")
    return registry, curriculum, manifest


@contextmanager
def locked(data_dir):
    path = data_dir / ".developer_pt.lock"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        yield
    finally:
        path.unlink()


def recover(data_dir):
    journal = data_dir / ".developer_pt_transaction.json"
    if not journal.exists():
        return
    tx = json.loads(journal.read_text(encoding="utf-8"))
    corpus = data_dir / "concept_blocks.jsonl"
    with corpus.open("rb") as f:
        f.seek(tx["size"])
        tail = f.read()
    expected = tx["append"].encode("utf-8")
    if not expected.startswith(tail):
        raise ValueError("Corpus changed after interrupted append; refusing recovery")
    for name, state in tx["metadata"].items():
        if Path(name).name != name:
            raise ValueError("Invalid transaction metadata path")
        value = (data_dir / name).read_text(encoding="utf-8")
        if value not in (state["before"], state["after"]):
            raise ValueError("Metadata changed after interrupted append; refusing recovery")
    with corpus.open("r+b") as f:
        f.truncate(tx["size"])
        f.flush()
        os.fsync(f.fileno())
    for name, state in tx["metadata"].items():
        atomic_bytes(data_dir / name, state["before"].encode())
    journal.unlink()


def append_rows(data_dir, name, rows):
    with locked(data_dir):
        recover(data_dir)
        registry, curriculum, manifest = load_target(data_dir, name)
        metadata_before = {filename: (data_dir / filename).read_text(encoding="utf-8")
                           for filename in ("curriculum_registry.json", f"{name}.json")}
        if json.loads(metadata_before["curriculum_registry.json"]) != registry:
            raise ValueError("Registry changed during validation")
        if json.loads(metadata_before[f"{name}.json"]) != manifest:
            raise ValueError("Manifest changed during validation")
        corpus = data_dir / "concept_blocks.jsonl"
        before_stat = corpus.stat()
        by_hash, by_id = {}, {}
        with corpus.open(encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)  # Never drop malformed existing records.
                h = text_hash(raw_text(row))
                if row["id"] in by_id:
                    raise ValueError(f"Duplicate existing ID at line {line_number}")
                by_id[row["id"]] = h
                by_hash.setdefault(h, row["id"])
        missing = set(curriculum["concept_block_ids"]) - by_id.keys()
        if missing:
            raise ValueError(f"PT references {len(missing)} missing corpus rows")
        members = set(curriculum["concept_block_ids"])
        added_ids, new_rows = [], []
        for row in rows:
            h = text_hash(row["raw"])
            if row["id"] in by_id and by_id[row["id"]] != h:
                raise ValueError("Content ID collision")
            block_id = by_hash.get(h)
            if block_id is None:
                block_id = row["id"]
                new_rows.append(row)
                by_hash[h] = block_id
                by_id[block_id] = h
            if block_id not in members:
                added_ids.append(block_id)
                members.add(block_id)
        if not added_ids:
            return {"new_rows": 0, "new_members": 0}
        course_id = "course_dev_pt_" + digest(curriculum["id"].encode())[:16]
        course = next((c for c in registry["courses"] if c["id"] == course_id), None)
        if course is None:
            course = {"id": course_id, "name": "Developer assistance", "concept_block_ids": []}
            registry["courses"].append(course)
        elif course_id not in curriculum["course_ids"]:
            raise ValueError("Developer course ID collision")
        course["concept_block_ids"].extend(added_ids)
        if course_id not in curriculum["course_ids"]:
            curriculum["course_ids"].append(course_id)
        courses = {c["id"]: c for c in registry["courses"]}
        curriculum["concept_block_ids"] = list(dict.fromkeys(
            i for c in curriculum["course_ids"] for i in courses[c]["concept_block_ids"]))
        curriculum["timestamp"] = int(time.time())
        manifest.update(concept_block_ids=curriculum["concept_block_ids"],
                        format_as_concept=False, training_stage="pt")
        metadata = {}
        for filename, value in (("curriculum_registry.json", registry), (f"{name}.json", manifest)):
            metadata[filename] = {"before": metadata_before[filename],
                                  "after": json_bytes(value).decode()}
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in new_rows)
        if payload and before_stat.st_size:
            with corpus.open("rb") as f:
                f.seek(-1, 2)
                if f.read(1) != b"\n":
                    payload = "\n" + payload
        now = corpus.stat()
        if (now.st_size, now.st_mtime_ns) != (before_stat.st_size, before_stat.st_mtime_ns):
            raise ValueError("Corpus changed during validation; retry with other writers stopped")
        if any((data_dir / filename).read_text(encoding="utf-8") != content
               for filename, content in metadata_before.items()):
            raise ValueError("Metadata changed during validation; retry with other writers stopped")
        journal = data_dir / ".developer_pt_transaction.json"
        atomic_bytes(journal, json_bytes({"size": before_stat.st_size, "append": payload, "metadata": metadata}))
        try:
            with corpus.open("ab") as f:
                f.write(payload.encode())
                f.flush()
                os.fsync(f.fileno())
            for filename, state in metadata.items():
                atomic_bytes(data_dir / filename, state["after"].encode())
            journal.unlink()
        except BaseException:
            recover(data_dir)
            raise
        return {"new_rows": len(new_rows), "new_members": len(added_ids)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--source", action="append", help="Select source IDs; repeatable")
    parser.add_argument("--input-jsonl", type=Path, help="Offline documents: text, title, source_url, revision, licenses")
    parser.add_argument("--vocab", type=Path, default=DATA / "vocab.bin")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--curriculum", default="Pre-Trainingv1")
    parser.add_argument("--output", type=Path, default=ROOT / "temp/developer_pt")
    parser.add_argument("--max-sequences", type=int, default=1000)
    parser.add_argument("--max-documents-per-source", type=int, default=20)
    parser.add_argument("--max-sequences-per-source", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--recover", action="store_true", help="Recover an interrupted append and exit; remove a stale lock only after its process is stopped")
    args = parser.parse_args()
    if args.recover:
        with locked(args.data_dir):
            recover(args.data_dir)
        print("Recovery complete")
        return
    if not (4 <= args.max_tokens <= 1024 and 0 < args.min_tokens <= args.max_tokens - 2
            and args.max_sequences > 0 and args.max_documents_per_source > 0
            and args.max_sequences_per_source > 0):
        parser.error("Invalid token/count limits")
    if Path(args.curriculum).name != args.curriculum or any(c in args.curriculum for c in '/\\:'):
        parser.error("Curriculum must be a name, not a path")
    tokenizer = PTTokenizer(args.vocab)
    load_target(args.data_dir, args.curriculum)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.input_jsonl:
        with args.input_jsonl.open(encoding="utf-8") as f:
            documents = [json.loads(line) for line in f if line.strip()]
        streams = [iter(documents)]
    else:
        sources = json.loads(args.sources.read_text(encoding="utf-8"))["sources"]
        if args.source:
            if set(args.source) - {s["id"] for s in sources}:
                parser.error("Unknown source ID")
            sources = [s for s in sources if s["id"] in args.source]
        streams = [iter(github_documents(s, args.output / "cache", args.max_documents_per_source)) for s in sources]
    stats, source_counts, rows, provenance, seen = Counter(), Counter(), [], [], set()
    document_metadata = {}
    source_cap = min(args.max_sequences_per_source,
                     max(1, args.max_sequences // max(1, len(streams))))
    # Round-robin documents so a large first source cannot consume the full run.
    while streams and len(rows) < args.max_sequences:
        remaining = []
        for stream in streams:
            try:
                doc = next(stream)
            except StopIteration:
                continue
            remaining.append(stream)
            for field in ("text", "title", "source_url", "revision", "licenses"):
                if not doc.get(field):
                    raise ValueError(f"Document missing {field}")
            stats["documents"] += 1
            source_id = doc.get("source_id", "local")
            if source_counts[source_id] >= source_cap:
                remaining.pop()
                continue
            text = clean_document(doc["text"])
            if ATOM_OPEN.search(text):
                stats["documents_with_atom_delimiters"] += 1
                continue
            document_id = digest((doc["source_url"] + "\n" + doc["revision"] + "\n" + digest(text.encode())).encode())
            document_metadata[document_id] = {k: v for k, v in doc.items() if k != "text"} | {
                "document_id": document_id, "document_sha256": digest(text.encode())}
            for chunk, offset, rejection in chunk_document(text, tokenizer, args.max_tokens):
                if rejection:
                    stats[rejection] += 1
                    continue
                tokens = tokenizer.count(chunk)
                if tokens < args.min_tokens:
                    stats["short_chunks"] += 1
                    continue
                if tokens + 2 > args.max_tokens:
                    raise AssertionError("Chunk exceeds token limit")
                h = text_hash(chunk)
                if h in seen:
                    stats["duplicate_chunks"] += 1
                    continue
                seen.add(h)
                block_id = "cb_pt_dev_" + h
                rows.append({"id": block_id, "name": doc["title"], "format_type": "raw",
                             "raw": chunk, "prompt": "", "answer": "", "explanation": [],
                             "intermediates": [], "intermediate_count": 0, "step_index": [],
                             "timestamp": 0, "source_sequence_id": doc["source_url"]})
                source_counts[source_id] += 1
                provenance.append({
                    "id": block_id, "document_id": document_id,
                    "character_offset": offset, "character_length": len(chunk),
                    "text_tokens": tokens, "tokens_with_boundaries": tokens + 2})
                if len(rows) >= args.max_sequences or source_counts[source_id] >= source_cap:
                    break
            print(f"Staged {len(rows)} sequences from {stats['documents']} documents", flush=True)
            if len(rows) >= args.max_sequences:
                break
        streams = remaining
    run_id = digest(json_bytes({"provenance": provenance, "documents": document_metadata,
                               "vocab": tokenizer.sha256, "max_tokens": args.max_tokens}))
    run_dir = args.output / "runs" / run_id
    for filename, records in (("candidates.jsonl", rows), ("provenance.jsonl", provenance),
                              ("documents.jsonl", list(document_metadata.values()))):
        atomic_bytes(run_dir / filename, b"".join(
            (json.dumps(row, ensure_ascii=False) + "\n").encode() for row in records))
    report = {"statistics": dict(stats), "sequences": len(rows), "sequences_by_source": dict(source_counts), "vocab_sha256": tokenizer.sha256,
              "tokenizer": "KTMG-v9-raw-CPU", "max_tokens": args.max_tokens,
              "text_tokens": sum(p["text_tokens"] for p in provenance),
              "max_observed_tokens_with_boundaries": max((p["tokens_with_boundaries"] for p in provenance), default=0),
              "mode": "append" if args.append else "stage", "artifacts": str(run_dir)}
    if args.append:
        if digest(args.vocab.read_bytes()) != tokenizer.sha256:
            raise ValueError("Vocabulary changed during generation; rerun before appending")
        # Keep provenance beside the corpus even when staging uses a temp folder.
        archive = args.data_dir / "developer_pt_provenance" / run_id
        archive.mkdir(parents=True, exist_ok=True)
        for filename in ("candidates.jsonl", "provenance.jsonl", "documents.jsonl"):
            shutil.copyfile(run_dir / filename, archive / filename)
        atomic_bytes(archive / "report.json", json_bytes(report))
        report["append"] = append_rows(args.data_dir, args.curriculum, rows)
        report["provenance_archive"] = str(archive.resolve())
        atomic_bytes(archive / "report.json", json_bytes(report))
    atomic_bytes(run_dir / "report.json", json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

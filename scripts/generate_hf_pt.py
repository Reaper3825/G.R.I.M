"""Bounded, revision-checked Hugging Face PT ingestion; stdlib only.

Uses dataset viewer rows, preserving its x-revision and complete responses.
Stage first; --append applies the staged candidates to three separate courses.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time
import urllib.parse
import urllib.request
import urllib.error

from generate_developer_pt import (DATA, atomic_bytes, json_bytes, digest,
    clean_document, chunk_document, text_hash, raw_text, append_rows, load_target, ATOM_OPEN)
from pt_vocab import PTTokenizer

SOURCES = [
    ("howto", "HuggingFaceTB/cosmopedia", "wikihow", "Practical how-to (Cosmopedia)", True),
    ("education", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "General education (Cosmopedia v2)", True),
    ("math", "HuggingFaceTB/finemath", "finemath-4plus", "Mathematics (FineMath 4+)", False),
]
SPAM = re.compile(r"write (?:my|a|your) (?:paper|essay) online|masterpapers\.com|buy essays|essay writing service|代写|代考", re.I)
PRACTICAL = re.compile(r"\b(?:organiz\w*|organis\w*|plan\w*|schedul\w*|budget\w*|travel\w*|pack\w*|cook\w*|meal\w*|grocer\w*|shop\w*|clean\w*|laundry|household|home|kitchen|email\w*|letter\w*|resume|meeting\w*|appointment\w*|calendar\w*|password\w*|repair\w*|maintain|polite\w*|communicat\w*|listen\w*|apolog\w*|thank|conversation\w*|time|priorit\w*|productiv\w*|goal\w*|study|studying|note\w*|recycl\w*|store|storage|save|saving|money|bank|bill\w*|rent\w*|interview\w*)\b", re.I)


MATH_HOSTS = {
    "math.stackexchange.com", "mathoverflow.net", "nrich.maths.org",
    "en.wikibooks.org", "en.m.wikibooks.org", "math.libretexts.org",
    "stats.libretexts.org", "openstax.org", "mathsisfun.com", "proofwiki.org",
    "statlect.com", "cut-the-knot.org", "mathinsight.org", "mathworld.wolfram.com",
    "artofproblemsolving.com", "ck12.org", "khanacademy.org", "mathforum.org",
    "mathopenref.com", "mathcentre.ac.uk", "ams.org", "maa.org",
}


def math_domain_allowed(url):
    host = (urllib.parse.urlsplit(url).hostname or "").lower().removeprefix("www.")
    return host in MATH_HOSTS or host.endswith(".edu") or host.endswith(".ac.uk")


def request_json(url):
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GRIM-PT-curation/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                blob = response.read(24 * 1024 * 1024 + 1)
                if len(blob) > 24 * 1024 * 1024:
                    raise ValueError("Response exceeds bounded download limit")
                return json.loads(blob), response.headers.get("x-revision")
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            delay = min(60, max(10, int(error.headers.get("Retry-After", "30"))))
            print(f"HF HTTP {error.code}; retrying in {delay}s", flush=True)
            time.sleep(delay)
        except OSError:
            if attempt == 7:
                raise
            time.sleep(min(30, 2 ** attempt))


def get_page(dataset, config, offset, length, revision, cache):
    query = dict(dataset=dataset, config=config, split="train", offset=offset, length=length)
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(query)
    path = cache / (digest((revision + url).encode()) + ".json")
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload, actual_revision = request_json(url)
        record = {"url": url, "revision": actual_revision, "payload": payload}
        if actual_revision != revision:
            raise ValueError("Dataset viewer revision differs from Hub; retry when synchronized")
        atomic_bytes(path, json_bytes(record))
    if record["revision"] != revision or record["url"] != url:
        raise ValueError("Invalid cached response identity")
    return record["payload"]


def rejection(row, truncated):
    if truncated:
        return "truncated_row"
    text = row.get("text", "")
    if not isinstance(text, str) or len(text) < 500:
        return "short_document"
    if ATOM_OPEN.search(text):
        return "atom_delimiters"
    if SPAM.search(text):
        return "essay_spam"
    if "remote host or network may be down" in text.lower():
        return "error_page_contamination"
    if text.lstrip().startswith("# Search by Topic"):
        return "topic_index"
    if len(re.findall(r"\bWeb\b", text)) >= 5 and "worksheet" in text.lower():
        return "worksheet_seo_text"
    if "\ufffd" in text or "\x00" in text:
        return "damaged_text"
    if row.get("language", "en") != "en":
        return "non_english"
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isascii() for c in letters) / len(letters) < .9:
        return "non_latin_heavy_text"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 10 and len(set(lines)) / len(lines) < .65:
        return "repeated_lines"
    if len(lines) >= 30 and sum(len(line) < 20 for line in lines) / len(lines) > .45:
        return "fragmented_extraction"
    return None


def source_pages(dataset, config, offsets, page_size, total, revision, cache):
    """Keep only four downloads in flight; return pages in deterministic order."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = []
        remaining = iter(offsets)
        def enqueue():
            offset = next(remaining, None)
            if offset is not None:
                pending.append(pool.submit(get_page, dataset, config, offset,
                    min(page_size, total-offset), revision, cache))
        for _ in range(4):
            enqueue()
        while pending:
            page = pending.pop(0).result()
            enqueue()
            yield page


def write_records(path, records):
    atomic_bytes(path, b"".join((json.dumps(r, ensure_ascii=False) + "\n").encode() for r in records))


def parquet_pages(spec, offsets, page_size, cache):
    """Read bounded pages from an integrity-checked, pinned first source shard."""
    import bisect
    import hashlib
    from functools import lru_cache
    from datetime import date, datetime
    import pyarrow.parquet as pq
    path = Path(spec["path"])
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    if h.hexdigest() != spec["sha256"] or path.stat().st_size != spec["size"]:
        raise ValueError("Parquet shard integrity mismatch")
    parquet = pq.ParquetFile(path)
    starts = [0]
    for i in range(parquet.num_row_groups):
        starts.append(starts[-1] + parquet.metadata.row_group(i).num_rows)
    @lru_cache(maxsize=4)
    def group_rows(group):
        return parquet.read_row_group(group).to_pylist()
    for offset in offsets:
        wrapped = []
        for index in range(offset, min(offset + page_size, starts[-1])):
            group = bisect.bisect_right(starts, index) - 1
            row = group_rows(group)[index-starts[group]]
            row = {k: v.isoformat() if isinstance(v, (date, datetime)) else v for k, v in row.items()}
            wrapped.append(dict(row_idx=index, row=row, truncated_cells=[]))
        payload = dict(rows=wrapped, num_rows_total=starts[-1])
        url = f"https://huggingface.co/datasets/{spec['dataset']}/resolve/{spec['revision']}/{spec['repository_path']}?" + urllib.parse.urlencode(
            dict(dataset=spec["dataset"], config=spec["config"], split="train", offset=offset, length=page_size))
        atomic_bytes(cache / (digest((spec["revision"] + url).encode()) + ".json"), json_bytes(
            dict(url=url, revision=spec["revision"], payload=payload, transport="parquet",
                 repository_path=spec["repository_path"], parquet_sha256=spec["sha256"])))
        yield payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--vocab", type=Path, default=DATA / "vocab.bin")
    parser.add_argument("--curriculum", default="Pre-Trainingv1")
    parser.add_argument("--output", type=Path, default=DATA / "hf_pt_provenance" / "pilot_v1")
    parser.add_argument("--per-source", type=int, default=400)
    for key, *_ in SOURCES:
        parser.add_argument(f"--{key}-sequences", type=int)
    parser.add_argument("--exclude-existing", action="store_true",
                        help="Count only content absent from the existing corpus toward quotas")
    parser.add_argument("--page-size", type=int, default=50, choices=(50, 100))
    parser.add_argument("--parquet-manifest", type=Path,
                        help="Use pinned local shards with an existing PyArrow environment")
    parser.add_argument("--max-pages", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    if args.per_source < 1 or args.max_pages < 1:
        parser.error("Counts must be positive")
    targets = {key: getattr(args, key + "_sequences") for key, *_ in SOURCES}
    targets = {key: value if value is not None else args.per_source for key, value in targets.items()}
    if any(value < 1 for value in targets.values()):
        parser.error("Source targets must be positive")
    tokenizer = PTTokenizer(args.vocab)
    # Fail before downloading if the target membership is already inconsistent.
    _, target, _ = load_target(args.data_dir, args.curriculum)
    before_count = len(target["concept_block_ids"])
    all_rows, provenance, documents, source_reports = [], [], [], {}
    seen = set()
    parquet_sources = json.loads(args.parquet_manifest.read_text(encoding="utf-8")) if args.parquet_manifest else {}
    if args.exclude_existing:
        with (args.data_dir / "concept_blocks.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    seen.add(text_hash(raw_text(json.loads(line))))
        print(f"Excluding {len(seen)} existing content hashes", flush=True)
    for key, dataset, config, course, synthetic in SOURCES:
        target_count = targets[key]
        spec = parquet_sources.get(key)
        if spec:
            if (spec["dataset"], spec["config"]) != (dataset, config):
                raise ValueError("Parquet manifest source mismatch")
            hub = spec["hub"]
        else:
            hub, _ = request_json("https://huggingface.co/api/datasets/" + dataset)
        revision = hub["sha"]
        atomic_bytes(args.output / "sources" / (key + "_" + revision + ".json"), json_bytes(hub))
        if spec:
            import pyarrow.parquet as pq
            if revision != spec["revision"]:
                raise ValueError("Parquet manifest revision mismatch")
            total = pq.ParquetFile(spec["path"]).metadata.num_rows
        else:
            first = get_page(dataset, config, 0, 1, revision, args.output / "cache")
            total = first["num_rows_total"]
        rng = random.Random(str(args.seed) + key)
        offsets = rng.sample(range(0, total, args.page_size),
                             min(args.max_pages, (total + args.page_size - 1) // args.page_size))
        stats = Counter()
        rows = []
        pages = (parquet_pages(spec, offsets, args.page_size, args.output / "cache") if spec else
                 source_pages(dataset, config, offsets, args.page_size, total, revision, args.output / "cache"))
        for page in pages:
            for wrapped in page["rows"]:
                stats["documents_examined"] += 1
                row = wrapped["row"]
                reason = rejection(row, wrapped.get("truncated_cells"))
                if reason:
                    stats[reason] += 1
                    continue
                if key == "math" and not math_domain_allowed(row.get("url", "")):
                    stats["outside_math_domains"] += 1
                    continue
                if key == "howto":
                    match = re.search(r'tutorial on "([^"]+)"', row.get("prompt", ""))
                    topic = match.group(1) if match else row["text"].strip().splitlines()[0]
                    if not PRACTICAL.search(topic):
                        stats["outside_practical_topics"] += 1
                        continue
                text = clean_document(row["text"])
                index = wrapped["row_idx"]
                locator = spec["repository_path"] if spec else "train"
                doc_id = digest(f"{dataset}/{revision}/{config}/{locator}/{index}".encode())
                source_url = f"https://huggingface.co/datasets/{dataset}/blob/{revision}/README.md"
                title = text.strip().splitlines()[0].lstrip("# ")[:180]
                accepted = 0
                for chunk, position, error in chunk_document(text, tokenizer, 1024):
                    if error:
                        stats[error] += 1
                        continue
                    tokens = tokenizer.count(chunk)
                    if tokens < 128:
                        stats["short_chunks"] += 1
                        continue
                    if tokens + 2 > 1024:
                        raise AssertionError("Sequence exceeds 1024")
                    h = text_hash(chunk)
                    if h in seen:
                        stats["duplicate_chunks"] += 1
                        continue
                    seen.add(h)
                    block_id = "cb_pt_hf_" + h
                    rows.append(dict(id=block_id, name=title, format_type="raw", raw=chunk,
                        prompt="", answer="", explanation=[], intermediates=[], intermediate_count=0,
                        step_index=[], timestamp=0, source_sequence_id=f"hf://datasets/{dataset}@{revision}/{config}/{locator}/{index}"))
                    provenance.append(dict(id=block_id, document_id=doc_id, source=key,
                        character_offset=position, character_length=len(chunk), text_tokens=tokens,
                        tokens_with_boundaries=tokens+2))
                    accepted += 1
                    # Avoid one long page dominating the pilot. No cross-document packing.
                    if len(rows) == target_count or accepted == 4:
                        break
                if accepted:
                    documents.append(dict(document_id=doc_id, dataset=dataset, config=config,
                        split="train", row_index=index, revision=revision, source_url=source_url,
                        synthetic=synthetic, declared_license=hub.get("cardData", {}).get("license"),
                        parquet_path=spec["repository_path"] if spec else None,
                        parquet_sha256=spec["sha256"] if spec else None,
                        document_sha256=digest(text.encode()),
                        upstream_metadata={k:v for k,v in row.items() if k != "text"}))
                if len(rows) == target_count:
                    break
            print(f"{key}: {len(rows)}/{target_count} sequences; {stats['documents_examined']} documents examined", flush=True)
            if len(rows) == target_count:
                break
        pages.close()
        if len(rows) != target_count:
            raise ValueError(f"Insufficient usable rows for {key}; increase --max-pages")
        all_rows.extend(rows)
        source_reports[key] = dict(dataset=dataset, config=config, revision=revision,
            course=course, sequences=len(rows), statistics=dict(stats),
            acquisition="pinned_first_parquet_shard" if spec else "dataset_viewer",
            parquet_path=spec["repository_path"] if spec else None)
    output_root = args.output
    run_id = digest(json_bytes(dict(provenance=provenance, vocab=tokenizer.sha256, sources=source_reports)))
    args.output = output_root / "runs" / run_id
    report = dict(seed=args.seed, source_reports=source_reports, sequences=len(all_rows),
        exclude_existing=args.exclude_existing, page_size=args.page_size, source_targets=targets,
        text_tokens=sum(p["text_tokens"] for p in provenance),
        max_tokens_with_boundaries=max(p["tokens_with_boundaries"] for p in provenance),
        vocab_sha256=tokenizer.sha256, tokenizer="KTMG-v9-raw-CPU", mode="stage",
        members_before=before_count, artifacts=str(args.output.resolve()),
        generator_sha256=digest(Path(__file__).read_bytes()), quality="Heuristic filtering; not fact-checked",
        split="All appended members are PT candidates; no held-out evaluation split created")
    write_records(args.output / "candidates.jsonl", all_rows)
    write_records(args.output / "provenance.jsonl", provenance)
    write_records(args.output / "documents.jsonl", documents)
    atomic_bytes(args.output / "report.json", json_bytes(report))
    if args.append:
        if digest(args.vocab.read_bytes()) != tokenizer.sha256:
            raise ValueError("Vocabulary changed during generation")
        report["mode"] = "append"
        report["append"] = {}
        for key, _, _, course, _ in SOURCES:
            ids = {p["id"] for p in provenance if p["source"] == key}
            report["append"][key] = append_rows(args.data_dir, args.curriculum,
                [r for r in all_rows if r["id"] in ids], "hf_" + key, course)
            # Each course is independently transactional and retry-safe.
            atomic_bytes(args.output / "report.json", json_bytes(report))
        _, target, _ = load_target(args.data_dir, args.curriculum)
        report["members_after"] = len(target["concept_block_ids"])
        atomic_bytes(args.output / "report.json", json_bytes(report))
    atomic_bytes(output_root / "report.json", json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

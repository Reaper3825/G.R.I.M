"""Verify staged HF chunks against cached source text; optionally append offline."""
import argparse
import json
from pathlib import Path
import urllib.parse

from generate_hf_pt import SOURCES, DATA, clean_document, digest, text_hash, PTTokenizer
from generate_developer_pt import append_rows, load_target, atomic_bytes, json_bytes


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_raw_entry(row):
    if row.get("format_type") != "raw" or not isinstance(row.get("raw"), str) or not row["raw"].strip():
        raise ValueError("PT entries require format_type=raw and nonempty raw text")
    for field in ("prompt", "answer", "explanation", "intermediates", "goal", "knowns",
                  "unknowns", "determine", "define", "execute", "update", "step_index", "intermediate_count"):
        if row.get(field):
            raise ValueError(f"Raw PT entry has nonempty structured field: {field}")


def verify(root, vocab):
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    artifacts = Path(report["artifacts"])
    rows = read_rows(artifacts / "candidates.jsonl")
    provenance = read_rows(artifacts / "provenance.jsonl")
    documents = read_rows(artifacts / "documents.jsonl")
    by_id = {r["id"]: r for r in rows}
    by_doc = {d["document_id"]: d for d in documents}
    if len(by_id) != len(rows) or len(by_doc) != len(documents):
        raise ValueError("Duplicate artifact IDs")
    if len(provenance) != len(rows) or {p["id"] for p in provenance} != set(by_id):
        raise ValueError("Candidate/provenance mismatch")
    tokenizer = PTTokenizer(vocab)
    if tokenizer.sha256 != report["vocab_sha256"]:
        raise ValueError("Vocabulary changed since staging")
    needed = {(d["dataset"], d["config"], d["revision"], d["row_index"]): d for d in documents}
    texts = {}
    for path in (root / "cache").glob("*.json"):
        cached = json.loads(path.read_text(encoding="utf-8"))
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(cached["url"]).query)
        for wrapped in cached["payload"]["rows"]:
            key = (query["dataset"][0], query["config"][0], cached["revision"], wrapped["row_idx"])
            if key in needed:
                doc = needed[key]
                if doc.get("parquet_path") != cached.get("repository_path"):
                    continue
                if doc.get("parquet_sha256") != cached.get("parquet_sha256"):
                    raise ValueError("Parquet source digest mismatch")
                if wrapped.get("truncated_cells"):
                    raise ValueError("Source response was truncated")
                text = clean_document(wrapped["row"]["text"])
                if digest(text.encode()) != doc["document_sha256"]:
                    raise ValueError("Source document digest mismatch")
                texts[doc["document_id"]] = text
    if set(texts) != set(by_doc):
        raise ValueError("Missing cached source documents")
    source_specs = {s[0]: s for s in SOURCES}
    total_tokens = 0
    for p in provenance:
        row, doc = by_id[p["id"]], by_doc[p["document_id"]]
        validate_raw_entry(row)
        _, dataset, config, _, _ = source_specs[p["source"]]
        if (doc["dataset"], doc["config"]) != (dataset, config):
            raise ValueError("Source attribution mismatch")
        start, length = p["character_offset"], p["character_length"]
        if start < 0 or length <= 0 or texts[p["document_id"]][start:start+length] != row["raw"]:
            raise ValueError("Chunk is not the documented contiguous source slice")
        if row["id"] != "cb_pt_hf_" + text_hash(row["raw"]) or row["format_type"] != "raw":
            raise ValueError("Candidate content identity mismatch")
        count = tokenizer.count(row["raw"])
        if count != p["text_tokens"] or count + 2 != p["tokens_with_boundaries"] or not 128 <= count <= 1022:
            raise ValueError("Token bound/count mismatch")
        total_tokens += count
    if total_tokens != report["text_tokens"] or len(rows) != report["sequences"]:
        raise ValueError("Report totals mismatch")
    report["verification"] = dict(sequences=len(rows), documents=len(documents),
        text_tokens=total_tokens, source_slices="verified", token_counts="verified", raw_pt_contract="verified")
    return report, rows, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA / "hf_pt_provenance" / "pilot_v1")
    parser.add_argument("--vocab", type=Path, default=DATA / "vocab.bin")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--curriculum", default="Pre-Trainingv1")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    report, rows, provenance = verify(args.output, args.vocab)
    print(json.dumps(report["verification"]), flush=True)
    artifacts = Path(report["artifacts"])
    if args.append:
        if digest(args.vocab.read_bytes()) != report["vocab_sha256"]:
            raise ValueError("Vocabulary changed during verification")
        report["mode"] = "append"
        report["append"] = {}
        _, curriculum, _ = load_target(args.data_dir, args.curriculum)
        report["members_before"] = len(curriculum["concept_block_ids"])
        for key, _, _, course, _ in SOURCES:
            ids = {p["id"] for p in provenance if p["source"] == key}
            report["append"][key] = append_rows(args.data_dir, args.curriculum,
                [r for r in rows if r["id"] in ids], "hf_" + key, course)
            atomic_bytes(artifacts / "report.json", json_bytes(report))
            atomic_bytes(args.output / "report.json", json_bytes(report))
            print(key, report["append"][key], flush=True)
        _, curriculum, _ = load_target(args.data_dir, args.curriculum)
        report["members_after"] = len(curriculum["concept_block_ids"])
    atomic_bytes(artifacts / "report.json", json_bytes(report))
    atomic_bytes(args.output / "report.json", json_bytes(report))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

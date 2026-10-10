"""Audit the 8,800-entry raw PT expansion against its saved baseline."""
import json
from pathlib import Path

from generate_hf_pt import DATA, SOURCES, digest
from generate_developer_pt import load_target, atomic_bytes, json_bytes
from verify_hf_pt import read_rows, validate_raw_entry


def main():
    root = DATA / "hf_pt_provenance" / "expansion_8800"
    baseline = json.loads((root / "baseline_pt.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    artifacts = Path(report["artifacts"])
    rows = read_rows(artifacts / "candidates.jsonl")
    provenance = read_rows(artifacts / "provenance.jsonl")
    expected = {r["id"]: r for r in rows}
    targets = {"howto": 3600, "education": 3600, "math": 1600}
    if len(rows) != 8800 or len(expected) != 8800:
        raise ValueError("Expected exactly 8,800 unique expansion entries")
    before = set(baseline["curriculum"]["concept_block_ids"])
    if before & expected.keys():
        raise ValueError("Expansion reuses baseline PT IDs")
    registry, curriculum, manifest = load_target(DATA, "Pre-Trainingv1")
    if set(curriculum["concept_block_ids"]) != before | expected.keys():
        raise ValueError("Final membership differs from baseline plus the requested expansion")
    courses = {c["id"]: c for c in registry["courses"]}
    for original in baseline["courses"]:
        current = courses[original["id"]]["concept_block_ids"]
        if current[:len(original["concept_block_ids"])] != original["concept_block_ids"]:
            raise ValueError("Existing course membership was changed")
    by_source = {}
    for key, _, _, name, _ in SOURCES:
        ids = [p["id"] for p in provenance if p["source"] == key]
        if len(ids) != targets[key] or len(set(ids)) != targets[key]:
            raise ValueError("Per-source expansion quota mismatch")
        course_id = "course_hf_" + key + "_pt_" + digest(curriculum["id"].encode())[:16]
        course = courses[course_id]
        if course["name"] != name or not set(ids) <= set(course["concept_block_ids"]):
            raise ValueError("Source course is missing new entry IDs")
        by_source[key] = ids
    found = set()
    with (DATA / "concept_blocks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in expected:
                if row["id"] in found or row != expected[row["id"]]:
                    raise ValueError("Duplicate or changed appended corpus entry")
                validate_raw_entry(row)
                found.add(row["id"])
    if found != set(expected):
        raise ValueError("Corpus is missing expansion entries")
    if (DATA / ".developer_pt.lock").exists() or (DATA / ".developer_pt_transaction.json").exists():
        raise ValueError("Unfinished append transaction")
    audit = dict(new_entries=len(found), members_before=len(before),
        members_after=len(curriculum["concept_block_ids"]), raw_pt_contract="verified",
        registry_manifest_course_membership="verified", baseline_preserved=True,
        entries_by_source={key: len(ids) for key, ids in by_source.items()})
    atomic_bytes(root / "appended_ids.json", json_bytes(by_source))
    atomic_bytes(root / "audit.json", json_bytes(audit))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

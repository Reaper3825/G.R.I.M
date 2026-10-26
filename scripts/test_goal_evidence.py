import json
import tempfile
import unittest
from pathlib import Path

from build_concept_blocks_flatbuffer import build_flatbuffer, default_flatc, write_dataset_envelope
from goal_evidence import validate_goal
from migrate_goal_evidence import migrate_entry, migrate_file


class GoalEvidenceTests(unittest.TestCase):
    def test_active_span_configuration_excludes_evidence(self):
        repo = Path(__file__).resolve().parent.parent
        fixture = json.loads((repo / "tests/fixtures/concept_spans.json").read_text())
        def check(nodes):
            for node in nodes:
                self.assertNotIn("evidence", node["source_path"].split("/"))
                check(node.get("children", []))
        check(fixture)
        config = repo / "ai_config.json"
        if config.exists():
            active = json.loads(config.read_text())["training"]["config"]["concept_spans"]
            check(active)
            def layout(nodes):
                return [{key: (layout(value) if key == "children" else value)
                         for key, value in node.items() if key != "supervision"} for node in nodes]
            self.assertEqual(layout(active), layout(fixture))

    def test_structured_and_optional_evidence(self):
        goal = {"success_criteria": [{"criterion": "Carry key", "evidence": {
            "predicate": "carrying", "subject": "key_1"}}, {"criterion": "Review"}],
            "constraints": [{"constraint": "Preserve apple", "evidence": {
                "predicate": "in", "subject": "apple_1", "target": "box_1"}}]}
        validate_goal(goal)
        original = {"id": "one", "prompt": "Retrieve it", "goal": goal}
        migrated, archive = migrate_entry(original)
        self.assertEqual(migrated, original)
        self.assertEqual(archive, [])

    def test_migration_archives_prose_without_interpreting_it(self):
        original = {"id": "one", "goal": {"success_criteria": [
            {"criterion": "Retrieve key", "evidence": "carrying(key_1)"},
            {"criterion": "Review", "evidence": ""}], "constraints": ["Stay safe"]}}
        migrated, archive = migrate_entry(original)
        self.assertEqual(migrated["goal"]["constraints"], [{"constraint": "Stay safe"}])
        self.assertNotIn("evidence", migrated["goal"]["success_criteria"][0])
        self.assertEqual(archive[0]["text"], "carrying(key_1)")
        self.assertEqual(archive[0]["path"], "/goal/success_criteria/0/evidence")
        self.assertEqual(original["goal"]["success_criteria"][0]["evidence"], "carrying(key_1)")

    def test_invalid_evidence_rejected(self):
        for evidence in (None, "", "prose", 5, [], {}, {"predicate": "carrying"},
                         {"predicate": " ", "subject": "key"},
                         {"predicate": "carrying", "subject": "key", "target": None},
                         {"predicate": "carrying", "subject": "key", "unknown": "bad"}):
            for collection, field in (("success_criteria", "criterion"), ("constraints", "constraint")):
                with self.subTest(evidence=evidence, collection=collection), self.assertRaises(ValueError):
                    validate_goal({collection: [{field: "Requirement", "evidence": evidence}]})

    def test_compiler_rejects_legacy_goal(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.jsonl"
            source.write_text(json.dumps({"id": "one", "prompt": "Task", "goal": {
                "success_criteria": [{"criterion": "Check", "evidence": "prose"}]}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "migrate"):
                write_dataset_envelope(source, Path(folder) / "envelope.json")

    def test_file_migration_preserves_source_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output, archive = [Path(folder) / name for name in ("source.jsonl", "new.jsonl", "archive.jsonl")]
            source.write_text(json.dumps({"id": "one", "prompt": "Task", "goal": {
                "success_criteria": [{"criterion": "Check", "evidence": "prose"}]}}), encoding="utf-8")
            before = source.read_bytes()
            self.assertEqual(migrate_file(source, output, archive), (1, 1))
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(json.loads(archive.read_text())["text"], "prose")
            with self.assertRaises(ValueError):
                migrate_file(source, output, archive)
            with self.assertRaises(ValueError):
                migrate_file(source, source, Path(folder) / "other.jsonl")

    def test_bad_later_entry_does_not_publish_partial_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output, archive = [Path(folder) / name for name in ("source.jsonl", "new.jsonl", "archive.jsonl")]
            source.write_text('{"id":"valid"}\n{"id":"invalid","goal":{"constraints":[5]}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                migrate_file(source, output, archive)
            self.assertFalse(output.exists())
            self.assertFalse(archive.exists())

    def test_flatbuffer_migration_and_recompile(self):
        import subprocess
        repo = Path(__file__).resolve().parent.parent
        flatc = default_flatc(repo)
        if not flatc.is_file():
            self.skipTest("Existing flatc unavailable")
        schema = repo / "DataCollection/concept_block.fbs"
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            legacy = {"schema_version": 2, "blocks": [{"id": "legacy", "prompt": "Task", "goal": {
                "success_criteria": [{"criterion": "Check", "legacy_evidence": "Archived proof"}],
                "legacy_constraints": ["Stay safe"]}}]}
            (folder / "old.json").write_text(json.dumps(legacy), encoding="utf-8")
            subprocess.run([str(flatc), "--binary", "-o", str(folder), str(schema), str(folder / "old.json")], check=True)
            output, archive = folder / "new.jsonl", folder / "archive.jsonl"
            self.assertEqual(migrate_file(folder / "old.fb", output, archive), (1, 1))
            self.assertEqual(json.loads(archive.read_text())["text"], "Archived proof")
            build_flatbuffer(output, folder / "new.fb", schema, flatc)
            roundtrip, unused = folder / "again.jsonl", folder / "again.archive.jsonl"
            self.assertEqual(migrate_file(folder / "new.fb", roundtrip, unused), (1, 0))
            self.assertEqual(json.loads(output.read_text())["goal"], json.loads(roundtrip.read_text())["goal"])

    def test_structured_compiler_roundtrip(self):
        repo = Path(__file__).resolve().parent.parent
        flatc = default_flatc(repo)
        if not flatc.is_file():
            self.skipTest("Existing flatc unavailable")
        goal = {"target_state": "Retrieve key", "success_criteria": [
            {"criterion": "Carry key", "evidence": {"predicate": "carrying", "subject": "key_1"}},
            {"criterion": "Review"}], "constraints": [
            {"constraint": "Keep apple out of box", "evidence": {
                "predicate": "in", "subject": "apple_1", "target": "box_1"}}]}
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "input.jsonl"
            source.write_text(json.dumps({"id": "structured", "prompt": "Task", "goal": goal}), encoding="utf-8")
            build_flatbuffer(source, folder / "input.fb", repo / "DataCollection/concept_block.fbs", flatc)
            migrate_file(folder / "input.fb", folder / "output.jsonl", folder / "archive.jsonl")
            self.assertEqual(json.loads((folder / "output.jsonl").read_text())["goal"], goal)
            self.assertEqual((folder / "archive.jsonl").read_text(), "")


if __name__ == "__main__":
    unittest.main()

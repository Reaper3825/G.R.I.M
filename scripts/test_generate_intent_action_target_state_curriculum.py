#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_intent_action_curriculum as source
import generate_intent_action_target_state_curriculum as curriculum
from intent_action_curriculum_content import BUILDERS


class IntentActionTargetStateCurriculumTests(unittest.TestCase):
    def test_copy_moves_answer_into_target_state(self) -> None:
        for index in range(len(BUILDERS) * 20):
            original = source.make_entry(index)
            copied = curriculum.make_entry(index)
            self.assertEqual(copied["prompt"], original["prompt"])
            self.assertEqual(copied["goal"], {"target_state": original["answer"]})
            self.assertEqual(copied["answer"], "")
            self.assertTrue(copied["id"].startswith(curriculum.ENTRY_PREFIX))

    def test_copy_has_unique_ids_and_prompts(self) -> None:
        entries = curriculum.generate_entries(len(BUILDERS) * 20)
        self.assertEqual(len({entry["id"] for entry in entries}), len(entries))
        self.assertEqual(len({entry["prompt"] for entry in entries}), len(entries))

    def test_owned_copy_rows_do_not_replace_source_rows(self) -> None:
        replacement = curriculum.generate_entries(10)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "concept_blocks.jsonl"
            original = [
                source.make_entry(0),
                curriculum.make_entry(99),
                {"id": "other_1", "prompt": "Keep me.", "answer": "Kept."},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in original),
                encoding="utf-8",
            )

            removed, kept = source.replace_owned_entries(
                path,
                replacement,
                dry_run=False,
                entry_prefix=curriculum.ENTRY_PREFIX,
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((removed, kept), (1, 2))
            self.assertEqual(rows[0]["id"], source.make_entry(0)["id"])
            self.assertEqual(rows[1]["id"], "other_1")
            self.assertEqual(
                [row["id"] for row in rows[2:]],
                [entry["id"] for entry in replacement],
            )

    def test_registry_adds_distinct_derived_curriculum(self) -> None:
        entries = curriculum.generate_entries(12)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "curriculum_registry.json"
            path.write_text(json.dumps({"curriculums": [{
                "id": source.CURRICULUM_ID,
                "name": source.CURRICULUM_NAME,
                "concept_block_ids": ["iav1_00000"],
            }]}), encoding="utf-8")
            source.update_registry(
                path,
                entries,
                dry_run=False,
                curriculum_id=curriculum.CURRICULUM_ID,
                curriculum_name=curriculum.CURRICULUM_NAME,
            )
            registry = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(registry["curriculums"]), 2)
            derived = next(
                item for item in registry["curriculums"]
                if item["id"] == curriculum.CURRICULUM_ID)
            self.assertEqual(derived["name"], curriculum.CURRICULUM_NAME)
            self.assertEqual(
                derived["concept_block_ids"],
                [entry["id"] for entry in entries],
            )


if __name__ == "__main__":
    unittest.main()

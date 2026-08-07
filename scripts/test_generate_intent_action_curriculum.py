#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_intent_action_curriculum as curriculum
from intent_action_curriculum_content import BUILDERS


class IntentActionCurriculumTests(unittest.TestCase):
    def test_generation_covers_every_family_with_unique_prompts(self) -> None:
        entries = curriculum.generate_entries(len(BUILDERS) * 20)
        prompts = {entry["prompt"] for entry in entries}
        self.assertEqual(len(prompts), len(entries))

    def test_rows_only_contain_target_state_training_fields(self) -> None:
        entries = curriculum.generate_entries(len(BUILDERS) * 20)
        for entry in entries:
            self.assertEqual(set(entry), {"id", "prompt", "answer", "goal"})
            self.assertEqual(entry["goal"], {"target_state": ""})
            self.assertFalse(any(character.isdigit() for character in entry["answer"]))

    def test_answers_vary_in_length(self) -> None:
        entries = curriculum.generate_entries(len(BUILDERS) * 20)
        lengths = {len(entry["answer"].split()) for entry in entries}
        self.assertTrue(any(length <= 3 for length in lengths))
        self.assertTrue(any(4 <= length <= 7 for length in lengths))
        self.assertTrue(any(length >= 8 for length in lengths))

    def test_owned_rows_are_replaced_and_unrelated_rows_are_preserved(self) -> None:
        replacement = curriculum.generate_entries(10)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "concept_blocks.jsonl"
            original = [
                {"id": "other_1", "prompt": "Keep me.", "answer": "Kept."},
                {"id": "iav1_legacy", "prompt": "Old.", "answer": "Old."},
                {"id": "other_2", "prompt": "Keep me too.", "answer": "Kept too."},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in original),
                encoding="utf-8",
            )

            removed, kept = curriculum.replace_owned_entries(path, replacement, dry_run=False)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((removed, kept), (1, 2))
            self.assertEqual([row["id"] for row in rows[:2]], ["other_1", "other_2"])
            self.assertEqual([row["id"] for row in rows[2:]], [entry["id"] for entry in replacement])

    def test_registry_keeps_one_authoritative_curriculum(self) -> None:
        entries = curriculum.generate_entries(12)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "curriculum_registry.json"
            path.write_text(json.dumps({"curriculums": [{
                "id": curriculum.CURRICULUM_ID,
                "name": curriculum.CURRICULUM_NAME,
                "concept_block_ids": ["iav1_old"],
            }]}), encoding="utf-8")
            curriculum.update_registry(path, entries, dry_run=False)
            registry = json.loads(path.read_text(encoding="utf-8"))
            matches = [item for item in registry["curriculums"] if item["id"] == curriculum.CURRICULUM_ID]
            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0]["concept_block_ids"], [entry["id"] for entry in entries])


if __name__ == "__main__":
    unittest.main()

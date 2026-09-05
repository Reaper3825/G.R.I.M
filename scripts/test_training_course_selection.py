"""Focused checks for the training diagnostic's course membership resolution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "resources/models/GRIM-text/training/rank_tokenizer_sequence_lengths.py"
spec = importlib.util.spec_from_file_location("rank_course_test", MODULE_PATH)
rank = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = rank
spec.loader.exec_module(rank)


class CourseSelectionTests(unittest.TestCase):
    def setUp(self):
        self.curriculum = {"name": "Example", "course_ids": ["b", "a"], "concept_block_ids": ["stale"]}
        self.registry = {"curriculums": [self.curriculum], "courses": [
            {"id": "a", "concept_block_ids": ["one", "shared"]},
            {"id": "b", "concept_block_ids": ["two", "shared"]},
            {"id": "unused", "concept_block_ids": ["exclude"]}]}

    def test_uses_only_assigned_courses_and_deduplicates(self):
        self.assertEqual(rank.course_concept_ids(self.registry, self.curriculum), {"one", "two", "shared"})

    def test_no_flat_compatibility_list_required(self):
        del self.curriculum["concept_block_ids"]
        self.assertEqual(rank.course_concept_ids(self.registry, self.curriculum), {"one", "two", "shared"})

    def test_invalid_references_and_membership_rejected(self):
        for field, value in [("course_ids", None), ("course_ids", ["missing"]),
                             ("course_ids", ["a", "a"]), ("course_ids", [3]),
                             ("course_ids", [])]:
            with self.subTest(value=value):
                curriculum = dict(self.curriculum, **{field: value})
                with self.assertRaises(RuntimeError):
                    rank.course_concept_ids(self.registry, curriculum)
        for ids in [None, "one", [""], [2]]:
            with self.subTest(ids=ids):
                registry = copy.deepcopy(self.registry)
                registry["courses"][0]["concept_block_ids"] = ids
                with self.assertRaises(RuntimeError):
                    rank.course_concept_ids(registry, self.curriculum)

    def test_duplicate_course_definition_rejected(self):
        self.registry["courses"].append(self.registry["courses"][0])
        with self.assertRaises(RuntimeError):
            rank.course_concept_ids(self.registry, self.curriculum)

    def test_empty_course_allowed_with_nonempty_sibling(self):
        self.registry["courses"][0]["concept_block_ids"] = []
        self.assertEqual(rank.course_concept_ids(self.registry, self.curriculum), {"two", "shared"})

    def test_file_selection_preserves_formatting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            for concept_mode in (True, False):
                self.curriculum["format_as_concept"] = concept_mode
                path.write_text(json.dumps(self.registry), encoding="utf-8")
                selection = rank.load_curriculum_selection(path, "Example")
                self.assertEqual(selection.selected_ids, {"one", "two", "shared"})
                self.assertEqual(bool(selection.concept_ids), concept_mode)
                self.assertEqual(bool(selection.plaintext_ids), not concept_mode)


if __name__ == "__main__":
    unittest.main()

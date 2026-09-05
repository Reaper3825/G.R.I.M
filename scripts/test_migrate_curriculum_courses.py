import copy
import unittest
from migrate_curriculum_courses import migrate_registry


class CourseMigrationTests(unittest.TestCase):
    def setUp(self):
        self.source = {"curriculums": [
            {"id": "a", "name": "Alpha", "concept_block_ids": ["2", "1"], "training_stage": "sft"},
            {"id": "b", "name": "Beta", "concept_block_ids": ["1"], "plaintext_block_ids": ["p"]},
            {"id": "c", "name": "Empty", "concept_block_ids": []}]}

    def test_preserves_membership_order_metadata_and_source(self):
        before = copy.deepcopy(self.source)
        result = migrate_registry(self.source)
        self.assertEqual(self.source, before)
        self.assertEqual(len(result["courses"]), 3)
        for original, migrated, course in zip(before["curriculums"], result["curriculums"], result["courses"]):
            self.assertEqual(migrated.pop("course_ids"), [course["id"]])
            self.assertEqual(original, migrated)
            self.assertEqual(course["concept_block_ids"], original["concept_block_ids"])

    def test_idempotent(self):
        result = migrate_registry(self.source)
        self.assertEqual(migrate_registry(result), result)

    def test_collision_rejected(self):
        self.source["courses"] = [{"id": "course_generic_a"}]
        with self.assertRaises(ValueError):
            migrate_registry(self.source)

    def test_missing_course_rejected(self):
        self.source["curriculums"][0]["course_ids"] = ["missing"]
        with self.assertRaises(KeyError):
            migrate_registry(self.source)

    def test_stale_projection_rejected(self):
        result = migrate_registry(self.source)
        result["courses"][0]["concept_block_ids"] = []
        with self.assertRaises(ValueError):
            migrate_registry(result)

    def test_shared_courses_flatten_in_order(self):
        result = migrate_registry(self.source)
        result["curriculums"][0]["course_ids"].append("course_generic_b")
        self.assertEqual(migrate_registry(result), result)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_execution_policy_curriculum as policy


class ExecutionPolicyCurriculumTests(unittest.TestCase):
    def test_counterfactual_generation_is_balanced_and_valid(self) -> None:
        pairs = policy.generate_pairs(120, 12345)
        report = policy.validate_pairs(pairs)
        self.assertEqual(report["pairs"], 120)
        self.assertEqual(report["entries"], 240)
        self.assertEqual(len(report["families"]), len(policy.SCENARIO_BUILDERS))
        self.assertLessEqual(
            report["top_lexical_gaps"][0][1], policy.MAX_LEXICAL_LABEL_GAP)
        self.assertGreaterEqual(
            len(report["first_token_distribution"]), len(policy.PROMPT_LAYOUTS) - 2)
        self.assertLessEqual(
            max(report["first_token_distribution"].values()) / report["pairs"], 0.25)
        self.assertGreaterEqual(report["request_first_fraction"], 0.25)
        self.assertLessEqual(report["request_first_fraction"], 0.75)
        self.assertLess(report["mean_common_prefix_ratio"], 0.65)
        for counts in report["families"].values():
            self.assertEqual(counts["execute"], counts["noop"])

    def test_pairs_have_matched_numeric_context_and_distinct_questions(self) -> None:
        for pair in policy.generate_pairs(240, 777):
            self.assertNotEqual(pair.execute["prompt"], pair.noop["prompt"])
            self.assertEqual(
                policy.NUMBER_RE.findall(pair.execute["prompt"]),
                policy.NUMBER_RE.findall(pair.noop["prompt"]),
            )
            self.assertIn("execution", pair.execute)
            self.assertNotIn("execution", pair.noop)
            self.assertEqual(pair.execute["format_type"], pair.noop["format_type"])

    def test_single_curriculum_registry_retires_v1(self) -> None:
        pairs = policy.generate_pairs(24, 101)
        entries = policy.flattened_entries(pairs, 101)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry_path = root / "curriculum_registry.json"
            registry_path.write_text(json.dumps({
                "curriculums": [
                    {"id": legacy, "name": legacy, "concept_block_ids": []}
                    for legacy in sorted(policy.LEGACY_CURRICULUM_IDS)
                ]
            }), encoding="utf-8")
            policy.update_registry(registry_path, entries, dry_run=False, retire_v1=True)
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            ids = {item["id"] for item in registry["curriculums"]}
            self.assertEqual(ids, {policy.CURRICULUM_ID})
            curriculum = registry["curriculums"][0]
            self.assertEqual(len(curriculum["concept_block_ids"]), 48)

    def test_append_is_idempotent(self) -> None:
        pairs = policy.generate_pairs(12, 202)
        entries = policy.flattened_entries(pairs, 202)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "concept_blocks.jsonl"
            self.assertEqual(policy.append_missing_entries(path, entries, dry_run=False), 24)
            self.assertEqual(policy.append_missing_entries(path, entries, dry_run=False), 0)
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 24)


if __name__ == "__main__":
    unittest.main()

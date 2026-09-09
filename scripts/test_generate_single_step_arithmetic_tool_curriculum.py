import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name(
    "generate_single_step_arithmetic_tool_curriculum.py")
SPEC = importlib.util.spec_from_file_location("single_step_arithmetic_tool", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class SingleStepArithmeticToolCurriculumTests(unittest.TestCase):
    def test_every_stratum_has_the_expected_contract(self):
        seen_ids = set()
        seen_prompts = set()
        entries = []
        for index in range(48):
            entry, metadata = MODULE.make_entry(index)
            MODULE.validate_entry(entry, metadata, seen_ids, seen_prompts)
            entries.append((entry, metadata))

        self.assertEqual({metadata["mode"] for _, metadata in entries}, set(MODULE.MODES))
        self.assertEqual(
            {metadata["operation"] for _, metadata in entries}, set(MODULE.OPERATIONS))
        self.assertEqual(
            {metadata["category"] for _, metadata in entries}, set(MODULE.CATEGORIES))
        self.assertEqual(sum(metadata["parenthesized"] for _, metadata in entries), 24)
        for entry, _ in entries:
            self.assertEqual(entry["update"], "")
            self.assertEqual(entry["execute"].count("<TOOL>"), 1)
            self.assertNotIn("=", entry["execute"])

    def test_tool_payload_is_grounded_while_determine_is_symbolic(self):
        entry, metadata = MODULE.make_entry(0)
        self.assertIn("<TOOL>(", entry["execute"])
        self.assertIn(MODULE.SYMBOLS[metadata["operation"]], entry["execute"])
        self.assertIn("=", entry["determine"])
        self.assertIn(entry["unknowns"][0], entry["determine"])
        self.assertNotIn(entry["unknowns"][0], entry["execute"])

    def test_count_values_are_integral(self):
        for index in range(60_000):
            _, metadata = MODULE.make_entry(index)
            if metadata["category"] == "count":
                self.assertEqual(metadata["lhs"], metadata["lhs"].to_integral())
                self.assertEqual(metadata["rhs"], metadata["rhs"].to_integral())
                self.assertEqual(metadata["result"], metadata["result"].to_integral())


if __name__ == "__main__":
    unittest.main()

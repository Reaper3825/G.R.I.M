import importlib.util
import collections
import json
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
            self.assertEqual(entry["execute"].count(" -> "), 1)

    def test_tool_payload_and_answer_use_variable_references(self):
        entry, metadata = MODULE.make_entry(0)
        self.assertIn("<TOOL>(", entry["execute"])
        self.assertIn(MODULE.SYMBOLS[metadata["operation"]], entry["execute"])
        self.assertNotIn("=", entry["determine"])
        self.assertEqual(entry["knowns"], [])
        self.assertEqual(entry["unknowns"], [])
        result_reference = MODULE.variable_reference(metadata["names"][2])
        self.assertTrue(entry["execute"].endswith(f" -> {result_reference}"))
        self.assertIn(result_reference, entry["answer"])
        self.assertNotIn(MODULE.format_decimal(metadata["result"]), entry["answer"])

        payload = entry["execute"].split("<TOOL>", 1)[1].split("</TOOL>", 1)[0]
        self.assertNotIn(MODULE.format_decimal(metadata["lhs"]), payload)
        self.assertNotIn(MODULE.format_decimal(metadata["rhs"]), payload)

    def test_count_values_are_integral(self):
        operations = collections.Counter()
        vocab = json.loads(MODULE.MANUAL_VOCAB_PATH.read_text(encoding="utf-8"))["tokens"]
        for index in range(60_000):
            entry, metadata = MODULE.make_entry(index)
            operations[metadata["operation"]] += 1
            names = metadata["names"]
            self.assertEqual(entry["define"].splitlines(), [
                "${" + names[0] + "} -> " + MODULE.format_decimal(metadata["lhs"]) + ";",
                "${" + names[1] + "} -> " + MODULE.format_decimal(metadata["rhs"]) + ";",
                "${" + names[2] + "};",
            ])
            for name in names:
                self.assertIn(name.split("_", 1)[0] + "_", vocab)
            if metadata["category"] == "count":
                self.assertEqual(metadata["lhs"], metadata["lhs"].to_integral())
                self.assertEqual(metadata["rhs"], metadata["rhs"].to_integral())
                self.assertEqual(metadata["result"], metadata["result"].to_integral())
        self.assertEqual(dict(operations), dict.fromkeys(MODULE.OPERATIONS, 15_000))

    def test_rejects_legacy_state_and_malformed_definitions(self):
        for field, value in (("knowns", ["start_liters = 120 liters"]),
                             ("unknowns", ["remaining_liters"]),
                             ("define", "${start_liters} -> 120")):
            with self.subTest(field=field):
                entry, metadata = MODULE.make_entry(0)
                entry[field] = value
                with self.assertRaises(ValueError):
                    MODULE.validate_entry(entry, metadata, set(), set())


if __name__ == "__main__":
    unittest.main()

import collections
from decimal import Decimal
import json
from pathlib import Path
import re
import tempfile
import unittest

import generate_single_step_arithmetic_tool_curriculum as curriculum
from single_step_arithmetic_scenarios import STORIES, make_case, render_case


class SingleStepArithmeticToolCurriculumTests(unittest.TestCase):
    def test_full_dataset_contract_and_coverage(self):
        ids, prompts = set(), set()
        operations, roles, modes = (collections.Counter() for _ in range(3))
        categories, words = collections.Counter(), collections.defaultdict(set)
        case_targets = collections.defaultdict(set)
        syntax = collections.Counter()
        vocab = json.loads(curriculum.MANUAL_VOCAB_PATH.read_text(encoding="utf-8"))["tokens"]
        for index in range(60_000):
            row, meta = curriculum.make_entry(index)
            curriculum.validate_entry(row, meta, ids, prompts)
            modes[meta["mode"]] += 1
            operations[meta["operation"]] += 1
            categories[meta["category"]] += 1
            syntax[meta["parenthesized"]] += 1
            if meta["mode"] == "contextual":
                roles[meta["relationship"], meta["target_role"]] += 1
                case_targets[meta["case_id"]].add(meta["target_role"])
                words[meta["relationship"], meta["target_role"]].add(meta["wording"])
            for name in meta["names"]:
                self.assertEqual(len(name.split("_")), 2)
                for piece in name.split("_"):
                    self.assertIn(piece + "_", vocab)
            # Independently confirm exactly the two supplied operands occur in
            # the prompt, despite question/fact ordering and unknown position.
            numbers = [Decimal(x) for x in re.findall(r"\d+(?:\.\d+)?", row["prompt"])]
            self.assertCountEqual(numbers, [meta["lhs"], meta["rhs"]])
            self.assertEqual(row["goal"]["success_criteria"][0]["evidence"], "")
            self.assertEqual(row["update"], "")
            if meta["category"] == "count":
                for value in (meta["lhs"], meta["rhs"], meta["result"]):
                    self.assertEqual(value, value.to_integral())
            if meta["target_role"] == "group_count":
                self.assertEqual(meta["result"], meta["result"].to_integral())
        self.assertEqual(modes, {"direct": 12_000, "contextual": 48_000})
        self.assertEqual(operations, {"add": 15_000, "sub": 27_000, "mul": 7_000, "div": 11_000})
        self.assertEqual(len(roles), 12)
        self.assertEqual(set(roles.values()), {4_000})
        self.assertEqual(set(map(len, case_targets.values())), {3})
        self.assertEqual(set(map(len, words.values())), {4})
        self.assertLessEqual(max(categories.values()) - min(categories.values()), 3)
        self.assertEqual(syntax, {True: 30_000, False: 30_000})

    def test_tank_regression_all_three_questions(self):
        case = make_case("decrease", 0, 7913, category="metric", story=STORIES["metric"][-1],
                         values=tuple(map(Decimal, (120, 36, 84))))
        expected = [
            ("add", 36, 84, 120, "start_volume", "starting water volume"),
            ("sub", 120, 84, 36, "change_volume", "water volume used"),
            ("sub", 120, 36, 84, "remainder_volume", "remaining water volume"),
        ]
        for target, (op, left, right, result, name, role) in enumerate(expected):
            row = render_case(case, target)
            self.assertEqual((row["operation"], row["lhs"], row["rhs"], row["result"]), (op, left, right, result))
            self.assertEqual(row["names"][2], name)
            self.assertIn(role, row["determine"])
            self.assertIn(role, row["answer"])
        used = render_case(case, 1)
        self.assertEqual(used["names"][:2], ("start_volume", "remainder_volume"))
        self.assertIn("120", used["prompt"])
        self.assertIn("84", used["prompt"])
        self.assertNotIn("36", used["prompt"])

    def test_inverse_relationships_and_dimension_of_group_count(self):
        expectations = {
            "increase": ((23, 19, 42), (("sub", 42, 19), ("sub", 42, 23), ("add", 23, 19))),
            "comparison": ((28, 15, 43), (("sub", 43, 15), ("sub", 43, 28), ("add", 28, 15))),
            "groups": ((6, Decimal("1.5"), 9), (("div", 9, Decimal("1.5")), ("div", 9, 6), ("mul", 6, Decimal("1.5")))),
        }
        for rel, (values, solutions) in expectations.items():
            case = make_case(rel, 0, 7913, category="metric", story=STORIES["metric"][-1],
                             values=tuple(map(Decimal, values)))
            for target, expected in enumerate(solutions):
                row = render_case(case, target)
                self.assertEqual((row["operation"], row["lhs"], row["rhs"]), expected)
                self.assertEqual(row["result"], values[target])
            if rel == "groups":
                row = render_case(case, 0)
                self.assertEqual(row["unit"], "containers")
                self.assertNotIn("liters", row["answer"])
                self.assertEqual(row["names"][2], "group_count")

    def test_rejects_wrong_roles_state_and_unapproved_name_parts(self):
        for field, value in (("knowns", ["start_volume = 120"]), ("unknowns", ["change_volume"]),
                             ("define", "${start_volume} -> 120"),
                             ("answer", "There are ${remainder_volume} liters remaining."),
                             ("determine", "Subtract the quantity used to find the remainder.")):
            with self.subTest(field=field):
                row, meta = curriculum.make_entry(2)
                row[field] = value
                with self.assertRaises(ValueError):
                    curriculum.validate_entry(row, meta, set(), set())
        for names in (("invented_volume", "change_volume", "total_volume"),
                      ("start_invented", "change_volume", "total_volume"),
                      ("start_volume_extra", "change_volume", "total_volume"),
                      ("start_volume", "start_volume", "total_volume")):
            with self.assertRaises(ValueError):
                curriculum.validate_variable_names(names)

    def test_seed_is_deterministic_and_changes_wording_or_values(self):
        self.assertEqual(curriculum.make_entry(402, 19), curriculum.make_entry(402, 19))
        self.assertNotEqual(curriculum.make_entry(402, 19), curriculum.make_entry(402, 20))

    def test_merge_preserves_unrelated_rows_and_handles_json_whitespace(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "target"
            source.mkdir()
            target.mkdir()
            new, _ = curriculum.make_entry(0)
            (source / "concept_blocks.jsonl").write_text(json.dumps(new) + "\n", encoding="utf-8")
            (source / "curriculum_registry.json").write_text(json.dumps(curriculum.generated_registry([new["id"]])), encoding="utf-8")
            unrelated = '{"id": "keep", "prompt": "unchanged"}\n'
            (target / "concept_blocks.jsonl").write_text(unrelated + json.dumps({"id": "ssatv1_000000", "prompt": "old"}) + "\n", encoding="utf-8")
            registry = curriculum.generated_registry(["ssatv1_000000"])
            registry["courses"].append({"id": "other_course", "concept_block_ids": ["keep"]})
            (target / "curriculum_registry.json").write_text(json.dumps(registry), encoding="utf-8")
            self.assertEqual(curriculum.merge_into_existing_dataset(source, target), (1, 1))
            lines = (target / "concept_blocks.jsonl").read_text().splitlines(keepends=True)
            self.assertEqual(lines[0], unrelated)
            self.assertEqual(json.loads(lines[1]), new)
            saved = json.loads((target / "curriculum_registry.json").read_text())
            self.assertEqual(saved["courses"][-1], registry["courses"][-1])


if __name__ == "__main__":
    unittest.main()

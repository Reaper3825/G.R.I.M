#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_generic_success_criteria_curriculum as curriculum
import generate_intent_action_curriculum as source
import generate_intent_action_target_state_curriculum as target_source
from intent_action_curriculum_content import BUILDERS, generate_content
from success_criterion_curriculum_content import CONTRACTS, criterion_for, evidence_for


class GenericSuccessCriteriaCurriculumTests(unittest.TestCase):
    def test_every_intent_family_has_a_generic_contract(self) -> None:
        families = {
            generate_content(index, source.DEFAULT_SEED).family
            for index in range(len(BUILDERS))
        }
        self.assertEqual(set(CONTRACTS), families)

    def test_rows_preserve_prompt_and_target_state(self) -> None:
        for index in range(len(BUILDERS) * 12):
            target = target_source.make_entry(index)
            rows = curriculum.make_entries(index)
            self.assertEqual(len(rows), 3)
            for row in rows:
                self.assertEqual(row["prompt"], target["prompt"])
                self.assertEqual(
                    row["goal"]["target_state"],
                    target["goal"]["target_state"],
                )

    def test_answers_are_plain_content_without_protocol_markers(self) -> None:
        rows = curriculum.generate_entries(len(BUILDERS) * 12)
        for row in rows:
            answer = row["answer"]
            suffix = row["id"].rsplit("_", 1)[-1]
            if suffix == "criterion":
                self.assertEqual(len(answer.splitlines()), 1)
            elif suffix == "evidence":
                self.assertEqual(len(answer.splitlines()), 1)
            else:
                self.assertEqual(len(answer.splitlines()), 2)
            self.assertNotIn("{", answer)
            self.assertNotIn("}", answer)
            self.assertNotIn('"criterion"', answer)
            self.assertNotIn('"evidence"', answer)
            self.assertNotIn('"complete"', answer)

    def test_evidence_rows_pin_criterion_without_evidence(self) -> None:
        for index in range(len(BUILDERS) * 12):
            evidence_row = curriculum.make_entries(index)[1]
            prior = evidence_row["goal"]["success_criteria"]
            self.assertEqual(len(prior), 1)
            self.assertEqual(set(prior[0]), {"criterion"})
            self.assertEqual(
                evidence_row["answer"],
                evidence_for(generate_content(index, source.DEFAULT_SEED).family,
                             index // len(BUILDERS)),
            )

    def test_contract_text_never_interpolates_prompt_or_target_state(self) -> None:
        for index in range(len(BUILDERS) * 40):
            content = generate_content(index, source.DEFAULT_SEED)
            target = target_source.make_entry(index)
            occurrence = index // len(BUILDERS)
            criterion = criterion_for(content.family, occurrence)
            evidence = evidence_for(content.family, occurrence)
            self.assertNotIn(target["prompt"], criterion)
            self.assertNotIn(target["prompt"], evidence)
            self.assertNotIn(target["goal"]["target_state"], criterion)
            self.assertNotIn(target["goal"]["target_state"], evidence)


if __name__ == "__main__":
    unittest.main()

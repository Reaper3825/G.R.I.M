"""Source-level prefix checks; does not build or run training/CUDA targets."""
import json
from pathlib import Path
import re
import unittest

import generate_single_step_arithmetic_tool_curriculum as curriculum


SOURCE = (Path(__file__).resolve().parents[1] /
          "resources/models/GRIM-text/training/Diagnostics/DiagnosticInference.cu")


class DiagnosticArithmeticContractTests(unittest.TestCase):
    def test_goal_matches_training_prefix_without_persisted_bindings(self):
        source = SOURCE.read_text(encoding="utf-8")
        initializer = re.search(
            r"reasoning_state\.goal = GRIM::ConceptBlockGoal\{(.*?)\n    \};",
            source, re.S).group(1)
        strings = [json.loads(s) for s in re.findall(r'"(?:[^"\\]|\\.)*"', initializer)]
        goal = curriculum.make_entry(2)[0]["goal"]
        self.assertEqual(strings, [goal["target_state"],
                                  goal["success_criteria"][0]["criterion"],
                                  goal["success_criteria"][0]["evidence"],
                                  *goal["constraints"]])
        self.assertNotRegex(source, r"reasoning_state\.(?:knowns|unknowns)\s*=")
        self.assertIn("reasoning_state.withPrompt(prompt)", source)
        self.assertIn("renderReasoningPrompt(inference_state, concept_spans)", source)

    def test_probe_rotation_covers_every_role_and_preserves_override(self):
        source = SOURCE.read_text(encoding="utf-8")
        cases = re.findall(r'\{"([a-z_]+)", "([^"\n]+)"\}', source)
        expected = {f"{relationship}_{role}"
                    for relationship, roles in curriculum.TARGETS.items() for role in roles}
        self.assertEqual(len(cases), 12)
        self.assertEqual({name for name, _ in cases}, expected)
        self.assertEqual(cases[0][0], "decrease_removed")
        # Probe values are independent of the generator's numerical sampling.
        given = [(120, 84), (36, 84), (120, 36), (19, 42), (23, 42), (23, 19),
                 (18, 50), (50, 32), (32, 18), (72, 9), (72, 8), (8, 9)]
        for (_, prompt), numbers in zip(cases, given):
            self.assertEqual(tuple(map(int, re.findall(r"\d+", prompt))), numbers)
        self.assertIn('readEnvString("GRIM_SAMPLE_PROMPT", diagnostic.prompt)', source)
        self.assertIn("optimizer_step / inference_diagnostic_interval - 1", source)
        self.assertIn("% kArithmeticPrompts.size()", source)


if __name__ == "__main__":
    unittest.main()

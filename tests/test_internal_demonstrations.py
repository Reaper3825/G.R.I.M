"""Replay, privacy, feedback ordering, and fixture checks in the existing image."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/textworld"))
from generate_internal_demonstrations import (clean_observation, generate, project_observation,
                                             replay, resolve_fixture, review_markdown, visible_goal)
from goal_evaluator import identity
from world_worker import Session, WorldError

OUTPUT = ROOT / "DataCollection/curated/grim_internal_state_v1/demonstrations"


class DemonstrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((OUTPUT / "host_fixture.json").read_text())
        cls.examples, cls.private = generate(ROOT, cls.fixture)

    def test_saved_examples_match_real_replay(self):
        for filename, expected in (("examples.jsonl", self.examples),
                                   ("verification.private.jsonl", self.private)):
            actual = [json.loads(line) for line in (OUTPUT / filename).read_text().splitlines()]
            self.assertEqual(actual, expected)
        self.assertEqual((OUTPUT / "examples.md").read_text(), review_markdown(self.examples))
        self.assertEqual(len(self.examples), 9)
        self.assertEqual(sum(event["span"] == "execute" for row in self.examples for event in row["events"]), 21)

    def test_feedback_order_masking_and_authoritative_outcome(self):
        for example, audit in zip(self.examples, self.private):
            self.assertEqual(identity(example), audit["example_identity"])
            self.assertEqual(audit["verification"]["status"], "succeeded")
            self.assertEqual(audit["verification"]["coverage"], "complete")
            events = example["events"]
            self.assertEqual(events[0]["span"], "determine")
            self.assertEqual(events[-1]["span"], "answer")
            self.assertEqual(example["input"]["reasoning_state"]["environment_context"],
                             project_observation(audit["receipts"][0]["resolved_state"]))
            receipt_index = 1
            for i in range(1, len(events) - 1, 2):
                command, feedback = events[i:i + 2]
                receipt = audit["receipts"][receipt_index]
                self.assertEqual(command["span"], "execute")
                self.assertEqual(command["content"], receipt["command"])
                self.assertTrue(command["supervised"])
                self.assertEqual(feedback["origin"], "environment")
                self.assertFalse(feedback["supervised"])
                self.assertEqual(feedback["content"], project_observation(receipt["resolved_state"]))
                receipt_index += 1
            self.assertEqual(receipt_index, len(audit["receipts"]))
            self.assertEqual([r["command"] for r in audit["receipts"][1:]], audit["witness"]["commands"])

    def test_private_fields_do_not_enter_model_context(self):
        forbidden_keys = {"evidence", "facts", "admissible_commands", "witness", "verification",
                          "room_id", "available_rooms", "episode", "engine", "observation_request"}
        def walk(value):
            if isinstance(value, dict):
                self.assertFalse(forbidden_keys.intersection(value))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)
        for row in self.examples:
            walk(row)
        state = {"description": "Visible room.", "location_path": "/tools",
                 "facts": "PRIVATE_SENTINEL", "admissible_commands": "PRIVATE_SENTINEL",
                 "episode": {"answer": "PRIVATE_SENTINEL"}}
        self.assertNotIn("PRIVATE_SENTINEL", project_observation(state))
        self.assertNotIn("evidence", json.dumps(visible_goal(self.private[0]["goal"])))

    def test_host_fixture_is_explicit_and_unknown_requests_fail(self):
        raw = {"feedback": "NATIVE_PLACEHOLDER", "location_path": "/tools",
               "observation_request": {"provider": "tool_catalog", "entity_id": "tool_catalog"}}
        with self.assertRaisesRegex(ValueError, "Resolve host observations"):
            project_observation(raw)
        resolved = resolve_fixture(raw, self.fixture)
        self.assertEqual(resolved["feedback"],
                         "Tool catalog:\nNo model tools are currently available.\n\nLOCATION: /tools")
        self.assertIn("observation_request", raw)  # Native receipt is immutable.
        self.assertNotIn("NATIVE_PLACEHOLDER", project_observation(resolved))
        raw["observation_request"]["provider"] = "tool_details"
        with self.assertRaisesRegex(ValueError, "Unsupported host observation"):
            resolve_fixture(raw, self.fixture)
        bad_fixture = dict(self.fixture, catalog_tools=["imaginary.tool"])
        with self.assertRaisesRegex(ValueError, "Only the explicit empty"):
            resolve_fixture({}, bad_fixture)

    def test_tampered_witness_cannot_become_a_demonstration(self):
        manifest = json.loads((OUTPUT.parent / "manifest.json").read_text())
        record = json.loads((ROOT / "resources/models/games/episodes" / manifest["episodes"][0]["record"]).read_text())
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory)
            try:
                session.handle({"op": "compile", "document": record["environment"],
                                "observation_providers": ["tool_catalog"]})
                bad = copy.deepcopy(record)
                bad["commands"] = []
                with self.assertRaisesRegex(WorldError, "do not verify Goal success"):
                    replay(bad, session, self.fixture)
                bad = copy.deepcopy(record)
                bad["witness"]["identity"]["goal"] = "stale"
                with self.assertRaisesRegex(WorldError, "Saved solution is stale"):
                    replay(bad, session, self.fixture)
            finally:
                session.close()

    def test_cleanup_preserves_tool_content_and_removes_terminal_chrome(self):
        self.assertEqual(clean_observation("\n-= Tools =- 0/0\nReal result.\n>\nLOCATION: /tools\n", "/tools"), "Real result.")
        self.assertEqual(clean_observation("ASCII diagram: /_\\\n\nUseful content."),
                         "ASCII diagram: /_\\\n\nUseful content.")


if __name__ == "__main__":
    unittest.main()

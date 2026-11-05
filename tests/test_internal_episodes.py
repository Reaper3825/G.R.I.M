"""Native generator checks; run in the existing grim-textworld:1.7.0 image."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/textworld"))
from generate_internal_episodes import DATASET, compact, game_document, generate, record_key
from world_worker import Session, WorldError


class InternalEpisodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads((ROOT / "resources/models/games/grim_internal.grimworld.json").read_text())
        cls.records, cls.blocks, cls.manifest = generate(cls.document)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.session = Session(self.directory.name)
        self.session.handle({"op": "compile", "document": game_document(self.document),
                             "observation_providers": ["tool_catalog"]})

    def tearDown(self):
        self.session.close()
        self.directory.cleanup()

    def start(self, record):
        selection = record["selection"]
        return self.session.handle({"op": "start_episode", "source_id": selection["source_id"],
                                    "goal": selection["goal"]})["state"]

    def test_complete_unique_noninitial_states_and_reproducible_files(self):
        self.assertEqual(self.manifest["reachable_states"], 10)
        self.assertEqual(len(self.records), 9)
        conditions = {compact(record["selection"]["goal"]["success_criteria"]) for _, record in self.records}
        self.assertEqual(len(conditions), 9)
        expected_lengths = {"closed": 1, "open": 4}
        for filename, record in self.records:
            room = record["selection"]["goal"]["success_criteria"][0]["evidence"]["subject"]
            state = record["selection"]["goal"]["success_criteria"][1]["evidence"]["predicate"]
            expected = expected_lengths[state]
            if state == "open":
                expected -= 2 if room == "tools" else 1 if room == "hub" else 0
            self.assertEqual(len(record["commands"]), expected)
            self.assertEqual(filename, record_key(record["witness"]["identity"]) + ".grimepisode.json")
            saved = json.loads((ROOT / "resources/models/games/episodes" / filename).read_text())
            self.assertEqual(saved, record)
        output = ROOT / "DataCollection/curated" / DATASET
        saved_blocks = [json.loads(line) for line in (output / "goals.jsonl").read_text().splitlines()]
        self.assertEqual(saved_blocks, self.blocks)
        self.assertEqual(json.loads((output / "manifest.json").read_text()), self.manifest)

    def test_every_proper_prefix_is_incomplete_and_full_witness_succeeds(self):
        for _, record in self.records:
            with self.subTest(source=record["selection"]["source_id"]):
                state = self.start(record)
                for command in record["commands"][:-1]:
                    state = self.session.handle({"op": "step", "command": command})["state"]
                self.assertEqual(state["episode"]["status"], "active")
                with self.assertRaisesRegex(WorldError, "do not verify Goal success"):
                    self.session.handle({"op": "verify_solution", "commands": record["commands"][:-1]})
                result = self.session.handle({"op": "verify_solution", "commands": record["commands"],
                                              "witness": record["witness"]})
                self.assertTrue(result["verified"])

    def test_forbidden_catalog_open_fails_and_stale_identity_is_rejected(self):
        for _, record in self.records:
            self.start(record)
            if record["selection"]["goal"]["constraints"]:
                state = self.session.handle({"op": "step", "command": "enter tools"})["state"]
                if state["episode"]["status"] == "succeeded":
                    with self.assertRaisesRegex(WorldError, "Episode is finished"):
                        self.session.handle({"op": "step", "command": "open tool catalog"})
                else:
                    state = self.session.handle({"op": "step", "command": "open tool catalog"})["state"]
                    self.assertEqual(state["episode"]["status"], "failed")
            self.start(record)
            witness = copy.deepcopy(record["witness"])
            witness["identity"]["starting_state"] = "stale"
            with self.assertRaisesRegex(WorldError, "Saved solution is stale"):
                self.session.handle({"op": "verify_solution", "commands": record["commands"], "witness": witness})

    def test_exhausted_search_does_not_return_a_partial_set(self):
        with self.assertRaisesRegex(ValueError, "State budget exhausted"):
            generate(self.document, max_states=1)


if __name__ == "__main__":
    unittest.main()

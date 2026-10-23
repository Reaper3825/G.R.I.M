"""Run inside the existing grim-textworld:1.7.0 image; no packages installed."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textworld"))
if Path("/worker").exists():
    sys.path.insert(0, "/worker")
from world_worker import Session, WorldError, build_world


def puzzle():
    return {"format": "grim-world", "version": 2, "spawn": "r1", "title": "Key puzzle",
            "objective": "Carry the brass key to the hall.",
            "entities": [
                {"id": "r1", "name": "Workshop", "kind": "room", "parent": ""},
                {"id": "r2", "name": "Hall", "kind": "room", "parent": ""},
                {"id": "c1", "name": "toolbox", "kind": "container", "parent": "r1", "state": "closed"},
                {"id": "k1", "name": "brass key", "kind": "key", "parent": "c1"},
                {"id": "f1", "name": "apple", "kind": "food", "parent": "r1"}],
            "passages": [{"id": "p1", "from": "r1", "to": "r2", "direction": "east",
                          "door_name": "oak door", "door_state": "locked", "key": "k1"}],
            "win_conditions": [{"predicate": "player_at", "subject": "r2"}, {"predicate": "carrying", "subject": "k1"}],
            "fail_conditions": [{"predicate": "eaten", "subject": "f1"}]}


SOLUTION = ["open toolbox", "take brass key from toolbox", "unlock oak door with brass key", "open oak door", "go east"]


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.session = Session(self.temp.name)

    def tearDown(self):
        self.session.close()
        self.temp.cleanup()

    def test_locked_puzzle_reset_recording_and_failure(self):
        initial = self.session.handle({"op": "compile", "document": puzzle()})["state"]
        self.assertEqual(initial["room_id"], "r1")
        self.assertFalse(initial["done"])
        invalid = self.session.handle({"op": "step", "command": "go east"})["state"]
        self.assertEqual(invalid["room_id"], "r1")
        for command in SOLUTION:
            state = self.session.handle({"op": "step", "command": command})["state"]
        self.assertTrue(state["done"] and state["won"])
        self.assertEqual(state["room_id"], "r2")
        verified = self.session.handle({"op": "verify_solution", "commands": SOLUTION})
        self.assertTrue(verified["verified"])
        self.assertTrue(self.session.done)  # Replay does not reset the interactive session.
        with self.assertRaises(WorldError):
            self.session.handle({"op": "verify_solution", "commands": ["look"]})
        reset = self.session.handle({"op": "reset"})["state"]
        self.assertEqual(reset["room_id"], "r1")
        self.assertFalse(reset["won"])
        self.session.handle({"op": "step", "command": "take apple"})
        state = self.session.handle({"op": "step", "command": "eat apple"})["state"]
        self.assertTrue(state["done"] and state["lost"])

    def test_failed_compile_preserves_session(self):
        self.session.handle({"op": "compile", "document": puzzle()})
        bad = puzzle()
        bad["entities"][2]["parent"] = "missing"
        with self.assertRaises(WorldError) as error:
            self.session.handle({"op": "compile", "document": bad})
        self.assertEqual(error.exception.entity_id, "c1")
        self.assertEqual(self.session.handle({"op": "step", "command": "look"})["state"]["room_id"], "r1")
        valid = self.session.handle({"op": "validate", "document": puzzle()})
        self.assertTrue(valid["validated"])
        self.assertIsNotNone(self.session.env)

    def test_adapter_rejects_invalid_worlds_and_inform_text(self):
        for mutate in (
            lambda d: d["entities"][0].update(name='Bad " room'),
            lambda d: d["entities"][2].update(parent="c1"),
            lambda d: d["passages"][0].update(key="f1"),
            lambda d: d["win_conditions"][0].update(subject="missing"),
            lambda d: d["entities"][1].update(name="Workshop"),
            lambda d: d["entities"][2].update(description="[some Inform substitution]"),
        ):
            world = puzzle()
            mutate(world)
            with self.assertRaises(WorldError):
                build_world(world)

    def test_win_all_fail_any_and_legacy_sandbox(self):
        world = puzzle()
        world["passages"][0]["door_state"] = "open"
        world["fail_conditions"].append({"predicate": "closed", "subject": "p1"})
        self.session.handle({"op": "compile", "document": world})
        state = self.session.handle({"op": "step", "command": "go east"})["state"]
        self.assertFalse(state["won"])  # Correct room alone does not satisfy ALL.
        self.session.handle({"op": "reset"})
        state = self.session.handle({"op": "step", "command": "close oak door"})["state"]
        self.assertTrue(state["lost"])  # Second failure condition alone is enough.
        world["version"] = 1
        world.pop("win_conditions")
        world.pop("fail_conditions")
        result = self.session.handle({"op": "compile", "document": world})
        self.assertTrue(result["warnings"])
        self.assertFalse(self.session.handle({"op": "step", "command": "look"})["state"]["done"])


if __name__ == "__main__":
    unittest.main()

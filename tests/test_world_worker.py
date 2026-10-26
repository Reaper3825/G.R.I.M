"""Run inside the existing grim-textworld:1.7.0 image; no packages installed."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "textworld"))
if Path("/worker").exists():
    sys.path.insert(0, "/worker")
from world_worker import Session, WorldError, build_world
from action_catalog import action_catalog
from goal_evaluator import GoalError


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


def goal(document):
    return {"target_state": document.get("objective", "Challenge"),
            "success_criteria": [{"criterion": "Win check", "evidence": c} for c in document.get("win_conditions", [])],
            "constraints": [{"constraint": "Do not violate", "evidence": c} for c in document.get("fail_conditions", [])]}


def compile_request(document):
    return {"op": "compile", "document": document, "source_id": "test-goal", "goal": goal(document)}


SOLUTION = ["open toolbox", "take brass key from toolbox", "unlock oak door with brass key", "open oak door", "go east"]


class WorkerTests(unittest.TestCase):
    def test_action_catalog_matches_native_snapshot(self):
        root = Path(__file__).resolve().parents[1]
        header = root / "ui/world_lab/textworld_action_catalog.generated.hpp"
        if Path("/catalog/textworld_action_catalog.generated.hpp").exists():
            header = Path("/catalog/textworld_action_catalog.generated.hpp")
        import json
        snapshot = json.loads(header.read_text().split('R"catalog(', 1)[1].split(')catalog"', 1)[0])
        self.assertEqual(action_catalog(), snapshot)
        catalog = {a["id"]: a for a in snapshot["actions"]}
        self.assertEqual(catalog["unlock/d"]["command"], "unlock {d} with {k}")
        self.assertEqual(catalog["unlock/d"]["adds"], ["closed(d)"])
        self.assertEqual(catalog["unlock/d"]["removes"], ["locked(d)"])
        self.assertIn("match(k, d)", catalog["unlock/d"]["requires"])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.session = Session(self.temp.name)

    def tearDown(self):
        self.session.close()
        self.temp.cleanup()

    def test_locked_puzzle_reset_recording_and_failure(self):
        compiled = self.session.handle(compile_request(puzzle()))
        self.assertEqual(compiled["action_catalog"], action_catalog())
        initial = compiled["state"]
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
        self.session.handle(compile_request(puzzle()))
        bad = puzzle()
        bad["entities"][2]["parent"] = "missing"
        with self.assertRaises(WorldError) as error:
            self.session.handle(compile_request(bad))
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
            with self.assertRaises((WorldError, GoalError)):
                self.session.handle(compile_request(world))

    def test_reuses_one_compilation_for_two_database_goals(self):
        from textworld.generator.game import Quest, QuestProgression
        compiled = self.session.handle(compile_request(puzzle()))
        native_a = self.session.episode.native
        self.assertIsInstance(native_a.quest, Quest)
        self.assertIsInstance(native_a.progression, QuestProgression)
        artifact = Path(self.session.game_file)
        before = artifact.read_bytes()
        for command in SOLUTION:
            self.session.handle({"op": "step", "command": command})
        self.assertTrue(self.session.done)
        self.assertTrue(native_a.progression.completed)
        new_goal = {"success_criteria": [{"criterion": "Eat apple", "evidence": {"predicate": "eaten", "subject": "f1"}}]}
        started = self.session.handle({"op": "start_episode", "source_id": "goal-b", "goal": new_goal})
        native_b = self.session.episode.native
        self.assertIsNot(native_b.progression, native_a.progression)
        self.assertFalse(native_b.progression.completed)
        self.assertEqual(started["build_id"], compiled["build_id"])
        self.assertEqual(Path(self.session.game_file), artifact)
        self.assertEqual(artifact.read_bytes(), before)
        self.assertEqual(started["state"]["room_id"], "r1")
        self.assertFalse(started["state"]["done"])
        for command in SOLUTION:
            state = self.session.handle({"op": "step", "command": command})["state"]
        self.assertFalse(state["done"])  # Old endings cannot terminate the new Goal.
        self.session.handle({"op": "step", "command": "go west"})
        self.session.handle({"op": "step", "command": "take apple"})
        state = self.session.handle({"op": "step", "command": "eat apple"})["state"]
        self.assertTrue(state["won"])
        self.assertTrue(native_b.progression.completed)
        self.assertFalse(state["engine"]["won"])

    def test_initial_success_failure_and_zero_command_witness(self):
        initial = {"predicate": "player_at", "subject": "r1"}
        request = {"op": "compile", "document": puzzle(), "source_id": "initial", "goal": {
            "success_criteria": [{"criterion": "At start", "evidence": initial}]}}
        state = self.session.handle(request)["state"]
        self.assertTrue(state["won"])
        witness = self.session.handle({"op": "verify_solution", "commands": []})["witness"]
        self.assertEqual(witness["commands"], [])
        with self.assertRaises(WorldError):
            self.session.handle({"op": "step", "command": "look"})
        failed = self.session.handle({"op": "start_episode", "source_id": "initial", "goal": {
            "success_criteria": request["goal"]["success_criteria"],
            "constraints": [{"constraint": "Avoid start", "evidence": initial}]}})["state"]
        self.assertTrue(failed["lost"])
        with self.assertRaises(WorldError):
            self.session.handle({"op": "verify_solution", "commands": [], "witness": witness})

    def test_frozen_goal_invalid_candidate_and_stale_witness(self):
        self.session.handle(compile_request(puzzle()))
        verified = self.session.handle({"op": "verify_solution", "commands": SOLUTION})
        self.assertFalse(self.session.done)
        self.assertEqual(self.session.episode.result["achievability"], "achievable")
        bad = {"success_criteria": [{"criterion": "Missing", "evidence": {"predicate": "carrying", "subject": "absent"}}]}
        previous = self.session.env
        with self.assertRaises(GoalError):
            self.session.handle({"op": "start_episode", "source_id": "bad", "goal": bad})
        self.assertIs(self.session.env, previous)
        self.assertEqual(self.session.handle({"op": "step", "command": "look"})["state"]["room_id"], "r1")
        self.assertEqual(self.session.episode.result["achievability"], "achievable")
        changed = goal(puzzle())
        changed["success_criteria"] = [{"criterion": "Eat", "evidence": {"predicate": "eaten", "subject": "f1"}}]
        self.session.handle({"op": "start_episode", "source_id": "different", "goal": changed})
        changed["success_criteria"].clear()
        self.assertEqual(len(self.session.episode.bound.goal["success_criteria"]), 1)
        with self.assertRaises(WorldError):
            self.session.handle({"op": "verify_solution", "commands": SOLUTION, "witness": verified["witness"]})

    def test_bind_arity_doorless_passage_and_unverified_goal(self):
        world = puzzle()
        world["passages"][0].update(door_name="", door_state="open", key="")
        self.session.handle({"op": "compile", "document": world})
        for evidence in ({"predicate": "open", "subject": "p1"},
                         {"predicate": "carrying", "subject": "k1", "target": "r1"}):
            with self.assertRaises(GoalError):
                self.session.handle({"op": "start_episode", "source_id": "invalid", "goal": {"success_criteria": [{"criterion": "Invalid", "evidence": evidence}]}})
        self.session.handle({"op": "start_episode", "source_id": "manual", "goal": {"success_criteria": [{"criterion": "Review manually"}]}})
        self.assertEqual(self.session.handle({"op": "step", "command": "look"})["state"]["episode"]["coverage"], "unverified")

    def test_win_all_fail_any_and_legacy_sandbox(self):
        world = puzzle()
        world["passages"][0]["door_state"] = "open"
        world["fail_conditions"].append({"predicate": "closed", "subject": "p1"})
        self.session.handle(compile_request(world))
        state = self.session.handle({"op": "step", "command": "go east"})["state"]
        self.assertFalse(state["won"])  # Correct room alone does not satisfy ALL.
        self.session.handle({"op": "reset"})
        state = self.session.handle({"op": "step", "command": "close oak door"})["state"]
        self.assertTrue(state["lost"])  # Second failure condition alone is enough.
        world["version"] = 1
        world.pop("win_conditions")
        world.pop("fail_conditions")
        result = self.session.handle(compile_request(world))
        self.assertTrue(result["warnings"])
        self.assertFalse(self.session.handle({"op": "step", "command": "look"})["state"]["done"])


if __name__ == "__main__":
    unittest.main()

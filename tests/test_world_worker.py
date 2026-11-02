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
from test_path_navigation import path_world


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
    def test_virtual_tools_require_unambiguous_accessible_catalogs(self):
        world = {"format": "grim-world", "version": 3, "spawn": "room", "entities": [
            {"id": "room", "name": "Office", "kind": "room"},
            {"id": "hidden", "name": "hidden catalog", "kind": "container", "parent": "room", "state": "closed", "open_provider": "tool_catalog"},
            {"id": "visible", "name": "visible catalog", "kind": "container", "parent": "room", "state": "closed", "open_provider": "tool_catalog"}],
            "passages": []}
        self.session.handle({"op": "compile", "document": world, "catalog_tools": ["foo"],
                             "observation_providers": ["tool_catalog", "tool_details"]})
        def step(command):
            return self.session.handle({"op": "step", "command": command, "catalog_tools": ["foo"]})["state"]
        self.assertNotIn("observation_request", step("examine foo"))
        step("open visible catalog")
        state = step("examine foo")
        self.assertEqual(state["observation_request"]["entity_id"], "visible")
        self.assertNotIn("examine foo in hidden catalog", state["admissible_commands"])
        self.assertNotIn("observation_request", step("examine foo in hidden catalog"))
        step("open hidden catalog")
        self.assertNotIn("observation_request", step("examine foo"))
        state = step("x foo in hidden catalog")
        self.assertEqual(state["observation_request"], {"provider": "tool_details", "entity_id": "hidden", "tool_id": "foo"})
        step("close visible catalog")
        self.assertEqual(step("examine foo")["observation_request"]["entity_id"], "hidden")

    def test_virtual_tools_scope_collisions_updates_and_replay(self):
        world = {"format": "grim-world", "version": 3, "spawn": "r1", "entities": [
            {"id": "r1", "name": "Entry", "kind": "room"},
            {"id": "r2", "name": "Office", "kind": "room"},
            {"id": "ledger", "name": "capability ledger", "kind": "container", "parent": "r2",
             "state": "closed", "open_provider": "tool_catalog"},
            {"id": "ordinary", "name": "open", "kind": "object", "parent": "r2",
             "description": "An ordinary object."}],
            "passages": [{"id": "p", "from": "r1", "to": "r2", "direction": "east"}]}
        roster = ["open", "ui.show", "alias info"]
        self.session.handle({"op": "compile", "document": world, "catalog_tools": roster,
                             "observation_providers": ["tool_catalog", "tool_details"]})
        def step(command, tools=None):
            return self.session.handle({"op": "step", "command": command,
                                        "catalog_tools": roster if tools is None else tools})["state"]
        self.assertNotIn("observation_request", step("examine ui.show"))
        state = step("go east")
        self.assertNotIn("examine ui.show in capability ledger", state["admissible_commands"])
        state = step("open capability ledger")
        self.assertIn("examine ui.show in capability ledger", state["admissible_commands"])
        self.assertIn("An ordinary object", step("examine open")["feedback"])
        for command, tool_id in (("examine open in capability ledger", "open"),
                                 ("x ui.show", "ui.show"), ("examine alias info", "alias info")):
            state = step(command)
            self.assertEqual(state["observation_request"],
                             {"provider": "tool_details", "entity_id": "ledger", "tool_id": tool_id})
        removed = step("examine ui.show", [])
        self.assertEqual(removed["observation_request"]["tool_id"], "ui.show")
        self.assertNotIn("examine ui.show in capability ledger", removed["admissible_commands"])
        self.assertNotIn("observation_request", step("examine absent"))
        step("go west")
        self.assertNotIn("observation_request", step("examine ui.show in capability ledger"))
        # A saved witness uses explicit catalog scope, independent of live roster.
        # Start in Office so inspection precedes reaching the destination.
        world["spawn"] = "r2"
        self.session.handle({"op": "compile", "document": world, "catalog_tools": roster,
            "observation_providers": ["tool_catalog", "tool_details"], "source_id": "catalog-replay", "goal": {
                "success_criteria": [{"criterion": "Arrive", "evidence": {"predicate": "player_at", "subject": "r1"}}]}})
        verified = self.session.handle({"op": "verify_solution", "commands": ["open capability ledger", "examine ui.show", "go west"],
                                        "catalog_tools": roster})
        self.assertEqual(verified["commands"], ["open capability ledger", "examine ui.show in capability ledger", "go west"])
        self.session.known_catalog_tools.clear()
        self.assertTrue(self.session.handle({"op": "verify_solution", "commands": verified["commands"],
                                            "witness": verified["witness"], "catalog_tools": []})["verified"])

    def test_host_open_uses_native_target_and_visibility(self):
        world = {"format": "grim-world", "version": 3, "spawn": "r1", "entities": [
            {"id": "r1", "name": "Entry", "kind": "room"},
            {"id": "r2", "name": "Office", "kind": "room"},
            {"id": "renamed_id", "name": "capability ledger", "kind": "container", "parent": "r2",
             "state": "closed", "open_provider": "tool_catalog", "description": "Ledger description."},
            {"id": "box", "name": "box", "kind": "container", "parent": "r2", "state": "closed"},
            {"id": "hidden", "name": "hidden ledger", "kind": "object", "parent": "box"},
            {"id": "ordinary", "name": "ordinary book", "kind": "object", "parent": "r2",
             "description": "An ordinary description."}],
            "passages": [{"id": "p", "from": "r1", "to": "r2", "direction": "east"}]}
        with self.assertRaises(WorldError):
            self.session.handle({"op": "compile", "document": world})
        self.session.handle({"op": "compile", "document": world, "observation_providers": ["tool_catalog"]})
        def step(command):
            return self.session.handle({"op": "step", "command": command})["state"]
        self.assertNotIn("observation_request", step("open capability ledger"))
        step("go east")
        description = step("examine capability ledger")
        self.assertNotIn("observation_request", description)
        self.assertIn("Ledger description", description["feedback"])
        for command in ("open capability ledger", "open capability ledger", "open the capability ledger"):
            state = step(command)
            self.assertEqual(state["observation_request"], {"entity_id": "renamed_id", "provider": "tool_catalog"})
        for command in ("open absent", "x capability ledger", "examine hidden ledger", "look", "inventory"):
            state = step(command)
            self.assertNotIn("observation_request", state)
            if command == "x capability ledger":
                self.assertIn("It is open", state["feedback"])
                self.assertNotIn("empty", state["feedback"])
        step("open box")
        self.assertNotIn("observation_request", step("x hidden ledger"))
        self.assertNotIn("observation_request", step("close capability ledger"))
        ordinary = step("examine ordinary book")
        self.assertNotIn("observation_request", ordinary)
        self.assertIn("An ordinary description", ordinary["feedback"])
        initial = self.session.handle({"op": "reset"})["state"]
        self.assertNotIn("observation_request", initial)
        self.assertIn({"predicate": "closed", "arguments": ["renamed_id"]}, initial["facts"])

    def test_native_path_movement_goal_replay_and_reset(self):
        world = path_world()
        request = {"op": "compile", "document": world, "source_id": "path-test", "goal": {
            "success_criteria": [{"criterion": "Arrive", "evidence": {"predicate": "player_at", "subject": "c"}}]}}
        initial = self.session.handle(request)["state"]
        self.assertEqual(initial["location_path"], "/")
        self.assertIn("enter alpha", initial["admissible_commands"])
        self.assertFalse(any(c.startswith("pathgo") for c in initial["admissible_commands"]))
        before = self.session.env.state
        for command in ("go /alpha/child", "back", "go /absent"):
            with self.assertRaises(WorldError):
                self.session.handle({"op": "step", "command": command})
            self.assertIs(self.session.env.state, before)
        state = self.session.handle({"op": "step", "command": "enter alpha"})["state"]
        self.assertEqual(state["room_id"], "a")
        self.assertEqual(state["location_path"], "/alpha")
        self.assertEqual(state["moves"], 1)
        self.assertIn({"predicate": "at", "arguments": ["P", "a"]}, state["facts"])
        self.assertIn("Room A", state["feedback"])
        self.assertEqual(self.session.handle({"op": "step", "command": "go /"})["state"]["room_id"], "root")
        commands = ["go alpha", "go ./child"]
        for command in commands:
            state = self.session.handle({"op": "step", "command": command})["state"]
        self.assertTrue(state["won"])
        self.assertEqual(state["location_path"], "/alpha/child")
        self.assertTrue(self.session.handle({"op": "verify_solution", "commands": commands})["verified"])
        self.assertEqual(self.session.handle({"op": "reset"})["state"]["location_path"], "/")
        self.session.handle({"op": "step", "command": "enter alpha"})
        self.assertEqual(self.session.handle({"op": "step", "command": "back"})["state"]["room_id"], "root")

    def test_path_branching_more_than_four_and_identity(self):
        from world_worker import environment_identity
        world = path_world()
        for i in range(6):
            ident = "extra" + str(i)
            world["entities"].append({"id": ident, "kind": "room", "name": "Extra room " + str(i)})
            world["navigation"]["rooms"].append({"id": ident, "parent": "root", "segment": ident})
        before = environment_identity(world)
        world["navigation"]["rooms"][-1]["segment"] = "last"
        self.assertNotEqual(before, environment_identity(world))
        state = self.session.handle({"op": "compile", "document": world})["state"]
        self.assertEqual(len(state["available_rooms"]), 8)
        for command in ("enter last", "go ..", "enter beta", "back"):
            state = self.session.handle({"op": "step", "command": command})["state"]
        self.assertEqual(state["room_id"], "root")
        bad = path_world()
        bad["navigation"]["rooms"][1]["parent"] = "c"
        with self.assertRaises(ValueError):
            self.session.handle({"op": "compile", "document": bad})
        self.assertEqual(self.session.handle({"op": "step", "command": "look"})["state"]["location_path"], "/")

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

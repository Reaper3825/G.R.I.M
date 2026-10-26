import copy
import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/textworld"))
from goal_evaluator import BoundGoal, Episode, GoalError


ENTITIES = {key: {"kind": kind, "symbol": (symbol, type_)} for key, kind, symbol, type_ in (
    ("room", "room", "r_0", "r"), ("key", "key", "k_0", "k"),
    ("apple", "food", "f_0", "f"), ("box", "container", "c_0", "c"),
    ("table", "supporter", "s_0", "s"), ("door", "door", "d_0", "d"))}


def requirement(pred="carrying", subject="key", target=None):
    evidence = {"predicate": pred, "subject": subject}
    if target is not None:
        evidence["target"] = target
    return {"criterion": "Requirement", "evidence": evidence}


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        if self._testMethodName != "test_invalid_bindings_have_source_locations" and importlib.util.find_spec("textworld") is None:
            self.skipTest("Native evaluation runs in the existing grim-textworld:1.7.0 image")

    def bind(self, entries=None, constraints=None):
        return BoundGoal({"success_criteria": entries if entries is not None else [requirement()],
                          "constraints": constraints or []}, "source", ENTITIES)

    def test_predicates_and_typed_identities(self):
        for pred, subject, target, fact in (
            ("player_at", "room", None, ("at", (("P", "P"), ("r_0", "r")))),
            ("carrying", "key", None, ("in", (("k_0", "k"), ("I", "I")))),
            ("at", "key", "room", ("at", (("k_0", "k"), ("r_0", "r")))),
            ("in", "apple", "box", ("in", (("f_0", "f"), ("c_0", "c")))),
            ("on", "apple", "table", ("on", (("f_0", "f"), ("s_0", "s")))),
            ("eaten", "apple", None, ("eaten", (("f_0", "f"),))),
            ("open", "door", None, ("open", (("d_0", "d"),))),
            ("closed", "box", None, ("closed", (("c_0", "c"),))),
            ("locked", "door", None, ("locked", (("d_0", "d"),))),
        ):
            with self.subTest(predicate=pred):
                bound = self.bind([requirement(pred, subject, target)])
                self.assertEqual(bound.evaluate({fact})["status"], "succeeded")
                self.assertEqual(bound.evaluate(set())["status"], "active")
                wrong_types = (fact[0], tuple((name, "wrong") for name, _ in fact[1]))
                self.assertEqual(bound.evaluate({wrong_types})["status"], "active")

    def test_invalid_bindings_have_source_locations(self):
        with self.assertRaises(GoalError):
            BoundGoal({"success_criteria": [requirement()]}, "", ENTITIES)
        for entry in (requirement("unknown"), requirement(subject="missing"), requirement("player_at", "key"),
                      requirement("carrying", "key", "box"), requirement("in", "key"),
                      requirement("in", "key", "room"), requirement("eaten", "key"),
                      requirement("open", "table"), requirement("at", "room", "room"),
                      {"criterion": "C", "evidence": "legacy"}):
            with self.subTest(entry=entry), self.assertRaises(GoalError) as error:
                self.bind([entry])
            self.assertEqual(error.exception.source_id, "source")
            self.assertEqual(error.exception.collection, "success_criteria")
            self.assertEqual(error.exception.index, 0)

    def test_failure_precedes_success(self):
        entry = requirement()
        bound = self.bind([entry], [{"constraint": "Avoid", "evidence": entry["evidence"]}])
        facts = {("in", (("k_0", "k"), ("I", "I")))}
        self.assertEqual(bound.evaluate(facts)["status"], "failed")

    def test_checks_are_simultaneous_not_latched(self):
        bound = self.bind([requirement(), requirement("player_at", "room")])
        first = {("in", (("k_0", "k"), ("I", "I")))}
        second = {("at", (("P", "P"), ("r_0", "r")))}
        self.assertEqual(bound.evaluate(first)["status"], "active")
        self.assertEqual(bound.evaluate(second)["status"], "active")
        self.assertEqual(bound.evaluate(first | second)["status"], "succeeded")
        episode = Episode(bound, "environment", set())
        episode.advance(first, command="take key")
        episode.advance(second, command="drop key and move")
        self.assertEqual(episode.result["status"], "active")
        episode.advance(first | second, command="take key")
        self.assertEqual(episode.result["status"], "succeeded")

    def test_native_quest_structure_and_progression(self):
        from textworld.generator.game import Event, Quest, QuestProgression

        bound = self.bind([requirement(), requirement("player_at", "room")], [
            {"constraint": "Do not eat", "evidence": {"predicate": "eaten", "subject": "apple"}},
            {"constraint": "Do not open", "evidence": {"predicate": "open", "subject": "door"}}])
        episode = Episode(bound, "environment", set())
        native = episode.native
        self.assertIsInstance(native.quest, Quest)
        self.assertIsInstance(native.progression, QuestProgression)
        self.assertEqual(len(native.quest.win_events), 1)
        self.assertIsInstance(native.quest.win_events[0], Event)
        self.assertEqual(len(native.quest.win_events[0].condition.preconditions), 2)
        self.assertEqual(len(native.quest.fail_events), 2)
        episode.advance({("open", (("d_0", "d"),))}, command="open door")
        self.assertTrue(native.progression.failed)
        self.assertEqual(episode.result["status"], "failed")

    def test_native_failure_precedence_and_missing_constraint_coverage(self):
        entry = requirement()
        facts = {("in", (("k_0", "k"), ("I", "I")))}
        episode = Episode(self.bind([entry], [{"constraint": "Avoid", "evidence": entry["evidence"]}]),
                          "environment", facts)
        self.assertTrue(episode.native.progression.completed)
        self.assertTrue(episode.native.progression.failed)
        self.assertEqual(episode.result["status"], "failed")
        episode = Episode(self.bind([entry], [{"constraint": "Unverified"}]), "environment", facts)
        self.assertEqual(episode.native.quest, None)
        self.assertEqual(episode.result["status"], "active")

    def test_unverified_requirements_prevent_whole_goal_success(self):
        fact = {("in", (("k_0", "k"), ("I", "I")))}
        for bound in (self.bind([requirement(), {"criterion": "Review"}]),
                      self.bind(constraints=[{"constraint": "Review"}])):
            self.assertEqual(bound.evaluate(fact)["status"], "active")
            self.assertEqual(bound.evaluate(fact)["coverage"], "unverified")
        self.assertEqual(self.bind([]).evaluate(set())["status"], "active")
        self.assertEqual(BoundGoal(None, "", ENTITIES).evaluate(set())["status"], "active")

    def test_snapshot_and_terminal_rules(self):
        goal = {"success_criteria": [requirement()]}
        before = copy.deepcopy(ENTITIES)
        bound = BoundGoal(goal, "source", ENTITIES)
        goal["success_criteria"].clear()
        episode = Episode(bound, "environment", set())
        self.assertEqual(ENTITIES, before)
        self.assertEqual(len(bound.goal["success_criteria"]), 1)
        episode.advance({("in", (("k_0", "k"), ("I", "I")))}, command="take key")
        self.assertEqual(episode.result["status"], "succeeded")
        with self.assertRaises(GoalError):
            episode.advance(set(), command="look")

    def test_engine_and_budget_stops_are_not_goal_failure(self):
        episode = Episode(self.bind(), "environment", set())
        episode.advance(set(), engine_done=True, command="quit")
        self.assertEqual(episode.result["status"], "stopped")
        self.assertEqual(episode.result["achievability"], "unknown")
        episode = Episode(self.bind(), "environment", set())
        for _ in range(1000):
            episode.advance(set(), command="look")
        self.assertEqual(episode.result["reason"], "command_limit")


if __name__ == "__main__":
    unittest.main()

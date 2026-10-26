"""Typed Goal binding and TextWorld-native Quest evaluation for reusable worlds."""
from dataclasses import dataclass
import copy
import hashlib
import json

# Bump when action/binding/evaluation semantics change, invalidating witnesses.
SEMANTICS = "grim-textworld-native-quest-v2"
PORTABLE = {"object", "key", "food"}


class GoalError(ValueError):
    def __init__(self, message, source_id="", collection="", index=-1, entity_id=""):
        super().__init__(message)
        self.source_id, self.collection, self.index, self.entity_id = source_id, collection, index, entity_id


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def fields(value, allowed):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError("Expected object with fields: " + ", ".join(sorted(allowed)))


@dataclass(frozen=True)
class Check:
    collection: str
    index: int
    text: str
    fact: object  # (predicate, tuple((runtime_id, runtime_type), ...)), or None.


class BoundGoal:
    def __init__(self, goal, source_id, entities, player=("P", "P"), inventory=("I", "I")):
        self.goal = copy.deepcopy(goal) if goal is not None else None
        self.source_id = source_id
        self.checks = []
        if not isinstance(source_id, str):
            raise GoalError("Goal source_id must be a string")
        if goal is None:
            return
        if not source_id.strip():
            raise GoalError("A selected Goal requires its source ConceptBlock ID")
        try:
            fields(goal, {"target_state", "success_criteria", "constraints"})
            if not isinstance(goal.get("target_state", ""), str):
                raise ValueError("target_state must be text")
        except ValueError as error:
            raise GoalError(str(error), source_id) from error
        for collection, text_key in (("success_criteria", "criterion"), ("constraints", "constraint")):
            entries = goal.get(collection, [])
            if not isinstance(entries, list) or len(entries) > 64:
                raise GoalError(collection + " must contain at most 64 requirements", source_id, collection)
            for index, entry in enumerate(entries):
                entity_id = ""
                try:
                    fields(entry, {text_key, "evidence"})
                    if not isinstance(entry.get(text_key), str) or not entry[text_key].strip():
                        raise ValueError(text_key + " must be non-empty text")
                    fact = None
                    if "evidence" in entry:
                        evidence = entry["evidence"]
                        fields(evidence, {"predicate", "subject", "target"})
                        pred, subject, target = (evidence.get(k, "") for k in ("predicate", "subject", "target"))
                        if not all(isinstance(v, str) for v in (pred, subject, target)):
                            raise ValueError("Evidence arguments must be strings")
                        entity_id = subject
                        entity = entities.get(subject)
                        if entity is None:
                            raise ValueError("Evidence references a missing entity or door")
                        kind, symbol = entity["kind"], tuple(entity["symbol"])
                        if pred in ("player_at", "carrying", "eaten", "open", "closed", "locked"):
                            if target:
                                raise ValueError("Unary evidence must not have a target")
                            required = {"player_at": {"room"}, "carrying": PORTABLE, "eaten": {"food"},
                                        "open": {"container", "door"}, "closed": {"container", "door"},
                                        "locked": {"container", "door"}}[pred]
                            if kind not in required:
                                raise ValueError("Invalid subject type for " + pred)
                            fact = (("at", (tuple(player), symbol)) if pred == "player_at" else
                                    ("in", (symbol, tuple(inventory))) if pred == "carrying" else
                                    (pred, (symbol,)))
                        elif pred in ("at", "in", "on"):
                            holder = entities.get(target)
                            entity_id = target or subject
                            if kind not in PORTABLE or holder is None or holder["kind"] != {"at": "room", "in": "container", "on": "supporter"}[pred]:
                                raise ValueError("Invalid subject/target types for " + pred)
                            fact = (pred, (symbol, tuple(holder["symbol"])))
                        else:
                            raise ValueError("Unsupported evidence predicate: " + pred)
                    self.checks.append(Check(collection, index, entry[text_key], fact))
                except (ValueError, TypeError, KeyError) as error:
                    raise GoalError(str(error), source_id, collection, index, entity_id) from error

    def evaluate(self, facts):
        """Evaluate an independent snapshot; episodes retain their own progression."""
        return NativeGoal(self).evaluate(facts)


class NativeGoal:
    def __init__(self, bound, kb=None):
        # Binding remains usable on the host without installing TextWorld.
        from textworld.generator.data import KnowledgeBase
        from textworld.generator.game import Event, Quest, QuestProgression
        from textworld.logic import Proposition, Variable

        self.kb = kb if kb is not None else KnowledgeBase.default()
        self.bound = bound
        self.events = [None if check.fact is None else Event(conditions=[
            Proposition(check.fact[0], [Variable(name, type_) for name, type_ in check.fact[1]])
        ]) for check in bound.checks]
        self.complete = all(event is not None for event in self.events)
        wins = [event for check, event in zip(bound.checks, self.events)
                if check.collection == "success_criteria"]
        self.covered = self.complete and bool(wins)
        # Native win_events are alternatives. One event holds ALL success
        # conditions so they must hold simultaneously, rather than latch apart.
        win_events = [Event(conditions=[fact for event in wins
                                       for fact in event.condition.preconditions])] if self.covered else []
        fail_events = [event for check, event in zip(bound.checks, self.events)
                       if check.collection == "constraints" and event is not None]
        self.quest = Quest(win_events=win_events, fail_events=fail_events) if win_events or fail_events else None
        self.progression = QuestProgression(self.quest, self.kb) if self.quest is not None else None

    def evaluate(self, facts):
        from textworld.logic import Proposition, State, Variable

        state = State(self.kb.logic, [Proposition(name, [Variable(n, t) for n, t in args])
                                     for name, args in facts])
        results = [{"collection": check.collection, "index": check.index, "text": check.text,
                    "status": "unverified" if event is None else
                              "satisfied" if event.is_triggering(state) else "unsatisfied"}
                   for check, event in zip(self.bound.checks, self.events)]
        if self.progression is not None:
            self.progression.update(state=state)
        failed = self.progression is not None and self.progression.failed
        succeeded = self.progression is not None and self.progression.completed
        # TextWorld can report both on the same transition; preserve the Goal
        # contract's failure precedence at the episode boundary.
        return {"status": "failed" if failed else "succeeded" if succeeded else "active",
                "coverage": "complete" if self.covered else "unverified", "requirements": results}


class Episode:
    def __init__(self, bound, environment_id, initial_facts, kb=None):
        self.bound = bound
        self.native = NativeGoal(bound, kb)
        self.commands = []
        self.witness_verified = False
        self.identity = {"environment": environment_id, "starting_state": identity(sorted(initial_facts)),
                         "source_id": bound.source_id, "goal": identity(bound.goal), "semantics": SEMANTICS}
        self.result = {}
        self.advance(initial_facts)

    def advance(self, facts, engine_done=False, command=None):
        if command is not None:
            if self.result.get("status") != "active":
                raise GoalError("Episode is finished; reset or start another Goal")
            self.commands.append(command)
        self.result = self.native.evaluate(facts)
        if self.result["status"] == "active" and (engine_done or len(self.commands) >= 1000):
            self.result.update(status="stopped", reason="engine_terminated" if engine_done else "command_limit")
        self.result["identity"] = self.identity
        # Initial success has a zero-command witness. Other episodes acquire a
        # reusable achievability witness through isolated solution replay.
        self.result["achievability"] = "achievable" if self.witness_verified or (self.result["status"] == "succeeded" and not self.commands) else "unknown"
        return copy.deepcopy(self.result)

"""Pure Goal binding/evaluation; no interpreter, database or model dependency."""
from dataclasses import dataclass
import copy
import hashlib
import json

# Bump when action/binding/evaluation semantics change, invalidating witnesses.
SEMANTICS = "grim-textworld-goal-v1"
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
        facts = set(facts)
        results = [{"collection": check.collection, "index": check.index, "text": check.text,
                    "status": "unverified" if check.fact is None else
                              "satisfied" if check.fact in facts else "unsatisfied"} for check in self.checks]
        failed = any(r["collection"] == "constraints" and r["status"] == "satisfied" for r in results)
        complete = all(r["status"] != "unverified" for r in results)
        wins = [r for r in results if r["collection"] == "success_criteria"]
        succeeded = bool(wins) and complete and all(r["status"] == "satisfied" for r in wins)
        return {"status": "failed" if failed else "succeeded" if succeeded else "active",
                "coverage": "complete" if complete and wins else "unverified", "requirements": results}


class Episode:
    def __init__(self, bound, environment_id, initial_facts):
        self.bound = bound
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
        self.result = self.bound.evaluate(facts)
        if self.result["status"] == "active" and (engine_done or len(self.commands) >= 1000):
            self.result.update(status="stopped", reason="engine_terminated" if engine_done else "command_limit")
        self.result["identity"] = self.identity
        # Initial success has a zero-command witness. Other episodes acquire a
        # reusable achievability witness through isolated solution replay.
        self.result["achievability"] = "achievable" if self.witness_verified or (self.result["status"] == "succeeded" and not self.commands) else "unknown"
        return copy.deepcopy(self.result)

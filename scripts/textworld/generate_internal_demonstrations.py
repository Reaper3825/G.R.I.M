"""Replay verified witnesses into ordered authoring examples, not trainer rows."""
import argparse
import copy
import json
from pathlib import Path
import re
import tempfile

from generate_internal_episodes import DATASET, compact, write_json
from goal_evaluator import identity
from world_worker import Session


VISIBLE_FIELDS = ("feedback", "description", "inventory", "location", "location_path")


def clean_observation(source, path=""):
    """Authoring mirror of Shared/Environment/EnvironmentObservation.hpp."""
    lines = source.splitlines()
    first = next((i for i, line in enumerate(lines) if line.strip()), len(lines))
    banner_end = 0
    if first < len(lines) and lines[first].strip() == "________  ________  __    __  ________":
        for i in range(first + 1, len(lines)):
            text = lines[i].strip()
            if re.fullmatch(r"-= .+ =-", text):
                banner_end = i
                break
            if any(c not in " _\\|/$><-" for c in text):
                break
    result, blank = "", False
    for line in lines[banner_end:]:
        text = line.strip()
        if (text == ">" or re.fullmatch(r">?[ \t]*-= .+ =-([ \t]*-?[0-9]+[ \t]*/[ \t]*[0-9]+)?", text)
                or (path and text == "LOCATION: " + path)):
            continue
        if not text:
            blank = bool(result)
            continue
        if result:
            result += "\n\n" if blank else "\n"
        result += line.rstrip(" \t\r")
        blank = False
    return result


def resolve_fixture(state, fixture):
    """Explicit empty-registry host fixture matching world_observation.hpp."""
    if fixture != {"format": "grim-host-observation-fixture", "version": 1,
                   "id": "empty-model-tool-catalog-v1", "catalog_tools": []}:
        raise ValueError("Only the explicit empty model-tool catalog fixture is supported")
    resolved = copy.deepcopy(state)
    request = resolved.pop("observation_request", None)
    if request is not None:
        if request != {"provider": "tool_catalog", "entity_id": "tool_catalog"}:
            raise ValueError("Unsupported host observation request; refusing fabricated feedback")
        resolved["feedback"] = "Tool catalog:\nNo model tools are currently available.\n"
        if "location_path" in resolved:
            resolved["feedback"] += "\nLOCATION: " + resolved["location_path"]
        resolved["tool_catalog"] = {"entity_id": "tool_catalog", "tools": []}
    return resolved


def project_observation(state):
    if "observation_request" in state:
        raise ValueError("Resolve host observations before projecting feedback")
    path = state.get("location_path", "")
    visible = {key: clean_observation(state.get(key, ""), path) for key in VISIBLE_FIELDS}
    if visible["feedback"] in (visible["description"], visible["inventory"]):
        visible["feedback"] = ""
    labels = ("Feedback", "Description", "Inventory", "Location", "Location path")
    return "\n\n".join(label + ":\n" + visible[key] for key, label in zip(VISIBLE_FIELDS, labels)
                         if visible[key])


def visible_goal(goal):
    return {"target_state": goal["target_state"],
            "success_criteria": [{"criterion": item["criterion"]} for item in goal["success_criteria"]],
            "constraints": [{"constraint": item["constraint"]} for item in goal["constraints"]]}


def authored_text(selection):
    # These statements depend only on the supplied Goal, not future responses.
    goal = selection["goal"]
    room = goal["success_criteria"][0]["evidence"]["subject"]
    location = "the System Hub" if room == "hub" else f"the {room} room"
    opened = goal["success_criteria"][1]["evidence"]["predicate"] == "open"
    if not opened:
        return (f"Move to {location} without opening the tool catalog.",
                f"I am in {location} and did not open the tool catalog.")
    if room == "tools":
        return ("Open the tool catalog in the tools room and remain there.",
                "The tool catalog is open, and I am in the tools room.")
    return (f"Open the tool catalog in the tools room, then {'return' if room == 'hub' else 'move'} to {location}.",
            f"The tool catalog is open, and I am {'back ' if room == 'hub' else ''}in {location}.")


def assistant_event(span, content):
    return {"origin": "assistant", "span": span, "content": content, "supervised": True}


def replay(record, session, fixture):
    selection = record["selection"]
    request = {"op": "start_episode", "source_id": selection["source_id"], "goal": selection["goal"]}
    initial = session.handle(request)["state"]
    if initial["episode"]["status"] != "active":
        raise ValueError("Expected a fresh, nonterminal episode")
    # Reject stale/tampered stored solutions before using them as demonstrations.
    session.handle({"op": "verify_solution", "commands": record["commands"], "witness": record["witness"]})
    determine, answer = authored_text(selection)
    events = [assistant_event("determine", determine)]
    receipts = [{"command": None, "native_state": initial,
                 "resolved_state": resolve_fixture(initial, fixture)}]
    latest = initial
    for command in record["commands"]:
        if latest["episode"]["status"] != "active":
            raise ValueError("Command after terminal state")
        events.append(assistant_event("execute", command))
        state = session.handle({"op": "step", "command": command})["state"]
        if state["moves"] <= latest["moves"]:
            raise ValueError("Witness contains an unsuccessful or non-progressing command")
        resolved = resolve_fixture(state, fixture)
        events.append({"origin": "environment", "span": "environment_context",
                       "content": project_observation(resolved), "supervised": False})
        receipts.append({"command": command, "native_state": state, "resolved_state": resolved})
        latest = state
    if latest["episode"]["status"] != "succeeded" or latest["episode"]["coverage"] != "complete":
        raise ValueError("Only fully evidence-verified successful trajectories may be exported")
    events.append(assistant_event("answer", answer))
    example = {"format": "grim-recurrent-authoring-example", "version": 1,
               "id": selection["source_id"],
               "input": {"prompt": selection["prompt"], "reasoning_state": {
                   "goal": visible_goal(selection["goal"]), "knowns": [], "unknowns": [],
                   "environment_context": project_observation(receipts[0]["resolved_state"])}},
               "events": events}
    private = {"id": selection["source_id"], "fixture_id": fixture["id"],
               "goal": selection["goal"], "witness": record["witness"], "receipts": receipts,
               "verification": latest["episode"], "example_identity": identity(example),
               "answer_source": "authored after evidence success; no model/controller grading run"}
    return example, private


def generate(root, fixture):
    directory = root / "DataCollection/curated" / DATASET
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    examples, private = [], []
    with tempfile.TemporaryDirectory(prefix="grim-demonstrations-") as temporary:
        session = Session(temporary)
        try:
            game = None
            for entry in manifest["episodes"]:
                if not re.fullmatch(r"[0-9a-f]+\.grimepisode\.json", entry["record"]):
                    raise ValueError("Invalid witness filename")
                record = json.loads((root / "resources/models/games/episodes" / entry["record"]).read_text(encoding="utf-8"))
                if record["selection"]["source_id"] != entry["source_id"] or not record["verified"]:
                    raise ValueError("Invalid witness selection")
                if game is None:
                    game = record["environment"]
                    session.handle({"op": "compile", "document": game,
                                    "observation_providers": ["tool_catalog"], "catalog_tools": []})
                elif record["environment"] != game:
                    raise ValueError("Mixed environments in demonstration set")
                example, audit = replay(record, session, fixture)
                examples.append(example)
                private.append(audit)
        finally:
            session.close()
    return examples, private


def review_markdown(examples):
    parts = ["# GRIM internal goal-following demonstrations\n\n"
             "Nine successful attempts. Catalog feedback uses the explicit empty-model-tool fixture.\n"
             "Input and environment observations are context; assistant spans are supervision targets.\n"
             "This is an authoring preview, not configured tokenizer output.\n"]
    for example in examples:
        initial = example["input"]
        goal = initial["reasoning_state"]["goal"]
        parts.append(f"## {example['id']}\n\n**Prompt:** {initial['prompt']}\n\n"
                     f"**Target state:** {goal['target_state']}\n\n**Success criteria:**\n")
        parts.extend("- " + item["criterion"] + "\n" for item in goal["success_criteria"])
        if goal["constraints"]:
            parts.append("\n**Constraints:**\n")
            parts.extend("- " + item["constraint"] + "\n" for item in goal["constraints"])
        parts.append("\n**Initial environment — context**\n\n```text\n" + initial["reasoning_state"]["environment_context"] + "\n```\n")
        for event in example["events"]:
            label = event["span"].capitalize() if event["origin"] == "assistant" else "Environment"
            policy = "supervised" if event["supervised"] else "context"
            parts.append(f"\n**{label} — {policy}**\n\n```text\n{event['content']}\n```\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = args.root / "DataCollection/curated" / DATASET / "demonstrations"
    fixture = json.loads((output / "host_fixture.json").read_text(encoding="utf-8"))
    examples, private = generate(args.root, fixture)
    for filename, rows in (("examples.jsonl", examples), ("verification.private.jsonl", private)):
        (output / filename).write_text("".join(compact(row) + "\n" for row in rows), encoding="utf-8")
    (output / "examples.md").write_text(review_markdown(examples), encoding="utf-8")
    write_json(output / "manifest.json", {"format": "grim-recurrent-authoring-set", "version": 1,
        "fixture_id": fixture["id"], "example_count": len(examples),
        "command_count": sum(e["span"] == "execute" for row in examples for e in row["events"]),
        "trainer_integration": "pending; event supervision flags are authoring metadata",
        "examples": [{"id": row["id"], "identity": identity(row)} for row in examples]})
    print(json.dumps({"examples": len(examples), "verified": True, "output": str(output)}))


if __name__ == "__main__":
    main()

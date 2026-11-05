"""Generate state-verified GRIM internal Goal assignments and native witnesses.

Run in the existing TextWorld image; never launches GRIM or a model. Search uses
real worker transitions, not a second implementation of TextWorld semantics.
These are Goal assignments/solution witnesses, not recurrent SFT transcripts.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import tempfile

from goal_evaluator import SEMANTICS
from world_worker import Session, environment_identity


DATASET = "grim_internal_state_v1"


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_key(value):
    # Match World Lab recordKey(identity.dump()) for its Check Saved lookup.
    result = 14695981039346656037
    for byte in compact(value).encode("utf-8"):
        result = ((result ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return format(result, "x")


def game_document(document):
    return {key: value for key, value in document.items() if key not in {
        "editor", "next_id", "objective", "solution", "win_conditions",
        "fail_conditions", "title", "environment_id"}}


def state_key(state):
    return compact(sorted(state["facts"], key=compact))


def state_actions(state):
    # Deliberately bounded to native location and container-state transitions.
    # Host UI/tools and observation-provider contents are not search targets.
    return sorted(command for command in state["admissible_commands"]
                  if command == "back" or command.startswith(("enter ", "open ", "close ")))


def discover(session, max_states):
    initial = session.handle({"op": "start_episode"})["state"]
    found = {state_key(initial): (initial, [])}
    pending = deque([state_key(initial)])
    while pending:
        state, prefix = found[pending.popleft()]
        for command in state_actions(state):
            session.handle({"op": "start_episode"})
            for previous in prefix:
                session.handle({"op": "step", "command": previous})
            result = session.handle({"op": "step", "command": command})["state"]
            key = state_key(result)
            if key in found:
                continue
            if len(found) >= max_states:
                raise ValueError("State budget exhausted; refusing to publish a partial set")
            found[key] = (result, prefix + [command])
            pending.append(key)
    return list(found.values())


def selection_for(state, document):
    room = state["room_id"]
    opened = {"predicate": "open", "arguments": ["tool_catalog"]} in state["facts"]
    catalog_state = "open" if opened else "closed"
    name = next(entity["name"] for entity in document["entities"] if entity["id"] == room)
    location = "the System Hub" if room == "hub" else f"the {name} room"
    target = f"You are in {location} with the tool catalog {catalog_state}."
    if not opened:
        prompt = f"Go to {location}. Do not open the tool catalog."
    elif room == "tools":
        prompt = "Go to the tools room and open the tool catalog. Stay in the tools room."
    elif room == "hub":
        prompt = "Open the tool catalog in the tools room, then return to the System Hub. Leave the catalog open."
    else:
        prompt = f"Open the tool catalog in the tools room, then go to {location}. Leave the catalog open."
    goal = {"target_state": target, "success_criteria": [
        {"criterion": f"You are in {location}.",
         "evidence": {"predicate": "player_at", "subject": room}},
        {"criterion": f"The tool catalog is {catalog_state}.",
         "evidence": {"predicate": catalog_state, "subject": "tool_catalog"}}],
        "constraints": []}
    if not opened:
        goal["constraints"].append({"constraint": "Do not open the tool catalog.",
                                    "evidence": {"predicate": "open", "subject": "tool_catalog"}})
    return {"source_id": f"{DATASET}_{room}_{catalog_state}",
            "name": f"GRIM internal: {name}, catalog {catalog_state}",
            "prompt": prompt,
            "goal": goal}


def generate(document, max_states=64):
    if document.get("title") != "GRIM" or document.get("spawn") != "hub":
        raise ValueError("This first generator requires the authored GRIM internal world")
    if {e["id"] for e in document["entities"]} != {
            "hub", "perception", "tools", "computer", "memory", "tool_catalog"}:
        raise ValueError("GRIM internal entity set changed; review the generator scope")
    game = game_document(document)
    records, blocks, entries = [], [], []
    with tempfile.TemporaryDirectory(prefix="grim-internal-episodes-") as directory:
        session = Session(directory)
        try:
            session.handle({"op": "compile", "document": game,
                            "observation_providers": ["tool_catalog"]})
            states = discover(session, max_states)
            for state, commands in states:
                if not commands:  # Exclude the already-satisfied initial state.
                    continue
                selection = selection_for(state, document)
                initial = session.handle({"op": "start_episode", "source_id": selection["source_id"],
                                          "goal": selection["goal"]})["state"]
                if initial["episode"]["status"] != "active" or initial["episode"]["coverage"] != "complete":
                    raise ValueError("Expected a nontrivial, fully verified Goal")
                final = initial
                for command in commands:
                    if final["episode"]["status"] != "active":
                        raise ValueError("Solution contains commands after termination")
                    final = session.handle({"op": "step", "command": command})["state"]
                if final["episode"]["status"] != "succeeded":
                    raise ValueError("Discovered solution failed bound Goal replay")
                verified = session.handle({"op": "verify_solution", "commands": commands})
                # Verify the saved identity as well as the fresh candidate replay.
                session.handle({"op": "verify_solution", "commands": verified["commands"],
                                "witness": verified["witness"]})
                filename = record_key(verified["witness"]["identity"]) + ".grimepisode.json"
                records.append((filename, {"format": "grim-episode", "version": 1,
                    "selection": selection, "environment": game, "commands": verified["commands"],
                    "witness": verified["witness"], "verified": True}))
                blocks.append({"id": selection["source_id"], "name": selection["name"],
                    "prompt": selection["prompt"], "goal": selection["goal"],
                    "knowns": [], "unknowns": [], "raw": "", "format_type": "chain_of_thought",
                    "source_sequence_id": DATASET, "timestamp": 0})
                entries.append({"source_id": selection["source_id"], "record": filename,
                    "commands": len(commands), "final_location": final["location_path"],
                    "requirements": final["episode"]["requirements"], "status": "succeeded"})
        finally:
            session.close()
    manifest = {"format": "grim-internal-state-episode-set", "version": 1, "dataset": DATASET,
                "environment_id": document["environment_id"], "environment_identity": environment_identity(game),
                "evaluator_semantics": SEMANTICS, "search": "breadth-first native worker replay",
                "reachable_states": len(states), "excluded_initial_states": 1,
                "episode_count": len(records), "episodes": entries}
    return records, blocks, manifest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=64)
    args = parser.parse_args()
    document = json.loads((args.root / "resources/models/games/grim_internal.grimworld.json").read_text(encoding="utf-8"))
    records, blocks, manifest = generate(document, args.max_states)
    # Publish only after every candidate has passed live and isolated replays.
    for filename, record in records:
        write_json(args.root / "resources/models/games/episodes" / filename, record)
    output = args.root / "DataCollection/curated" / DATASET
    output.mkdir(parents=True, exist_ok=True)
    (output / "goals.jsonl").write_text("".join(compact(block) + "\n" for block in blocks), encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"episodes": len(records), "reachable_states": manifest["reachable_states"],
                      "verified": True, "output": str(output)}))


if __name__ == "__main__":
    main()

"""Generate Goal-only Navigate assignments from a live navigation snapshot.

Standard library only. Does not execute GRIM, modify the active dataset, or
invent an evaluator for LiveWorld. Durable provider bindings stay in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
COURSE_ID = "course_environment_navigate_v1"
DATASET = "live_navigation_goals_v1"


def text(value, label):
    if not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n\x00"):
        raise ValueError(f"{label} must be non-empty single-line text")
    return value


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def generate(snapshot):
    """Return Goal-only blocks, a course record, and private authoring metadata.

    IDs depend only on provider type/name, surviving restarts, path changes,
    provider replacement, and additions. Paths/text refresh on regeneration.
    """
    if not isinstance(snapshot, dict) or snapshot.get("format") != "grim-live-navigation-snapshot":
        raise ValueError("Expected a grim-live-navigation-snapshot")
    if type(snapshot.get("version")) is not int or snapshot["version"] != 1:
        raise ValueError("Unsupported navigation snapshot version")
    session_id = text(snapshot.get("session_id"), "session_id")
    hub_id = text(snapshot.get("hub_id"), "hub_id")
    rooms = snapshot.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        raise ValueError("Snapshot requires rooms, including its hub")
    by_id, paths, providers = {}, set(), set()
    for room in rooms:
        if not isinstance(room, dict):
            raise ValueError("Each navigation room must be an object")
        room_id = text(room.get("room_id"), "room_id")
        text(room.get("display_name"), "display_name")
        path = text(room.get("path"), "path")
        if room_id in by_id or path in paths:
            raise ValueError("Room IDs and paths must be unique")
        by_id[room_id] = room
        paths.add(path)
        if room_id == hub_id:
            if path != "/" or room.get("parent_id") != "" or "provider" in room:
                raise ValueError("Hub must be the unowned root room")
            continue
        if room.get("parent_id") != hub_id or not re.fullmatch(r"/[a-zA-Z0-9_-]{1,80}", path):
            raise ValueError("Live provider rooms must be direct hub children with one path segment")
        provider = room.get("provider")
        if not isinstance(provider, dict) or provider.get("type") not in ("builtin", "plugin"):
            raise ValueError("Provider room requires a builtin/plugin identity")
        key = (provider["type"], text(provider.get("name"), "provider name"))
        if not isinstance(provider.get("instance"), str):
            raise ValueError("Provider instance must be text (empty is allowed)")
        if key in providers:
            raise ValueError("Duplicate provider identity")
        providers.add(key)
    if hub_id not in by_id:
        raise ValueError("Snapshot hub is missing")

    blocks, assignments = [], []
    destinations = sorted((r for r in rooms if r["room_id"] != hub_id),
                          key=lambda r: (r["provider"]["type"], r["provider"]["name"]))
    for room in destinations:
        provider = room["provider"]
        destination = {"type": provider["type"], "name": provider["name"]}
        block_id = "live_navigation_v1_" + identity(destination)
        label = room["display_name"]
        goal = {"target_state": f"You are in the {label} room.",
                "success_criteria": [{"criterion": f"You are in the {label} room."}],
                "constraints": []}
        blocks.append({"id": block_id, "name": f"Navigate to the {label} room",
                       "prompt": f'From {by_id[hub_id]["display_name"]}, go to the {label} room.',
                       "goal": goal, "knowns": [], "unknowns": [], "raw": "",
                       "format_type": "chain_of_thought", "source_sequence_id": DATASET,
                       "timestamp": 0})
        assignments.append({"source_id": block_id, "destination_provider": destination,
                            "snapshot_room_id": room["room_id"], "snapshot_path": room["path"],
                            "snapshot_provider_instance": provider["instance"],
                            "start_room_id": hub_id, "minimum_moves": 1})
    course = {"id": COURSE_ID, "name": "Navigate", "environment_id": "",
              "concept_block_ids": [block["id"] for block in blocks]}
    manifest = {"format": "grim-live-navigation-goal-set", "version": 1,
                "backend": "live_world", "snapshot_identity": identity(snapshot),
                "snapshot_session_id": session_id, "hub_id": hub_id,
                "room_count": len(rooms), "goal_count": len(blocks),
                "validation": "snapshot structure and hub adjacency; no runtime evaluation",
                "verification_status": "unverified: live Goal binding/evaluation not implemented",
                "course_binding_status": "unbound: live sessions are not registered TextWorld environments",
                "order": "provider type, then exact provider name",
                "assignments": assignments}
    return blocks, course, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path,
                        default=ROOT / "resources/models/games/live_navigation.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "DataCollection/curated/live_navigation_goals_v1")
    args = parser.parse_args()
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8-sig"))
    blocks, course, manifest = generate(snapshot)
    manifest["source_snapshot"] = str(args.snapshot.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "goals.jsonl").write_text(
        "".join(json.dumps(block, ensure_ascii=False, sort_keys=True) + "\n" for block in blocks),
        encoding="utf-8")
    for filename, value in (("course.json", course), ("manifest.json", manifest)):
        (args.output_dir / filename).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"goals": len(blocks), "output": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()

"""Live navigation Goal authoring checks; no running host required."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/textworld"))
from generate_navigation_course import generate


def fixture():
    return {"format": "grim-live-navigation-snapshot", "version": 1,
            "session_id": "session-1", "hub_id": "system_hub", "rooms": [
                {"room_id": "system_hub", "parent_id": "", "display_name": "System Hub", "path": "/"},
                {"room_id": "provider_1", "parent_id": "system_hub", "display_name": "UI", "path": "/UI",
                 "provider": {"type": "plugin", "name": "UI Plugin", "instance": "first"}}]}


class NavigationCourseTests(unittest.TestCase):
    def test_goal_only_records_and_private_bindings(self):
        blocks, course, manifest = generate(fixture())
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["prompt"], "From System Hub, go to the UI room.")
        self.assertEqual(blocks[0]["goal"]["success_criteria"],
                         [{"criterion": "You are in the UI room."}])
        self.assertFalse(set(blocks[0]) & {"determine", "execute", "answer", "update", "define"})
        self.assertEqual(course["concept_block_ids"], [blocks[0]["id"]])
        self.assertEqual(course["environment_id"], "")
        self.assertEqual(manifest["assignments"][0]["destination_provider"],
                         {"type": "plugin", "name": "UI Plugin"})

    def test_new_rooms_discovered_and_old_ids_preserved(self):
        snapshot = fixture()
        old = generate(snapshot)[0][0]["id"]
        room = copy.deepcopy(snapshot["rooms"][1])
        room.update(room_id="provider_2", display_name="Input", path="/Input")
        room["provider"].update(name="Input Plugin")
        snapshot["rooms"].append(room)
        blocks, _, manifest = generate(snapshot)
        self.assertEqual(len(blocks), 2)
        self.assertIn(old, [b["id"] for b in blocks])
        self.assertEqual(manifest["goal_count"], 2)
        self.assertEqual(generate({**snapshot, "rooms": list(reversed(snapshot["rooms"]))})[:2],
                         generate(snapshot)[:2])

    def test_restart_replacement_and_path_changes_preserve_assignment_identity(self):
        snapshot = fixture()
        original = generate(snapshot)[0][0]
        snapshot["session_id"] = "new-session"
        snapshot["rooms"][1].update(room_id="provider_99", path="/UI-collision", display_name="New display")
        snapshot["rooms"][1]["provider"]["instance"] = "second"
        refreshed = generate(snapshot)[0][0]
        self.assertEqual(original["id"], refreshed["id"])
        self.assertEqual(refreshed["prompt"], "From System Hub, go to the New display room.")

    def test_removed_provider_disappears_and_empty_hub_is_valid(self):
        snapshot = fixture()
        snapshot["rooms"].pop()
        blocks, course, manifest = generate(snapshot)
        self.assertEqual(blocks, [])
        self.assertEqual(course["concept_block_ids"], [])
        self.assertEqual(manifest["goal_count"], 0)

    def test_provider_types_distinguish_identical_names(self):
        snapshot = fixture()
        room = copy.deepcopy(snapshot["rooms"][1])
        room.update(room_id="provider_2", path="/UI-builtin")
        room["provider"]["type"] = "builtin"
        snapshot["rooms"].append(room)
        self.assertEqual(len({b["id"] for b in generate(snapshot)[0]}), 2)

    def test_invalid_snapshots_rejected(self):
        for mutate in (
            lambda s: s.update(version=2),
            lambda s: s.update(session_id=""),
            lambda s: s.update(hub_id="missing"),
            lambda s: s["rooms"][1].update(parent_id="missing"),
            lambda s: s["rooms"][1].update(path="/UI/nested"),
            lambda s: s["rooms"][1].update(display_name=""),
            lambda s: s["rooms"][1]["provider"].update(type="unknown"),
            lambda s: s["rooms"][1]["provider"].pop("instance"),
            lambda s: s["rooms"].append(copy.deepcopy(s["rooms"][1])),
            lambda s: s["rooms"][0].update(provider={}),
        ):
            snapshot = fixture()
            mutate(snapshot)
            with self.assertRaises(ValueError):
                generate(snapshot)


if __name__ == "__main__":
    unittest.main()

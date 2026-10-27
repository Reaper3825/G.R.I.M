"""Pure address tests; engine integration runs in test_world_worker.py."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/textworld"))
if Path("/worker").exists():
    sys.path.insert(0, "/worker")
from path_navigation import PathNavigation


def path_world():
    rooms = [("root", "", ""), ("a", "root", "alpha"), ("b", "root", "beta"), ("c", "a", "child")]
    return {"format": "grim-world", "version": 4, "spawn": "root", "passages": [],
            "entities": [{"id": ident, "kind": "room", "name": "Room " + ident, "parent": ""}
                         for ident, _, _ in rooms],
            "navigation": {"mode": "path", "rooms": [dict(id=ident, parent=parent, segment=segment)
                                                       for ident, parent, segment in rooms]}}


class PathTests(unittest.TestCase):
    def test_paths_aliases_and_single_edge_policy(self):
        nav = PathNavigation(path_world())
        self.assertEqual(nav.paths["c"], "/alpha/child")
        for command in ("enter alpha", "go /alpha", "go alpha", "go ./alpha/"):
            self.assertEqual(nav.resolve("root", command), "a")
        for command in ("back", "go ..", "go /", "go ../"):
            self.assertEqual(nav.resolve("a", command), "root")
        for room, command in (("root", "back"), ("root", "go /alpha/child"),
                              ("a", "go /beta"), ("a", "enter /alpha/child"),
                              ("root", "go missing"), ("root", "go /../alpha")):
            with self.assertRaises(ValueError, msg=command):
                nav.resolve(room, command)
        self.assertIsNone(nav.resolve("root", "look"))

    def test_bad_hierarchies(self):
        for mutate in (
            lambda d: d["navigation"]["rooms"].pop(),
            lambda d: d["navigation"]["rooms"].append(dict(d["navigation"]["rooms"][0])),
            lambda d: d["navigation"]["rooms"][1].update(parent="c"),
            lambda d: d["navigation"]["rooms"][1].update(parent="missing"),
            lambda d: d["navigation"]["rooms"][1].update(parent="", segment=""),
            lambda d: d["navigation"]["rooms"][1].update(segment="beta"),
            lambda d: d["navigation"]["rooms"][1].update(segment=".."),
            lambda d: d["navigation"]["rooms"][1].update(segment="a/b"),
            lambda d: d["navigation"]["rooms"][1].update(segment="café"),
            lambda d: d.update(version=3),
            lambda d: d.update(passages=[{}]),
        ):
            document = copy.deepcopy(path_world())
            mutate(document)
            with self.assertRaises(ValueError):
                PathNavigation(document)


if __name__ == "__main__":
    unittest.main()

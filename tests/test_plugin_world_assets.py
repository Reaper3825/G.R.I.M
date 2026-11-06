"""Validate authored plugin assets against provider declarations without loading DLLs."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "resources/models/world_assets"


class PluginWorldAssetsTests(unittest.TestCase):
    def test_provider_coverage_and_reusable_resources(self):
        configured = re.findall(r"add_subdirectory\((\w+_plugin)\)",
                                 (ROOT / "plugins/CMakeLists.txt").read_text(encoding="utf-8"))
        expected = set()
        for directory in configured:
            source = ROOT / "plugins" / directory / (directory + ".cpp")
            if not source.exists():
                source = ROOT / "plugins" / (directory + ".cpp")
            text = source.read_text(encoding="utf-8")
            declaration = re.search(r'(?:GrimPluginInfo\s+\w+\s*\{\s*GRIM_API_VERSION\s*,|'
                                    r'GrimPluginInfo\s+\w+\s*=\s*\{\s*GRIM_API_VERSION\s*,|'
                                    r'info\.name\s*=)\s*"([^"]+)"', text)
            self.assertIsNotNone(declaration, str(source))
            expected.add(declaration.group(1))
        rooms = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((LIBRARY / "rooms").glob("plugin_*.grimasset.json"))]
        self.assertEqual(len(rooms), len(expected))
        self.assertEqual({a["entities"][0]["plugin_binding"]["provider_name"] for a in rooms}, expected)
        objects = {}
        for path in (LIBRARY / "objects").glob("plugin_*.grimasset.json"):
            asset = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(asset["entities"]), 1)
            entity = asset["entities"][0]
            self.assertEqual(entity["parent"], "")
            self.assertEqual(entity["id"], asset["root"])
            identity = entity["asset_source"]["asset_id"]
            self.assertNotIn(identity, objects)
            objects[identity] = entity
        used = set()
        for asset in rooms:
            room, *resources = asset["entities"]
            self.assertEqual(room["id"], asset["root"])
            self.assertEqual(room["kind"], "room")
            self.assertEqual(room["plugin_binding"]["contract"], "grim-plugin-room")
            keys = set()
            for entity in resources:
                self.assertEqual(entity["parent"], room["id"])
                self.assertEqual(entity["kind"], "supporter")
                binding = entity["resource_binding"]
                key = (binding["type"], binding["key"])
                self.assertNotIn(key, keys)
                keys.add(key)
                self.assertIsInstance(entity["initial_state"], dict)
                identity = entity["asset_source"]["asset_id"]
                standalone = dict(objects[identity], parent=room["id"])
                self.assertEqual(entity, standalone)
                used.add(identity)
            if room["plugin_binding"]["provider_name"] == "UI Plugin":
                panels = {e["resource_binding"]["key"]: e for e in resources
                          if e["resource_binding"]["type"] == "ui.panel"}
                self.assertEqual(set(panels), {"console", "settings"})
                for key, entity in panels.items():
                    source = ROOT / "ui" / ("console_panel.cpp" if key == "console" else "ui_settings_menu.cpp")
                    self.assertRegex(source.read_text(encoding="utf-8"), r'UIPanel\("[^"]+", true, "' + key + r'"\)')
                    self.assertIsInstance(entity["initial_state"]["visible"], bool)
        self.assertEqual(used, set(objects))


if __name__ == "__main__":
    unittest.main()

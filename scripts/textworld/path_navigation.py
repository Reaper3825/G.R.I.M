"""Generic hierarchical room addresses and native TextWorld movement rules.

Addresses are independent of entity containment and display names. One command
crosses one parent/child edge; absolute addresses never grant teleportation.
"""
import re


class PathNavigation:
    def __init__(self, document):
        navigation = document.get("navigation")
        self.rooms = {}
        self.paths = {}
        self.by_path = {}
        if navigation is None:
            return
        if document.get("version") != 4 or not isinstance(navigation, dict) or navigation.get("mode") != "path":
            raise ValueError("Path navigation requires a version-4 path world")
        if document.get("passages"):
            raise ValueError("Path worlds cannot contain compass passages")
        entries = navigation.get("rooms")
        if not isinstance(entries, list):
            raise ValueError("Expected navigation rooms")
        room_ids = {e["id"] for e in document["entities"] if e["kind"] == "room"}
        for entry in entries:
            ident, parent, segment = entry["id"], entry["parent"], entry["segment"]
            if ident not in room_ids or ident in self.rooms or not isinstance(parent, str) or not isinstance(segment, str):
                raise ValueError("Invalid or duplicate navigation room")
            if parent and not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", segment):
                raise ValueError("Path segments use 1-80 ASCII letters, digits, underscores or hyphens")
            if not parent and segment:
                raise ValueError("Root segment must be empty")
            self.rooms[ident] = (parent, segment)
        if set(self.rooms) != room_ids or not room_ids:
            raise ValueError("Every room must have a navigation entry")
        roots = [ident for ident, (parent, _) in self.rooms.items() if not parent]
        if len(roots) != 1:
            raise ValueError("Path navigation requires exactly one root")
        self.root = roots[0]
        for ident in self.rooms:
            chain, node, segments = set(), ident, []
            while node:
                if node in chain or node not in self.rooms:
                    raise ValueError("Navigation cycle or missing parent")
                chain.add(node)
                parent, segment = self.rooms[node]
                if parent:
                    segments.append(segment)
                node = parent
            path = "/" + "/".join(reversed(segments))
            if len(path) > 509 or path in self.by_path:
                raise ValueError("Duplicate path or path exceeds 509 bytes")
            self.paths[ident] = path
            self.by_path[path] = ident

    def adjacent(self, current):
        parent = self.rooms[current][0]
        return ([parent] if parent else []) + [ident for ident, (p, _) in self.rooms.items() if p == current]

    def resolve(self, current, command):
        """Return a neighboring room ID, or None for a non-navigation command."""
        if command == "back":
            target = ".."
        elif command.startswith("enter "):
            target = command[6:].strip()
            if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", target):
                raise ValueError("enter expects one child path segment")
        elif command.startswith("go "):
            target = command[3:].strip()
        else:
            return None
        parts = [] if target.startswith("/") else self.paths[current].strip("/").split("/")
        parts = [part for part in parts if part]
        for part in target.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                if not parts:
                    raise ValueError("Already at root; path cannot ascend above it")
                parts.pop()
            else:
                parts.append(part)
        destination = self.by_path.get("/" + "/".join(parts))
        if destination is None:
            raise ValueError("Unknown room path")
        if destination not in self.adjacent(current):
            raise ValueError("Navigation requires a directly connected parent or child room")
        return destination


def path_knowledge_base():
    """Extend a private KB copy, leaving default compass games untouched."""
    from textworld.generator.data import KnowledgeBase
    from textworld.logic import GameLogic

    base = KnowledgeBase.default()
    source = base.logic.serialize()
    start = source.index("type r {")
    room = source[start:]
    room = room.replace("predicates {", "predicates {\n        path_link(r, r);", 1)
    room = room.replace("rules {", "rules {\n        go/path :: at(P, r) & $path_link(r, r') -> at(P, r');", 1)
    room = room.replace("reverse_rules {", "reverse_rules {\n        go/path :: go/path;", 1)
    room = room.replace('            at(P, r) ::', '            path_link(r, r\') :: "The {r} path-connects to the {r\'}";\n            at(P, r) ::', 1)
    room = room.replace("commands {", 'commands {\n            go/path :: "pathgo {r\'}" :: "pathgoing the -= {r\'} =-";', 1)
    kb = KnowledgeBase(GameLogic.parse(source[:start] + room), base.text_grammars_path)
    kb.inform7_addons_code += '''
Path-connecting relates various rooms to various rooms.
The verb to path-connect to means the path-connecting relation.
Pathgoing is an action applying to one visible thing.
Understand "pathgo [any room]" as pathgoing.
Check pathgoing:
    unless the location path-connects to the noun:
        say "That room is not directly accessible." instead.
Carry out pathgoing:
    move the player to the noun, without printing a room description.
Report pathgoing:
    try looking.
'''
    return kb

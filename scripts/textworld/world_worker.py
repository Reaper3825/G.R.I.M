"""World Lab's TextWorld 1.7 adapter and single-session file-mailbox worker.

Only the session directory is writable. No GRIM/model/runtime code is imported.
Requests and responses are published by rename so partial JSON is never consumed.
"""
import argparse
import json
import re
import time
import traceback
from pathlib import Path

TYPES = {"room": "r", "object": "o", "key": "k", "container": "c",
         "supporter": "s", "food": "f"}
OPPOSITE = {"north": "south", "south": "north", "east": "west", "west": "east"}
PORTABLE = {"object", "key", "food"}
MAX_BYTES = 16 * 1024 * 1024


class WorldError(ValueError):
    def __init__(self, message, entity_id=""):
        super().__init__(message)
        self.entity_id = entity_id


def need(ok, message, entity_id=""):
    if not ok:
        raise WorldError(message, entity_id)


def game_text(value, entity_id, name=False):
    # TextWorld embeds these strings in Inform source, including room identifiers.
    # Reject Inform substitutions/source delimiters rather than silently altering text.
    need(isinstance(value, str), "Expected text", entity_id)
    need(not any(c in value for c in '"[]\\\r\n\x00'),
         'Names/descriptions cannot contain quotes, brackets, backslashes or line breaks.', entity_id)
    if name:
        need(bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 -]{0,79}", value)),
             "Playable names must start with a letter and use letters, digits, spaces or hyphens (80 characters max).", entity_id)
    return value


def build_world(document):
    import textworld
    from textworld.generator.game import Event, Quest

    need(textworld.__version__ == "1.7.0", "World Lab requires TextWorld 1.7.0")
    need(document.get("format") == "grim-world" and document.get("version") in (1, 2), "Unsupported world format")
    entities = document.get("entities", [])
    passages = document.get("passages", [])
    need(isinstance(entities, list) and 0 < len(entities) <= 1000, "Playable worlds support 1-1000 entities")
    need(isinstance(passages, list) and len(passages) <= 2000, "Too many passages")
    source = {}
    names = set()
    for e in entities:
        ident = e["id"]
        need(isinstance(ident, str) and ident and ident not in source, "Duplicate/invalid entity ID", str(ident))
        need(e["kind"] in TYPES, "Unsupported entity kind", ident)
        name = game_text(e["name"], ident, True)
        game_text(e.get("description", ""), ident)
        need(name.casefold() not in names, "Playable entity names must be unique", ident)
        names.add(name.casefold())
        source[ident] = e
    need(document.get("spawn") in source and source[document["spawn"]]["kind"] == "room", "Select a player-start room")
    options = textworld.GameOptions()
    options.seeds = 1234
    maker = textworld.GameMaker(options)
    mapped = {}
    for ident, e in source.items():
        mapped[ident] = maker.new(type=TYPES[e["kind"]], name=e["name"], desc=e.get("description") or None)
    for ident, e in source.items():
        if e["kind"] == "room":
            need(not e.get("parent"), "A room cannot be inside another entity", ident)
            continue
        parent = source.get(e.get("parent"))
        need(parent is not None and parent["kind"] in ("room", "container", "supporter"), "Invalid object location", ident)
        need(e["kind"] in PORTABLE or parent["kind"] == "room",
             "TextWorld containers/supporters must be directly in a room", ident)
        visited = {ident}
        ancestor = parent
        while ancestor:
            need(ancestor["id"] not in visited, "Containment cycle", ident)
            visited.add(ancestor["id"])
            ancestor = source.get(ancestor.get("parent"))
        mapped[parent["id"]].add(mapped[ident])
        if e["kind"] == "container":
            need(e.get("state", "open") in ("open", "closed"), "Invalid container state", ident)
            mapped[ident].add_property(e.get("state", "open"))
    exits = set()
    for p in passages:
        ident = p["id"]
        need(ident not in source, "Duplicate passage ID", ident)
        need(p["from"] in source and source[p["from"]]["kind"] == "room" and
             p["to"] in source and source[p["to"]]["kind"] == "room" and p["from"] != p["to"], "Invalid passage endpoints", ident)
        direction = p["direction"]
        need(direction in OPPOSITE, "Invalid compass direction", ident)
        for exit_ in ((p["from"], direction), (p["to"], OPPOSITE[direction])):
            need(exit_ not in exits, "Compass exit is already occupied", ident)
            exits.add(exit_)
        path = maker.connect(getattr(mapped[p["from"]], direction), getattr(mapped[p["to"]], OPPOSITE[direction]))
        source[ident] = {"id": ident, "kind": "passage"}
        if p.get("door_name"):
            name = game_text(p["door_name"], ident, True)
            need(name.casefold() not in names, "Door names must be unique", ident)
            names.add(name.casefold())
            door = maker.new_door(path, name=name)
            mapped[ident] = door
            state = p.get("door_state", "open")
            need(state in ("open", "closed", "locked"), "Invalid door state", ident)
            door.add_property(state)
            key = p.get("key", "")
            need(state != "locked" or key, "Locked doors need a matching key", ident)
            if key:
                need(key in source and source[key]["kind"] == "key", "Door key reference is invalid", ident)
                maker.add_fact("match", mapped[key], door)
    maker.set_player(mapped[document["spawn"]])

    def condition(c):
        ident = c["subject"]
        need(ident in mapped, "Condition references a missing entity or door", ident)
        kind = source[ident]["kind"]
        subject = mapped[ident]
        pred = c["predicate"]
        if pred == "player_at":
            need(kind == "room", "Player location must be a room", ident)
            return maker.new_fact("at", maker.player, subject)
        if pred == "carrying":
            need(kind in PORTABLE, "Only portable objects can be carried", ident)
            return maker.new_fact("in", subject, maker.inventory)
        if pred == "eaten":
            need(kind == "food", "Only food can be eaten", ident)
            return maker.new_fact("eaten", subject)
        if pred in ("open", "closed", "locked"):
            need(kind in ("container", "passage"), "State condition needs a container or door", ident)
            return maker.new_fact(pred, subject)
        target = c.get("target", "")
        target_kind = {"at": "room", "in": "container", "on": "supporter"}.get(pred)
        need(kind in PORTABLE and target_kind and target in mapped and source[target]["kind"] == target_kind,
             "Invalid location condition", ident)
        return maker.new_fact(pred, subject, mapped[target])

    wins = document.get("win_conditions", [])
    fails = document.get("fail_conditions", [])
    need(len(wins) <= 64 and len(fails) <= 64, "Too many conditions")
    if wins or fails:
        objective = game_text(document.get("objective") or "Complete the authored challenge.", "")
        maker.quests = [Quest(win_events=[Event(conditions=[condition(c) for c in wins])] if wins else [],
                             fail_events=[Event(conditions=[condition(c)]) for c in fails], desc=objective)]
    maker.validate()
    return maker, mapped


class Session:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.env = None
        self.game_file = None
        self.mapping = {}
        self.done = False
        self.commands = []
        self.build_id = 0

    def close(self):
        if self.env:
            self.env.close()
        self.env = None

    @staticmethod
    def open_game(path):
        import textworld
        return textworld.start(str(path), request_infos=textworld.EnvInfos(
            won=True, lost=True, score=True, moves=True, location=True,
            inventory=True, description=True, facts=True, admissible_commands=True))

    def observation(self, state, score=0, done=False):
        facts = [{"predicate": f.name, "arguments": [self.mapping.get(a.name, a.name) for a in f.arguments]}
                 for f in state.get("facts", [])]
        room = next((f["arguments"][1] for f in facts
                     if f["predicate"] == "at" and f["arguments"][0] == "P"), "")
        return {"feedback": state.feedback, "description": state.get("description", ""),
                "inventory": state.get("inventory", ""), "location": state.get("location", ""),
                "room_id": room, "score": score, "moves": state.get("moves", 0),
                "done": bool(done), "won": bool(state.get("won")), "lost": bool(state.get("lost")),
                "facts": facts, "admissible_commands": state.get("admissible_commands", [])}

    def handle(self, request):
        op = request["op"]
        if op in ("compile", "validate"):
            maker, mapping = build_world(request["document"])
            self.build_id += 1
            path = self.directory / ("game-%d.z8" % self.build_id)
            game_file = maker.compile(str(path))
            candidate = self.open_game(game_file)
            try:
                state = candidate.reset()
            except BaseException:
                candidate.close()
                raise
            warnings = []
            if not request["document"].get("win_conditions"):
                warnings.append("No winning conditions: exploration only.")
            if op == "validate":
                candidate.close()
                return {"validated": True, "warnings": warnings}
            self.close()
            self.env = candidate
            self.game_file = game_file
            self.mapping = {entity.id: ident for ident, entity in mapping.items()}
            # EnvInfos exposes human-readable variables in facts in TextWorld 1.7.
            self.mapping.update({entity.name: ident for ident, entity in mapping.items()})
            self.commands = []
            self.done = False
            return {"state": self.observation(state), "warnings": warnings}
        need(self.env is not None, "Compile and play a world first")
        if op == "reset":
            state = self.env.reset()
            self.commands = []
            self.done = False
            return {"state": self.observation(state)}
        if op == "step":
            need(not self.done, "Episode is finished; reset to play again")
            command = request.get("command", "").strip()
            need(0 < len(command) <= 512 and not any(c in command for c in "\r\n\x00"), "Enter one command (512 characters max)")
            need(len(self.commands) < 1000, "Episode reached 1000 commands; reset to continue")
            state, score, self.done = self.env.step(command)
            self.commands.append(command)
            return {"state": self.observation(state, score, self.done)}
        if op == "verify_solution":
            commands = request.get("commands", [])
            need(isinstance(commands, list) and 0 < len(commands) <= 1000, "A solution needs 1-1000 commands")
            replay = self.open_game(self.game_file)
            try:
                replay.reset()
                done = False
                for command in commands:
                    need(not done, "Solution contains commands after the episode ended")
                    need(isinstance(command, str) and 0 < len(command) <= 512 and not any(c in command for c in "\r\n\x00"), "Invalid solution command")
                    state, _, done = replay.step(command)
                need(done and state.get("won"), "Recorded commands do not reach a win")
                return {"verified": True, "commands": commands}
            finally:
                replay.close()
        raise WorldError("Unknown operation: " + str(op))


def serve(directory):
    directory = Path(directory)
    session = Session(directory)
    last_request = time.monotonic()
    try:
        while not (directory / "stop").exists():
            request_path = directory / "request.json"
            if not request_path.exists():
                if time.monotonic() - last_request > 900:
                    return
                time.sleep(0.03)
                continue
            request_id = None
            try:
                need(request_path.stat().st_size <= MAX_BYTES, "Request exceeds 16 MB")
                request = json.loads(request_path.read_text(encoding="utf-8"))
                request_id = request["id"]
                need(request.get("protocol") == 1, "Unsupported worker protocol")
                response = {"protocol": 1, "id": request_id, "ok": True, "result": session.handle(request)}
            except Exception as error:
                traceback.print_exc()
                response = {"protocol": 1, "id": request_id, "ok": False,
                            "error": str(error), "entity_id": getattr(error, "entity_id", "")}
            request_path.unlink(missing_ok=True)
            temporary = directory / "response.tmp"
            temporary.write_text(json.dumps(response, ensure_ascii=True), encoding="utf-8")
            temporary.replace(directory / "response.json")
            last_request = time.monotonic()
    finally:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    serve(parser.parse_args().session)

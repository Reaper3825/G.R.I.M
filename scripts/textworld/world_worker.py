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
from action_catalog import action_catalog
from goal_evaluator import BoundGoal, Episode, GoalError, identity, SEMANTICS
from path_navigation import PathNavigation, path_knowledge_base

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

    need(textworld.__version__ == "1.7.0", "World Lab requires TextWorld 1.7.0")
    need(document.get("format") == "grim-world" and document.get("version") in (1, 2, 3, 4), "Unsupported world format")
    navigation = PathNavigation(document)
    need(document.get("version") != 4 or navigation.rooms, "Version-4 worlds require path navigation")
    if navigation.rooms:
        need(not any(field in document for field in ("objective", "solution", "win_conditions", "fail_conditions")),
             "Path worlds store Goals and solutions separately")
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
    if navigation.rooms:
        options.kb = path_knowledge_base()
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
    for ident, (parent, _) in navigation.rooms.items():
        if parent:
            maker.add_fact("path_link", mapped[parent], mapped[ident])
            maker.add_fact("path_link", mapped[ident], mapped[parent])
            # TextWorld's room emitter only serializes compass relations. Emit
            # this extension's relations explicitly using generated engine IDs.
            options.kb.inform7_addons_code += (
                f"\nThe {mapped[parent].id} path-connects to the {mapped[ident].id}.\n"
                f"The {mapped[ident].id} path-connects to the {mapped[parent].id}.\n")
    maker.set_player(mapped[document["spawn"]])

    # Native Quests are attached to episode progression, not compiled endings.
    maker.validate()
    return maker, mapped


def raw_facts(state):
    # TextWorld 1.7's public facts replace IDs with human-readable names.
    # Require its tracked symbolic state; never fall back to prose/name matching.
    need(state.get("_facts") is not None, "TextWorld authoritative symbolic facts unavailable")
    return {(f.name, tuple((a.name, a.type) for a in f.arguments)) for f in state["_facts"]}


def environment_identity(document):
    content = {"semantics": SEMANTICS, "spawn": document["spawn"],
               "entities": document["entities"], "passages": document.get("passages", [])}
    if "navigation" in document:
        content["navigation"] = document["navigation"]
        content["navigation_semantics"] = "path-v1-adjacent"
    return identity(content)


class Session:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.env = None
        self.game_file = None
        self.mapping = {}
        self.bindings = {}
        self.episode = None
        self.environment_id = ""
        self.build_id = 0
        self.navigation = None
        self.room_names = {}

    @property
    def done(self):
        return self.episode is not None and self.episode.result["status"] != "active"

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

    def observation(self, state, engine_score=0, engine_done=False):
        facts = [{"predicate": name, "arguments": [self.mapping.get(symbol, symbol[0]) for symbol in args]}
                 for name, args in sorted(raw_facts(state))]
        room = next((f["arguments"][1] for f in facts
                     if f["predicate"] == "at" and f["arguments"][0] == "P"), "")
        episode = self.episode.result
        result = {"feedback": state.feedback, "description": state.get("description", ""),
                "inventory": state.get("inventory", ""), "location": state.get("location", ""),
                "room_id": room, "moves": state.get("moves", 0),
                "done": self.done, "won": episode["status"] == "succeeded", "lost": episode["status"] == "failed",
                "score": int(episode["status"] == "succeeded"), "episode": episode,
                "engine": {"done": bool(engine_done), "score": engine_score,
                           "won": bool(state.get("won")), "lost": bool(state.get("lost"))},
                "facts": facts, "admissible_commands": state.get("admissible_commands", [])}
        if self.navigation and self.navigation.rooms:
            nav = self.navigation
            result["location_path"] = nav.paths[room]
            result["available_rooms"] = [{"id": ident, "path": nav.paths[ident], "name": self.room_names[ident]}
                                         for ident in nav.adjacent(room)]
            commands = [c for c in result["admissible_commands"] if not c.startswith("pathgo ")]
            commands += ["go " + nav.paths[ident] for ident in nav.adjacent(room)]
            commands += ["enter " + segment for _, (parent, segment) in nav.rooms.items() if parent == room]
            if nav.rooms[room][0]:
                commands += ["back", "go .."]
            result["admissible_commands"] = sorted(set(commands))
            result["feedback"] += "\nLOCATION: " + nav.paths[room]
        return result

    def step(self, env, command):
        """Shared dispatch for live play and deterministic witness replay."""
        if self.navigation and self.navigation.rooms:
            state = env.state
            current = next(self.mapping[args[1]] for name, args in raw_facts(state)
                           if name == "at" and args[0][1] == "P")
            try:
                destination = self.navigation.resolve(current, command)
            except ValueError as error:
                raise WorldError(str(error)) from error
            need(not command.casefold().startswith("pathgo"), "Use enter, go <path>, or back")
            if destination:
                command = "pathgo " + self.room_names[destination]
        return env.step(command)

    @staticmethod
    def command(value):
        need(isinstance(value, str) and 0 < len(value.strip()) <= 512 and not any(c in value for c in "\r\n\x00"),
             "Enter one command (512 characters max)")
        return value.strip()

    def candidate_episode(self, game_file, bound, environment_id):
        candidate = self.open_game(game_file)
        try:
            state = candidate.reset()
            episode = Episode(bound, environment_id, raw_facts(state),
                              kb=state["_game_progression"].game.kb)
            return candidate, state, episode
        except BaseException:
            candidate.close()
            raise

    def handle(self, request):
        op = request["op"]
        if op in ("compile", "validate"):
            document = request["document"]
            navigation = PathNavigation(document)
            maker, mapping = build_world(document)
            next_build = self.build_id + 1
            path = self.directory / ("environment-%d.z8" % next_build)
            sources = {e["id"]: e for e in document["entities"]}
            sources.update({p["id"]: {"kind": "door"} for p in document.get("passages", []) if p.get("door_name")})
            bindings = {ident: {"kind": sources[ident]["kind"], "symbol": (entity.id, entity.type)}
                        for ident, entity in mapping.items()}
            # Goals travel separately; document-owned legacy endings are never applied.
            bound = BoundGoal(request.get("goal"), request.get("source_id", ""), bindings)
            game_file = maker.compile(str(path))
            env_id = environment_identity(document)
            candidate, state, episode = self.candidate_episode(game_file, bound, env_id)
            warnings = (["Achievability unknown until a successful replay; manual playtest only."]
                        if episode.result["achievability"] == "unknown" else [])
            if document.get("win_conditions") or document.get("fail_conditions"):
                warnings.append("Legacy world conditions ignored. Import them as a ConceptBlock Goal explicitly.")
            if op == "validate":
                candidate.close()
                return {"validated": True, "warnings": warnings, "episode": episode.result}
            self.close()
            self.env, self.game_file, self.bindings = candidate, game_file, bindings
            self.mapping = {(entity.id, entity.type): ident for ident, entity in mapping.items()}
            self.environment_id, self.episode, self.build_id = env_id, episode, next_build
            self.navigation = navigation
            self.room_names = {e["id"]: e["name"] for e in document["entities"] if e["kind"] == "room"}
            return {"state": self.observation(state), "warnings": warnings,
                    "action_catalog": action_catalog(), "build_id": self.build_id, "environment_id": env_id}
        need(self.env is not None, "Compile an environment first")
        if op in ("start_episode", "reset"):
            bound = (BoundGoal(request.get("goal"), request.get("source_id", ""), self.bindings)
                     if op == "start_episode" else self.episode.bound)
            candidate, state, episode = self.candidate_episode(self.game_file, bound, self.environment_id)
            self.close()
            self.env, self.episode = candidate, episode
            return {"state": self.observation(state), "build_id": self.build_id, "environment_id": self.environment_id}
        if op == "step":
            need(not self.done, "Episode is finished; reset or start another Goal")
            command = self.command(request.get("command", ""))
            state, score, engine_done = self.step(self.env, command)
            self.episode.advance(raw_facts(state), engine_done, command)
            return {"state": self.observation(state, score, engine_done)}
        if op == "verify_solution":
            commands = request.get("commands", [])
            need(isinstance(commands, list) and len(commands) <= 1000, "A solution supports 0-1000 commands")
            commands = [self.command(command) for command in commands]
            witness = request.get("witness")
            if witness is not None:
                need(isinstance(witness, dict) and witness.get("identity") == self.episode.identity,
                     "Saved solution is stale for the environment, starting state, Goal, or evaluator")
            replay, state, episode = self.candidate_episode(self.game_file, self.episode.bound, self.environment_id)
            try:
                need(episode.identity == self.episode.identity, "Replay starting state differs from the active episode")
                for command in commands:
                    need(episode.result["status"] == "active", "Solution contains commands after the episode ended")
                    state, _, done = self.step(replay, command)
                    episode.advance(raw_facts(state), done, command)
                need(episode.result["status"] == "succeeded", "Recorded commands do not verify Goal success")
                self.episode.witness_verified = True
                self.episode.result["achievability"] = "achievable"
                return {"verified": True, "commands": commands, "achievability": "achievable",
                        "witness": {"identity": episode.identity, "commands": commands}}
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
                need(request.get("protocol") == 2, "Unsupported worker protocol; restart/recompile with the current client")
                response = {"protocol": 2, "id": request_id, "ok": True, "result": session.handle(request)}
            except Exception as error:
                traceback.print_exc()
                response = {"protocol": 2, "id": request_id, "ok": False,
                            "error": str(error), "entity_id": getattr(error, "entity_id", ""),
                            "source_id": getattr(error, "source_id", ""),
                            "collection": getattr(error, "collection", ""), "index": getattr(error, "index", -1)}
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

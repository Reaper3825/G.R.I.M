"""Verify the installed compiler, interpreter, reset and step APIs without GRIM."""

import json
import tempfile
from pathlib import Path

import textworld


def main():
    if textworld.__version__ != "1.7.0":
        raise RuntimeError(f"Unexpected TextWorld version: {textworld.__version__}")
    with tempfile.TemporaryDirectory(prefix="grim-textworld-") as directory:
        options = textworld.GameOptions()
        options.seeds = 1234
        options.nb_rooms = 2
        options.nb_objects = 3
        options.quest_length = 2
        options.path = str(Path(directory) / "smoke.z8")
        options.file_ext = ".z8"
        game_file, _ = textworld.make(options)
        infos = textworld.EnvInfos(won=True, extras=["walkthrough"])
        env = textworld.start(game_file, request_infos=infos)
        try:
            initial = env.reset()
            commands = initial.get("extra.walkthrough")
            if not initial.feedback or not commands:
                raise RuntimeError("Missing initial observation or test walkthrough")
            state, score, done = env.step("look")
            if not state.feedback or done:
                raise RuntimeError("Unexpected look response")
            for command in commands:
                state, score, done = env.step(command)
            if not done or not state.won:
                raise RuntimeError("Generated game's walkthrough did not win")
            if not env.reset().feedback:
                raise RuntimeError("Environment failed to reset after completion")
            print(json.dumps({"status": "passed", "textworld": textworld.__version__,
                              "seed": 1234, "walkthrough_steps": len(commands),
                              "score": score}))
        finally:
            env.close()


if __name__ == "__main__":
    main()

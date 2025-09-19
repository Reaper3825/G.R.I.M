#!/usr/bin/env python3
import sys
import os
import json
import tempfile
import time
from pathlib import Path
from TTS.api import TTS

MODEL_NAME = "tts_models/en/vctk/vits"
tts = TTS(MODEL_NAME, progress_bar=False, gpu=False)

temp_dir = Path(tempfile.gettempdir()) / "grim_tts"
temp_dir.mkdir(parents=True, exist_ok=True)

def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

def debug(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

def synthesize(text, speaker=None, speed=1.0):
    ts = int(time.time() * 1000)
    out_path = temp_dir / f"grim_tts_{ts}.wav"
    try:
        if speaker:
            tts.tts_to_file(text=text, speaker=speaker, speed=speed, file_path=str(out_path))
        else:
            tts.tts_to_file(text=text, speed=speed, file_path=str(out_path))
        send({"file": str(out_path)})
    except Exception as e:
        send({"error": str(e)})

# -------------------------------------------------
# Input handling
# -------------------------------------------------
if len(sys.argv) > 1:
    debug("Mode: CLI args")
    text = " ".join(sys.argv[1:])
    synthesize(text)
else:
    debug("Mode: stdin")
    raw_input = sys.stdin.read().strip()
    if raw_input:
        for line in raw_input.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                text = req.get("text", "")
                speaker = req.get("speaker", None)
                speed = req.get("speed", 1.0)
                if not text:
                    send({"error": "empty_text"})
                    continue
                synthesize(text, speaker, speed)
            except Exception as e:
                send({"error": str(e)})
    else:
        send({"error": "No input provided"})

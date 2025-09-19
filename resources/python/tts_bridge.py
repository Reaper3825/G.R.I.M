#!/usr/bin/env python3
import sys
import os
import json
import tempfile
import time
from pathlib import Path
from TTS.api import TTS

# -------------------------------------------------
# Load model once into memory
# -------------------------------------------------
MODEL_NAME = "tts_models/en/vctk/vits"  # default; can make configurable
tts = TTS(MODEL_NAME, progress_bar=False, gpu=False)

# Ensure temp dir exists
temp_dir = Path(tempfile.gettempdir()) / "grim_tts"
temp_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Helper: safe JSON write
# -------------------------------------------------
def send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()

# -------------------------------------------------
# Helper: run synthesis
# -------------------------------------------------
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
# Mode 1: If JSON arrives on stdin, process line by line
# -------------------------------------------------
if not sys.stdin.isatty():
    for line in sys.stdin:
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
    sys.exit(0)

# -------------------------------------------------
# Mode 2: No stdin → fallback to CLI args
# -------------------------------------------------
if len(sys.argv) > 1:
    text = " ".join(sys.argv[1:])
    synthesize(text)
else:
    send({"error": "No input provided"})

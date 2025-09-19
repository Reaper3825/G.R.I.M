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
# Main loop: read JSON from stdin
# -------------------------------------------------
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

        # Unique filename
        ts = int(time.time() * 1000)
        out_path = temp_dir / f"grim_tts_{ts}.wav"

        # Run inference (speaker optional)
        if speaker:
            tts.tts_to_file(text=text, speaker=speaker, file_path=str(out_path))
        else:
            tts.tts_to_file(text=text, file_path=str(out_path))

        send({"file": str(out_path)})

    except Exception as e:
        send({"error": str(e)})

#!/usr/bin/env python3
import sys
import os
import json
import uuid
from pathlib import Path

# ------------------------------
# Portable model path (fix Resources double nesting)
# ------------------------------
os.environ["TTS_HOME"] = str(Path(__file__).resolve().parent.parent.parent / "Resources" / "models")

# ------------------------------
# JSON-safe output helpers
# ------------------------------
def send(obj):
    real_out = sys.__stdout__
    real_out.write(json.dumps(obj) + "\n")
    real_out.flush()

def debug(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

# ------------------------------
# Startup signal
# ------------------------------
send({"status": "LOADING"})

# ------------------------------
# Heavy imports
# ------------------------------
from TTS.api import TTS

MODEL_NAME = "tts_models/en/vctk/vits"
USE_GPU = False  # force CPU

debug(f"[bridge] Initializing model {MODEL_NAME} (gpu={USE_GPU})")
tts = TTS(MODEL_NAME, progress_bar=False, gpu=USE_GPU)
debug(f"[bridge] Loaded {MODEL_NAME}")

# Fetch available speakers
speakers = []
try:
    speakers = tts.speakers if hasattr(tts, "speakers") else []
except Exception as e:
    debug(f"[bridge-warning] Could not fetch speakers: {e}")

# Pick default
DEFAULT_SPEAKER = "p225" if "p225" in speakers else (speakers[0] if speakers else None)

# Output dir
out_dir = Path(__file__).resolve().parent.parent / "tts_out"
out_dir.mkdir(parents=True, exist_ok=True)

# Confirm ready
send({
    "status": "READY",
    "model": MODEL_NAME,
    "speakers": speakers,
    "default_speaker": DEFAULT_SPEAKER
})

# ------------------------------
# Helpers
# ------------------------------
def synthesize(text, speaker=None, speed=1.0):
    out_file = out_dir / f"{uuid.uuid4().hex}.wav"
    try:
        # Guarantee a speaker
        if not speaker:
            speaker = DEFAULT_SPEAKER or (speakers[0] if speakers else None)
        if not speaker:
            raise RuntimeError("No available speaker in model")

        debug(f"[bridge] Synthesizing text='{text}' speaker={speaker} speed={speed}")
        tts.tts_to_file(text=text, speaker=speaker, speed=speed, file_path=str(out_file))
        send({"file": str(out_file), "speaker": speaker})
    except Exception as e:
        debug(f"[bridge-error] {repr(e)}")
        send({"error": str(e)})

# ------------------------------
# Main loop
# ------------------------------
debug("[bridge] Mode: persistent stdin")
for raw_input in sys.stdin:
    line = raw_input.strip()
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
        debug(f"[bridge-error] {repr(e)}")
        send({"error": str(e)})

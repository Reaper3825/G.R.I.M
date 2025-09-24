#!/usr/bin/env python3
import sys
import os
import json
import uuid
import argparse
from pathlib import Path

# ------------------------------
# Environment / Model Path
# ------------------------------
os.environ["TTS_HOME"] = str(Path(__file__).resolve().parent.parent.parent / "Resources" / "models")

# Ensure UTF-8 for stdout/stderr
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

# ------------------------------
# Helpers
# ------------------------------
def send(obj: dict):
    """Send JSON object to stdout (for C++ to parse)."""
    try:
        real_out = sys.__stdout__
        real_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        real_out.flush()
    except Exception as e:
        debug(f"[bridge-error] Failed to send JSON: {e}")

def debug(msg: str):
    """Send debug/info messages to stderr (not parsed by C++)."""
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass

# ------------------------------
# CLI Args
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--oneshot", action="store_true", help="Run once then exit")
parser.add_argument("--text", type=str, help="Text to synthesize")
parser.add_argument("--speaker", type=str, default=None, help="Speaker ID")
parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
parser.add_argument("--out", type=str, default=None, help="Output wav path")
args, unknown = parser.parse_known_args()

# ------------------------------
# Startup signal
# ------------------------------
send({"status": "LOADING"})

# ------------------------------
# Heavy imports
# ------------------------------
try:
    from TTS.api import TTS
except Exception as e:
    debug(f"[bridge-error] Failed to import Coqui TTS: {e}")
    send({"error": f"import_failed: {e}"})
    sys.exit(1)

# ------------------------------
# Model Init
# ------------------------------
MODEL_NAME = "tts_models/en/vctk/vits"
USE_GPU = False  # force CPU for compatibility

try:
    debug(f"[bridge] Initializing model {MODEL_NAME} (gpu={USE_GPU})")
    tts = TTS(MODEL_NAME, progress_bar=False, gpu=USE_GPU)
    debug(f"[bridge] Loaded {MODEL_NAME}")
except Exception as e:
    debug(f"[bridge-error] Model load failed: {e}")
    send({"error": f"model_load_failed: {e}"})
    sys.exit(1)

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
# Synthesis function
# ------------------------------
def synthesize(text: str, speaker: str = None, speed: float = 1.0, out_file: Path = None):
    if out_file is None:
        out_file = out_dir / f"{uuid.uuid4().hex}.wav"
    try:
        # Guarantee a speaker
        if not speaker:
            speaker = DEFAULT_SPEAKER or (speakers[0] if speakers else None)
        if not speaker:
            raise RuntimeError("No available speaker in model")

        debug(f"[bridge] Synthesizing text='{text}' speaker={speaker} speed={speed}")
        tts.tts_to_file(text=text, speaker=speaker, speed=speed, file_path=str(out_file))

        if not out_file.exists():
            raise RuntimeError(f"TTS synthesis failed, file not created: {out_file}")

        send({"file": str(out_file), "speaker": speaker, "speed": speed, "done": True})
        debug(f"[bridge] Successfully wrote file {out_file}")
    except Exception as e:
        debug(f"[bridge-error] {repr(e)}")
        send({"error": str(e), "done": False})

# ------------------------------
# One-Shot Mode
# ------------------------------
if args.oneshot:
    debug("ONE-Shot mode Begin")
    if not args.text:
        send({"error": "no_text", "done": False})
        sys.exit(1)

    out_file = Path(args.out) if args.out else (out_dir / f"{uuid.uuid4().hex}.wav")
    synthesize(args.text, args.speaker, args.speed, out_file)
    sys.exit(0)   # <-- make sure we quit after one-shot


# ------------------------------
# Persistent stdin Mode (default)
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
            send({"error": "empty_text", "done": False})
            continue
        synthesize(text, speaker, speed)
    except Exception as e:
        debug(f"[bridge-error] {repr(e)}")
        send({"error": str(e), "done": False})

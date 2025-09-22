import sys
import json
import argparse
from TTS.api import TTS
import os

# ---------- Helpers ----------
def log(msg):
    """Log messages to stderr (never stdout)."""
    print(msg, file=sys.stderr, flush=True)

def send(obj):
    """Send JSON protocol messages to stdout."""
    print(json.dumps(obj), flush=True)

# ---------- Persistent Mode ----------
def persistent_loop(model_name, speaker):
    try:
        tts = TTS(model_name)
        log(f"Model loaded: {model_name}")
    except Exception as e:
        send({"status": "error", "message": str(e)})
        return

    send({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            log(f"Invalid JSON from GRIM: {line} ({e})")
            continue

        cmd = req.get("command")
        if cmd == "exit":
            send({"status": "bye"})
            break
        elif cmd == "speak":
            text = req.get("text", "")
            spk = req.get("speaker", speaker)
            speed = float(req.get("speed", 1.0))
            out_path = req.get("out", "output.wav")

            try:
                tts.tts_to_file(text=text, file_path=out_path, speaker=spk, speed=speed)
                send({"status": "ok", "file": out_path})
            except Exception as e:
                send({"status": "error", "message": str(e)})
        else:
            send({"status": "error", "message": f"Unknown command {cmd}"})


# ---------- One-shot Mode ----------
def oneshot_mode(args):
    try:
        tts = TTS(args.model)
        log(f"Model loaded (oneshot): {args.model}")
        tts.tts_to_file(
            text=args.text,
            file_path=args.out,
            speaker=args.speaker,
            speed=args.speed
        )
        send({"status": "ok", "file": args.out})
    except Exception as e:
        send({"status": "error", "message": str(e)})


# ---------- Entry ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent", action="store_true",
                        help="Run in persistent stdin/stdout mode")

    # 🔹 Use a multi-speaker model so p225, p226, etc. work
    parser.add_argument("--model",
                        default="tts_models/en/vctk/vits",
                        help="TTS model name or local path")

    parser.add_argument("--speaker", default="p225", help="Speaker ID")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--out", help="Output file path (oneshot only)")
    parser.add_argument("text", nargs="?", help="Text to speak (oneshot only)")
    args = parser.parse_args()


    if args.persistent:
        persistent_loop(args.model, args.speaker)
    else:
        if not args.text or not args.out:
            send({"status": "error", "message": "Oneshot mode requires text and --out"})
        else:
            oneshot_mode(args)

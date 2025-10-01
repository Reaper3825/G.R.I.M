
# =================================================================================================
# G.R.I.M. Wake Word Detector
#
# Description:
#   This script uses the openwakeword library to listen for a specific wake word from a
#   microphone input stream. When the wake word is detected, it outputs a JSON object
#   to standard output, which can be captured by the calling C++ application.
#
# Dependencies:
#   - openwakeword
#   - pyaudio
#   - numpy
#
# Installation:
#   pip install openwakeword pyaudio numpy
#
# Usage:
#   python wakeword_detector.py <model_name>
#
#   <model_name>: The name of the wake word model to use (e.g., "hey_jarvis").
#                 Models are downloaded automatically by the openwakeword library.
#
# Output Format:
#   The script outputs JSON objects to stdout in the following format upon detection:
#   {
#     "event": "wake_word_detected",
#     "model": "<model_name>",
#     "score": <detection_confidence_score>
#   }
#
# =================================================================================================

import sys
import json
import argparse
import traceback

try:
    import openwakeword
    from openwakeword.model import Model
    from openwakeword.vad import VAD
    import numpy as np
    import pyaudio
except ImportError as e:
    error_message = {
        "event": "error",
        "type": "import_error",
        "message": f"A required Python library is not installed: {e}. Please run 'pip install openwakeword pyaudio numpy'."
    }
    print(json.dumps(error_message), flush=True)
    sys.exit(1)

# --- Configuration ---
CHUNK_SIZE = 1280  # Samples per frame
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sample rate (required by openwakeword)

def main(model_name):
    """
    Initializes the wake word detector and starts the listening loop.
    """
    # Log to stderr to avoid polluting stdout for the parent process
    print(f"Initializing wake word detector with model '{model_name}'...", file=sys.stderr)
    sys.stderr.flush()

    try:
        # 1. Initialize PyAudio
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

        # 2. Initialize openwakeword Model
        # Use built-in pre-trained models
        owwModel = Model(wakeword_models=[model_name], inference_framework="tflite")
        vad = VAD()

        print("Wake word listener started. Listening for wake word...", file=sys.stderr)
        sys.stderr.flush()

        # 3. Main listening loop
        while True:
            # Read audio from the microphone
            audio_chunk = stream.read(CHUNK_SIZE)

            # Perform voice activity detection
            if vad.is_speech(audio_chunk):
                # Feed audio to openwakeword
                prediction = owwModel.predict(audio_chunk)

                # Check for wake word activation
                for model, score in prediction.items():
                    if score > 0.5:  # Confidence threshold
                        output = {
                            "event": "wake_word_detected",
                            "model": model,
                            "score": score
                        }
                        # Output JSON to stdout and flush to ensure parent process receives it immediately
                        print(json.dumps(output), flush=True)
                        # Reset after detection to avoid re-triggering
                        owwModel.reset()


    except Exception as e:
        error_message = {
            "event": "error",
            "type": "runtime_error",
            "message": f"An exception occurred: {e}",
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_message), flush=True)
        sys.exit(1)

    finally:
        # Clean up resources
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        if 'p' in locals():
            p.terminate()
        print("Wake word listener stopped.", file=sys.stderr)
        sys.stderr.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="G.R.I.M. Wake Word Detector")
    parser.add_argument("model_name", nargs='?', default="hey_jarvis",
                        help="The name of the wake word model to use (e.g., 'hey_jarvis').")

    args = parser.parse_args()

    main(args.model_name)

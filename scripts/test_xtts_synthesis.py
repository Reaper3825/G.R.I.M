# Test XTTS v2 synthesis to diagnose tuple index error

import sys
import os

# Set env before imports
os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')
os.environ['COQUI_TOS_AGREED'] = '1'

print("Importing TTS...", flush=True)
from TTS.api import TTS

print("Loading XTTS v2 model...", flush=True)
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda')

print("Model loaded successfully!", flush=True)

# Test with default voice
reference_audio = "D:/G.R.I.M/resources/voices/default.wav"
output_file = "D:/G.R.I.M/resources/tts_out/temp/test_xtts.wav"

print(f"Reference audio: {reference_audio}", flush=True)
print(f"Output file: {output_file}", flush=True)

if not os.path.exists(reference_audio):
    print(f"ERROR: Reference audio not found!", flush=True)
    sys.exit(1)

print("Attempting synthesis...", flush=True)

try:
    tts.tts_to_file(
        text="Hello, this is a test of XTTS version two.",
        file_path=output_file,
        speaker_wav=reference_audio,
        language="en"
    )
    print(f"SUCCESS! Generated: {output_file}", flush=True)
    
except IndexError as e:
    print(f"IndexError (tuple index out of range): {e}", flush=True)
    import traceback
    traceback.print_exc()
    
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()

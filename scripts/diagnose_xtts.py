# XTTS v2 Diagnostic Tool
# Checks for common issues causing "tuple index out of range"

import sys
import os

print("="*60)
print("XTTS v2 Diagnostic Tool")
print("="*60)

# Check 1: PyTorch version
print("\n[1/6] Checking PyTorch version...")
try:
    import torch
    print(f"  ? PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ? CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("  ? CUDA not available (using CPU)")
except Exception as e:
    print(f"  ? ERROR: {e}")
    sys.exit(1)

# Check 2: TTS installation
print("\n[2/6] Checking TTS installation...")
try:
    import TTS
    from TTS.api import TTS as TTS_API
    print(f"  ? TTS library installed")
except Exception as e:
    print(f"  ? ERROR: {e}")
    print("  Run: pip install TTS>=0.22.0")
    sys.exit(1)

# Check 3: Reference audio exists
print("\n[3/6] Checking reference audio...")
ref_path = "D:/G.R.I.M/resources/voices/default.wav"
if not os.path.exists(ref_path):
    print(f"  ? ERROR: default.wav not found at {ref_path}")
    print("  Run: .\\scripts\\download_default_voice.ps1")
    sys.exit(1)
else:
    print(f"  ? Reference audio exists")

# Check 4: Validate reference audio
print("\n[4/6] Validating reference audio...")
try:
    import soundfile as sf
    data, sr = sf.read(ref_path)
    duration = len(data) / sr
    print(f"  ? Duration: {duration:.2f}s")
    print(f"  ? Sample rate: {sr}Hz")
    print(f"  ? Samples: {len(data)}")
    
    if duration < 3.0:
        print(f"  ? WARNING: Audio is short ({duration:.2f}s). XTTS v2 works best with 3-30s.")
    
    # Check if stereo (should be mono)
    if len(data.shape) > 1:
        print(f"  ? WARNING: Audio is stereo. Converting to mono recommended.")
        
except Exception as e:
    print(f"  ? ERROR: {e}")
    sys.exit(1)

# Check 5: Monkey-patch torch.load
print("\n[5/6] Applying PyTorch 2.6+ compatibility patch...")
_original_load = torch.load

def _patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load
print("  ? torch.load patched to use weights_only=False")

# Check 6: Load XTTS v2 model
print("\n[6/6] Loading XTTS v2 model...")
try:
    os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')
    os.environ['COQUI_TOS_AGREED'] = '1'
    
    print("  Loading model (this may take 30-60s)...", flush=True)
    tts = TTS_API('tts_models/multilingual/multi-dataset/xtts_v2')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts = tts.to(device)
    
    print(f"  ? Model loaded on {device}")
    
    # Try synthesis
    print("\n[BONUS] Testing synthesis...")
    output_path = "D:/G.R.I.M/resources/tts_out/temp/diagnostic_test.wav"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print("  Generating audio...", flush=True)
    tts.tts_to_file(
        text="This is a diagnostic test of XTTS version two.",
        file_path=output_path,
        speaker_wav=ref_path,
        language="en"
    )
    
    print(f"  ? SUCCESS! Generated: {output_path}")
    
    # Check output
    if os.path.exists(output_path):
        out_data, out_sr = sf.read(output_path)
        out_duration = len(out_data) / out_sr
        print(f"  ? Output duration: {out_duration:.2f}s at {out_sr}Hz")
    
except IndexError as e:
    print(f"  ? TUPLE INDEX ERROR: {e}")
    print("\nPossible causes:")
    print("  1. Reference audio is corrupted")
    print("  2. PyTorch/CUDA version mismatch")
    print("  3. TTS version incompatibility")
    print("\nTry:")
    print("  pip install --force-reinstall TTS>=0.22.0")
    print("  .\\scripts\\download_default_voice.ps1")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    
except Exception as e:
    print(f"  ? ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("All checks passed! XTTS v2 is working correctly.")
print("="*60)

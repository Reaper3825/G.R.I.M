# Test XTTS v2 with Cached Embedding
# This should be MUCH faster (0.5-1s instead of 60s)

import sys
import os
import time

os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')
os.environ['COQUI_TOS_AGREED'] = '1'

# Monkey-patch torch.load
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

print("Loading XTTS v2 model...")
from TTS.api import TTS
import numpy as np

tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda')
print("Model loaded!")

# Load the cached embedding
embedding_path = "D:/G.R.I.M/resources/voices/embeddings/default.npz"

if not os.path.exists(embedding_path):
    print(f"ERROR: Embedding not found at {embedding_path}")
    print("Run: python scripts/create_default_embedding.py")
    sys.exit(1)

print(f"\nLoading cached embedding: {embedding_path}")
data = np.load(embedding_path)
gpt_cond_latent = torch.from_numpy(data['gpt_cond_latent']).to('cuda')
speaker_embedding = torch.from_numpy(data['speaker_embedding']).to('cuda')

print("Embedding loaded successfully!")

# Test synthesis with cached embedding
text = "This is a test using the cached speaker embedding. It should be very fast!"
output_file = "D:/G.R.I.M/resources/tts_out/temp/embedding_test.wav"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

print(f"\nSynthesizing: {text}")
print("Starting timer...")

start = time.time()

# Direct synthesis with cached embedding
model = tts.synthesizer.tts_model
wav = model.inference(
    text=text,
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding
)

# Save
import soundfile as sf

# XTTS returns a 1D array, soundfile expects (samples,) or (samples, channels)
if isinstance(wav, dict):
    # Some models return a dict with 'wav' key
    wav = wav['wav']

# Ensure it's a numpy array
if torch.is_tensor(wav):
    wav = wav.cpu().numpy()

# Reshape if needed
if len(wav.shape) == 1:
    # Already correct shape for mono
    pass
elif len(wav.shape) == 2 and wav.shape[0] == 1:
    # (1, samples) -> (samples,)
    wav = wav.squeeze()

sf.write(output_file, wav, model.config.audio.sample_rate)

elapsed = time.time() - start

print(f"\n? SUCCESS!")
print(f"? Time: {elapsed:.2f} seconds")
print(f"? Output: {output_file}")

if elapsed < 2.0:
    print(f"\n?? EXCELLENT! Embedding cache is working perfectly!")
    print(f"   Synthesis was {60/elapsed:.1f}x faster than the slow method!")
elif elapsed < 5.0:
    print(f"\n? Good! Embedding is working, but slightly slower than expected")
else:
    print(f"\n? Warning: Still slow ({elapsed:.1f}s). Embedding might not be working correctly.")

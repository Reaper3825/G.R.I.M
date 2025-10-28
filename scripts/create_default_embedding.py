# Create Default Speaker Embedding for Faster TTS
# Run this ONCE to create a cached embedding for the default voice

import sys
import os

# Set environment variables
os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')
os.environ['COQUI_TOS_AGREED'] = '1'

print("="*60)
print("Creating Default Speaker Embedding")
print("="*60)

# Monkey-patch torch.load for PyTorch 2.6+ compatibility
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

print("\nLoading XTTS v2 model...")
from TTS.api import TTS
tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to('cuda')

print("Model loaded successfully!")

# Paths
ref_audio = "D:/G.R.I.M/resources/voices/default.wav"
output_dir = "D:/G.R.I.M/resources/voices/embeddings"
output_file = os.path.join(output_dir, "default.npz")

if not os.path.exists(ref_audio):
    print(f"\nERROR: Reference audio not found: {ref_audio}")
    sys.exit(1)

os.makedirs(output_dir, exist_ok=True)

print(f"\nReference audio: {ref_audio}")
print(f"Output: {output_file}")

print("\nComputing speaker embedding...")
print("(This will take 30-60 seconds the first time)")

try:
    # Access the XTTS model directly
    model = tts.synthesizer.tts_model
    
    # Compute conditioning latents (this is what takes time)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=ref_audio
    )
    
    print("Embedding computed successfully!")
    
    # Save to disk
    import numpy as np
    
    gpt_cond_np = gpt_cond_latent.cpu().numpy()
    spk_emb_np = speaker_embedding.cpu().numpy()
    
    np.savez(
        output_file,
        gpt_cond_latent=gpt_cond_np,
        speaker_embedding=spk_emb_np
    )
    
    file_size_kb = os.path.getsize(output_file) / 1024
    
    print(f"\n? SUCCESS!")
    print(f"? Embedding saved: {output_file}")
    print(f"? Size: {file_size_kb:.2f} KB")
    print(f"\nNow all TTS requests will use this cached embedding!")
    print("Expected speed: ~0.5-1 second instead of 60+ seconds")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

"""Inspect XTTS v2 model structure to understand the architecture."""

import os
import torch

os.environ['NUMBA_CACHE_DIR'] = os.path.expanduser('~/.numba_cache')

# PyTorch 2.6+ compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.api import TTS

print("Loading XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")
model = tts.synthesizer.tts_model

print("\n" + "="*60)
print("XTTS v2 Model Structure:")
print("="*60)

print("\nModel type:", type(model).__name__)
print("\nModel attributes:")
for attr in dir(model):
    if not attr.startswith('_'):
        obj = getattr(model, attr)
        if isinstance(obj, torch.nn.Module):
            print(f"  - {attr}: {type(obj).__name__}")

print("\n" + "="*60)
print("Submodules:")
print("="*60)
for name, module in model.named_children():
    print(f"\n{name}:")
    print(f"  Type: {type(module).__name__}")
    if hasattr(module, 'forward'):
        print(f"  Has forward method: Yes")

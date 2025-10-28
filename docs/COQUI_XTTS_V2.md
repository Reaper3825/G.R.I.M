# Coqui XTTS v2 Integration Guide

## Overview

G.R.I.M now uses **Coqui XTTS v2**, a state-of-the-art text-to-speech system with:

- **Voice Cloning**: Clone any voice from a short audio sample
- **Multilingual Support**: 17 languages supported
- **High Quality**: Natural-sounding speech synthesis
- **GPU Acceleration**: Fast synthesis with NVIDIA CUDA
- **Streaming Capable**: Real-time audio generation

## What Changed from Old Coqui

### Old System (VCTK/VITS)
- Model: `tts_models/en/vctk/vits`
- Fixed speaker IDs (p225, p226, etc.)
- English only
- CPU-based

### New System (XTTS v2)
- Model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Voice cloning from reference audio
- 17 languages supported
- GPU accelerated (optional)
- Higher quality output

## Installation

### 1. Run Setup Script

```powershell
cd D:\G.R.I.M\scripts
.\setup_coqui_xtts_v2.ps1
```

This will:
- Install PyTorch with CUDA support (if GPU available)
- Install Coqui TTS v0.22+
- Download XTTS v2 model (~1.8GB)
- Verify installation

### 2. Clean Old Data (Optional)

```powershell
.\cleanup_old_coqui.ps1
```

This removes old model files and temp audio.

## Configuration

Edit `D:\G.R.I.M\resources\ai_config.json`:

```json
{
  "voice": {
    "engine": "coqui",
    "speaker": "default",
    "speed": 1.0,
    "language": "en",
    "output_dir": "D:/G.R.I.M/resources/tts_out"
  }
}
```

### Supported Languages

XTTS v2 supports:
- `en` - English
- `es` - Spanish
- `fr` - French
- `de` - German
- `it` - Italian
- `pt` - Portuguese
- `pl` - Polish
- `tr` - Turkish
- `ru` - Russian
- `nl` - Dutch
- `cs` - Czech
- `ar` - Arabic
- `zh-cn` - Chinese (Mandarin)
- `hu` - Hungarian
- `ko` - Korean
- `ja` - Japanese
- `hi` - Hindi

## Voice Cloning

### Adding Custom Voices

1. Record a 6-30 second audio sample of the target voice
2. Save as WAV file (16kHz, mono recommended)
3. Update `coqui_bridge.py`:

```python
VOICE_REFERENCES = {
    "default": None,
    "custom_voice": "D:/G.R.I.M/resources/voices/my_voice.wav",
}
```

4. Use in code:
```cpp
Voice::coquiSpeak("Hello world", "custom_voice", 1.0);
```

### Runtime Voice Updates

Send command to bridge:
```json
{
  "command": "set_voice_reference",
  "speaker": "new_voice",
  "reference_path": "path/to/audio.wav"
}
```

### ? Speaker Embeddings (NEW)

**Cache voice profiles for 10x faster synthesis!**

Speaker embeddings pre-compute and cache the voice analysis, making subsequent TTS requests much faster.

#### Create Embedding

```
create_embedding austin D:/G.R.I.M/resources/voices/my_voice.wav
```

#### List Cached Embeddings

```
list_embeddings
```

#### Performance

| Method | Synthesis Time |
|--------|---------------|
| Without embedding | ~5-8 seconds |
| With cached embedding | **~0.5-1 second** ? |

#### Usage

Once created, just use the speaker ID in config:
```json
{
  "voice": {
    "speaker": "austin"  // Uses cached embedding automatically
  }
}
```

**See [SPEAKER_EMBEDDINGS.md](SPEAKER_EMBEDDINGS.md) for full documentation.**

## Performance Optimization

### GPU Acceleration

XTTS v2 benefits significantly from GPU:
- **CPU**: ~5-10 seconds per sentence
- **GPU**: ~0.5-2 seconds per sentence

#### Enable GPU:
```powershell
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Check GPU Usage:
```powershell
nvidia-smi
```

### Caching

XTTS v2 uses the same cache system:
- **Permanent cache**: System messages, greetings
- **Temp cache**: Dynamic responses (cleared after 24h)
- **Cache location**: `D:/G.R.I.M/resources/tts_out/cache/`

### Model Loading

First launch takes 30-60 seconds to load model. Use persistent mode:
```cpp
// Launched automatically in Voice::initTTS()
// Bridge stays alive, model stays in memory
```

## API Changes

### C++ Interface

```cpp
// Old (still works)
Voice::coquiSpeak("Hello", "p225", 1.0);

// New (recommended)
Voice::coquiSpeak("Hello", "default", 1.0);

// Set language
Voice::setLanguage("es");  // Switch to Spanish
Voice::coquiSpeak("Hola mundo", "default", 1.0);

// Check if XTTS v2 is loaded
if (Voice::isXTTSv2Enabled()) {
    // XTTS v2 specific features available
}
```

### Python Bridge Commands

```json
// Speak command
{
  "command": "speak",
  "text": "Hello world",
  "speaker": "default",
  "speed": 1.0,
  "language": "en",
  "out": "output.wav"
}

// List supported languages
{
  "command": "list_languages"
}

// Set voice reference
{
  "command": "set_voice_reference",
  "speaker": "my_voice",
  "reference_path": "path/to/sample.wav"
}
```

## Troubleshooting

### Model Download Fails
```powershell
# Manually download model
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### GPU Not Detected
```powershell
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### ? PyTorch 2.6+ "Weights Only Load Failed" Error

**Symptom:** 
```
_pickle.UnpicklingError: Weights only load failed...
GLOBAL TTS.tts.configs.xtts_config.XttsConfig was not an allowed global
```

**Cause:** PyTorch 2.6+ changed default `weights_only=True`, blocking TTS model loading.

**Fix:** The `coqui_bridge.py` now automatically patches `torch.load` to use `weights_only=False`. If you still see this error:

```powershell
# Option 1: Downgrade PyTorch (if needed)
pip install "torch<2.6.0" "torchaudio<2.6.0"

# Option 2: Verify the patch is loaded
python -c "from resources.python.coqui_bridge import *; print('Patch loaded')"
```

### ? "Tuple Index Out of Range" Error

**Symptom:**
```
[ERROR][Voice/Bridge] Coqui TTS error: tuple index out of range
```

**Causes & Fixes:**

1. **Reference audio too short** (< 3 seconds)
   ```powershell
   # Check duration
   python -c "import soundfile as sf; d,sr=sf.read('resources/voices/default.wav'); print(f'{len(d)/sr:.1f}s')"
   
   # Re-download default voice
   .\scripts\download_default_voice.ps1
   ```

2. **Corrupted reference audio**
   ```powershell
   # Re-create default voice
   Remove-Item resources\voices\default.wav
   .\scripts\download_default_voice.ps1
   ```

3. **Missing speaker reference**
   ```powershell
   # Ensure default.wav exists
   Test-Path resources\voices\default.wav
   
   # If false, run setup
   .\scripts\setup_coqui_xtts_v2.ps1
   ```

4. **PyTorch/CUDA version mismatch**
   ```powershell
   # Check versions
   python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   
   # Reinstall if needed
   pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Slow Synthesis
- Enable GPU acceleration
- Check model is cached (first run is slower)
- Reduce text length (XTTS v2 is slower for very long text)
- ? Use speaker embeddings for 10x faster synthesis

### Low Quality Output
- Check language setting matches text language
- Use appropriate speed (0.8-1.2 recommended)
- Ensure reference audio is high quality for voice cloning
- Reference audio should be 6-30 seconds of clear speech

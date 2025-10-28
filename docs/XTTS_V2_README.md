# Coqui XTTS v2 Upgrade - README

## ?? What's New

G.R.I.M now uses **Coqui XTTS v2** - a cutting-edge text-to-speech system featuring:

- ?? **Voice Cloning** - Clone any voice from a short audio sample
- ?? **17 Languages** - Multi-language support (English, Spanish, French, German, and more)
- ?? **GPU Accelerated** - 10x faster synthesis with NVIDIA CUDA
- ?? **Premium Quality** - State-of-the-art neural speech synthesis
- ?? **Smart Caching** - Instant playback for repeated phrases

## ? Quick Start

### 1. Install (one command)
```powershell
cd D:\G.R.I.M\scripts
.\setup_coqui_xtts_v2.ps1
```

### 2. Update Config
Edit `D:\G.R.I.M\resources\ai_config.json`:
```json
{
  "voice": {
    "engine": "coqui",
    "speaker": "default",
    "language": "en"
  }
}
```

### 3. Build & Test
```powershell
cd D:\G.R.I.M\out\build
cmake --build . --config Release
```

Launch G.R.I.M and test:
```
test_tts Hello, this is XTTS version two!
```

## ?? Performance

| Mode | Speed per Phrase | Quality |
|------|-----------------|---------|
| XTTS v2 + GPU | 0.5-2s | ????? |
| XTTS v2 + CPU | 5-10s | ????? |
| Old Coqui | 3-5s | ???? |
| Cached | < 0.1s | ????? |

## ?? Supported Languages

English, Spanish, French, German, Italian, Portuguese, Polish, Turkish, Russian, Dutch, Czech, Arabic, Chinese, Hungarian, Korean, Japanese, Hindi

## ?? New C++ API

```cpp
// Basic usage (same as before)
Voice::coquiSpeak("Hello world", "default", 1.0);

// Change language
Voice::setLanguage("es");
Voice::coquiSpeak("Hola mundo", "default", 1.0);

// Check if XTTS v2 enabled
if (Voice::isXTTSv2Enabled()) {
    // Use XTTS v2 features
}
```

## ?? What Changed

### Modified Files
- `resources/python/coqui_bridge.py` - Upgraded to XTTS v2
- `voice/voice_speak.cpp` - Added language support
- `voice/voice_speak.hpp` - New API functions
- `commands/commands_voice.cpp` - Updated commands
- `voice/tts_cache.cpp` - Enhanced caching

### New Files
- `resources/python/requirements.txt` - Python dependencies
- `scripts/setup_coqui_xtts_v2.ps1` - Auto-installer
- `scripts/cleanup_old_coqui.ps1` - Cleanup utility
- `docs/COQUI_XTTS_V2.md` - Full documentation
- `docs/XTTS_V2_MIGRATION.md` - Migration guide
- `docs/XTTS_V2_QUICK_REF.md` - Quick reference

### CMake
**No changes!** Coqui is pure Python (pip managed).

## ? Backward Compatibility

**100% compatible** with existing code:
- Old speaker IDs still work
- No API changes required
- Existing cache preserved
- Fallback to legacy models supported

## ?? Troubleshooting

### Model won't download?
```powershell
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### GPU not detected?
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Check GPU usage:
```powershell
nvidia-smi
```

## ?? Documentation

- **Full Guide**: [COQUI_XTTS_V2.md](./COQUI_XTTS_V2.md)
- **Migration**: [XTTS_V2_MIGRATION.md](./XTTS_V2_MIGRATION.md)
- **Quick Ref**: [XTTS_V2_QUICK_REF.md](./XTTS_V2_QUICK_REF.md)
- **Summary**: [UPGRADE_SUMMARY.md](./UPGRADE_SUMMARY.md)

## ?? Next Steps

1. ? Run `setup_coqui_xtts_v2.ps1`
2. ? Update `ai_config.json`
3. ? Test with `test_tts`
4. ? Verify GPU usage (if applicable)
5. ? Clean old data with `cleanup_old_coqui.ps1`
6. ?? Explore multi-language and voice cloning!

## ?? Pro Tips

- **First launch takes 30-60s** (model loads once)
- **Use GPU for 10x speedup** (requires NVIDIA CUDA)
- **Cache hits are instant** (system remembers common phrases)
- **Try different languages** with `Voice::setLanguage()`
- **Custom voices via voice cloning** (see docs)

## ?? Rollback

If needed, revert to old Coqui:
1. Edit `voice_speak.cpp` line 150
2. Change model to `tts_models/en/vctk/vits`
3. Rebuild

## ?? Success!

You now have state-of-the-art TTS with:
- Premium voice quality
- Multi-language support
- Voice cloning capability
- GPU acceleration
- Smart caching

Enjoy the upgrade! ??

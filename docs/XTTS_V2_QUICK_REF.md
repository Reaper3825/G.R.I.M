# XTTS v2 Quick Reference

## Installation
```powershell
cd D:\G.R.I.M\scripts
.\setup_coqui_xtts_v2.ps1
```

## Configuration
```json
// ai_config.json
{
  "voice": {
    "engine": "coqui",
    "speaker": "default",
    "speed": 1.0,
    "language": "en"
  }
}
```

## C++ API

```cpp
// Basic usage
Voice::coquiSpeak("Hello world", "default", 1.0);

// Change language
Voice::setLanguage("es");
Voice::coquiSpeak("Hola", "default", 1.0);

// Check XTTS v2 status
if (Voice::isXTTSv2Enabled()) {
    // XTTS v2 features available
}

// Get current language
std::string lang = Voice::getLanguage();
```

## Python Bridge Commands

```json
// Speak
{"command": "speak", "text": "Hello", "speaker": "default", "speed": 1.0, "language": "en", "out": "output.wav"}

// List languages
{"command": "list_languages"}

// Set voice reference
{"command": "set_voice_reference", "speaker": "my_voice", "reference_path": "voice.wav"}

// Exit
{"command": "exit"}
```

## Supported Languages
en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi

## Performance
- **First run**: 30-60s (model load)
- **GPU**: 0.5-2s per phrase
- **CPU**: 5-10s per phrase
- **Cached**: instant

## Troubleshooting

### Model not loading
```powershell
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### GPU not detected
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Check GPU usage
```powershell
nvidia-smi
```

## Files Changed
- `resources/python/coqui_bridge.py` - XTTS v2 bridge
- `voice/voice_speak.cpp` - C++ integration
- `voice/voice_speak.hpp` - API additions
- `commands/commands_voice.cpp` - Command updates

## New Files
- `resources/python/requirements.txt` - Python dependencies
- `scripts/setup_coqui_xtts_v2.ps1` - Setup script
- `scripts/cleanup_old_coqui.ps1` - Cleanup script
- `docs/COQUI_XTTS_V2.md` - Full documentation
- `docs/XTTS_V2_MIGRATION.md` - Migration guide

## Cleanup Old Version
```powershell
cd D:\G.R.I.M\scripts
.\cleanup_old_coqui.ps1
```

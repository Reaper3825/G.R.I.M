# Speaker Switching - How It Works

## Overview

G.R.I.M's Coqui XTTS v2 integration supports **dynamic speaker switching** without requiring bridge restarts. This document explains the architecture and data flow.

## Architecture

### Components

1. **Settings Menu** (`ui/ui_settings_menu.cpp`)
   - Provides "Speaker" button that cycles through available speakers
   - Updates `ai_config.json` when saved
   - Calls `Voice::setSpeaker()` to update runtime variable

2. **Voice Module** (`voice/voice_speak.cpp`)
   - Maintains `g_speaker` runtime variable
   - Reads initial speaker from config at startup
   - Can be updated dynamically via `Voice::setSpeaker()`

3. **Coqui Bridge** (`resources/python/coqui_bridge.py`)
   - Runs persistently with XTTS v2 model loaded
   - Accepts `speaker` parameter per TTS request
   - Loads appropriate embedding on-demand

## Data Flow

### Startup
```
1. voice_speak.cpp::initTTS()
   ├─> Reads ai_config.json
   ├─> Sets g_speaker = "p226" (or default)
   └─> Launches coqui_bridge.py with model

2. coqui_bridge.py::persistent_loop()
   ├─> Loads XTTS v2 model once
   ├─> Warms up with default embedding
   └─> Waits for TTS requests
```

### Runtime Speaker Change
```
User: Settings > Speaker: p226 -> default > Save & Close

1. UISettingsMenu::applyChanges()
   ├─> Updates pendingConfig["voice"]["speaker"] = "default"
   ├─> Calls saveConfig() → writes ai_config.json
   └─> Calls Voice::setSpeaker("default")

2. Voice::setSpeaker("default")
   └─> Updates g_speaker = "default"

3. Next TTS Request
   └─> coquiSpeak(text, g_speaker, g_speed)
       └─> Sends {"speaker": "default", ...} to bridge

4. coqui_bridge.py
   ├─> Receives speaker = "default"
   ├─> Loads embedding from embeddings/default.npz
   └─> Synthesizes with new voice
```

## No Restart Required!

### Why This Works

XTTS v2 is a **multi-speaker model** that uses **speaker embeddings** to clone voices. The model itself doesn't need to change - only the embedding does.

**Key Points:**
- ✅ Model loads **once** at startup (slow, ~10-30s)
- ✅ Speaker embeddings load **per request** (fast, <0.1s)
- ✅ Changing speakers = swapping embeddings, not models
- ✅ Bridge stays alive across speaker changes

### What Requires Restart

Only changing the **model** itself requires restarting the bridge:
- Switching from XTTS v2 to a different TTS model
- Changing from GPU to CPU mode
- Updating Python dependencies

## Config Structure

```json
{
  "voice": {
    "engine": "coqui",
    "speaker": "p226",                    // ← Current active speaker
    "available_speakers": [               // ← Speakers in cycle rotation
      "default",
      "p226"
    ],
    "speed": 1.5
  }
}
```

## Speaker Embedding Cache

Embeddings are stored in `resources/voices/embeddings/`:
```
embeddings/
├── default.npz     (130 KB)
├── p226.npz        (130 KB)
└── custom.npz      (130 KB)
```

**Loading Process:**
1. Bridge receives speaker ID (e.g., "p226")
2. Checks cache: `embeddings/p226.npz`
3. If exists → load from disk (0.1s)
4. If missing → compute from reference audio (5-8s, then cache)

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **Bridge startup** | 10-30s | Model load (one-time) |
| **Speaker change** | <0.1s | Embedding swap |
| **First use of speaker** | 5-8s | Computes embedding |
| **Subsequent uses** | 0.5-1s | Uses cached embedding |

## Voice References

Each speaker needs a reference audio file:
```python
VOICE_REFERENCES = {
    "default": "D:/G.R.I.M/resources/voices/default.wav",
    "p226": "D:/G.R.I.M/resources/voices/p226_reference.wav",
}
```

## Adding New Speakers

### Method 1: Command
```bash
create_embedding custom_voice D:/path/to/voice_sample.wav
```

### Method 2: Manual Setup

1. **Prepare audio** (10-15 seconds, clean speech)
   ```bash
   D:/G.R.I.M/resources/voices/my_voice.wav
   ```

2. **Run setup script**
   ```python
   python scripts/setup_speaker.py my_voice D:/G.R.I.M/resources/voices/my_voice.wav
   ```

3. **Update config**
   ```json
   {
     "voice": {
       "available_speakers": ["default", "p226", "my_voice"]
     }
   }
   ```

4. **Use in UI**
   - Open Settings
   - Click "Speaker" to cycle to "my_voice"
   - Save & Close

## Technical Details

### Embedding Format
```python
# embeddings/p226.npz contains:
{
  'gpt_cond_latent': np.array([1, 32, 1024]),  # GPT conditioning
  'speaker_embedding': np.array([1, 512, 1])   # Speaker features
}
```

### Bridge Communication
```python
# Request
{
  "command": "speak",
  "text": "Hello world",
  "speaker": "p226",           # ← Dynamic per request
  "language": "en",
  "use_embedding": True         # Use cached embedding
}

# Response
{
  "status": "ok",
  "file": "D:/G.R.I.M/resources/tts_out/temp/abc123.wav"
}
```

## Troubleshooting

### Speaker not changing
**Check:**
1. Did you click "Save & Close" in settings?
2. Is the embedding file present in `embeddings/`?
3. Check logs for `Voice speaker updated to: <name>`

### Voice sounds wrong
**Possible causes:**
- Wrong reference audio used during embedding creation
- Reference audio quality is poor
- Speaker name mismatch in config vs embeddings

**Fix:**
```bash
# Delete old embedding
rm D:/G.R.I.M/resources/voices/embeddings/p226.npz

# Recreate with better reference
python scripts/setup_p226_speaker.py
```

### Slow synthesis after speaker change
**First use is slow** - XTTS v2 computes embedding from reference audio.
- First synthesis: 5-8s (computes + caches embedding)
- Subsequent: 0.5-1s (uses cached embedding)

**Solution:** Be patient on first use, it will cache automatically.

## Best Practices

1. **Pre-cache embeddings** - Run setup scripts before using speakers
2. **Quality matters** - Use clean, clear reference audio (10-15s)
3. **Test before deploying** - Verify embeddings with `test_tts` command
4. **Keep references** - Don't delete original reference audio files
5. **Name clearly** - Use descriptive speaker IDs (e.g., "formal_male", "casual_female")

## Summary

✅ **Speaker changes are instant** - no bridge restart needed
✅ **XTTS v2 handles it** - model supports infinite speakers via embeddings  
✅ **UI makes it easy** - cycle through speakers with one button
✅ **Embeddings cache** - first use is slow, subsequent uses are fast
✅ **Config persists** - changes save to `ai_config.json`

---

**Last Updated:** November 3, 2025  
**Related Docs:** `SPEAKER_EMBEDDINGS.md`, `COQUI_XTTS_V2.md`

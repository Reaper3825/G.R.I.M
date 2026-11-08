# XTTS v2 Speaker Embeddings

## Overview

Speaker embeddings allow you to **create and cache voice profiles** for faster and more consistent voice cloning in XTTS v2.

### Benefits
- ? **10x faster synthesis** - Embeddings are pre-computed and reused
- ?? **Consistent voice quality** - Same embedding = same voice every time
- ?? **Efficient storage** - ~100KB per voice profile
- ?? **Easy management** - Simple commands to create, list, and delete

## How It Works

1. **First synthesis** - XTTS v2 analyzes reference audio to create an embedding
2. **Embedding cached** - Stored in `resources/voices/embeddings/`
3. **Subsequent uses** - Pre-computed embedding is loaded instantly
4. **Faster TTS** - Skip audio processing, go straight to synthesis

## Commands

### Create Embedding

```
create_embedding <speaker_id> <path/to/voice.wav>
```

**Example:**
```
create_embedding austin D:/G.R.I.M/resources/voices/my_voice.wav
```

Creates a cached voice embedding from your audio file.

### List Embeddings

```
list_embeddings
```

Shows all cached speaker embeddings with file sizes.

### ? Select Embedding (UI Method - NEW!)

Instead of editing config files, use the **Settings Menu**:

1. Open settings: `settings` command or press `~`
2. Find **Speaker** dropdown (under Voice section)
3. Select your speaker from the list
4. Click **Save & Close**

**See [SPEAKER_UI_SETTINGS.md](SPEAKER_UI_SETTINGS.md) for UI guide.**

### Use Embedding (Config Method)

Alternatively, set the speaker in `ai_config.json`:

```json
{
  "voice": {
    "speaker": "austin"
  }
}
```

G.R.I.M will automatically use the cached embedding.

## Technical Details

### Embedding Format

Embeddings are stored as NumPy `.npz` files containing:
- `gpt_cond_latent` - GPT conditioning vectors (512-dim)
- `speaker_embedding` - Speaker identity embedding (192-dim)

### Storage Location

```
D:/G.R.I.M/resources/voices/embeddings/
??? default.npz
??? austin.npz
??? custom_voice.npz
```

### Performance

| Method | First Synthesis | Cached Synthesis |
|--------|----------------|------------------|
| **No embedding** | ~5-8s | ~5-8s (recomputes every time) |
| **With embedding** | ~5-8s (+ cache) | **~0.5-1s** ? |

### Voice Reference Requirements

- **Format:** WAV, MP3, or FLAC
- **Duration:** 3-30 seconds recommended
- **Quality:** Clean, clear speech (no music/noise)
- **Content:** Any speech works (doesn't need to match TTS text)

## API Usage (Python Bridge)

The Coqui bridge supports these commands:

### Create Embedding
```json
{
  "command": "create_embedding",
  "speaker": "speaker_id",
  "reference_path": "/path/to/voice.wav"
}
```

### List Embeddings
```json
{
  "command": "list_embeddings"
}
```

**Response:**
```json
{
  "status": "ok",
  "embeddings": [
    {
      "speaker": "default",
      "path": "...",
      "size_kb": 98.5,
      "modified": 1698765432
    }
  ],
  "count": 1
}
```

### Delete Embedding
```json
{
  "command": "delete_embedding",
  "speaker": "speaker_id"
}
```

### Synthesis with Embedding
```json
{
  "command": "speak",
  "text": "Hello world",
  "speaker": "austin",
  "use_embedding": true,  // ? Enable cached embedding
  "language": "en"
}
```

Set `use_embedding: false` to force recomputation from reference audio.

## Best Practices

### 1. Use High-Quality Reference Audio
- Record in quiet environment
- Speak naturally and clearly
- 10-15 seconds is ideal

### 2. Test Before Caching
```
test_tts Hello, this is a voice test.
```

If it sounds good, create the embedding.

### 3. Name Embeddings Meaningfully
```
create_embedding austin_formal resources/voices/formal_speech.wav
create_embedding austin_casual resources/voices/casual_chat.wav
```

### 4. Clean Up Unused Embeddings
Embeddings take ~100KB each. Delete unused ones to save space.

## Troubleshooting

### "Model does not support embeddings"
- **Cause:** Not using XTTS v2 model
- **Fix:** Check `ai_config.json` - ensure model is `tts_models/multilingual/multi-dataset/xtts_v2`

### "Failed to save embedding"
- **Cause:** Missing numpy or write permissions
- **Fix:** Run `pip install numpy` and check folder permissions

### "Embedding not found"
- **Cause:** Speaker ID doesn't match filename
- **Fix:** Run `list_embeddings` to see available speakers

### Poor Voice Quality with Embedding
- **Cause:** Low-quality reference audio
- **Fix:** Re-record reference with better quality, create new embedding

## Advanced: Custom Embedding Directory

Edit `coqui_bridge.py`:

```python
EMBEDDING_DIR = "D:/custom/path/embeddings"
```

Then restart G.R.I.M.

## Integration with TTS Cache

Embeddings work seamlessly with the TTS cache:

1. **First synthesis:** Creates embedding + caches TTS output
2. **Same text again:** Uses cached WAV file (instant)
3. **New text, same voice:** Uses cached embedding (fast synthesis)

This gives you the best of both worlds! ??

# Speaker Embedding - Complete Feature Set

## ?? Quick Reference

### Create Voice Profile
```
create_embedding <name> <audio_file>
```

### List All Voices
```
list_embeddings
```

### Select Voice (UI)
```
settings ? Speaker dropdown ? Select ? Save
```

### Select Voice (Config)
```json
{ "voice": { "speaker": "name" } }
```

## ?? Performance

| Method | Time |
|--------|------|
| Without embedding | ~5-8s |
| With embedding | ~0.5-1s |
| **Speedup** | **10x faster** ? |

## ??? File Locations

```
resources/
??? voices/
?   ??? default.wav              ? Reference audio
?   ??? your_voice.wav           ? Your recordings
?   ??? embeddings/              ? Cached profiles
?       ??? default.npz          (100KB each)
?       ??? custom.npz
```

## ?? Commands

| Command | Description |
|---------|-------------|
| `create_embedding <id> <file>` | Create voice profile |
| `list_embeddings` | Show cached profiles |
| `settings` | Open UI (select via dropdown) |

## ??? UI Quick Steps

1. Type `settings`
2. Click **Speaker** dropdown
3. Select voice
4. Click **Save & Close**

## ?? Best Practices

? **DO:**
- Use 10-15 second audio samples
- Record in quiet environment
- Test with `test_tts` first
- Name descriptively (`austin_formal`)

? **DON'T:**
- Use noisy/low-quality audio
- Skip testing before caching
- Delete default embedding
- Forget to save settings

## ?? Troubleshooting

| Problem | Solution |
|---------|----------|
| No custom voices in dropdown | Run `create_embedding` |
| Voice doesn't change | Click "Save & Close" in settings |
| Poor quality | Use better reference audio |
| Can't find embedding | Check `embeddings/` folder |

## ?? Documentation

- **Full Guide:** `SPEAKER_EMBEDDINGS.md`
- **UI Guide:** `SPEAKER_UI_SETTINGS.md`
- **Quick Start:** `EMBEDDINGS_QUICKSTART.md`

## ?? That's It!

**Creating custom voices is now as simple as:**
```
create_embedding my_voice audio.wav
settings ? Speaker: my_voice ? Save
test_tts Hello in my voice!
```

**Enjoy your voice-cloned GRIM! ??**

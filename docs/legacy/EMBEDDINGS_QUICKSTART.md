# Speaker Embeddings - Quick Start

## What Are Speaker Embeddings?

Pre-computed voice profiles that make XTTS v2 synthesis **10x faster**.

## Basic Workflow

### 1. Create an Embedding

```bash
create_embedding my_voice resources/voices/sample.wav
```

### 2. List Embeddings

```bash
list_embeddings
```

### 3. Use It

Set in `ai_config.json`:
```json
{
  "voice": {
    "speaker": "my_voice"
  }
}
```

That's it! G.R.I.M will automatically use the cached embedding.

## Commands

| Command | Description |
|---------|-------------|
| `create_embedding <id> <file>` | Create voice profile from audio file |
| `list_embeddings` | Show all cached embeddings |

## Benefits

- ? **10x faster** - ~0.5s vs ~5s synthesis time
- ?? **Consistent** - Same voice every time
- ?? **Small** - ~100KB per voice
- ?? **Simple** - Create once, use forever

## Tips

- **Reference audio**: 10-15 seconds of clear speech
- **Quality matters**: Better source = better voice
- **Test first**: Use `test_tts` before creating embedding
- **Name clearly**: Use descriptive IDs like `austin_formal`

## Troubleshooting

### Command not found
**Fix:** Rebuild G.R.I.M after upgrading to XTTS v2

### Embedding creation fails
**Fix:** Ensure audio file exists and numpy is installed:
```bash
pip install numpy
```

### "Model does not support embeddings"
**Fix:** Verify XTTS v2 model in use:
```json
{
  "voice": {
    "engine": "coqui"  // Must be using Coqui
  }
}
```

## Full Documentation

See [SPEAKER_EMBEDDINGS.md](SPEAKER_EMBEDDINGS.md) for complete guide.

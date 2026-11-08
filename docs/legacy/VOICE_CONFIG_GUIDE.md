# Voice Recognition Configuration Guide

## Overview

This guide explains how to configure GRIM's voice recognition system to reduce hallucinations and improve accuracy.

## Configuration File

All voice settings are in `ai_config.json`, which is automatically created with defaults on first run.

## Key Configuration Sections

### 1. **Whisper Parameters** (Speech-to-Text)

Located at `ai_config.json ? whisper`:

```json
{
  "whisper": {
    "whisper_model": "ggml-base.en.bin",
    "temperature": 0.0,
    "beam_size": 5,
    "best_of": 5,
    "no_speech_threshold": 0.6,
    "min_speech_ms": 500,
    "min_silence_ms": 1200,
    "max_len": 1,
    "suppress_blank": true,
    "initial_prompt": "Voice commands: open notepad, close window, show time"
  }
}
```

#### Parameter Explanations:

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `temperature` | 0.0 | Controls randomness (0.0 = deterministic, 1.0 = creative) | **Keep at 0.0** for commands to reduce hallucinations |
| `beam_size` | 5 | Number of alternative paths to explore | Higher = more accurate but slower. Try 3-7 |
| `no_speech_threshold` | 0.6 | Probability threshold to reject noise/silence | Increase to 0.7-0.8 in noisy environments |
| `min_speech_ms` | 500 | Minimum speech duration to process (ms) | Increase to filter out accidental activations |
| `min_silence_ms` | 1200 | Silence duration before finalizing transcript | Decrease for faster response, increase if commands get cut off |
| `max_len` | 1 | Prefer shorter outputs | Keep at 1 for commands |
| `suppress_blank` | true | Filter empty/blank transcripts | Always true |
| `initial_prompt` | (see above) | Guide the model toward expected commands | Add your most common commands here |

### 2. **Voice Input Settings**

Located at `ai_config.json ? voice`:

```json
{
  "voice": {
    "silence_threshold": 0.02,
    "silence_timeout_ms": 1200,
    "confidence_threshold": 0.6,
    "filter_low_confidence": true,
    "require_multi_command_confirmation": true,
    "max_commands_per_input": 3,
    "input_device_index": -1
  }
}
```

#### Parameter Explanations:

| Parameter | Default | Description | Tuning Advice |
|-----------|---------|-------------|---------------|
| `silence_threshold` | 0.02 | RMS energy below this = silence | Increase in noisy environments (0.03-0.05) |
| `silence_timeout_ms` | 1200 | Max silence duration during speech | Increase if commands get cut off mid-sentence |
| `confidence_threshold` | 0.6 | Reject transcripts below this confidence | Increase to 0.7-0.8 to filter uncertain results |
| `filter_low_confidence` | true | Enable confidence filtering | Always true for production use |
| `require_multi_command_confirmation` | true | Ask before executing multiple commands | True for safety, false for convenience |
| `max_commands_per_input` | 3 | Maximum commands allowed in one input | Safety limit to prevent runaway execution |
| `input_device_index` | -1 | Mic device (-1 = system default) | Use specific index for multi-mic setups |

## Common Issues & Solutions

### Issue 1: Extra Commands Detected

**Symptom**: You say "open notepad" but system hears "open notepad, close window"

**Solutions**:
1. **Increase `temperature` to 0.0** (more deterministic)
2. **Increase `no_speech_threshold` to 0.7**
3. **Add your common commands to `initial_prompt`**
4. **Enable `require_multi_command_confirmation`**

**Example Config**:
```json
{
  "whisper": {
    "temperature": 0.0,
    "no_speech_threshold": 0.7,
    "initial_prompt": "Voice commands: open notepad, list directory, show time"
  },
  "voice": {
    "require_multi_command_confirmation": true
  }
}
```

### Issue 2: Commands Get Cut Off

**Symptom**: "open note..." gets finalized before you finish saying "notepad"

**Solutions**:
1. **Increase `min_silence_ms` to 1500-2000**
2. **Increase `silence_timeout_ms` to 2000**

**Example Config**:
```json
{
  "whisper": {
    "min_silence_ms": 1800
  },
  "voice": {
    "silence_timeout_ms": 2000
  }
}
```

### Issue 3: Background Noise Triggers Commands

**Symptom**: GRIM responds to TV, music, or other background sounds

**Solutions**:
1. **Increase `silence_threshold` to 0.04-0.06**
2. **Increase `no_speech_threshold` to 0.75**
3. **Increase `min_speech_ms` to 700**

**Example Config**:
```json
{
  "whisper": {
    "no_speech_threshold": 0.75,
    "min_speech_ms": 700
  },
  "voice": {
    "silence_threshold": 0.05
  }
}
```

### Issue 4: GRIM Doesn't Hear You

**Symptom**: You speak clearly but no response

**Solutions**:
1. **Decrease `silence_threshold` to 0.01**
2. **Decrease `min_speech_ms` to 300**
3. **Check microphone levels in Windows**

**Example Config**:
```json
{
  "whisper": {
    "min_speech_ms": 300
  },
  "voice": {
    "silence_threshold": 0.01
  }
}
```

## Recommended Configurations

### Quiet Environment (Office, Home)
```json
{
  "whisper": {
    "temperature": 0.0,
    "beam_size": 5,
    "no_speech_threshold": 0.6,
    "min_speech_ms": 500,
    "min_silence_ms": 1200
  },
  "voice": {
    "silence_threshold": 0.02,
    "confidence_threshold": 0.6
  }
}
```

### Noisy Environment (Open Office, Background Music)
```json
{
  "whisper": {
    "temperature": 0.0,
    "beam_size": 7,
    "no_speech_threshold": 0.75,
    "min_speech_ms": 700,
    "min_silence_ms": 1500
  },
  "voice": {
    "silence_threshold": 0.05,
    "confidence_threshold": 0.75
  }
}
```

### Fast Response (Reduce Latency)
```json
{
  "whisper": {
    "temperature": 0.0,
    "beam_size": 3,
    "min_speech_ms": 300,
    "min_silence_ms": 800
  },
  "voice": {
    "silence_timeout_ms": 1000
  }
}
```

### Maximum Accuracy (Slower but More Reliable)
```json
{
  "whisper": {
    "temperature": 0.0,
    "beam_size": 10,
    "best_of": 10,
    "no_speech_threshold": 0.7,
    "min_speech_ms": 600,
    "min_silence_ms": 1800
  },
  "voice": {
    "confidence_threshold": 0.75,
    "require_multi_command_confirmation": true
  }
}
```

## Advanced: Custom Initial Prompt

The `initial_prompt` guides Whisper toward your most common commands. Customize it with your actual usage patterns:

```json
{
  "whisper": {
    "initial_prompt": "Voice commands: open notepad, open chrome, list directory, show time, close window, minimize window, open file manager, search for"
  }
}
```

**Tips**:
- Include your 10-15 most common commands
- Use exact phrasing you typically say
- Keep it under 200 characters
- Update it over time as your usage changes

## Testing Your Configuration

1. **Make a small change** (e.g., increase `temperature` from 0.0 to 0.1)
2. **Restart GRIM** (config is loaded at startup)
3. **Test with same command 5 times**
4. **Check logs** for detailed transcription info:
   ```
   [DEBUG][Voice] Whisper params: temp=0.0 beam=5 no_speech_thold=0.6
   [DEBUG][Voice] Heard speech: "open notepad"
   ```
5. **Iterate** based on results

## Logging & Debugging

Enable detailed voice logging:

```json
{
  "log_level": "DEBUG"
}
```

Key log messages to watch:
- `[DEBUG][Voice] Whisper params: ...` - Active configuration
- `[DEBUG][Voice] Heard speech: "..."` - Final transcript
- `[DEBUG][HandleCommand] Detected N commands` - Multi-command detection
- `[DEBUG][Voice] Speech started/ended` - Detection timing

## Performance vs Accuracy Trade-offs

| Setting | Faster ? | More Accurate ?? |
|---------|----------|------------------|
| `beam_size` | 3 | 10 |
| `temperature` | 0.2 | 0.0 |
| `min_silence_ms` | 800 | 1800 |
| `min_speech_ms` | 300 | 700 |
| `no_speech_threshold` | 0.5 | 0.75 |

## Multi-Command Safety

To prevent accidental execution of hallucinated commands:

```json
{
  "voice": {
    "require_multi_command_confirmation": true,  // Ask before executing
    "max_commands_per_input": 3,                 // Hard limit
    "confidence_threshold": 0.7                  // Only process high-confidence
  }
}
```

This will:
1. Limit to 3 commands maximum
2. Ask "I heard 2 commands: ... Should I execute them?"
3. Filter out any transcript with <70% confidence

## Support

If you're still experiencing issues:

1. Check GRIM logs in `logs/` directory
2. Verify microphone works in Windows Sound Settings
3. Test with different `temperature` values (0.0, 0.1, 0.2)
4. Try a different Whisper model (see `whisper_model` parameter)

## Config File Location

- **Default**: `D:\G.R.I.M\ai_config.json`
- **Auto-created** if missing
- **Auto-patched** with new defaults on updates
- **Backup** recommended before major changes

---

**Last Updated**: 2025-01-21  
**GRIM Version**: 2.0+

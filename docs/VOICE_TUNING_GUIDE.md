# Voice Recognition Tuning Guide

## Problem: Whisper Hallucinations

Whisper sometimes adds extra words or misinterprets short commands:
- **Input**: "open notepad"
- **Heard**: "She's open now, pal."

This happens because Whisper tries to create grammatically correct sentences.

## Solution: Optimized Parameters

### Key Settings (in `voice.cpp`)

```cpp
wparams.temperature = 0.0f;  // Most important! Reduces creativity/hallucinations
wparams.max_len = 1;         // Prefer shorter outputs
wparams.suppress_blank = true;  // Skip blank/silence segments
wparams.initial_prompt = "Voice commands: open notepad, close window, show time";
```

### What Each Parameter Does:

1. **`temperature = 0.0`** (was 0.2)
   - **Lower = more deterministic** (less creative, more accurate)
   - Range: 0.0 to 1.0
   - **Recommendation**: 0.0 for commands, 0.2-0.4 for dictation

2. **`max_len = 1`**
   - Limits output length (fewer hallucinated words)
   - Forces Whisper to be concise

3. **`suppress_blank = true`**
   - Skips segments that are just silence
   - Reduces "[BLANK_AUDIO]" artifacts

4. **`initial_prompt`**
   - Guides Whisper toward command-style speech
   - Examples should match your use case
   - Update this with your common commands!

## Additional Tuning Options

### In `ai_config.json`:

```json
{
  "whisper": {
    "sampling_strategy": "greedy",  // or "beam_search" for better quality
    "temperature": 0.0,
    "min_speech_ms": 500,    // Minimum speech duration
    "min_silence_ms": 1200   // How long to wait for more speech
  },
  "silence_threshold": 0.02,  // Lower = more sensitive
  "silence_timeout_ms": 4000  // Time before cutting off
}
```

### Tuning Tips:

**If Whisper adds extra words:**
- ? Lower `temperature` (try 0.0)
- ? Add common commands to `initial_prompt`
- ? Use smaller model (base.en instead of medium)

**If Whisper cuts off mid-sentence:**
- ? Increase `min_silence_ms` (1500-2000)
- ? Increase `silence_timeout_ms` (5000-6000)

**If Whisper triggers on background noise:**
- ? Increase `silence_threshold` (0.03-0.05)
- ? Run calibration first (happens automatically)

**If recognition is too slow:**
- ? Use smaller model (`ggml-tiny.en.bin` or `ggml-base.en.bin`)
- ? Enable GPU acceleration (already enabled)

## Testing Workflow

1. **Test a command**: Say "open notepad"
2. **Check logs**: Look for `[Voice] Heard speech: "..."`
3. **If hallucinated**:
   - Lower temperature
   - Update initial_prompt with that command
4. **Repeat** until accuracy improves

## Model Selection

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| `tiny.en` | 75MB | Very Fast | Lower | Quick testing |
| `base.en` | 142MB | Fast | Good | **Commands** ? |
| `small.en` | 466MB | Medium | Better | Mixed use |
| `medium.en` | 1.5GB | Slow | Best | Dictation |

**Recommendation**: Use `base.en` for voice commands. It's fast and accurate enough.

## Example Custom Prompts

### For Home Automation:
```cpp
wparams.initial_prompt = "Voice commands: turn on lights, set temperature, lock door";
```

### For PC Control:
```cpp
wparams.initial_prompt = "Commands: open app, close window, screenshot, volume up";
```

### For GRIM:
```cpp
wparams.initial_prompt = "Voice commands: open notepad, close window, show time, search google";
```

## Advanced: Beam Search

For higher accuracy (slower):
```cpp
whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);
wparams.beam_search.beam_size = 5;  // Higher = more accurate but slower
wparams.temperature = 0.0f;
```

Only use for dictation/transcription, not real-time commands.

## Troubleshooting

### Still Getting Hallucinations?

1. **Check microphone quality** - Background noise causes errors
2. **Speak clearly** - Mumbling confuses Whisper
3. **Use keywords** - Say "open notepad" not "could you open notepad"
4. **Update prompt** - Add problematic phrases to `initial_prompt`
5. **Try smaller model** - Sometimes less is more

### Command Not Recognized?

1. **Check NLP rules** - Command might be filtered
2. **Check aliases** - App name might be wrong
3. **Add to prompt** - Guide Whisper toward that command
4. **Check logs** - See what Whisper actually heard

## Summary

**Best Settings for Commands:**
- Temperature: `0.0`
- Model: `base.en`
- Prompt: Add your common commands
- Silence threshold: `0.02` (calibrate first)

This should eliminate most hallucinations! ??

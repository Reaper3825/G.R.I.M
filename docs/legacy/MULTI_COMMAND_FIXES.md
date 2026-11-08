# Multi-Command and AI Backend Issues - Troubleshooting Guide

## Issues Identified

### 1. **Multiple Feedback Prompts (Fixed ?)**
**Problem**: System asks "Was that what you wanted?" 3 times when processing multi-command input.

**Root Cause**: Feedback prompt was triggered during each recursive `handleCommand()` call in multi-command processing.

**Solution Implemented**: Added `g_isMultiCommandContext` flag to suppress individual feedback prompts and consolidate them into a single prompt after all commands complete.

### 2. **AI Backend Failure (Needs Fix ?)**
**Problem**: 
```
[ERROR][AI] Exception: [json.exception.type_error.306] cannot use value() with discarded
[ERROR][AI] Interpretation failed — non-JSON response: [AI] Backend call failed
```

**Root Cause**: In `ai.cpp`, `ai_interpret()` function:
1. Parses AI response as JSON
2. If parsing fails, `j.is_discarded()` returns true
3. Code tries to call `j.value("intent", "conversation")` on the discarded object
4. This throws `json.exception.type_error.306`

**Location**: `..\..\ai\ai.cpp` line ~200-250 in `ai_interpret()`

**Fix Needed**:
```cpp
// BEFORE (Buggy):
nlohmann::json j = nlohmann::json::parse(reply, nullptr, false);
if (j.is_discarded()) {
    LOG_ERROR("AI", "Interpretation failed — non-JSON response: " + reply);
    result.message = "[AI] Could not interpret input.";
    result.voice = "Sorry, I couldn't interpret that.";
    return result;  // ? Still tries to access j later!
}

std::string intent = j.value("intent", "conversation");  // ? Throws exception if j is discarded!

// AFTER (Fixed):
nlohmann::json j = nlohmann::json::parse(reply, nullptr, false);
if (j.is_discarded() || !j.is_object()) {
    LOG_ERROR("AI", "Interpretation failed — non-JSON response: " + reply);
    result.message = "[AI] Could not interpret input.";
    result.voice = "Sorry, I couldn't interpret that.";
    result.success = false;  // ? Ensure failure state
    return result;  // ? Early return to avoid accessing discarded JSON
}

// Only access j after validation
std::string intent = j.value("intent", "conversation");
```

### 3. **Voice Recognition Accuracy (Configuration Issue ??)**
**Problem**: Voice recognition picks up " open notepad, close window," when you only said "open notepad"

**Root Cause**: This is NOT a code bug - it's a **voice recognition accuracy issue**. The speech-to-text system (likely Whisper) is either:
- Picking up background noise
- Hallucinating extra words (common with AI speech recognition)
- Using too low of a confidence threshold

**Solutions**:

#### Option A: Increase Confidence Threshold
In `ai_config.json`:
```json
{
  "voice": {
    "confidence_threshold": 0.8,  // ? Increase from default (0.5-0.6)
    "filter_low_confidence": true
  }
}
```

#### Option B: Add Voice Confirmation for Multi-Command
In `commands_core.cpp`, modify multi-command detection:
```cpp
if (commands.size() > 1)
{
    LOG_DEBUG("HandleCommand", "Detected " + std::to_string(commands.size()) + " commands");
    
    // Ask for confirmation before executing multiple commands
    std::string ask = "I heard " + std::to_string(commands.size()) + " commands: ";
    for (size_t i = 0; i < commands.size(); ++i) {
        ask += "\"" + commands[i] + "\"";
        if (i < commands.size() - 1) ask += ", ";
    }
    ask += ". Is that correct?";
    
    // ... confirmation logic
}
```

#### Option C: Adjust Whisper Parameters
In `ai_config.json`:
```json
{
  "whisper": {
    "temperature": 0.0,        // ? More deterministic (less hallucination)
    "beam_size": 5,            // ? More careful decoding
    "best_of": 5,             // ? Pick best of N candidates
    "no_speech_threshold": 0.6 // ? Filter out noise
  }
}
```

## Testing & Verification

### Test 1: Multi-Command Handling
```
Input: "open notepad, close window"
Expected: 
- Log shows 2 commands detected
- Both commands execute
- Only ONE feedback prompt appears
```

### Test 2: Single Command (Regression Test)
```
Input: "open notepad"
Expected:
- No multi-command detection
- Command executes normally
- One feedback prompt appears (if enabled)
```

### Test 3: AI Backend Failure Handling
```
Scenario: Ollama returns non-JSON
Expected:
- Error logged
- User sees "Sorry, I couldn't interpret that."
- No exception thrown
- No crash
```

## Monitoring

### Key Log Messages to Watch:

**Multi-Command Detection**:
```
[DEBUG][HandleCommand] Detected 2 commands in voice input
  [1] "open notepad"
  [2] "close window"
```

**Context Flag**:
```
[DEBUG][Feedback] Opened feedback prompt for multi-command batch
```

**AI Parsing Failure**:
```
[ERROR][AI] Interpretation failed — non-JSON response: [AI] Backend call failed
```

**Voice Recognition Quality**:
```
[DEBUG][WakeVoice] Captured voice transcript: " open notepad, close window,"
[DEBUG][Voice] Heard speech: " open notepad, close window,"
```

## Configuration Recommendations

### For Your Setup (Based on Logs):

1. **Enable Voice Confidence Logging**
   - Add confidence scores to voice transcripts
   - Helps identify when speech recognition is guessing

2. **Tune Ollama Parameters**
   ```json
   {
     "ollama_url": "http://127.0.0.1:11434",
     "default_model": "mistral:latest",
     "temperature": 0.3,
     "top_p": 0.9,
     "stream": false
   }
   ```

3. **Add Voice Input Sanitization**
   - Trim whitespace
   - Remove trailing commas
   - Filter out filler words

## Next Steps

1. ? **Multi-command feedback suppression** - Already implemented
2. ?? **Fix AI JSON parsing bug** - Needs implementation
3. ?? **Tune voice recognition** - Configuration adjustments
4. ?? **Add telemetry** - Track command accuracy over time

## Related Files

- `commands_core.cpp` - Multi-command handling + feedback logic
- `ai.cpp` - AI interpretation + JSON parsing
- `voice.cpp` / `voice_stream.cpp` - Speech recognition
- `ai_config.json` - Configuration settings

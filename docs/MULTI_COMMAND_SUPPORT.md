# Multi-Command Voice Input Support

## Overview
GRIM now supports processing multiple commands in a single voice input, separated by commas. This enhancement addresses voice recognition scenarios where multiple commands are captured in one transcript.

## Changes Made

### 1. Input Parser Enhancement (`input_parser.cpp`)
Added `splitCommands()` function that:
- Splits input by commas
- Trims whitespace from each command
- Returns a vector of individual commands

```cpp
std::vector<std::string> splitCommands(const std::string& input)
```

### 2. Command Handler Update (`commands_core.cpp`)
Modified `handleCommand()` to:
- Detect multi-command inputs using `splitCommands()`
- Log all detected commands for debugging
- Process each command sequentially via recursive calls
- Maintain all existing functionality (feedback, clarification, RL, AI, etc.)

## Usage Examples

### Voice Input
When you say: **"open notepad, set timer 5"**

The system will:
1. Detect 2 commands in the input
2. Log: `[DEBUG][HandleCommand] Detected 2 commands in voice input`
3. Log: `  [1] "open notepad"`
4. Log: `  [2] "set timer 5"`
5. Execute `open notepad`
6. Execute `set timer 5`

### Debugging Voice Recognition Issues
If voice recognition incorrectly captures commands (like " open notepad, close window," when you only said "open notepad"), the logs will now show:

```
[DEBUG][HandleCommand] Detected 2 commands in voice input
  [1] "open notepad"
  [2] "close window"
```

This helps identify:
- What the voice recognition system actually heard
- Whether the issue is voice recognition accuracy vs. command processing
- Patterns in misrecognized commands

## Addressing Your Specific Issue

### The Problem
Voice recognition captured: `" open notepad, close window,"`
You actually said: `"open notepad"`

### What Happens Now
1. System detects and logs both commands
2. Executes "open notepad" ?
3. Attempts "close window" (misheard command) ?
4. Logs clearly show what was detected

### Recommendations

#### Short-term: Review Logs
Check the debug logs to see what commands are being detected. This helps you identify if:
- Voice recognition is picking up background noise
- Commands are being misheard
- The microphone is too sensitive

#### Long-term: Voice Recognition Tuning
Consider adjusting in `voice.cpp` or `voice_stream.cpp`:
1. **Confidence threshold**: Increase minimum confidence for recognized speech
2. **Noise cancellation**: Enable/enhance background noise filtering
3. **Pause detection**: Add delay before processing to catch incomplete phrases
4. **Confirmation mode**: Ask for confirmation before executing multi-command inputs

## Configuration

### Enable/Disable Multi-Command Processing
If you want to disable multi-command processing temporarily, you can add a config flag in `ai_config.json`:

```json
{
  "voice": {
    "multi_command_enabled": true,
    "multi_command_separator": ","
  }
}
```

### Future Enhancements
- Add confidence scores to each detected command
- Implement "undo last command" functionality
- Add voice confirmation: "I heard 2 commands: open notepad and close window. Is that correct?"
- Filter out low-confidence commands automatically

## Testing

### Manual Test
1. Type in console: `open notepad, ls`
2. Both commands should execute sequentially

### Voice Test
1. Speak: "open notepad, show help"
2. Check logs for command detection
3. Verify both commands execute

### Edge Cases
- Empty commands: `"open,, close"` ? filters out empty strings
- Single command: `"open notepad"` ? works as before (no change)
- Trailing comma: `"open notepad,"` ? handled correctly

## Logging

All multi-command processing is logged with these tags:
- `[DEBUG][HandleCommand]` - Command detection and parsing
- `[TRACE][HandleCommand]` - Detailed execution flow

Example log output:
```
[TRACE][HandleCommand] START line=" open notepad, close window,"
[DEBUG][Core] Ensuring core plugins registered...
[DEBUG][HandleCommand] Detected 2 commands in voice input
  [1] "open notepad"
  [2] "close window"
[TRACE][HandleCommand] START line="open notepad"
...
```

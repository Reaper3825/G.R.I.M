# GRIM Input Control - User Guide

## How GRIM Uses Input Control

GRIM can now control your mouse and keyboard through natural language commands. The system integrates perception (seeing what's on screen) with input control (interacting with what it sees).

## Voice/Text Commands GRIM Understands

### Mouse Control

**Move Mouse:**
- "move mouse to 500 300"
- "position mouse 100 200"
- "put the mouse at 800 600"

**Click:**
- "click" (left click at current position)
- "click left"
- "click right"
- "click middle"
- "press the right mouse button"

**Click on UI Elements:**
- "click on Submit" (finds "Submit" text via OCR and clicks it)
- "click on text OK"
- "click on object button" (uses object detection)

### Keyboard Control

**Type Text:**
- "type hello world"
- "write test message"
- "enter my email address"

**Press Keys:**
- "press enter"
- "hit escape"
- "tap tab"

**Keyboard Shortcuts:**
- "press ctrl+c" (copy)
- "press ctrl+v" (paste)
- "press alt+f4" (close window)
- "press win+r" (run dialog)

### Combined Perception + Input

**Intelligent Clicking:**
1. GRIM reads the screen (OCR)
2. Finds the text/object you mentioned
3. Clicks on it automatically

Example workflow:
```
You: "what do you see?"
GRIM: [analyzes screen, reads text]
You: "click on the Submit button"
GRIM: [finds "Submit" text, clicks it]
```

## Direct Command Syntax

If you prefer direct commands instead of natural language:

### Commands Available:

| Command | Syntax | Example |
|---------|--------|---------|
| `move_mouse` | `move_mouse <x> <y>` | `move_mouse 500 300` |
| `click` | `click [left\|right\|middle]` | `click right` |
| `type` | `type <text>` | `type Hello GRIM!` |
| `key` | `key <keyname>` or `key <combo>` | `key enter` or `key ctrl+c` |
| `click_on` | `click_on <target>` | `click_on text Submit` |

### Perception Commands:

| Command | Description |
|---------|-------------|
| `perception_what_see` | Analyze and describe what's on screen |
| `perception_read_text` | Read all text via OCR |
| `perception_detect_objects` | Detect UI objects |

## Code Integration Examples

### From AI/NLP Intent System

The input controller is automatically triggered when GRIM detects certain intents:

```cpp
// In your AI command handler
if (intent == "click_on") {
    // Automatically routed to cmdInputClickOn
    // which uses perception + input control
}
```

### Direct API Usage

```cpp
#include "perception/perception_context.hpp"

using namespace GRIM::Perception;

// Ensure context manager is initialized
if (g_contextManager) {
    // Move and click
    g_contextManager->moveMouseTo(500, 300);
    g_contextManager->clickMouse("left");
    
    // Type naturally
    g_contextManager->typeText("Hello!", 15);
    
    // Keyboard shortcuts
    g_contextManager->pressKeyCombo({"ctrl", "c"});
    
    // Perception-based clicking
    g_contextManager->clickOnText("Submit");
}
```

## NLP Rules

The following patterns are recognized (from `nlp_rules.json`):

1. **move_mouse**: Matches "move mouse to X Y", "position mouse X Y"
2. **click**: Matches "click", "click left/right/middle"
3. **type**: Matches "type TEXT", "write TEXT", "enter TEXT"
4. **key**: Matches "press KEY", "hit KEY", "tap KEY"
5. **click_on**: Matches "click on TARGET"

## Workflow Examples

### Example 1: Open Notepad and Type
```
User: "press win+r"
GRIM: ✓ Pressed key combo: win+r

User: "type notepad"
GRIM: ✓ Typed: notepad

User: "press enter"
GRIM: ✓ Pressed key: enter

[wait for notepad to open]

User: "type Hello from GRIM!"
GRIM: ✓ Typed: Hello from GRIM!
```

### Example 2: Use Vision to Click Button
```
User: "what do you see?"
GRIM: [Screen shows: Desktop with Chrome browser, Submit button visible...]

User: "click on text Submit"
GRIM: ✓ Clicking on text 'Submit' at estimated position

[GRIM finds "Submit" via OCR and clicks it]
```

### Example 3: Automated Form Filling
```
User: "move mouse to 300 200"
User: "click"
User: "type john.doe@email.com"
User: "press tab"
User: "type MyPassword123"
User: "press enter"
```

## Safety Features

1. **Logging**: All input actions are logged via `logDebug()` for audit trail
2. **Validation**: Coordinates and parameters are validated before execution
3. **Natural Timing**: Human-like delays prevent detection as automation
4. **Error Handling**: Graceful failure with informative error messages

## Initialization

The input control system is automatically available when:
- ✓ Perception system is initialized (`g_contextManager` exists)
- ✓ Core plugins are loaded (happens at startup)
- ✓ Commands are registered (automatic via `core_plugin.cpp`)

No manual initialization required - it works out of the box!

## Troubleshooting

**"Input system not available"**
- Check if perception context manager is initialized
- Verify in logs: `[Plugin] Registering input control commands`

**Clicks not hitting target**
- Use `perception_read_text` first to verify OCR is working
- Add delays between commands for UI to respond
- Use `move_mouse` + `click` for precise control

**Keys not working**
- Check key name is correct (see InputController documentation)
- Try direct command instead of NLP: `key ctrl+c`
- Check logs for "Unknown key" errors

## Advanced Usage

### Combine with Memory System
```cpp
// GRIM remembers button locations
g_contextManager->storeContextInMemory(ctx, "Submit button at (500, 300)");

// Later, recall and use
auto ctx = g_contextManager->getCurrentContext();
g_contextManager->clickOnText("Submit");
```

### Multi-Monitor Support
```cpp
// Click on something on monitor 2
auto monitor2Ctx = g_contextManager->getMonitorContext(1, true);
g_contextManager->clickAt(100, 100, "left");
```

### Continuous Monitoring + Auto-Click
```cpp
// React to screen changes automatically
ContinuousCaptureConfig config;
config.frameSkip = 30;
g_contextManager->startContinuousCapture(config);

while (running) {
    auto ctx = g_contextManager->getLatestContext();
    if (ctx.hasText && ctx.screenText.find("Error") != std::string::npos) {
        g_contextManager->clickOnText("OK");
    }
}
```

## See Also

- `core/input/README.md` - Technical documentation
- `perception/IMPROVEMENTS.md` - Perception layer details
- `nlp_rules.json` - NLP pattern definitions
- `commands/commands_perception.cpp` - Implementation details

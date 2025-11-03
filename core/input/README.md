# GRIM Input Control System

## Overview
GRIM now has native Windows input control capabilities that work seamlessly with the perception layer. This allows GRIM to see what's on screen and interact with it through keyboard and mouse simulation.

## Architecture

### Core Components
- **InputController** (`core/input/InputController.hpp/cpp`): Low-level Windows API wrapper for input simulation
- **PerceptionContextManager** (`perception/perception_context.hpp/cpp`): High-level interface combining perception + input

### Integration Points
The input control system is integrated into the perception layer, allowing GRIM to:
1. **See** what's on screen (OCR, object detection, vision AI)
2. **Act** on what it sees (mouse clicks, keyboard input)
3. **Verify** actions worked (continuous monitoring)

## Usage Examples

### Basic Mouse Control

```cpp
#include "perception/perception_context.hpp"

using namespace GRIM::Perception;

// Move mouse to specific coordinates
g_contextManager->moveMouseTo(500, 300);

// Click at current position
g_contextManager->clickMouse("left");   // or "right", "middle"

// Double-click
g_contextManager->doubleClickMouse("left");

// Scroll
g_contextManager->scrollMouse(120);  // positive = up, negative = down
```

### Basic Keyboard Control

```cpp
// Type text with natural human-like delays
g_contextManager->typeText("Hello, World!", 15); // 15ms delay between keys

// Individual key presses
g_contextManager->tapKey("enter");
g_contextManager->tapKey("esc");
g_contextManager->tapKey("tab");

// Key combinations
g_contextManager->pressKeyCombo({"ctrl", "c"});  // Copy
g_contextManager->pressKeyCombo({"ctrl", "v"});  // Paste
g_contextManager->pressKeyCombo({"alt", "f4"});  // Close window
g_contextManager->pressKeyCombo({"win", "r"});   // Run dialog
```

### Supported Keys

#### Special Keys
- `enter`, `return`
- `esc`, `escape`
- `tab`
- `space`
- `backspace`
- `delete`, `del`
- `home`, `end`
- `pageup`, `pgup`, `pagedown`, `pgdn`

#### Arrow Keys
- `left`, `right`, `up`, `down`

#### Modifier Keys
- `shift`
- `ctrl`, `control`
- `alt`
- `win`, `windows`

#### Function Keys
- `f1` through `f12`

#### Alphanumeric
- Single letters: `"a"`, `"b"`, ..., `"z"`
- Single digits: `"0"`, `"1"`, ..., `"9"`

### High-Level Perception-Based Actions

```cpp
// Click at specific coordinates
g_contextManager->clickAt(250, 400, "left");

// Find text on screen and click it (uses OCR)
g_contextManager->clickOnText("Submit Button", "left");

// Find object and click it (uses object detection)
g_contextManager->clickOnObject("button", "left");
```

### Complete Workflow Example

```cpp
#include "perception/perception_context.hpp"

using namespace GRIM::Perception;

// Example: Open Notepad and type a message
void openNotepadAndType() {
    // 1. Open Run dialog
    g_contextManager->pressKeyCombo({"win", "r"});
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // 2. Type "notepad"
    g_contextManager->typeText("notepad");
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    // 3. Press Enter
    g_contextManager->tapKey("enter");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // 4. Type a message
    g_contextManager->typeText("Hello from GRIM! This is automated input.", 20);
    
    // 5. Verify what we typed (using perception)
    auto ctx = g_contextManager->getCurrentContext(true);
    if (ctx.hasText && ctx.screenText.find("Hello from GRIM") != std::string::npos) {
        Logger::log("Successfully typed and verified message!");
    }
}

// Example: Click on a specific UI element using vision
void clickOnButton() {
    // 1. Capture and analyze current screen
    auto ctx = g_contextManager->getCurrentContext(true);
    
    // 2. Use OCR to find button text
    if (ctx.hasText) {
        Logger::log("Screen text: " + ctx.screenText);
        g_contextManager->clickOnText("OK", "left");
    }
    
    // Or use object detection
    if (!ctx.detectedObjects.empty()) {
        for (const auto& obj : ctx.detectedObjects) {
            Logger::log("Found object: " + obj.label);
        }
        g_contextManager->clickOnObject("button", "left");
    }
}
```

## Advanced Features

### Natural Human-Like Input
The InputController automatically adds:
- Random delays between key presses (50-100ms for clicks)
- Variable typing speed (configurable delay)
- Proper key press/release sequences

This makes the input undetectable by most anti-automation systems.

### Multi-Monitor Support
```cpp
// Get context from specific monitor
auto monitor2Ctx = g_contextManager->getMonitorContext(1, true); // 0-indexed

// Click on something on monitor 2
// (coordinates are relative to the specific monitor)
if (monitor2Ctx.isValid) {
    g_contextManager->clickAt(100, 100, "left");
}
```

### Continuous Monitoring + Input Loop
```cpp
// Start continuous screen capture
ContinuousCaptureConfig config;
config.frameSkip = 30;
config.changeThreshold = 0.05f;
g_contextManager->startContinuousCapture(config);

// React to screen changes
while (running) {
    auto ctx = g_contextManager->getLatestContext();
    
    if (ctx.hasText && ctx.screenText.find("Error") != std::string::npos) {
        // Error detected! Click OK button
        g_contextManager->clickOnText("OK", "left");
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

g_contextManager->stopContinuousCapture();
```

## Safety Considerations

### Coordinate Validation
Always validate coordinates are within screen bounds:
```cpp
auto ctx = g_contextManager->getCurrentContext();
int x = 500, y = 300;

if (x >= 0 && x < ctx.screenWidth && y >= 0 && y < ctx.screenHeight) {
    g_contextManager->clickAt(x, y);
} else {
    Logger::log("Invalid coordinates", Logger::Level::WARNING);
}
```

### Error Handling
```cpp
try {
    g_contextManager->clickOnText("Button");
} catch (const std::exception& e) {
    Logger::log("Input error: " + std::string(e.what()), Logger::Level::ERROR);
}
```

### User Control
Always provide a way to stop automated input:
```cpp
// Reserve a hotkey for emergency stop
// Check for ESC key press before each action
if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
    Logger::log("User stopped automation");
    return;
}
```

## Technical Details

### Windows API Used
- `SendInput()` - Hardware-level input injection
- `SetCursorPos()` - Mouse positioning
- `VkKeyScanA()` - Character to virtual key mapping
- `INPUT` structures - Keyboard and mouse events

### Undetectability
The implementation uses:
- Hardware-level input (not window messages)
- Random timing variations
- Proper key press/release sequences
- Natural mouse movement

This works with:
- ✅ Web browsers (Chrome, Firefox, Edge)
- ✅ Desktop applications
- ✅ Games (most)
- ✅ Protected applications

## Integration with AI/NLP

```cpp
// Example: GRIM command handler
void handleCommand(const std::string& command) {
    if (command.find("click on") != std::string::npos) {
        // Extract what to click from command
        // e.g., "click on the submit button"
        
        auto ctx = g_contextManager->getCurrentContext(true);
        
        // Use NLP to extract target
        std::string target = extractTarget(command);
        
        // Try OCR first
        g_contextManager->clickOnText(target, "left");
    }
    else if (command.find("type") != std::string::npos) {
        // e.g., "type hello world"
        std::string text = extractText(command);
        g_contextManager->typeText(text);
    }
}
```

## Troubleshooting

### Input Not Working
1. **Check if application is running as admin** - Some apps require elevated privileges
2. **Verify coordinates are correct** - Use `getCurrentContext()` to check screen dimensions
3. **Check logs** - All input actions are logged via `Logger::log()`

### Clicks Missing Targets
1. **Add delays** - Give UI time to respond
2. **Use higher precision** - Get exact coordinates from OCR bounding boxes
3. **Verify screen capture** - Ensure perception is capturing the right monitor

### Text Not Typing Correctly
1. **Check keyboard layout** - `VkKeyScanA()` uses current keyboard layout
2. **Increase delay** - Some apps need slower typing (increase `delayMs`)
3. **Use Unicode** - For non-ASCII characters, extend `typeText()` with `KEYEVENTF_UNICODE`

## Future Enhancements

Potential improvements:
- [ ] OCR bounding box support for precise text clicking
- [ ] Image-based template matching for UI element detection
- [ ] Mouse movement smoothing/interpolation
- [ ] Recording and playback of input sequences
- [ ] Visual feedback overlay showing what GRIM is interacting with

## License & Usage
This input control system is part of GRIM and follows the same license. Use responsibly and ethically - never use for malicious automation or to violate terms of service.

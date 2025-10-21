# GRIM Console Integration

## Overview
The GRIM console now integrates with the popup UI and supports keyboard shortcuts for quick access.

## Features

### 1. **Popup Click Integration**
When you click the GRIM popup overlay, it will show/bring to front the console window.

**Implementation:**
- `popup_ui/pop_ui.cpp` - Mouse click detection using `Mouse::wasPressed()`
- `popup_ui/popup_window.cpp` - WM_LBUTTONDOWN handler
- Both call `GRIMConsole::showConsole()`

### 2. **Console Visibility Controls**

**API Functions:**
```cpp
GRIMConsole::showConsole();     // Show and bring console to front
GRIMConsole::hideConsole();     // Hide console window
GRIMConsole::toggleConsole();   // Toggle visibility
```

**Window Management:**
- Console integrates with `WindowManager` for tracking
- Properly handles window state (SW_RESTORE, SW_HIDE)
- SetForegroundWindow ensures focus when shown

### 3. **Keyboard Shortcuts**

**Console Toggle Key: ` (Grave/Tilde)**
- Press the grave/tilde key (usually below ESC) to toggle console
- Works globally even when console is hidden
- Implemented in `wake/wake_key.cpp`

**Voice Command Key: Right Ctrl**
- Existing functionality - hold to capture voice command
- Still works independently

## Technical Details

### Console Rendering
- Uses Win32 GDI (CreateFontW, TextOutW) for stable rendering
- Runs on separate thread to avoid BGFX conflicts
- Self-contained window with WM_PAINT handling

### Mouse Integration
```cpp
// In popup_ui/pop_ui.cpp
if (Mouse::wasPressed(MouseButton::Left))
{
    POINT cursorPos = Mouse::getPosition();
    HWND hwndUnderCursor = WindowFromPoint(cursorPos);
    if (hwndUnderCursor == g_hwnd)
    {
        GRIMConsole::showConsole();
    }
}
```

### Keyboard Integration
```cpp
// In wake/wake_key.cpp
Key::onPress(KeyCode::Grave, [](KeyCode code) {
    GRIMConsole::toggleConsole();
});
```

## Usage

### Show Console
- **Click** the GRIM popup overlay
- **Press** ` (grave/tilde key)
- **Call** `GRIMConsole::showConsole()` from code

### Hide Console
- **Press** ` (grave/tilde key) again
- **Press** ESC in console window
- **Click** the X (close button) on console window
- **Call** `GRIMConsole::hideConsole()` from code

### Type Commands
- Console accepts text input
- Press ENTER to execute commands
- History scrolls automatically

## Behavior Notes

- **ESC key**: Hides console (does NOT exit program)
- **Close button (X)**: Hides console (does NOT exit program)
- **Console stays running**: Even when hidden, console thread continues
- **Toggle anytime**: Use ` key to show/hide from anywhere

## Files Modified

### Headers
- `ui/console_ui.hpp` - Added show/hide/toggle API

### Implementation
- `ui/console_ui.cpp` - Implemented visibility controls
- `popup_ui/pop_ui.cpp` - Added console show on click
- `popup_ui/popup_window.cpp` - Added WM_LBUTTONDOWN handler
- `wake/wake_key.cpp` - Added console toggle hotkey

### Dependencies
- `helpers/mouse.cpp` - Mouse button detection
- `helpers/key.cpp` - Keyboard input system
- `core/window_manager.hpp` - Window state management

## Future Enhancements

Possible improvements:
- [ ] Customizable hotkey via config file
- [ ] Smooth show/hide animations
- [ ] Console auto-hide on focus loss
- [ ] Multi-monitor position persistence
- [ ] Opacity/transparency controls

## Testing

Build successful ?
- Console renders correctly with GDI
- Popup click shows console
- Tilde key toggles console
- No BGFX threading conflicts
- Mouse and keyboard systems integrated

## Notes

- Console uses Win32 GDI instead of BGFX to avoid threading issues
- Popup uses BGFX for rendering (separate from console)
- Main thread handles BGFX frame() calls
- Console thread handles Win32 message loop independently

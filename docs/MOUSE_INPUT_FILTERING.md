# Mouse Input Filtering System

## ?? Overview

The mouse input filtering system prevents accidental clicks on the game/desktop underneath GRIM by only allowing mouse input when:
1. **Popup UI** is visible (the floating Oreo character)
2. **UI Panel** is open (settings menu, console, etc.)

## ?? Implementation

### Files Modified

1. **`core/input_parser.hpp`**
   - Added `mouseInputEnabled` flag to `InputState`
   - Added `shouldProcessMouseInput()` static method

2. **`core/input_parser.cpp`**
   - Implemented `shouldProcessMouseInput()` check
   - Modified `capture()` to filter mouse buttons
   - Modified `captureFromHWND()` to filter mouse buttons
   - Clears mouse state when input is disabled

### How It Works

```cpp
bool InputState::shouldProcessMouseInput()
{
    // Check if popup is visible
    bool popupVisible = isPopupVisible();
    
    // Check if cursor is over a UI panel
    bool uiPanelVisible = UIRoot::get().shouldReceiveInputAt(cursorX, cursorY);
    
    // Allow input only if either is true
    return popupVisible || uiPanelVisible;
}
```

### Input States

| Condition | Mouse Buttons | Mouse Position | Keyboard |
|-----------|---------------|----------------|----------|
| **Popup/Panel visible** | ? Active | ? Tracked | ? Active |
| **Nothing visible** | ? Disabled | ? Tracked | ? Active |

### Key Features

? **Prevents click-through** - Mouse clicks don't affect underlying apps  
? **Position always tracked** - Cursor position still available  
? **Keyboard always active** - Hotkeys work regardless  
? **Smooth transitions** - State resets prevent false events  
? **Debug logging** - Can monitor filtering state  

## ?? User Experience

### Before
```
User clicks on desktop ? Desktop app activates ??
User drags mouse ? Accidental selection in game ??
```

### After
```
User clicks on desktop ? Nothing happens ?
UI panel opens ? Mouse clicks now active ?
UI panel closes ? Mouse clicks disabled again ?
```

## ?? Debugging

### Check Current State

The debug log shows mouse input state every 60 frames:

```
[InputState] LButton: down=1 prev=0 pressed=1 enabled=1  ? Mouse active
[InputState] LButton: down=1 prev=0 pressed=0 enabled=0  ? Mouse filtered
```

### Manual Testing

1. **Test with no UI visible:**
   ```
   - Close all panels
   - Hide popup
   - Try clicking desktop ? Nothing should happen
   ```

2. **Test with popup visible:**
   ```
   - Say "Hey GRIM" (popup appears)
   - Click anywhere ? Popup should respond
   ```

3. **Test with settings open:**
   ```
   - Type: settings
   - Click on setting ? Should work
   - Click outside panel ? Should pass through to desktop
   ```

## ?? Performance

- **Negligible overhead** - Single boolean check per frame
- **No polling** - Uses existing visibility state
- **Thread-safe** - All checks are atomic or mutex-protected

## ?? Known Limitations

1. **Position-based check** - If cursor is far from panels but panels are open, input still disabled
   - **Solution:** Could change to "any panel visible" instead of "cursor over panel"

2. **No window focus check** - Input disabled even if GRIM window has focus
   - **Future enhancement:** Could add focus-aware mode

## ?? Configuration

Currently hardcoded behavior. Future enhancements could include:

```json
{
  "input": {
    "mouse_filter_mode": "smart",  // "smart", "always", "never"
    "require_panel_focus": false,
    "click_through_margin": 10     // pixels outside panel to still allow
  }
}
```

## ?? Testing Checklist

- [ ] Mouse disabled when nothing visible
- [ ] Mouse enabled when popup shows
- [ ] Mouse enabled when settings menu opens
- [ ] Mouse disabled when all UI closes
- [ ] Keyboard always works
- [ ] Mouse position always tracked
- [ ] No false click events on state transitions
- [ ] Works with multiple panels open
- [ ] Works with popup + panel simultaneously

## ?? Future Enhancements

1. **Visual indicator** - Show when mouse is filtered
2. **Per-window filtering** - Different rules for different windows
3. **Configurable modes** - User can choose filtering behavior
4. **Smart focus** - Detect when user wants to interact with background
5. **Gesture detection** - Allow specific gestures even when filtered

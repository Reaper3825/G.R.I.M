# Mouse Input Filtering - Quick Reference

## ?? What It Does

**Before:** Clicking anywhere on screen could trigger unwanted actions in games/apps underneath GRIM.

**After:** Mouse input **only works** when GRIM UI is visible:
- ? Popup (Oreo character) is shown
- ? Settings menu is open
- ? Console panel is visible
- ? Otherwise, clicks pass through (no accidental clicks!)

## ?? Key Points

| Feature | Behavior |
|---------|----------|
| **Mouse clicks** | Disabled when UI hidden |
| **Mouse position** | Always tracked |
| **Keyboard** | Always active |
| **UI interaction** | Works normally |
| **Click-through** | Prevented automatically |

## ?? How to Test

### Test 1: No UI (clicks should be ignored)
```
1. Close all GRIM panels
2. Hide popup (wait for timeout)
3. Try clicking on desktop ? Nothing happens ?
```

### Test 2: Popup visible (clicks should work)
```
1. Say "Hey GRIM" or use test_tts
2. Popup appears
3. Click on popup ? Console opens ?
```

### Test 3: Settings menu (clicks should work)
```
1. Type: settings
2. Settings menu opens
3. Click on buttons ? They respond ?
4. Close settings
5. Try clicking desktop ? Nothing happens ?
```

### Test 4: Playing a game
```
1. Launch a game (fullscreen or windowed)
2. Close all GRIM UI
3. Play game normally
4. Click around ? Game receives clicks, GRIM doesn't interfere ?
5. Say "Hey GRIM"
6. Popup appears over game
7. Click popup ? GRIM responds (game doesn't) ?
```

## ?? Debug Info

Watch the console log for:
```
[InputState] LButton: down=1 enabled=1  ? Mouse active
[InputState] LButton: down=1 enabled=0  ? Mouse filtered
```

## ??? Technical Details

**Function:** `InputState::shouldProcessMouseInput()`

**Logic:**
```cpp
bool enabled = isPopupVisible() || uiRoot.shouldReceiveInputAt(cursorX, cursorY);
```

**When TRUE:**
- Popup is showing, OR
- Cursor is over a visible UI panel

**When FALSE:**
- No UI visible
- Mouse state reported as all buttons up
- Click events never generated

## ? Benefits

1. **No accidental clicks** - Can't accidentally click desktop/game
2. **Clean interaction** - UI only responds when intentional
3. **Seamless gaming** - GRIM doesn't interfere with your game
4. **Better UX** - Clear separation between GRIM and system

## ?? If Issues

**Problem:** Mouse not working in UI
- ? Check if panel is visible: `settings`
- ? Check if popup is showing: `test_tts Hello`
- ? Check logs for "enabled=1" when clicking

**Problem:** Clicks still going through
- ? Verify build is up to date
- ? Check `InputState::mouseInputEnabled` flag
- ? Restart GRIM

**Problem:** Can't interact with anything
- ? Open a panel: `settings`
- ? Show popup: Voice command or test_tts
- ? Check if `shouldProcessMouseInput()` returns true

---

**Status:** ? Implemented and working  
**Build:** Successful  
**Version:** Mouse input filtering active

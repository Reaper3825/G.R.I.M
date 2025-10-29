# Settings Menu Config Save Fix

## Problem

The settings menu was not properly updating the global `aiConfig` variable when saving changes. This meant that:
- Changes were saved to `ai_config.json` file correctly
- BUT the running application continued to use the old values
- User had to restart the application for changes to take effect

## Root Cause

In `ui_settings_menu.cpp`, the `saveConfig()` function was only writing to the file:

```cpp
void UISettingsMenu::saveConfig() {
    try {
        std::ofstream f("ai_config.json");
        f << config.dump(4);
        LOG_DEBUG("UISettingsMenu", "Saved ai_config.json");
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to save ai_config.json");
    }
}
```

It wasn't updating the global `aiConfig` variable that the rest of the application uses.

## Solution

Add two lines to `saveConfig()`:

```cpp
void UISettingsMenu::saveConfig() {
    // At the top of the file, add:
    extern nlohmann::json aiConfig;
    
    try {
        std::ofstream f("ai_config.json");
        f << config.dump(4);
        f.close();  // Ensure file is flushed
        
        // ? UPDATE: Apply to global config immediately
        aiConfig = config;
        
        LOG_DEBUG("UISettingsMenu", "Saved ai_config.json and updated global aiConfig");
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to save ai_config.json");
    }
}
```

## Changes Required

### File: `ui/ui_settings_menu.cpp`

1. **Add extern declaration at top of file** (after includes):
```cpp
#include "ui_settings_menu.hpp"
#include "overlay_renderer.hpp"
#include "logger.hpp"
#include "input_parser.hpp"
#include <fstream>
#include <functional>
#include <filesystem>

// ? ADD THIS LINE
extern nlohmann::json aiConfig;
```

2. **Update `saveConfig()` function**:
```cpp
void UISettingsMenu::saveConfig() {
    try {
        std::ofstream f("ai_config.json");
        f << config.dump(4);
        f.close();  // ? ADD: Ensure file is flushed
        
        // ? ADD: Update global config
        aiConfig = config;
        
        LOG_DEBUG("UISettingsMenu", "Saved ai_config.json and updated global aiConfig");
    } catch (...) {
        LOG_ERROR("UISettingsMenu", "Failed to save ai_config.json");
    }
}
```

3. **Optionally enhance `applyChanges()` with debug logging**:
```cpp
void UISettingsMenu::applyChanges() {
    if (!hasChanges) {
        LOG_DEBUG("UISettingsMenu", "No changes to apply");
        return;
    }
    
    // ? ADD: Debug logging for whisper settings
    if (pendingConfig.contains("whisper")) {
        LOG_DEBUG("UISettingsMenu", "Applying whisper settings:");
        LOG_DEBUG("UISettingsMenu", "  temperature: " + 
                  std::to_string(pendingConfig["whisper"].value("temperature", 0.0f)));
        LOG_DEBUG("UISettingsMenu", "  beam_size: " + 
                  std::to_string(pendingConfig["whisper"].value("beam_size", 5)));
        LOG_DEBUG("UISettingsMenu", "  suppress_blank: " + 
                  std::string(pendingConfig["whisper"].value("suppress_blank", true) ? "true" : "false"));
    }
    
    config = pendingConfig;
    saveConfig();
    hasChanges = false;
    
    LOG_DEBUG("UISettingsMenu", "Settings applied and saved");
}
```

## Testing

1. Open settings menu
2. Change temperature slider
3. Change beam size slider
4. Toggle suppress_blank
5. Click "Save & Close"
6. Check logs for:
   ```
   [DEBUG][UISettingsMenu] Applying whisper settings:
   [DEBUG][UISettingsMenu]   temperature: 0.5
   [DEBUG][UISettingsMenu]   beam_size: 7
   [DEBUG][UISettingsMenu]   suppress_blank: true
   [DEBUG][UISettingsMenu] Saved ai_config.json and updated global aiConfig
   ```
7. Use voice input immediately (without restarting)
8. Verify new settings are active

## Impact

? **Immediate Effect**: Changes take effect immediately without restart  
? **Whisper Settings**: All whisper parameters properly applied  
? **Voice Settings**: Speaker, engine changes work instantly  
? **Personality**: Personality prompt changes apply immediately  

## Verification

Check `ai_config.json` file after saving - all values should match what was set in the UI.

Use voice input right after changing whisper settings - the new temperature/beam_size should be active.

---

**Status**: Fix documented  
**Files to modify**: `ui/ui_settings_menu.cpp`  
**Lines to add**: 3 lines total

# Speaker Embedding UI Settings

## Overview

You can now select speaker embeddings directly from the **Settings Menu** in G.R.I.M!

## Accessing Settings

### Method 1: Command
```
settings
```

### Method 2: Keyboard
Press `~` (tilde) to open console, then type `settings`

### Method 3: Wake Word
```
"Hey GRIM, open settings"
```

## Using the Speaker Dropdown

### 1. Open Settings

The settings menu will appear as an overlay.

### 2. Locate Voice Section

You'll see:
```
Voice: coqui         [Click to cycle]
Speaker: default     [Dropdown ?]
```

### 3. Select Speaker

Click the **Speaker** dropdown to see all available voice embeddings:
```
?? Speaker ??????????
? ? default         ?
?   austin          ?
?   custom_voice    ?
?   (no custom...) ?  ? Shows if no embeddings exist
???????????????????-?
```

### 4. Save Changes

Click **Save & Close** to apply the new speaker.

## Features

### ? Auto-Detection
- Automatically scans `resources/voices/embeddings/` folder
- Shows all `.npz` embedding files
- Updates list when new embeddings are created

### ? Smart Display
- Only shows when **Voice Engine = Coqui**
- Hides for SAPI (doesn't use embeddings)
- Highlights current selection with ?

### ? Real-Time Preview
- Changes take effect on next TTS request
- No need to restart G.R.I.M

## Creating New Speakers

### From Console
```
create_embedding my_voice resources/voices/sample.wav
```

### Refresh Settings
Close and reopen settings menu to see the new speaker in dropdown.

## Workflow Example

```bash
# 1. Create embedding
create_embedding austin D:/G.R.I.M/resources/voices/my_voice.wav

# 2. Open settings
settings

# 3. Select "Speaker" dropdown

# 4. Choose "austin"

# 5. Click "Save & Close"

# 6. Test it
test_tts Hello, this uses my voice!
```

## Settings Persistence

Speaker selection is saved to `ai_config.json`:

```json
{
  "voice": {
    "engine": "coqui",
    "speaker": "austin"  ? Your selection
  }
}
```

## Troubleshooting

### Dropdown shows "(no custom voices)"
**Cause:** No embeddings found  
**Fix:** Create embeddings using `create_embedding` command

### Speaker not in dropdown
**Cause:** Embedding file not in correct location  
**Fix:** Check `D:/G.R.I.M/resources/voices/embeddings/` for `.npz` files

### Selection not saving
**Cause:** Settings not applied  
**Fix:** Make sure to click **"Save & Close"** (not just close window)

### Voice doesn't change
**Cause:** TTS cache using old voice  
**Fix:** Wait a few seconds or test with new text

## UI Controls

| Element | Action | Result |
|---------|--------|--------|
| **Speaker Dropdown** | Click | Opens speaker list |
| **Speaker Item** | Click | Selects speaker |
| **Save & Close** | Click | Applies and saves settings |
| **Cancel** | Click | Discards changes |
| **Scroll Wheel** | Scroll | Navigate long lists |

## Advanced

### Keyboard Navigation
- **Arrow keys** - Navigate dropdown items
- **Enter** - Select item
- **Escape** - Close dropdown

### Multiple Voices
Create multiple embeddings for different moods:
```
create_embedding austin_formal voice_formal.wav
create_embedding austin_casual voice_casual.wav
create_embedding austin_excited voice_excited.wav
```

Then switch between them easily via dropdown!

## Integration

The dropdown UI integrates with:
- ? Speaker embedding system
- ? TTS cache (automatic cache key updates)
- ? Coqui XTTS v2 bridge
- ? Settings persistence

All changes are immediately reflected in the TTS system without requiring restarts.

## Benefits

- ??? **No config file editing** - Visual selection
- ?? **Auto-discovery** - Finds all embeddings automatically
- ?? **Persistent** - Saves selection across sessions
- ?? **Simple** - One click to change voice
- ?? **Fast** - Changes apply immediately

Enjoy easy voice switching! ??

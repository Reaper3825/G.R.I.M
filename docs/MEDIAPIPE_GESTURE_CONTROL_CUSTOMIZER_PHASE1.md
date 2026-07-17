# MediaPipe Gesture Control Customizer - Phase 1

Phase 1 turns GRIM's fixed MediaPipe control mappings into an offline,
operator-editable binding profile. It customizes how recognized labels become
semantic control actions; it does not train new gesture models or invoke a
hosted service.

## Runtime contract

Every binding has a stable id, gesture label, allowlisted action, trigger phase,
hand filter, minimum hold, cooldown, armed requirement, enabled flag, and
priority. Higher-priority bindings are evaluated first when two enabled rows
overlap. Pointer and mouse actions are always forced to require an armed control
session even if imported configuration says otherwise.

The action allowlist is:

- `control.arm`
- `control.disarm`
- `pointer.move`
- `mouse.left_click`
- `mouse.right_click`
- `voice.wake`

No binding can contain a shell command, URL, prompt, or arbitrary command id.

## UI

Physical Environment -> Interaction now has Live and Bindings views. Both keep
the live camera and landmark preview visible on the left. Live shows runtime
diagnostics on the right; Bindings replaces that side panel with a scrollable
binding list and compact selected-binding editor. Bindings can be added, edited,
disabled, deleted, restored to defaults, populated from the strongest current
live gesture, and tested with global dry-run mode. Apply persists immediately
through GRIM's canonical runtime-config merge path.

## Persistence

Configuration is stored under:

```json
{
  "physical_interaction": {
    "gesture_control": {
      "schema": 1,
      "enabled": true,
      "dry_run": false,
      "bindings": [
        {
          "id": "wake",
          "gesture": "ILoveYou",
          "action": "voice.wake",
          "trigger": "held",
          "hand": "either",
          "minimum_hold_ms": 1100,
          "cooldown_ms": 10000,
          "requires_armed": false,
          "enabled": true,
          "priority": 80
        }
      ]
    }
  }
}
```

A missing section uses the built-in safe defaults. An invalid schema or unknown
action is rejected as a whole during startup; GRIM continues with the defaults.
There is no network path in loading, editing, validating, applying, or saving
these bindings.

## Deferred work

Capturing examples and recognizing genuinely new static poses from MediaPipe
landmarks is the next phase. Offline `.task` training/export remains optional
and is not part of this phase.

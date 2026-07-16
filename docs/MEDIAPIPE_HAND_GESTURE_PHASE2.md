# MediaPipe Hand Gesture Integration — Phase 2

Phase 2 turns Phase 1's frame-level hand recognition into guarded local computer
control. It remains offline and does not project gesture data into LLM context.

## Pipeline

1. `PhysicalHandGestureBus` publishes the latest completed inference result.
2. `PhysicalGestureControlLoop` processes each source frame exactly once.
3. Confidence hysteresis, activation dwell, release grace, and result-staleness
   checks produce `Started`, `Held`, and `Released` events.
4. `PhysicalGestureEventBus` retains the latest 128 semantic events so edges are
   not lost like they could be in a latest-value slot.
5. The guarded router sends only explicit mouse or wake actions through
   `PlatformInput` and `WakeKey`.

Raw MediaPipe labels never execute actions directly.

## Default control vocabulary

| Gesture | Behavior | Guard |
|---|---|---|
| `Victory` | Arm computer control | Hold 800 ms |
| `Open_Palm` | Disarm | Hold 700 ms |
| `Pointing_Up` | Move cursor using index fingertip | Armed session required |
| `Closed_Fist` | Left click | Armed + 650 ms click cooldown |
| `Thumb_Down` | Right click | Armed + 650 ms click cooldown |
| `ILoveYou` | Request wake and voice capture | Hold 1100 ms + 10 s cooldown |

Pointer/click control automatically disarms after 15 seconds without routed
control activity. Wake is deliberately usable while unarmed by default because
it is itself an activation mechanism, but it has a long hold, one-shot latch,
and cooldown. Set `wake_requires_armed=true` for stricter installations.

## Pointer behavior

The index fingertip is landmark 8. Motion is relative rather than absolute, so
entering pointer mode does not teleport the cursor. Normalized camera-space
deltas pass through a deadzone, exponential smoothing, gain, and a maximum
per-result step before OS injection.

Switching away from `Pointing_Up`, disarming, losing the hand, stale inference,
or controller shutdown resets pointer history. Phase 2 emits complete clicks;
it does not hold mouse buttons down, so a lost camera cannot leave the OS in a
dragging state.

## Configuration surface

Call `RequestConfigurePhysicalGestureControl()` with a
`PhysicalGestureControlConfig`. It exposes:

- Preferred hand, entry/exit confidence, dwell, release, and stale timers
- Gesture-to-action labels
- Arm/disarm timing and armed-session timeout
- Pointer gain, smoothing, deadzone, inversion, and maximum step
- Click and wake cooldowns
- Whether wake requires an armed session

The Physical Environment Interaction diagnostics show read-only controller
state, arming, stable event, action counts, blocked reasons, and injection
errors. Configuration intentionally remains outside the UI in this phase.

## Platform output boundary

- Windows uses `SendInput` for relative movement and complete button clicks.
- macOS posts CoreGraphics mouse events. Accessibility permission may be
  required; rejected injection is reported as an action failure.
- X11 uses pointer warping and synthetic button events. Wayland or clients that
  reject `send_event` input will report failure or may require a compositor-
  specific input portal in a later platform adapter.

Wake gestures call `WakeKey::requestWake("physical_gesture")`, which publishes a
central motion wake event and enters the same voice-capture path as the saved
wake-key binding.

## Acceptance checks

1. An incidental gesture cannot move or click the pointer before arming.
2. One noisy classification frame cannot arm, click, or wake the system.
3. Holding `Pointing_Up` moves smoothly without cursor teleportation.
4. A stable click gesture emits one complete click, not repeated down events.
5. Hand loss releases the stable event and resets pointer history.
6. Armed control times out and disarms without further activity.
7. A rejected OS injection is visible in logs and controller status.
8. Phase 1 recognition and the rest of physical perception continue when the
   controller is disabled.

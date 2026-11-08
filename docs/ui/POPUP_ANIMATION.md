# Popup Animation and Presentation

The current popup presents rendered 3D frames. It is separate from overlay
panel styling; sprite-era window scaling and fragment-shader replacement
tutorials are not the current animation workflow.

## Speech-driven presentation

[pop_ui.cpp](../../popup_ui/pop_ui.cpp) reads `Voice::isSpeaking()`, updates
`PopupAnimState`, and switches persistent `presence_idle` / `speech_pulse`
poses when speech state changes. Visibility requests use `load_in` / `fade_out`
transitions; speech can request that a hidden popup be shown.

The loop provides the 3D renderer's base transform scale:

```text
scale = (1 + breathe * 0.02) * (1 + pulse * 0.08)
```

It supplies speech intensity as emissive input, while render-side clips layer
their own scale/alpha effects. `updateVoiceAnim()` in
[popup_anim.cpp](../../popup_ui/popup_anim.cpp) resets and decays a speech-start
pulse, smooths intensity, and maintains an idle oscillation. It is not the old
3-Hz perpetual pulse, "breathing removed", or fixed 20-FPS sprite contract.

[popup_3d_renderer.cpp](../../popup_ui/popup_3d_renderer.cpp) evaluates animation
and produces frames. `popupMailboxConsume()` must deliver a new generation
before `presentPopup3DFrame()` updates the native layered window.
Changing animation inputs alone does not certify that a fresh frame was
rendered/presented.

## Presets and style changes

[popup_anim_presets.hpp](../../popup_ui/popup_anim_presets.hpp) distinguishes
one-shot clips from persistent blend poses. Presets can layer scalar
alpha/scale/emissive effects, spin, pose offsets, and mesh-cache animation.
Initialization loads built-ins and configured JSON overrides.

Use the current preset/asset contract for style changes; do not replace a
sprite shader or invent a new animation branch from old snippets.
Alternative fade/flash/wave styles, amplitude-driven effects, and waveform
visualization are design possibilities, not advertised implemented settings.
Any chosen extension belongs here or in the separate [3D design](POPUP_3D.md).

## Troubleshooting

1. Check actual visibility and transition state.
2. Check TTS readiness/playback independently of animation; failed synthesis
   is not necessarily a render failure.
3. Check speaking-state changes, loop delta time, and renderer initialization.
4. If state changes but the image freezes, inspect render submission and mailbox
   generation before adding window-resampling code.
5. Inspect actual asset/preset configuration rather than old sprite paths.

Provider selection belongs to
[voice configuration](../VOICE_TUNING_GUIDE.md); the selected CosyVoice path
is not a SAPI/XTTS animation requirement.

## Historical performance notes

The old GDI `StretchBlt` fix described a sprite-scaling pipeline and unverified
2-to-60-FPS/GPU-acceleration claims. It is not proof of the current 3D readback
or presentation cost and must not be reapplied as the active renderer design.
Likewise, a 16-ms loop sleep is not a measured 60-FPS guarantee.

Measure current render, readback/mailbox, presentation, and visibility costs
separately before claiming performance. This consolidation ran no host,
voice, renderer, or model runtime.

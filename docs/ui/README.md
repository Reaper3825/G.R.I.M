# UI Documentation

Shared UI contracts live here. Keep implementation guidance with its owning
contract rather than creating another plan or fix-summary document.

| Guide | Scope |
|---|---|
| [Rendering and theme](RENDERING.md) | Overlay renderer modules, glass/backdrop, theme, and styling |
| [Widgets, input, and layout](WIDGETS_INPUT_LAYOUT.md) | Click/capture semantics, containers, Settings refresh, and canvas geometry |
| [Panel operations](PANEL_OPERATIONS.md) | Stable keys, UI-thread operations, plugin requests, completion, and dynamic surfaces |
| [Console](CONSOLE.md) | Prompt entry, Settings access, shortcuts, and panel visibility |
| [Popup animation](POPUP_ANIMATION.md) | Speech-reactive 3D presentation, presets, mailbox, and troubleshooting |
| [Popup 3D design](POPUP_3D.md) | Separate native popup renderer/asset design and implementation plan |
| [Observatory](OBSERVATORY.md) | Inspection/capture contract, user controls, and developer primitives |
| [Training panel](TRAINING_PANEL.md) | Actual tokenizer/config-compiler connections and legacy control-server gaps |
| [OSINT panel](OSINT_PANEL.md) | OSINT-specific result presentation |

## Related feature contracts

- [Desktop input](../input/README.md) owns registered OS-input tools and
  cursor/capture policy; UI widget routing remains in this category.
- [Model/training](../model/README.md) owns math and runtime state behind the
  inspection and training-panel features.
- [OSINT](../osint/README.md) owns scan/tool/storage behavior behind its panel.
- [Voice and voice settings](../VOICE_TUNING_GUIDE.md) owns provider/recognition
  behavior and voice-specific Settings controls.
- [World Lab](../WORLD_LAB.md), [World Lab reference](../WORLD_LAB_REFERENCE.md),
  and [Live World](../LIVE_WORLD.md) stay at the feature boundary: they also
  describe environment dispatch, Goal evaluation, and host resource contracts,
  not only UI widgets.
- [Plugin API](../PLUGIN_API.md) owns ABI/service contracts.
- [Physical perception](../PHYSICAL_PERCEPTION_SUBSYSTEM.md) owns cameras,
  gesture control, and device state.
- [Legacy documents](../legacy/) retain retired architectures and snippets.

## Organization rules

The rendering and widget guides replace the old glassmorphism, polish,
container, button-fix, and canvas documents. Popup animation replaces the
overlapping sprite-era style/performance notes. Observatory prerequisites are
included with the Observatory contract; voice Settings integration belongs in
the voice guide.

Existing feature plans retain their own implemented/planned boundaries.
Moving a plan here does not certify that every proposed feature is implemented.
New design work should use a clearly marked section of its relevant guide.
Do not treat speculative colors, timings, or old performance tables as current
runtime guarantees.

This organization changes documentation only. It does not change UI/source
behavior, build or start the application, or introduce a new UI architecture.

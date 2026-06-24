#pragma once

#include "popup_3d_types.hpp"
#include <string>
#include <cstdint>

// ===========================================================
// Popup animation presets + clip engine
// ===========================================================
// A "preset" is a named animation clip that can drive any combination of:
//   - a baked geometry-node mesh-cache clip (.gmc), and/or
//   - scalar tweens for alpha / scale / emissive, and/or
//   - a Y-axis spin.
// Presets are evaluated on the render thread and layered on top of the
// existing per-frame PopupRenderInput (voice/breathe stay intact).

enum class PopupEasing : uint8_t
{
    Linear,
    EaseInCubic,
    EaseOutCubic,
    EaseInOutCubic,
    EaseOutBack,    // slight overshoot — nice for "pop in"
    EaseInBack
};

// A scalar A->B tween over the normalized clip time [0,1].
struct PopupTween
{
    bool        enabled = false;
    float       from    = 0.0f;
    float       to      = 1.0f;
    PopupEasing easing  = PopupEasing::EaseOutCubic;
};

// A named animation preset.
struct PopupAnimPreset
{
    std::string name;
    float       durationSec   = 0.45f;  // duration of the tween portion
    bool        loop          = false;
    bool        holdLastValue = true;   // when finished & not looping, hold end state

    PopupTween  alpha;     // multiplies onto input.alphaMul
    PopupTween  scale;     // multiplies onto input.transform.scale
    PopupTween  emissive;  // adds to input.emissiveMul
    float       spinTurns = 0.0f;  // full Y revolutions across the clip (additive)

    std::string meshCache;  // optional .gmc filename (relative to popup_3d dir); "" = none
};

// Output of evaluating the active preset for the current time.
struct PopupClipEval
{
    bool  active = false;        // a preset is currently playing
    float alphaMul    = 1.0f;    // multiply onto input alpha
    float scaleMul    = 1.0f;    // multiply onto input scale
    float emissiveAdd = 0.0f;    // add to input emissive
    float spinY       = 0.0f;    // additive Y rotation (radians)

    // Non-null when a baked mesh-cache frame should be drawn this frame.
    const PopupMeshFrame* frame = nullptr;

    bool finished = false;       // tween reached its end (non-looping)
};

// Opaque engine (owns presets, loaded caches, and playback state).
struct PopupClipEngine;

PopupClipEngine* popupClipEngineCreate();
void             popupClipEngineDestroy(PopupClipEngine* engine);

// Populate built-in presets ("load_in", "fade_out", "idle_spin").
void popupClipEngineLoadDefaults(PopupClipEngine* engine);

// Load presets from a JSON file (see resources/popup_3d/anim_presets.json).
// Any referenced .gmc mesh caches are preloaded from `popup3dDir`.
// Returns true on success; on failure the engine is left with its current presets.
bool popupClipEngineLoadJson(PopupClipEngine* engine,
                             const std::string& jsonPath,
                             const std::string& popup3dDir,
                             uint32_t defaultColorABGR);

// Request playback of a preset by name (thread-safe; safe to call from the UI thread).
// No-op if the name is unknown.
void popupClipEngineTrigger(PopupClipEngine* engine, const char* presetName);

// Evaluate the active preset at `nowSeconds` (steady-clock seconds).
// Returns true if a preset is active and `out` was filled.
bool popupClipEngineEvaluate(PopupClipEngine* engine, double nowSeconds, PopupClipEval& out);

// Largest per-frame vertex/index counts across all loaded mesh caches
// (used to size the dynamic GPU buffers). Zero if no caches are loaded.
void popupClipEngineMaxBuffer(const PopupClipEngine* engine,
                              uint32_t* maxVerts, uint32_t* maxIndices);

// True if any loaded preset references a mesh cache.
bool popupClipEngineHasGeometry(const PopupClipEngine* engine);

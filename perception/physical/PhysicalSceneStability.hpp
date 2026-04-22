#pragma once

#include <cstdint>
#include <string>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalSceneStability
//
//  A single per-frame signal produced by PhysicalFrameConditioner that lets
//  every downstream operator decide whether the scene meaningfully changed
//  since its last inference. Carried alongside the model image on
//  PhysicalFrameBus so the gate is computed exactly once.
//
//  Computation (in PhysicalFrameConditioner, after the model image is built):
//    1. Downscale the conditioned BGR image to a small grayscale thumbnail
//       (config.thumbnail_width × thumbnail_height, INTER_AREA).
//    2. motion_magnitude = mean absolute difference vs the previous
//       thumbnail, in 0..255 luma units.
//    3. scene_hash_64 = 64-bit average-hash of the thumbnail (8x8
//       luma-vs-mean bitmap when thumbnail >= 8x8, else exact pixel bits).
//    4. is_stable = (motion_magnitude <= motion_threshold) AND
//                   (popcount(scene_hash_64 XOR previous_hash) <= hash_hamming_threshold)
//    5. frames_since_change advances by 1 each stable frame; resets to 0
//       when is_stable is false. Capped at max_stable_streak_frames so a
//       static-scene gate can be force-broken (prevents permanent skip if a
//       cached result drifts out of date for some other reason).
//    6. change_reason is populated whenever is_stable is false, with a
//       human-readable explanation. Empty when stable.
//
//  Rule 20 contract: every consumer that decides to REUSE a cached result
//  based on this signal MUST surface that decision on its output via
//  PhysicalCacheStatus below — never silently re-publish stale data.
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalSceneStabilityConfig {
    bool     enable                      = true;

    // Thumbnail used for both MAD and average-hash. 64x36 ≈ 2k pixels —
    // negligible compared to a 640x360 model image.
    int      thumbnail_width             = 64;
    int      thumbnail_height            = 36;

    // Mean-absolute-difference threshold in 0..255 luma units. Anything
    // above this is treated as motion. 4.0 ≈ ~1.5% of luma range — small
    // enough to pick up a hand wave at the edge of frame, large enough to
    // ignore sensor noise on a tripod.
    double   motion_threshold            = 4.0;

    // Maximum Hamming distance (0..64) between consecutive 64-bit average
    // hashes that still counts as the "same scene". 8 is conservative —
    // a single subtle lighting shift typically flips ~2-4 bits.
    int      hash_hamming_threshold      = 8;

    // Hard cap on consecutive stable frames before the gate is force-broken
    // and operators are required to refresh their caches. 600 frames at
    // 30 fps = 20s. Prevents indefinite drift if a scene happens to be
    // bit-identical (e.g. webcam pointed at a static poster).
    uint32_t max_stable_streak_frames    = 600;
};

struct PhysicalSceneStability {
    // True only when the conditioner actually computed the signal this
    // frame. If false, every other field is meaningless and consumers MUST
    // behave as if the scene changed (i.e. always run inference). This is
    // the loud-failure path required by Rule 20: operators do NOT silently
    // skip inference when the signal is missing.
    bool     valid                       = false;

    double   motion_magnitude            = 0.0;     // 0..255 luma units (MAD)
    uint64_t scene_hash_64               = 0;       // average-hash bitmap
    int      hamming_vs_previous         = 0;       // 0..64

    bool     is_stable                   = false;
    uint32_t frames_since_change         = 0;       // capped per config
    bool     stable_streak_capped        = false;   // true when frames_since_change hit cap

    // Empty when is_stable; otherwise a short reason describing why the
    // gate broke (motion / hash / first-frame / streak-capped / disabled).
    std::string change_reason;
};

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalCacheStatus
//
//  Attached to every cache-aware operator output envelope. Loud-by-default:
//  if cache_hit is true, the consumer KNOWS the result was reused without
//  fresh inference and can choose to render it differently (e.g. dim the
//  overlay, show "cached" badge).
//
//  cache_age_frames counts the number of frames between the inference that
//  produced the cached result and the frame this envelope is published for
//  (0 means "fresh", N means "this result was last computed N frames ago").
//
//  cache_reason is a short tag from a fixed vocabulary so log analysis is
//  trivial:
//    ""                    — fresh inference (cache_hit == false)
//    "stable_scene"        — reused because PhysicalSceneStability.is_stable
//    "cadence_floor"       — reused because min_period_ms not yet elapsed
//    "stable_and_cadence"  — both conditions held
//    "no_signal"           — scene-stability signal absent; operator chose
//                            to run anyway (cache_hit == false). Logged for
//                            observability.
// ─────────────────────────────────────────────────────────────────────────────
struct PhysicalCacheStatus {
    bool        cache_hit         = false;
    uint32_t    cache_age_frames  = 0;
    std::string cache_reason;
};

// Per-operator cadence knobs embedded in each operator's *Config struct.
// Defaults preserve the pre-optimization behaviour exactly: every frame,
// no reuse — so adding the field is a pure no-op until a config is loaded
// that turns it on.
struct PhysicalOperatorCadenceConfig {
    // Minimum wall time between successive fresh inferences, in
    // milliseconds. 0 means "no floor — run as fast as the loop ticks".
    uint32_t    min_period_ms          = 0;

    // When true, AND PhysicalSceneStability.is_stable is true on the
    // current frame, the loop driver will reuse the last fresh result
    // and mark cache_hit=true. When false, the operator runs every
    // frame regardless of scene stability (cadence floor still applies).
    bool        reuse_on_stable_scene  = false;
};

}}} // namespace GRIM::Perception::Physical

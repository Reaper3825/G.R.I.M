#pragma once

#include "PhysicalImageOperatorState.hpp"
#include "PhysicalPerceptionPrimitiveResult.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalClassPolicy — Stage-2.5 class normalisation + ranking layer.
//
//  Sits BETWEEN the per-operator results (object detector, entity tracker,
//  instance segmenter, image classifier) and the model-context-matrix
//  builder. Two responsibilities, one pass:
//
//    1. CLASS-MERGE MAP — collapse synonymous class labels emitted by
//       different detectors into a single CANONICAL label. e.g.:
//           "tv" + "monitor" + "laptop_screen" → "screen"
//           "person" + "pedestrian"            → "person"
//       Every signal in PhysicalPerceptionPrimitiveResults that carries a
//       class_label is rewritten in-place. Per-detector class_id values
//       are preserved so the original detector identity is not lost.
//
//    2. CLASS-PRIORITY POLICY — assign each canonical class a
//       priority_rank (lower = higher priority) and a confidence_floor.
//       Items below the floor are dropped from the per-operator vectors;
//       items whose canonical class has a rank greater than
//       emit_only_top_rank (when > 0) are also dropped. The surviving
//       items are summarised into a sorted vector of
//       PhysicalClassPolicyClassSummary rows that downstream consumers
//       (model-context builder, UI) iterate over directly.
//
//  Self-owned, internally managed. Mainloop integration point is exactly
//  one call from PhysicalPerceptionPrimitivesLoop:
//
//      ApplyPhysicalClassPolicyToPerceptionResults(results, results.class_policy);
//
//  Numerical-precision contract:
//    * priority_rank is an int — never converted via float.
//    * confidence_floor is in [0, 1]; comparisons use >=, not >, so a
//      floor of 0.0 is a strict pass-through.
//    * Mutation of the per-operator vectors uses stable_partition so the
//      original per-operator ordering is preserved among survivors.
//    * When the rule set is EMPTY, Apply* is a guaranteed no-op apart from
//      writing inference_count / last_apply_ms / last_frame_counter. Zero
//      labels rewritten, zero items dropped, zero ranked rows.
//
//  Rule 20 / Rule 3:
//    * Configure throws on conflicting rules (alias mapped to two different
//      canonicals, canonical declared with two different ranks, etc.).
//    * Apply catches and reports via out.last_error_reason / state ==
//      InferenceFailed. Never silently swallows.
// ─────────────────────────────────────────────────────────────────────────────

// One merge rule: rewrite every detection/track/mask/classification whose
// class_label is in source_labels to canonical_label. If canonical_label is
// itself in source_labels it is a self-rule (allowed; lets you declare a
// canonical class without aliases just to attach a priority to it).
struct PhysicalClassMergeRule {
    std::string              canonical_label;
    std::vector<std::string> source_labels;
};

// One priority rule for a single canonical class.
//   priority_rank     : 1 = highest priority. Unranked classes get
//                       cfg.default_priority_rank.
//   confidence_floor  : items below this confidence are dropped. Unranked
//                       classes get cfg.default_confidence_floor.
struct PhysicalClassPriorityRule {
    std::string canonical_label;
    int32_t     priority_rank    = 1000;
    float       confidence_floor = 0.0f;
};

struct PhysicalClassPolicyConfig {
    std::vector<PhysicalClassMergeRule>    merge_rules;
    std::vector<PhysicalClassPriorityRule> priority_rules;

    // Defaults applied to canonical classes that have no priority rule.
    int32_t  default_priority_rank    = 1000;
    float    default_confidence_floor = 0.0f;

    // When > 0, items whose canonical class has priority_rank greater than
    // this are dropped before the ranked summary is built. 0 disables the
    // cutoff entirely (everything that survives the confidence floor is
    // emitted). MUST be >= 0; throws otherwise.
    int32_t  emit_only_top_rank       = 0;

    // Cross-class de-duplication. After merge-relabel, two items that came
    // from DIFFERENT original classes (e.g. "chair" and "couch") will share
    // the same canonical label and now spatially overlap on the same
    // physical object. This is the IoU threshold above which a lower-ranked
    // overlapping item is suppressed. Applied per-section to:
    //   * results.object_detector.detections      (using model_box)
    //   * results.entity_tracker.tracks            (using smoothed_model_box)
    //   * results.instance_segmenter ... .instances (using prompt_model_box)
    // Tie-break ranking inside same canonical class:
    //   higher confidence wins (tracks: smoothed_confidence; dets: confidence;
    //   masks: detection_confidence). 0.0f disables the pass entirely.
    //   MUST be in [0, 1]; throws otherwise.
    float    post_merge_nms_iou       = 0.0f;
};

class PhysicalClassPolicy {
public:
    PhysicalClassPolicy();
    ~PhysicalClassPolicy();

    // Apply (or re-apply) configuration. Throws on:
    //   * empty canonical_label in any merge rule
    //   * empty source_labels in any merge rule
    //   * the same alias mapped to two different canonicals
    //   * the same canonical declared in two different priority rules
    //     with different ranks/floors
    //   * confidence_floor outside [0, 1]
    //   * emit_only_top_rank < 0
    // Calling with the default-constructed config (empty rule set) is the
    // canonical pass-through "ready" state — it transitions directly to
    // ModelLoaded.
    void ConfigurePhysicalClassPolicy(const PhysicalClassPolicyConfig& cfg);

    // Per-frame entry point. Never throws — failures land in `out`.
    // Mutates results.object_detector.detections, results.entity_tracker
    // .tracks, results.instance_segmenter.segmentation.instances, and
    // results.image_classifier.top_k IN PLACE: relabels by canonical and
    // drops items that fail the confidence floor or the priority cutoff.
    // Pass `out` as `results.class_policy` from the loop — Apply does not
    // read class_policy itself, only writes its summary fields.
    void ApplyPhysicalClassPolicyToPerceptionResults(
        PhysicalPerceptionPrimitiveResults& results,
        PhysicalClassPolicyOutput& out);

    // Drop the rule set; returns to NoModelConfigured.
    void                       ResetPhysicalClassPolicy();

    PhysicalImageOperatorState GetPhysicalClassPolicyState() const;
    std::string                GetPhysicalClassPolicyLastError() const;
    bool                       IsPhysicalClassPolicyReady() const;

    // Read-only inspector — used by the UI panel to render the active rule
    // set without re-querying the loop's pending config.
    PhysicalClassPolicyConfig  GetPhysicalClassPolicyConfig() const;

private:
    mutable std::mutex            mutex_;
    PhysicalClassPolicyConfig     cfg_{};

    // Indexes built once at Configure for O(1) per-item lookup.
    //   alias_to_canonical_  : every source_label → canonical_label
    //                          (every canonical_label maps to itself by
    //                          default, so an unaliased pass-through still
    //                          lands on the priority side).
    //   canonical_to_rank_   : canonical_label → priority_rank
    //   canonical_to_floor_  : canonical_label → confidence_floor
    std::unordered_map<std::string, std::string> alias_to_canonical_;
    std::unordered_map<std::string, int32_t>     canonical_to_rank_;
    std::unordered_map<std::string, float>       canonical_to_floor_;

    PhysicalImageOperatorState    state_ = PhysicalImageOperatorState::NoModelConfigured;
    std::string                   last_error_reason_;
    uint64_t                      inference_count_ = 0;
};

}}} // namespace GRIM::Perception::Physical

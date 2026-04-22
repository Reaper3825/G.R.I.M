#pragma once

#include "PhysicalPerceptionPrimitiveResult.hpp"
#include "PhysicalSpatialGroundingResult.hpp"
#include "PhysicalWorldStateResult.hpp"

namespace GRIM { namespace Perception { namespace Physical {

// ─────────────────────────────────────────────────────────────────────────────
//  PhysicalWorldStateBuilder — pure, deterministic fusion of the Stage-2 +
//  Stage-3 surfaces into a Stage-4 PhysicalWorldStateSnapshot.
//
//  No threads, no buses, no I/O — just a function. The PhysicalWorldStateLoop
//  owns the threading + bus plumbing; this builder is unit-testable in
//  isolation.
//
//  Algorithm (all O(N²) bounded by entity count, which is small):
//    1. SEED entities directly from perc.entity_tracker.tracks. Identity is
//       track_id; nothing else may invent or merge identities.
//    2. ENRICH WITH GROUNDING — for each entity, look up
//       ground.grounded_entities by track_id and copy depth / surface /
//       motion fields. Missing grounding ⇒ has_depth = false.
//    3. ATTACH MASKS — for each entity, find the instance mask with the
//       highest IoU(entity.model_box, mask.mask_model_bbox). Sets
//       has_instance_mask + instance_mask_pixel_count.
//    4. ATTACH SCENE TEXT — for each scene-text line, compute the centroid
//       of model_quad. Append text to whichever entity's model_box contains
//       the centroid (single-owner; no double-attach).
//    5. COMPUTE OCCLUSION — for every ordered pair (A, B) where both have
//       depth and B is closer than A by depth_difference_for_nearer, if
//       intersection(A.model_box, B.model_box) / area(A.model_box) >=
//       min_overlap_for_occlusion, then A.occluded_by_object_id = B.object_id
//       (the most-occluding B wins). Sets visible_area_fraction.
//    6. SET VISIBILITY — Coasting ⇒ Coasting; else occluded_by != 0 ⇒
//       Occluded; else Visible.
//    7. COMPUTE RELATIONS — for every ordered pair, emit at most one of
//       {Contains, ContainedBy, Overlaps, LeftOf, RightOf, Above, Below}
//       based on box geometry, plus {NearerThan, FartherThan} when both
//       have depth. Per-entity relations are sorted by other_object_id ASC
//       and capped to cfg.max_relations_per_entity (closest centres win).
//    8. SORT entities by object_id ASC for stable diffing.
//
//  Numerical-precision contract:
//    * Centres are computed as box.x + box.width/2 (no float drift over
//      multiple frames — pure function of the snapshot box).
//    * raw_centre uses the SAME affine transform stored in raw_to_model;
//      we never invert and re-derive.
//    * Relation strengths are clamped to [0, 1].
// ─────────────────────────────────────────────────────────────────────────────

struct PhysicalWorldStateBuilderConfig {
    // Cap on relations per entity. We keep the K nearest other entities
    // (by centre-distance in MODEL pixels) so the snapshot is bounded even
    // in busy scenes.
    int   max_relations_per_entity   = 4;

    // Box-overlap fraction needed to declare A occluded by B (where B is
    // closer than A). Computed as intersection / area_self. Range (0, 1].
    float min_overlap_for_occlusion  = 0.30f;

    // Minimum |range_self - range_other| required to assert NearerThan /
    // FartherThan (or to count B as "closer" for occlusion). The unit is
    // whatever DepthUnits the entity uses (Relative ⇒ normalised; Meters
    // ⇒ metres). Same threshold for both — the caller picks the unit.
    float depth_difference_for_nearer = 0.05f;

    // Axis dominance for LeftOf/RightOf vs Above/Below. We emit a
    // horizontal relation when |dx| / (|dx| + |dy| + eps) >= this; vertical
    // when |dy| / (|dx| + |dy| + eps) >= this. Range (0.5, 1.0].
    float position_axis_dominance     = 0.60f;
};

// Pure function. Throws std::invalid_argument on out-of-range cfg
// (Rule 20 — fail loud at the boundary). Never throws on missing inputs;
// missing grounding / masks / text just produces a sparser snapshot.
void BuildPhysicalWorldStateSnapshot(
    const PhysicalPerceptionPrimitiveResults& perc,
    const PhysicalSpatialGroundingResults&    ground,
    const PhysicalWorldStateBuilderConfig&    cfg,
    PhysicalWorldStateSnapshot&               out);

}}} // namespace GRIM::Perception::Physical

#include "PhysicalWorldStateBuilder.hpp"

#include "PhysicalWorldStateLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// ─── Geometry helpers ──────────────────────────────────────────────────────

float RectArea(const cv::Rect2f& r) {
    return std::max(0.0f, r.width) * std::max(0.0f, r.height);
}

float IntersectionArea(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float x1 = std::max(a.x, b.x);
    const float y1 = std::max(a.y, b.y);
    const float x2 = std::min(a.x + a.width,  b.x + b.width);
    const float y2 = std::min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    return (x2 - x1) * (y2 - y1);
}

float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    const float inter = IntersectionArea(a, b);
    if (inter <= 0.0f) return 0.0f;
    const float uni = RectArea(a) + RectArea(b) - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

bool RectContains(const cv::Rect2f& outer, const cv::Rect2f& inner) {
    // Strict containment with a 1-px floating-point slack so a perfectly
    // coincident edge still counts as "Contains".
    constexpr float kSlack = 1.0f;
    return inner.x      >= outer.x      - kSlack
        && inner.y      >= outer.y      - kSlack
        && inner.x + inner.width  <= outer.x + outer.width  + kSlack
        && inner.y + inner.height <= outer.y + outer.height + kSlack;
}

cv::Point2f BoxCentre(const cv::Rect2f& r) {
    return cv::Point2f(r.x + r.width  * 0.5f, r.y + r.height * 0.5f);
}

cv::Point2f QuadCentroid(const PhysicalQuad& q) {
    return cv::Point2f(
        (q.p0.x + q.p1.x + q.p2.x + q.p3.x) * 0.25f,
        (q.p0.y + q.p1.y + q.p2.y + q.p3.y) * 0.25f);
}

bool ModelBoxContainsPoint(const cv::Rect2f& box, const cv::Point2f& p) {
    return p.x >= box.x && p.x <  box.x + box.width
        && p.y >= box.y && p.y <  box.y + box.height;
}

float Clamp01(float v) { return std::max(0.0f, std::min(1.0f, v)); }

void ValidateBuilderConfig(const PhysicalWorldStateBuilderConfig& cfg) {
    if (cfg.max_relations_per_entity < 0) {
        throw std::invalid_argument(
            "PhysicalWorldStateBuilderConfig.max_relations_per_entity < 0");
    }
    if (!(cfg.min_overlap_for_occlusion > 0.0f && cfg.min_overlap_for_occlusion <= 1.0f)) {
        throw std::invalid_argument(
            "PhysicalWorldStateBuilderConfig.min_overlap_for_occlusion must be in (0, 1]");
    }
    if (cfg.depth_difference_for_nearer < 0.0f) {
        throw std::invalid_argument(
            "PhysicalWorldStateBuilderConfig.depth_difference_for_nearer < 0");
    }
    if (!(cfg.position_axis_dominance > 0.5f && cfg.position_axis_dominance <= 1.0f)) {
        throw std::invalid_argument(
            "PhysicalWorldStateBuilderConfig.position_axis_dominance must be in (0.5, 1]");
    }
}

cv::Point2f ApplyRawToModelInverse(const PhysicalSignalRawToModelTransform& t,
                                   const cv::Point2f& model_pt)
{
    // We need raw_centre. The forward transform is
    //   model = raw * scale + offset
    // so raw = (model - offset) / scale. scale must be non-zero — the
    // conditioner guarantees this; assert defensively.
    const double sx = (t.scale_x != 0.0) ? t.scale_x : 1.0;
    const double sy = (t.scale_y != 0.0) ? t.scale_y : 1.0;
    return cv::Point2f(
        static_cast<float>((model_pt.x - t.offset_x) / sx),
        static_cast<float>((model_pt.y - t.offset_y) / sy));
}

// ─── Per-entity helpers ────────────────────────────────────────────────────

PhysicalWorldEntity SeedEntityFromTrack(const PhysicalEntityTrack& tr) {
    PhysicalWorldEntity e;
    e.object_id              = tr.track_id;
    e.class_id               = tr.class_id;
    e.class_label            = tr.class_label;
    e.confidence             = tr.smoothed_confidence;
    e.track_state            = tr.state;
    e.model_box              = tr.smoothed_model_box;
    e.raw_box                = tr.smoothed_raw_box;
    e.model_centre           = BoxCentre(tr.smoothed_model_box);
    e.raw_centre             = BoxCentre(tr.smoothed_raw_box);
    e.velocity_model_px_per_sec_x = static_cast<float>(tr.velocity_px_per_sec_x);
    e.velocity_model_px_per_sec_y = static_cast<float>(tr.velocity_px_per_sec_y);
    e.first_seen_steady_ns   = tr.first_seen_steady_ns;
    e.last_seen_steady_ns    = tr.last_update_steady_ns;
    e.last_seen_frame_counter = tr.last_update_frame_counter;
    e.age_in_frames          = tr.age_in_frames;
    e.hit_streak             = tr.hit_streak;
    e.miss_streak            = tr.miss_streak;
    // Coasting tracks have no fresh visual evidence; mark visibility upfront
    // so step 6 only has to override Visible/Occluded for non-coasting tracks.
    if (tr.state == PhysicalEntityTrackState::Coasting) {
        e.visibility = PhysicalEntityVisibility::Coasting;
    }
    return e;
}

void EnrichWithGrounding(PhysicalWorldEntity& e,
                         const PhysicalGroundedEntity& g)
{
    e.has_depth                  = true;
    e.depth_units                = g.units;
    e.range_value                = g.range_value;
    e.range_value_meters         = g.range_value_meters;
    e.range_confidence           = g.range_confidence;
    e.support_surface            = g.support_surface;
    e.support_surface_score      = g.support_surface_score;
    e.path_blocked               = g.path_blocked;
    e.path_block_score           = g.path_block_score;
    e.depth_velocity_units_per_sec = g.depth_velocity_units_per_sec;
    e.motion_state               = g.motion_state;
    e.moved_since_last_frame     = g.moved_since_last_frame;
    // The grounded entity's velocity is more authoritative if present
    // (it copies from the same track but is recomputed at fusion time).
    e.velocity_model_px_per_sec_x = g.velocity_model_px_per_sec_x;
    e.velocity_model_px_per_sec_y = g.velocity_model_px_per_sec_y;
}

void AttachBestInstanceMask(PhysicalWorldEntity& e,
                            const std::vector<PhysicalInstanceMask>& masks)
{
    if (masks.empty()) return;
    float    best_iou        = 0.0f;
    int32_t  best_pixel_cnt  = 0;
    for (const auto& m : masks) {
        // mask_model_bbox is integer; widen to float for IoU.
        const cv::Rect2f mb(
            static_cast<float>(m.mask_model_bbox.x),
            static_cast<float>(m.mask_model_bbox.y),
            static_cast<float>(m.mask_model_bbox.width),
            static_cast<float>(m.mask_model_bbox.height));
        const float iou = IoU(e.model_box, mb);
        if (iou > best_iou) {
            best_iou       = iou;
            best_pixel_cnt = m.mask_pixel_count;
        }
    }
    if (best_iou > 0.0f) {
        e.has_instance_mask         = true;
        e.instance_mask_pixel_count = best_pixel_cnt;
    }
}

void AttachSceneText(std::vector<PhysicalWorldEntity>& entities,
                     const std::vector<PhysicalSceneTextLine>& lines)
{
    for (const auto& line : lines) {
        if (line.text.empty()) continue;
        const cv::Point2f c = QuadCentroid(line.model_quad);
        for (auto& e : entities) {
            if (ModelBoxContainsPoint(e.model_box, c)) {
                e.text_on_object.push_back(line.text);
                break;  // single-owner — first containing entity wins
            }
        }
    }
}

bool IsNearerThan(const PhysicalWorldEntity& a,
                  const PhysicalWorldEntity& b,
                  float depth_threshold)
{
    if (!a.has_depth || !b.has_depth) return false;
    if (a.depth_units != b.depth_units) return false;  // refuse cross-unit comparison
    return (b.range_value < a.range_value)
        && ((a.range_value - b.range_value) >= depth_threshold);
}

void ComputeOcclusion(std::vector<PhysicalWorldEntity>& entities,
                      const PhysicalWorldStateBuilderConfig& cfg)
{
    for (auto& a : entities) {
        if (a.visibility == PhysicalEntityVisibility::Coasting) continue;
        const float a_area = RectArea(a.model_box);
        if (a_area <= 0.0f) {
            a.visibility            = PhysicalEntityVisibility::Visible;
            a.visible_area_fraction = 1.0f;
            continue;
        }
        float    max_overlap_frac    = 0.0f;
        uint64_t most_occluding_id   = 0;
        for (const auto& b : entities) {
            if (b.object_id == a.object_id) continue;
            if (!IsNearerThan(a, b, cfg.depth_difference_for_nearer)) continue;
            const float inter = IntersectionArea(a.model_box, b.model_box);
            if (inter <= 0.0f) continue;
            const float frac = inter / a_area;
            if (frac > max_overlap_frac) {
                max_overlap_frac  = frac;
                most_occluding_id = b.object_id;
            }
        }
        a.visible_area_fraction = Clamp01(1.0f - max_overlap_frac);
        if (max_overlap_frac >= cfg.min_overlap_for_occlusion) {
            a.occluded_by_object_id = most_occluding_id;
            a.visibility            = PhysicalEntityVisibility::Occluded;
        } else {
            a.visibility            = PhysicalEntityVisibility::Visible;
        }
    }
}

PhysicalEntityRelationKind ClassifyGeometricRelation(
    const PhysicalWorldEntity& a,
    const PhysicalWorldEntity& b,
    float position_axis_dominance,
    float& strength_out)
{
    const float inter = IntersectionArea(a.model_box, b.model_box);
    if (inter > 0.0f) {
        if (RectContains(a.model_box, b.model_box)) {
            strength_out = Clamp01(inter / std::max(1e-6f, RectArea(b.model_box)));
            return PhysicalEntityRelationKind::Contains;
        }
        if (RectContains(b.model_box, a.model_box)) {
            strength_out = Clamp01(inter / std::max(1e-6f, RectArea(a.model_box)));
            return PhysicalEntityRelationKind::ContainedBy;
        }
        const float min_area = std::max(1e-6f,
            std::min(RectArea(a.model_box), RectArea(b.model_box)));
        strength_out = Clamp01(inter / min_area);
        return PhysicalEntityRelationKind::Overlaps;
    }
    // No overlap — choose dominant axis between centres.
    const float dx = b.model_centre.x - a.model_centre.x;
    const float dy = b.model_centre.y - a.model_centre.y;
    const float adx = std::fabs(dx);
    const float ady = std::fabs(dy);
    const float denom = adx + ady + 1e-6f;
    const float horiz_score = adx / denom;
    const float vert_score  = ady / denom;
    if (horiz_score >= position_axis_dominance) {
        strength_out = Clamp01(horiz_score);
        return (dx > 0.0f) ? PhysicalEntityRelationKind::LeftOf
                           : PhysicalEntityRelationKind::RightOf;
        // dx > 0 ⇒ b is to the right of a ⇒ a is LeftOf b.
    }
    if (vert_score >= position_axis_dominance) {
        strength_out = Clamp01(vert_score);
        return (dy > 0.0f) ? PhysicalEntityRelationKind::Above
                           : PhysicalEntityRelationKind::Below;
        // dy > 0 ⇒ b is below a ⇒ a is Above b.
    }
    strength_out = 0.0f;
    return PhysicalEntityRelationKind::None;
}

void ComputeRelations(std::vector<PhysicalWorldEntity>& entities,
                      const PhysicalWorldStateBuilderConfig& cfg)
{
    if (cfg.max_relations_per_entity == 0) return;
    const size_t n = entities.size();
    if (n < 2) return;

    // For each entity, collect candidate (distance, relation) pairs, then
    // keep the K nearest. Two relations may emit per pair (geometric +
    // depth) — both count toward the cap.
    struct Candidate {
        float                           centre_distance;
        PhysicalEntitySpatialRelation   relation;
    };

    for (size_t i = 0; i < n; ++i) {
        std::vector<Candidate> candidates;
        candidates.reserve((n - 1) * 2);
        const auto& a = entities[i];
        for (size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            const auto& b = entities[j];
            const float ddx = b.model_centre.x - a.model_centre.x;
            const float ddy = b.model_centre.y - a.model_centre.y;
            const float dist = std::sqrt(ddx * ddx + ddy * ddy);

            float geo_strength = 0.0f;
            const auto geo_kind = ClassifyGeometricRelation(
                a, b, cfg.position_axis_dominance, geo_strength);
            if (geo_kind != PhysicalEntityRelationKind::None) {
                PhysicalEntitySpatialRelation r;
                r.other_object_id = b.object_id;
                r.kind            = geo_kind;
                r.strength        = geo_strength;
                candidates.push_back({dist, r});
            }

            // Depth ordering — only when both have depth and units agree.
            if (a.has_depth && b.has_depth && a.depth_units == b.depth_units) {
                const float ddepth = a.range_value - b.range_value;
                if (std::fabs(ddepth) >= cfg.depth_difference_for_nearer) {
                    PhysicalEntitySpatialRelation r;
                    r.other_object_id = b.object_id;
                    r.kind = (ddepth > 0.0f)
                                ? PhysicalEntityRelationKind::FartherThan
                                : PhysicalEntityRelationKind::NearerThan;
                    r.strength = Clamp01(std::fabs(ddepth));
                    candidates.push_back({dist, r});
                }
            }
        }

        // Sort by centre distance ASC, keep top-K.
        const size_t cap = static_cast<size_t>(cfg.max_relations_per_entity);
        if (candidates.size() > cap) {
            std::nth_element(
                candidates.begin(), candidates.begin() + cap, candidates.end(),
                [](const Candidate& lhs, const Candidate& rhs) {
                    return lhs.centre_distance < rhs.centre_distance;
                });
            candidates.resize(cap);
        }

        // Now sort by other_object_id ASC for stable diffing per the
        // PhysicalWorldStateResult contract.
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& lhs, const Candidate& rhs) {
                      if (lhs.relation.other_object_id != rhs.relation.other_object_id)
                          return lhs.relation.other_object_id < rhs.relation.other_object_id;
                      return static_cast<uint8_t>(lhs.relation.kind)
                           < static_cast<uint8_t>(rhs.relation.kind);
                  });

        entities[i].relations.clear();
        entities[i].relations.reserve(candidates.size());
        for (auto& c : candidates) entities[i].relations.push_back(c.relation);
    }
}

} // anonymous namespace

void BuildPhysicalWorldStateSnapshot(
    const PhysicalPerceptionPrimitiveResults& perc,
    const PhysicalSpatialGroundingResults&    ground,
    const PhysicalWorldStateBuilderConfig&    cfg,
    PhysicalWorldStateSnapshot&               out)
{
    ValidateBuilderConfig(cfg);

    out = PhysicalWorldStateSnapshot{};
    out.source_frame_counter              = perc.source_frame_counter;
    out.source_perception_results_counter = perc.source_frame_counter;
    out.source_grounding_results_counter  = ground.source_frame_counter;
    out.model_image_width                 = perc.model_image_width;
    out.model_image_height                = perc.model_image_height;
    out.raw_image_width                   = perc.raw_image_width;
    out.raw_image_height                  = perc.raw_image_height;
    out.raw_to_model                      = perc.raw_to_model;

    // 1. Seed from tracks.
    const auto& tracks = perc.entity_tracker.tracks;
    out.entities.reserve(tracks.size());
    for (const auto& tr : tracks) {
        if (tr.track_id == 0) continue; // sentinel; defensively skip
        out.entities.push_back(SeedEntityFromTrack(tr));
        // Recompute raw_centre via the explicit raw_to_model inverse so it
        // matches the snapshot's coordinate contract even if the tracker
        // somehow stored an inconsistent raw box.
        auto& e = out.entities.back();
        e.raw_centre = ApplyRawToModelInverse(perc.raw_to_model, e.model_centre);
    }

    // 2. Enrich with grounding (only when both buses agree on counter —
    // the loop guarantees this; we still join by track_id defensively).
    if (ground.source_frame_counter == perc.source_frame_counter) {
        std::unordered_map<uint64_t, const PhysicalGroundedEntity*> by_id;
        by_id.reserve(ground.grounded_entities.size());
        for (const auto& g : ground.grounded_entities) by_id[g.track_id] = &g;
        for (auto& e : out.entities) {
            const auto it = by_id.find(e.object_id);
            if (it != by_id.end()) EnrichWithGrounding(e, *it->second);
        }
    }

    // 3. Attach masks.
    for (auto& e : out.entities) {
        AttachBestInstanceMask(e, perc.instance_segmenter.segmentation.instances);
    }

    // 4. Attach scene text.
    AttachSceneText(out.entities, perc.scene_text_reader.lines);

    // 5+6. Occlusion + visibility.
    ComputeOcclusion(out.entities, cfg);

    // 7. Relations.
    ComputeRelations(out.entities, cfg);

    // 8. Sort + counters.
    std::sort(out.entities.begin(), out.entities.end(),
              [](const PhysicalWorldEntity& a, const PhysicalWorldEntity& b) {
                  return a.object_id < b.object_id;
              });
    for (const auto& e : out.entities) {
        switch (e.visibility) {
            case PhysicalEntityVisibility::Visible:  ++out.num_visible_entities;  break;
            case PhysicalEntityVisibility::Occluded: ++out.num_occluded_entities; break;
            case PhysicalEntityVisibility::Coasting: ++out.num_coasting_entities; break;
            case PhysicalEntityVisibility::Unknown:                                break;
        }
    }

    out.built_at_steady_ns = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

}}} // namespace GRIM::Perception::Physical

#include "PhysicalEntityTracker.hpp"

#include "PhysicalPerceptionPrimitivesLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

// Steady-clock now, in nanoseconds since epoch. Monotonic — safe for
// velocity arithmetic across midnight, NTP slews, etc.
int64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Standard intersection-over-union for two axis-aligned boxes. Returns 0
// for degenerate or non-overlapping inputs.
float IntersectionOverUnion(const cv::Rect2f& a, const cv::Rect2f& b) {
    if (a.width <= 0.0f || a.height <= 0.0f || b.width <= 0.0f || b.height <= 0.0f) {
        return 0.0f;
    }
    const float ax2 = a.x + a.width;
    const float ay2 = a.y + a.height;
    const float bx2 = b.x + b.width;
    const float by2 = b.y + b.height;
    const float ix1 = std::max(a.x, b.x);
    const float iy1 = std::max(a.y, b.y);
    const float ix2 = std::min(ax2, bx2);
    const float iy2 = std::min(ay2, by2);
    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.0f || ih <= 0.0f) return 0.0f;
    const float inter = iw * ih;
    const float uni   = a.width * a.height + b.width * b.height - inter;
    if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

// Apply the inverse of the conditioner's letterbox-affine transform to
// recover the raw-image box for a model-space box. raw = (model - offset) / scale.
// Both scales must be positive — we enforce this on every call so a bogus
// transform fails loudly via Rule 3 rather than silently drawing nonsense.
cv::Rect2f BackProjectModelBoxToRawSpace(
    const cv::Rect2f& model_box,
    const PhysicalSignalRawToModelTransform& raw_to_model,
    int raw_image_width,
    int raw_image_height)
{
    if (raw_to_model.scale_x <= 0.0f || raw_to_model.scale_y <= 0.0f) {
        throw std::runtime_error(
            "BackProjectModelBoxToRawSpace: raw_to_model has non-positive scale "
            "(scale_x=" + std::to_string(raw_to_model.scale_x)
            + " scale_y=" + std::to_string(raw_to_model.scale_y) + ")");
    }
    const float rx = (model_box.x - raw_to_model.offset_x) / raw_to_model.scale_x;
    const float ry = (model_box.y - raw_to_model.offset_y) / raw_to_model.scale_y;
    const float rw = model_box.width  / raw_to_model.scale_x;
    const float rh = model_box.height / raw_to_model.scale_y;
    cv::Rect2f raw{rx, ry, rw, rh};
    if (raw_image_width > 0 && raw_image_height > 0) {
        // Clip into the actual raw frame so the UI / consumers never see
        // negative or out-of-bounds coordinates from numerical drift.
        const float x1 = std::max(0.0f, raw.x);
        const float y1 = std::max(0.0f, raw.y);
        const float x2 = std::min(static_cast<float>(raw_image_width),
                                  raw.x + raw.width);
        const float y2 = std::min(static_cast<float>(raw_image_height),
                                  raw.y + raw.height);
        raw.x      = x1;
        raw.y      = y1;
        raw.width  = std::max(0.0f, x2 - x1);
        raw.height = std::max(0.0f, y2 - y1);
    }
    return raw;
}

cv::Point2f BoxCentre(const cv::Rect2f& b) {
    return {b.x + b.width * 0.5f, b.y + b.height * 0.5f};
}

} // anonymous namespace

PhysicalEntityTracker::PhysicalEntityTracker()  = default;
PhysicalEntityTracker::~PhysicalEntityTracker() = default;

void PhysicalEntityTracker::ConfigurePhysicalEntityTracker(
    const PhysicalEntityTrackerConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);
    // Validate up front — Rule 20: silent fallbacks forbidden.
    if (!(cfg.smoothing_alpha > 0.0f && cfg.smoothing_alpha <= 1.0f)) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalEntityTracker: smoothing_alpha must be in (0, 1] (got "
                             + std::to_string(cfg.smoothing_alpha) + ")";
        throw std::runtime_error(last_error_reason_);
    }
    if (!(cfg.min_iou_for_match >= 0.0f && cfg.min_iou_for_match <= 1.0f)) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalEntityTracker: min_iou_for_match must be in [0, 1] (got "
                             + std::to_string(cfg.min_iou_for_match) + ")";
        throw std::runtime_error(last_error_reason_);
    }
    if (!(cfg.cross_track_nms_iou >= 0.0f && cfg.cross_track_nms_iou <= 1.0f)) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalEntityTracker: cross_track_nms_iou must be in [0, 1] (got "
                             + std::to_string(cfg.cross_track_nms_iou) + ")";
        throw std::runtime_error(last_error_reason_);
    }
    if (cfg.max_predict_seconds < 0.0) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalEntityTracker: max_predict_seconds must be >= 0";
        throw std::runtime_error(last_error_reason_);
    }
    cfg_   = cfg;
    state_ = PhysicalImageOperatorState::ModelLoaded;
    last_error_reason_.clear();
    LOG_DEBUG(PHYSICAL_PERC_PRIM_LOG_TAG,
              std::string("ConfigurePhysicalEntityTracker: alpha=")
              + std::to_string(cfg_.smoothing_alpha)
              + " min_iou=" + std::to_string(cfg_.min_iou_for_match)
              + " confirm_hits=" + std::to_string(cfg_.min_hits_to_confirm)
              + " max_age_misses=" + std::to_string(cfg_.max_age_misses));
}

void PhysicalEntityTracker::RouteDetectionsToPhysicalEntityTracker(
    const std::vector<PhysicalObjectDetection>& detections,
    const PhysicalSignalRawToModelTransform& raw_to_model,
    int  raw_image_width,
    int  raw_image_height,
    uint64_t source_frame_counter,
    PhysicalEntityTrackerOutput& out)
{
    std::lock_guard<std::mutex> lk(mutex_);

    out = PhysicalEntityTrackerOutput{};
    out.state                  = state_;
    out.last_error_reason      = last_error_reason_;
    out.inference_count        = inference_count_;
    out.last_frame_counter     = source_frame_counter;
    out.total_tracks_spawned   = total_tracks_spawned_;
    out.total_tracks_confirmed = total_tracks_confirmed_;
    out.total_tracks_culled    = total_tracks_culled_;

    if (state_ != PhysicalImageOperatorState::ModelLoaded) {
        // Not ready — surface state and stop. No silent skip.
        return;
    }

    const auto t_route_start = std::chrono::steady_clock::now();

    try {
        const int64_t now_ns = SteadyNowNs();

        // ── 1. PREDICT ──────────────────────────────────────────────────
        for (auto& lt : live_tracks_) {
            const int64_t prev_ns = lt.snapshot.last_update_steady_ns;
            const double  dt_raw  = (now_ns - prev_ns) / 1e9;
            const double  dt      = std::clamp(dt_raw, 0.0, cfg_.max_predict_seconds);
            cv::Rect2f predicted = lt.snapshot.smoothed_model_box;
            predicted.x += static_cast<float>(lt.snapshot.velocity_px_per_sec_x * dt);
            predicted.y += static_cast<float>(lt.snapshot.velocity_px_per_sec_y * dt);
            lt.predicted_model_box = predicted;
        }

        // Filter detections by confidence floor.
        std::vector<size_t> det_indices;
        det_indices.reserve(detections.size());
        for (size_t i = 0; i < detections.size(); ++i) {
            if (detections[i].confidence >= cfg_.detection_confidence_floor) {
                det_indices.push_back(i);
            }
        }

        // ── 2. ASSOCIATE (greedy by ascending cost, IoU-gated, class-gated) ─
        struct Candidate {
            float    cost;        // 1 - IoU
            size_t   track_idx;
            size_t   det_pos;     // index into det_indices
        };
        std::vector<Candidate> candidates;
        candidates.reserve(live_tracks_.size() * det_indices.size());
        const float cost_gate = 1.0f - cfg_.min_iou_for_match;
        for (size_t t = 0; t < live_tracks_.size(); ++t) {
            const auto& trk = live_tracks_[t];
            for (size_t dp = 0; dp < det_indices.size(); ++dp) {
                const auto& det = detections[det_indices[dp]];
                if (det.class_id != trk.snapshot.class_id) continue;
                const float iou  = IntersectionOverUnion(trk.predicted_model_box, det.model_box);
                const float cost = 1.0f - iou;
                if (cost > cost_gate) continue;
                candidates.push_back({cost, t, dp});
            }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& a, const Candidate& b){ return a.cost < b.cost; });

        std::vector<int> matched_det_for_track(live_tracks_.size(), -1);
        std::vector<bool> det_taken(det_indices.size(), false);
        std::vector<bool> trk_taken(live_tracks_.size(), false);
        for (const auto& c : candidates) {
            if (trk_taken[c.track_idx] || det_taken[c.det_pos]) continue;
            matched_det_for_track[c.track_idx] = static_cast<int>(c.det_pos);
            trk_taken[c.track_idx] = true;
            det_taken[c.det_pos]   = true;
        }

        // ── 3. UPDATE matched tracks ────────────────────────────────────
        const float alpha = cfg_.smoothing_alpha;
        for (size_t t = 0; t < live_tracks_.size(); ++t) {
            auto& lt = live_tracks_[t];
            const int dp = matched_det_for_track[t];
            ++lt.snapshot.age_in_frames;

            if (dp < 0) {
                // Missed this frame — coast. Predicted box becomes the
                // working box so consumers see continuous motion.
                ++lt.snapshot.miss_streak;
                ++lt.snapshot.total_misses;
                lt.snapshot.hit_streak = 0;
                lt.snapshot.smoothed_model_box = lt.predicted_model_box;
                lt.snapshot.smoothed_raw_box   = BackProjectModelBoxToRawSpace(
                    lt.snapshot.smoothed_model_box, raw_to_model,
                    raw_image_width, raw_image_height);
                if (lt.snapshot.state == PhysicalEntityTrackState::Confirmed) {
                    lt.snapshot.state = PhysicalEntityTrackState::Coasting;
                }
                lt.snapshot.last_update_frame_counter = source_frame_counter;
                lt.snapshot.last_update_steady_ns     = now_ns;
                continue;
            }

            const auto& det = detections[det_indices[dp]];
            const int64_t prev_ns = lt.snapshot.last_update_steady_ns;
            const double  dt      = (now_ns - prev_ns) / 1e9;

            // Smoothed box (EMA in model space).
            cv::Rect2f& sb = lt.snapshot.smoothed_model_box;
            sb.x      = (1.0f - alpha) * sb.x      + alpha * det.model_box.x;
            sb.y      = (1.0f - alpha) * sb.y      + alpha * det.model_box.y;
            sb.width  = (1.0f - alpha) * sb.width  + alpha * det.model_box.width;
            sb.height = (1.0f - alpha) * sb.height + alpha * det.model_box.height;

            // Velocity from centre delta — guarded against tiny dt.
            const cv::Point2f new_centre = BoxCentre(sb);
            if (dt >= cfg_.min_velocity_dt_seconds) {
                lt.snapshot.velocity_px_per_sec_x =
                    (new_centre.x - lt.last_centre.x) / dt;
                lt.snapshot.velocity_px_per_sec_y =
                    (new_centre.y - lt.last_centre.y) / dt;
            } else {
                lt.snapshot.velocity_px_per_sec_x = 0.0;
                lt.snapshot.velocity_px_per_sec_y = 0.0;
            }
            lt.last_centre = new_centre;

            lt.snapshot.smoothed_raw_box = BackProjectModelBoxToRawSpace(
                sb, raw_to_model, raw_image_width, raw_image_height);

            lt.snapshot.last_detection_confidence = det.confidence;
            lt.snapshot.smoothed_confidence =
                (1.0f - alpha) * lt.snapshot.smoothed_confidence + alpha * det.confidence;

            ++lt.snapshot.hit_streak;
            ++lt.snapshot.total_hits;
            lt.snapshot.miss_streak = 0;
            lt.snapshot.last_update_frame_counter  = source_frame_counter;
            lt.snapshot.last_matched_frame_counter = source_frame_counter;
            lt.snapshot.last_update_steady_ns      = now_ns;

            // Promote.
            if (lt.snapshot.state == PhysicalEntityTrackState::Tentative
                && lt.snapshot.hit_streak >= cfg_.min_hits_to_confirm) {
                lt.snapshot.state = PhysicalEntityTrackState::Confirmed;
                ++total_tracks_confirmed_;
            } else if (lt.snapshot.state == PhysicalEntityTrackState::Coasting) {
                lt.snapshot.state = PhysicalEntityTrackState::Confirmed;
            }
        }

        // ── 4. SPAWN unmatched detections as Tentative ─────────────────
        for (size_t dp = 0; dp < det_indices.size(); ++dp) {
            if (det_taken[dp]) continue;
            const auto& det = detections[det_indices[dp]];
            LiveTrack lt{};
            lt.snapshot.track_id   = next_track_id_++;
            lt.snapshot.class_id   = det.class_id;
            lt.snapshot.class_label= det.class_label;
            lt.snapshot.state      = PhysicalEntityTrackState::Tentative;
            lt.snapshot.smoothed_model_box        = det.model_box;
            lt.snapshot.smoothed_raw_box          = BackProjectModelBoxToRawSpace(
                det.model_box, raw_to_model, raw_image_width, raw_image_height);
            lt.snapshot.last_detection_confidence = det.confidence;
            lt.snapshot.smoothed_confidence       = det.confidence;
            lt.snapshot.age_in_frames             = 1;
            lt.snapshot.hit_streak                = 1;
            lt.snapshot.total_hits                = 1;
            lt.snapshot.first_seen_frame_counter  = source_frame_counter;
            lt.snapshot.last_update_frame_counter = source_frame_counter;
            lt.snapshot.last_matched_frame_counter= source_frame_counter;
            lt.snapshot.first_seen_steady_ns      = now_ns;
            lt.snapshot.last_update_steady_ns     = now_ns;
            lt.predicted_model_box                = det.model_box;
            lt.last_centre                        = BoxCentre(det.model_box);
            live_tracks_.push_back(std::move(lt));
            ++total_tracks_spawned_;
        }

        // ── 5. CULL aged / dead tracks ─────────────────────────────────
        const size_t before_cull = live_tracks_.size();
        live_tracks_.erase(
            std::remove_if(live_tracks_.begin(), live_tracks_.end(),
                [&](const LiveTrack& lt){
                    if (lt.snapshot.state == PhysicalEntityTrackState::Tentative
                        && lt.snapshot.miss_streak >= cfg_.max_tentative_misses) {
                        return true;
                    }
                    return lt.snapshot.miss_streak >= cfg_.max_age_misses;
                }),
            live_tracks_.end());
        total_tracks_culled_ += (before_cull - live_tracks_.size());

        // ── 5b. CROSS-TRACK NMS (same-class) ──────────────────
        // Detectors (notably YOLO at low conf thresholds) regularly spawn
        // two near-identical boxes for one physical object. Each becomes a
        // separate track, and downstream consumers see ghost duplicates.
        // Greedy NMS over (state, total_hits, smoothed_confidence, age).
        if (cfg_.cross_track_nms_iou > 0.0f && live_tracks_.size() >= 2) {
            // Rank index: lower index == "keep" priority.
            std::vector<size_t> order(live_tracks_.size());
            std::iota(order.begin(), order.end(), size_t{0});
            auto rank_tuple = [&](size_t i) {
                const auto& s = live_tracks_[i].snapshot;
                // Confirmed=2, Coasting=1, Tentative=0 — higher beats lower.
                int state_rank = 0;
                switch (s.state) {
                    case PhysicalEntityTrackState::Confirmed: state_rank = 2; break;
                    case PhysicalEntityTrackState::Coasting:  state_rank = 1; break;
                    case PhysicalEntityTrackState::Tentative: state_rank = 0; break;
                }
                return std::make_tuple(state_rank,
                                       static_cast<int>(s.total_hits),
                                       s.smoothed_confidence,
                                       static_cast<int>(s.age_in_frames));
            };
            std::sort(order.begin(), order.end(),
                      [&](size_t a, size_t b){ return rank_tuple(a) > rank_tuple(b); });

            std::vector<bool> suppressed(live_tracks_.size(), false);
            for (size_t ai = 0; ai < order.size(); ++ai) {
                const size_t a = order[ai];
                if (suppressed[a]) continue;
                const auto& sa = live_tracks_[a].snapshot;
                for (size_t bi = ai + 1; bi < order.size(); ++bi) {
                    const size_t b = order[bi];
                    if (suppressed[b]) continue;
                    const auto& sb = live_tracks_[b].snapshot;
                    if (sb.class_id != sa.class_id) continue;
                    const float iou = IntersectionOverUnion(
                        sa.smoothed_model_box, sb.smoothed_model_box);
                    if (iou >= cfg_.cross_track_nms_iou) {
                        suppressed[b] = true;
                    }
                }
            }
            // Erase suppressed tracks; preserve original order otherwise.
            const size_t before_nms = live_tracks_.size();
            std::vector<LiveTrack> kept;
            kept.reserve(live_tracks_.size());
            for (size_t i = 0; i < live_tracks_.size(); ++i) {
                if (!suppressed[i]) kept.push_back(std::move(live_tracks_[i]));
            }
            live_tracks_ = std::move(kept);
            total_tracks_culled_ += (before_nms - live_tracks_.size());
        }

        // ── 6. EMIT snapshot ───────────────────────────────────────────
        out.tracks.reserve(live_tracks_.size());
        for (const auto& lt : live_tracks_) {
            out.tracks.push_back(lt.snapshot);
        }

        ++inference_count_;
        out.state                  = state_;
        out.last_error_reason.clear();
        last_error_reason_.clear();
        out.inference_count        = inference_count_;
        out.total_tracks_spawned   = total_tracks_spawned_;
        out.total_tracks_confirmed = total_tracks_confirmed_;
        out.total_tracks_culled    = total_tracks_culled_;
    } catch (const std::exception& e) {
        last_error_reason_   = std::string("RouteDetectionsToPhysicalEntityTracker failed: ") + e.what();
        state_               = PhysicalImageOperatorState::InferenceFailed;
        out.state            = state_;
        out.last_error_reason= last_error_reason_;
        LOG_ERROR(PHYSICAL_PERC_PRIM_LOG_TAG, last_error_reason_);
        return;
    }

    const auto t_route_end = std::chrono::steady_clock::now();
    out.last_route_ms = std::chrono::duration<double, std::milli>(
                            t_route_end - t_route_start).count();
}

void PhysicalEntityTracker::ResetPhysicalEntityTracker() {
    std::lock_guard<std::mutex> lk(mutex_);
    live_tracks_.clear();
    cfg_                   = PhysicalEntityTrackerConfig{};
    state_                 = PhysicalImageOperatorState::NoModelConfigured;
    last_error_reason_.clear();
    inference_count_       = 0;
    next_track_id_         = 1;
    total_tracks_spawned_  = 0;
    total_tracks_confirmed_= 0;
    total_tracks_culled_   = 0;
}

PhysicalImageOperatorState PhysicalEntityTracker::GetPhysicalEntityTrackerState() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_;
}

std::string PhysicalEntityTracker::GetPhysicalEntityTrackerLastError() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_reason_;
}

bool PhysicalEntityTracker::IsPhysicalEntityTrackerReady() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_ == PhysicalImageOperatorState::ModelLoaded;
}

}}} // namespace GRIM::Perception::Physical

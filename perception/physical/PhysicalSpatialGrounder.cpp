#include "PhysicalSpatialGrounder.hpp"

#include "PhysicalSpatialGroundingLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

int64_t SteadyNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Clamp a model-space rectangle to the depth-map bounds. Returns an empty
// rect if the input lies entirely outside.
cv::Rect ClampModelBoxToMapBounds(const cv::Rect2f& m, int map_w, int map_h) {
    if (map_w <= 0 || map_h <= 0) return cv::Rect{};
    const int x1 = std::max(0,                  static_cast<int>(std::floor(m.x)));
    const int y1 = std::max(0,                  static_cast<int>(std::floor(m.y)));
    const int x2 = std::min(map_w,              static_cast<int>(std::ceil (m.x + m.width)));
    const int y2 = std::min(map_h,              static_cast<int>(std::ceil (m.y + m.height)));
    if (x2 <= x1 || y2 <= y1) return cv::Rect{};
    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

// Compute median + finite-fraction over a CV_32FC1 ROI. Returns false if
// no finite samples were found. `out_median` is the median over finite
// samples; `out_finite_frac` is (#finite / total).
bool MedianOverFiniteRoi(const cv::Mat& depth, const cv::Rect& roi,
                         float& out_median, float& out_finite_frac)
{
    out_median = 0.0f;
    out_finite_frac = 0.0f;
    if (depth.empty() || roi.empty()) return false;
    if (depth.type() != CV_32FC1) return false;

    cv::Mat sub = depth(roi);
    std::vector<float> vals;
    vals.reserve(static_cast<size_t>(sub.rows) * static_cast<size_t>(sub.cols));
    int total = 0;
    for (int y = 0; y < sub.rows; ++y) {
        const float* r = sub.ptr<float>(y);
        for (int x = 0; x < sub.cols; ++x) {
            ++total;
            const float v = r[x];
            if (std::isfinite(v)) vals.push_back(v);
        }
    }
    if (total <= 0) return false;
    out_finite_frac = static_cast<float>(vals.size()) / static_cast<float>(total);
    if (vals.empty()) return false;
    const size_t n  = vals.size();
    const size_t mid = n / 2;
    std::nth_element(vals.begin(), vals.begin() + static_cast<long>(mid), vals.end());
    out_median = vals[mid];
    return true;
}

// Mean over a strip ROI. Returns false if no finite samples.
bool MeanOverFiniteRoi(const cv::Mat& depth, const cv::Rect& roi, float& out_mean)
{
    out_mean = 0.0f;
    if (depth.empty() || roi.empty()) return false;
    if (depth.type() != CV_32FC1) return false;
    cv::Mat sub = depth(roi);
    double acc = 0.0;
    int    n   = 0;
    for (int y = 0; y < sub.rows; ++y) {
        const float* r = sub.ptr<float>(y);
        for (int x = 0; x < sub.cols; ++x) {
            const float v = r[x];
            if (std::isfinite(v)) { acc += v; ++n; }
        }
    }
    if (n <= 0) return false;
    out_mean = static_cast<float>(acc / n);
    return true;
}

} // anonymous namespace

PhysicalSpatialGrounder::PhysicalSpatialGrounder()  = default;
PhysicalSpatialGrounder::~PhysicalSpatialGrounder() = default;

void PhysicalSpatialGrounder::ConfigurePhysicalSpatialGrounder(
    const PhysicalSpatialGrounderConfig& cfg)
{
    std::lock_guard<std::mutex> lk(mutex_);

    auto fail = [&](const std::string& reason) {
        state_             = PhysicalImageOperatorState::ModelLoadFailed;
        last_error_reason_ = "ConfigurePhysicalSpatialGrounder: " + reason;
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    };

    if (!(cfg.min_range_confidence >= 0.0f && cfg.min_range_confidence <= 1.0f))
        fail("min_range_confidence must be in [0,1]");
    if (cfg.support_strip_height_px <= 0)
        fail("support_strip_height_px must be > 0");
    if (cfg.floor_below_delta < 0.0f || cfg.table_step < 0.0f || cfg.wall_flatness < 0.0f)
        fail("support deltas must be >= 0");
    if (cfg.wall_min_aspect <= 0.0f)
        fail("wall_min_aspect must be > 0");
    if (!(cfg.nav_cone_half_width_frac >= 0.0f && cfg.nav_cone_half_width_frac <= 1.0f))
        fail("nav_cone_half_width_frac must be in [0,1]");
    if (!(cfg.nav_cone_min_y_frac >= 0.0f && cfg.nav_cone_min_y_frac <= 1.0f))
        fail("nav_cone_min_y_frac must be in [0,1]");
    if (!(cfg.path_block_threshold >= 0.0f && cfg.path_block_threshold <= 1.0f))
        fail("path_block_threshold must be in [0,1]");
    if (cfg.max_path_meters <= 0.0)
        fail("max_path_meters must be > 0");
    if (cfg.moving_threshold_px_per_sec < 0.0f || cfg.moving_threshold_inv_per_sec < 0.0f)
        fail("moving_threshold_* must be >= 0");

    cfg_   = cfg;
    state_ = PhysicalImageOperatorState::ModelLoaded;
    last_error_reason_.clear();
    LOG_DEBUG(PHYSICAL_SPATIAL_GROUND_LOG_TAG,
              std::string("ConfigurePhysicalSpatialGrounder: min_range_conf=")
              + std::to_string(cfg_.min_range_confidence)
              + " path_block_thresh=" + std::to_string(cfg_.path_block_threshold)
              + " moving_px/s=" + std::to_string(cfg_.moving_threshold_px_per_sec));
}

void PhysicalSpatialGrounder::RouteDepthAndTracksToPhysicalSpatialGrounder(
    const PhysicalDepthMap&                 depth_map,
    const PhysicalEntityTrackerOutput&      tracks,
    int                                     model_image_width,
    int                                     model_image_height,
    std::vector<PhysicalGroundedEntity>&    out_grounded,
    PhysicalImageOperatorState&             out_state,
    std::string&                            out_error,
    double&                                 out_grounding_ms)
{
    out_grounded.clear();
    out_grounding_ms = 0.0;

    std::lock_guard<std::mutex> lk(mutex_);
    out_state = state_;
    out_error = last_error_reason_;

    if (state_ != PhysicalImageOperatorState::ModelLoaded) return;

    const auto t0 = std::chrono::steady_clock::now();
    try {
        if (depth_map.empty()) {
            // Not an error — depth estimator may be NoModelConfigured. But
            // surface this explicitly so the consumer can decide.
            out_error = "PhysicalSpatialGrounder: depth_map is empty — nothing to ground";
            ++run_count_;
            out_grounding_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            return;
        }
        if (depth_map.inverse_depth_image.type() != CV_32FC1) {
            throw std::runtime_error("depth_map.inverse_depth_image is not CV_32FC1");
        }
        if (model_image_width <= 0 || model_image_height <= 0) {
            throw std::runtime_error("model_image dims must be positive (got "
                                     + std::to_string(model_image_width) + "x"
                                     + std::to_string(model_image_height) + ")");
        }
        // The depth map MUST be at MODEL resolution — the estimator
        // contract guarantees this; verify loudly.
        if (depth_map.map_width  != model_image_width ||
            depth_map.map_height != model_image_height) {
            throw std::runtime_error(
                "depth_map size " + std::to_string(depth_map.map_width) + "x"
                + std::to_string(depth_map.map_height) + " != model "
                + std::to_string(model_image_width) + "x"
                + std::to_string(model_image_height));
        }

        const cv::Mat& dimg   = depth_map.inverse_depth_image;
        const bool     metric = (depth_map.units == DepthUnits::Meters)
                                && !depth_map.metric_depth_image.empty()
                                && depth_map.metric_depth_image.type() == CV_32FC1;
        const cv::Mat* mimg   = metric ? &depth_map.metric_depth_image : nullptr;

        const float frame_w_f = static_cast<float>(model_image_width);
        const float frame_h_f = static_cast<float>(model_image_height);
        const float frame_mid_x = frame_w_f * 0.5f;
        const float nav_half    = cfg_.nav_cone_half_width_frac * frame_w_f;
        const float nav_min_y   = cfg_.nav_cone_min_y_frac      * frame_h_f;

        const int64_t now_ns = SteadyNowNs();
        std::vector<uint64_t> active_track_ids;
        active_track_ids.reserve(tracks.tracks.size());

        out_grounded.reserve(tracks.tracks.size());

        for (const auto& tr : tracks.tracks) {
            active_track_ids.push_back(tr.track_id);

            cv::Rect box = ClampModelBoxToMapBounds(tr.smoothed_model_box,
                                                    depth_map.map_width,
                                                    depth_map.map_height);
            if (box.empty()) continue;

            float median_inv = 0.0f;
            float finite_frac = 0.0f;
            if (!MedianOverFiniteRoi(dimg, box, median_inv, finite_frac)) continue;
            if (finite_frac < cfg_.min_range_confidence) continue;

            PhysicalGroundedEntity g;
            g.track_id          = tr.track_id;
            g.class_id          = tr.class_id;
            g.class_label       = tr.class_label;
            g.model_box         = tr.smoothed_model_box;
            g.raw_box           = tr.smoothed_raw_box;
            g.units             = depth_map.units;
            g.range_value       = median_inv;
            g.range_confidence  = finite_frac;
            if (metric) {
                float median_m = 0.0f;
                float frac_m   = 0.0f;
                MedianOverFiniteRoi(*mimg, box, median_m, frac_m);
                g.range_value_meters = median_m;
            }

            // ── Support-surface heuristic ──────────────────────────────
            const int strip_h = std::max(1, std::min(cfg_.support_strip_height_px,
                                                     box.height));
            cv::Rect bottom_strip(box.x, box.y + box.height - strip_h, box.width, strip_h);
            cv::Rect top_strip   (box.x, box.y,                          box.width, strip_h);

            const int below_y = box.y + box.height;
            cv::Rect below_strip(box.x, below_y,
                                 box.width,
                                 std::min(cfg_.support_strip_height_px,
                                          depth_map.map_height - below_y));
            if (below_strip.height < 0) below_strip.height = 0;

            float bottom_mean = median_inv;
            float top_mean    = median_inv;
            float below_mean  = std::numeric_limits<float>::quiet_NaN();
            MeanOverFiniteRoi(dimg, bottom_strip, bottom_mean);
            MeanOverFiniteRoi(dimg, top_strip,    top_mean);
            const bool have_below = (below_strip.height > 0)
                                    && MeanOverFiniteRoi(dimg, below_strip, below_mean);

            // Inverse depth: larger value = closer. So:
            //   "below the box is FARTHER than the bottom of the box" =>
            //       below_mean < bottom_mean  (smaller inv depth ⇒ farther)
            //   "abrupt step (table edge): bottom is much closer than below"
            //       (bottom_mean - below_mean) > table_step
            const float box_aspect = (box.width > 0)
                                     ? static_cast<float>(box.height) / static_cast<float>(box.width)
                                     : 0.0f;
            const float vertical_inv_delta = bottom_mean - top_mean;
            const float flatness_abs       = std::fabs(vertical_inv_delta);

            PhysicalSupportSurfaceClass cls = PhysicalSupportSurfaceClass::Unknown;
            float cls_score = 0.0f;

            if (have_below) {
                const float floor_disc = (bottom_mean - below_mean) - cfg_.floor_below_delta;
                const float table_disc = (bottom_mean - below_mean) - cfg_.table_step;

                if (table_disc >= 0.0f) {
                    cls       = PhysicalSupportSurfaceClass::Table;
                    cls_score = std::min(1.0f, table_disc / std::max(1e-6f, cfg_.table_step));
                } else if (floor_disc >= 0.0f && vertical_inv_delta > 0.0f) {
                    // Bottom is closer than below by floor_below_delta, AND
                    // depth ramps near→far as we go up the box (top is farther,
                    // i.e. smaller inv) ⇒ floor.
                    cls       = PhysicalSupportSurfaceClass::Floor;
                    cls_score = std::min(1.0f, floor_disc / std::max(1e-6f, cfg_.floor_below_delta));
                }
            }
            if (cls == PhysicalSupportSurfaceClass::Unknown) {
                if (flatness_abs <= cfg_.wall_flatness && box_aspect >= cfg_.wall_min_aspect) {
                    cls       = PhysicalSupportSurfaceClass::Wall;
                    cls_score = std::min(1.0f, 1.0f - (flatness_abs / std::max(1e-6f, cfg_.wall_flatness)));
                }
            }
            g.support_surface       = cls;
            g.support_surface_score = std::max(0.0f, std::min(1.0f, cls_score));

            // ── Path obstruction ───────────────────────────────────────
            const float box_cx     = tr.smoothed_model_box.x + tr.smoothed_model_box.width  * 0.5f;
            const float box_bottom = tr.smoothed_model_box.y + tr.smoothed_model_box.height;
            const bool  in_cone    = (std::fabs(box_cx - frame_mid_x) <= nav_half)
                                     && (box_bottom >= nav_min_y);
            float closeness = 0.0f;
            if (metric && g.range_value_meters > 0.0f) {
                closeness = 1.0f - static_cast<float>(
                                std::min(1.0,
                                         g.range_value_meters / cfg_.max_path_meters));
                if (closeness < 0.0f) closeness = 0.0f;
            } else {
                // Inverse-depth normalised: larger = closer. Use directly.
                closeness = std::max(0.0f, std::min(1.0f, median_inv));
            }
            g.path_block_score = in_cone ? closeness : 0.0f;
            g.path_blocked     = (g.path_block_score >= cfg_.path_block_threshold);

            // ── Motion ─────────────────────────────────────────────────
            g.velocity_model_px_per_sec_x = static_cast<float>(tr.velocity_px_per_sec_x);
            g.velocity_model_px_per_sec_y = static_cast<float>(tr.velocity_px_per_sec_y);

            auto hist_it = depth_history_.find(tr.track_id);
            float depth_velocity = 0.0f;
            bool  moved_depth    = false;
            if (hist_it != depth_history_.end()) {
                const auto& h = hist_it->second;
                const double dt_s = std::max(0.0,
                                             (now_ns - h.last_seen_steady_ns) * 1e-9);
                if (dt_s > 0.0) {
                    if (metric) {
                        depth_velocity = static_cast<float>(
                            (g.range_value_meters - h.last_metric_meters) / dt_s);
                        moved_depth = std::fabs(g.range_value_meters - h.last_metric_meters)
                                      > cfg_.moved_quantum_inv;
                    } else {
                        depth_velocity = static_cast<float>(
                            (median_inv - h.last_inverse_depth) / dt_s);
                        moved_depth = std::fabs(median_inv - h.last_inverse_depth)
                                      > cfg_.moved_quantum_inv;
                    }
                }
            }
            g.depth_velocity_units_per_sec = depth_velocity;

            const float speed_2d_px_s = std::sqrt(
                g.velocity_model_px_per_sec_x * g.velocity_model_px_per_sec_x +
                g.velocity_model_px_per_sec_y * g.velocity_model_px_per_sec_y);
            const bool moving_2d    = speed_2d_px_s     > cfg_.moving_threshold_px_per_sec;
            const bool moving_depth = std::fabs(depth_velocity) > cfg_.moving_threshold_inv_per_sec;

            if (tr.state == PhysicalEntityTrackState::Coasting) {
                g.motion_state = PhysicalEntityMotionState::Coasted;
            } else if (hist_it == depth_history_.end()) {
                g.motion_state = PhysicalEntityMotionState::Unknown;
            } else if (moving_2d || moving_depth) {
                g.motion_state = PhysicalEntityMotionState::Moving;
            } else {
                g.motion_state = PhysicalEntityMotionState::Static;
            }

            const float per_frame_2d_px = speed_2d_px_s * 0.0f; // see comment below
            (void)per_frame_2d_px;
            // moved_since_last_frame: we don't have wall-time delta here,
            // so use the 2D-velocity quantum × the tracker dt (encoded in
            // velocity_px_per_sec). A simpler, equivalent rule: if the 2D
            // speed exceeds the quantum (≥ 1px change at 1Hz) OR depth moved.
            const bool moved_2d = speed_2d_px_s > cfg_.moved_quantum_px;
            g.moved_since_last_frame = moved_2d || moved_depth;

            // Update history.
            DepthHistoryEntry e;
            e.last_inverse_depth      = median_inv;
            e.last_metric_meters      = g.range_value_meters;
            e.last_seen_steady_ns     = now_ns;
            e.last_seen_frame_counter = tracks.last_frame_counter;
            e.last_units              = depth_map.units;
            depth_history_[tr.track_id] = e;

            out_grounded.push_back(std::move(g));
        }

        // Evict history for tracks that no longer appear in the live set.
        // O(N+M); acceptable at the few-dozen-track scale we expect.
        if (!depth_history_.empty()) {
            std::vector<uint64_t> to_drop;
            to_drop.reserve(depth_history_.size());
            for (const auto& kv : depth_history_) {
                if (std::find(active_track_ids.begin(), active_track_ids.end(), kv.first)
                    == active_track_ids.end()) {
                    to_drop.push_back(kv.first);
                }
            }
            for (uint64_t id : to_drop) depth_history_.erase(id);
        }

        ++run_count_;
        out_grounding_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        out_state = PhysicalImageOperatorState::ModelLoaded;
        out_error.clear();
        last_error_reason_.clear();
    } catch (const std::exception& e) {
        last_error_reason_ = std::string("RouteDepthAndTracksToPhysicalSpatialGrounder failed: ") + e.what();
        state_             = PhysicalImageOperatorState::InferenceFailed;
        out_state          = PhysicalImageOperatorState::InferenceFailed;
        out_error          = last_error_reason_;
        out_grounded.clear();
        LOG_ERROR(PHYSICAL_SPATIAL_GROUND_LOG_TAG, last_error_reason_);
    }
}

void PhysicalSpatialGrounder::ResetPhysicalSpatialGrounder() {
    std::lock_guard<std::mutex> lk(mutex_);
    cfg_              = PhysicalSpatialGrounderConfig{};
    last_error_reason_.clear();
    run_count_        = 0;
    depth_history_.clear();
    state_            = PhysicalImageOperatorState::NoModelConfigured;
}

PhysicalImageOperatorState PhysicalSpatialGrounder::GetPhysicalSpatialGrounderState() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_;
}

std::string PhysicalSpatialGrounder::GetPhysicalSpatialGrounderLastError() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return last_error_reason_;
}

bool PhysicalSpatialGrounder::IsPhysicalSpatialGrounderReady() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return state_ == PhysicalImageOperatorState::ModelLoaded;
}

uint64_t PhysicalSpatialGrounder::GetPhysicalSpatialGrounderRunCount() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return run_count_;
}

}}} // namespace GRIM::Perception::Physical

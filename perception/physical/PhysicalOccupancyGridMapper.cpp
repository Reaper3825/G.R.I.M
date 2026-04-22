#include "PhysicalOccupancyGridMapper.hpp"

#include "PhysicalLocalizationLogTag.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace GRIM { namespace Perception { namespace Physical {

namespace {

void ValidateConfigOrThrow(const PhysicalOccupancyGridMapperConfig& cfg) {
    if (cfg.resolution_meters <= 0.0) {
        throw std::runtime_error(
            "PhysicalOccupancyGridMapper: resolution_meters must be > 0 (got "
            + std::to_string(cfg.resolution_meters) + ")");
    }
    if (cfg.initial_cols <= 0 || cfg.initial_rows <= 0) {
        throw std::runtime_error(
            "PhysicalOccupancyGridMapper: initial_cols/rows must be > 0");
    }
    if (cfg.free_log_odds_increment <= 0.0f) {
        throw std::runtime_error(
            "PhysicalOccupancyGridMapper: free_log_odds_increment must be > 0");
    }
    if (cfg.max_abs_log_odds <= cfg.free_log_odds_increment) {
        throw std::runtime_error(
            "PhysicalOccupancyGridMapper: max_abs_log_odds must exceed free_log_odds_increment");
    }
}

// Convert a world-XY position into integer cell coordinates. Returns false
// when the point lies outside the grid; out values are NOT written.
bool WorldXyToCell(const PhysicalOccupancyGrid2D& g,
                   double x_meters, double y_meters,
                   int& out_col, int& out_row)
{
    const double dx = x_meters - g.world_origin_meters[0];
    const double dy = y_meters - g.world_origin_meters[1];
    const int    c  = static_cast<int>(std::floor(dx / g.resolution_meters + 0.5));
    const int    r  = static_cast<int>(std::floor(dy / g.resolution_meters + 0.5));
    if (c < 0 || r < 0 || c >= g.cols || r >= g.rows) return false;
    out_col = c; out_row = r;
    return true;
}

// Bresenham line walk from (c0,r0) to (c1,r1). Calls visit(col,row) for
// every cell on the line. Cells outside the grid are skipped (Rule 20:
// no silent grid extension here either).
template <typename Fn>
void WalkLineBresenham(int c0, int r0, int c1, int r1,
                       int cols, int rows, Fn&& visit)
{
    int dc = std::abs(c1 - c0);
    int dr = -std::abs(r1 - r0);
    int sc = (c0 < c1) ? 1 : -1;
    int sr = (r0 < r1) ? 1 : -1;
    int err = dc + dr;
    while (true) {
        if (c0 >= 0 && r0 >= 0 && c0 < cols && r0 < rows) visit(c0, r0);
        if (c0 == c1 && r0 == r1) break;
        const int e2 = 2 * err;
        if (e2 >= dr) { err += dr; c0 += sc; }
        if (e2 <= dc) { err += dc; r0 += sr; }
    }
}

} // anonymous namespace

PhysicalOccupancyGridMapper::PhysicalOccupancyGridMapper() {
    cfg_ = PhysicalOccupancyGridMapperConfig{};
    ValidateConfigOrThrow(cfg_);
    grid_.cols                  = cfg_.initial_cols;
    grid_.rows                  = cfg_.initial_rows;
    grid_.resolution_meters     = cfg_.resolution_meters;
    grid_.world_origin_meters[0] = cfg_.world_origin_meters[0];
    grid_.world_origin_meters[1] = cfg_.world_origin_meters[1];
    grid_.cells_log_odds.assign(static_cast<size_t>(grid_.cols) * grid_.rows, 0.0f);
}

void PhysicalOccupancyGridMapper::ConfigurePhysicalOccupancyGridMapper(
    const PhysicalOccupancyGridMapperConfig& cfg)
{
    ValidateConfigOrThrow(cfg);
    cfg_  = cfg;
    grid_ = PhysicalOccupancyGrid2D{};
    grid_.cols                  = cfg_.initial_cols;
    grid_.rows                  = cfg_.initial_rows;
    grid_.resolution_meters     = cfg_.resolution_meters;
    grid_.world_origin_meters[0] = cfg_.world_origin_meters[0];
    grid_.world_origin_meters[1] = cfg_.world_origin_meters[1];
    grid_.cells_log_odds.assign(static_cast<size_t>(grid_.cols) * grid_.rows, 0.0f);
    has_prior_pose_     = false;
    last_error_reason_.clear();
    LOG_DEBUG(PHYSICAL_LOCALIZATION_LOG_TAG,
              "ConfigurePhysicalOccupancyGridMapper: cfg accepted ("
              + std::to_string(grid_.cols) + "x" + std::to_string(grid_.rows)
              + " @ " + std::to_string(grid_.resolution_meters) + "m)");
}

void PhysicalOccupancyGridMapper::ResetPhysicalOccupancyGridMapper() {
    std::fill(grid_.cells_log_odds.begin(), grid_.cells_log_odds.end(), 0.0f);
    grid_.cells_updated_this_frame = 0;
    grid_.total_cells_observed     = 0;
    has_prior_pose_ = false;
    last_error_reason_.clear();
}

void PhysicalOccupancyGridMapper::CopyOccupancyGridSnapshot(PhysicalOccupancyGrid2D& out) const {
    out = grid_;   // vector + scalars copy
}

uint64_t PhysicalOccupancyGridMapper::RouteCameraPoseToPhysicalOccupancyGridMapper(
    const cv::Mat& T_world_camera, PhysicalLocalizationPoseScaleState scale_state)
{
    grid_.cells_updated_this_frame = 0;
    last_error_reason_.clear();

    if (T_world_camera.empty() || T_world_camera.rows != 4 || T_world_camera.cols != 4
        || T_world_camera.type() != CV_64F)
    {
        last_error_reason_ = "RouteCameraPoseToPhysicalOccupancyGridMapper: T_world_camera "
                             "must be 4x4 CV_64F (got "
                             + std::to_string(T_world_camera.rows) + "x"
                             + std::to_string(T_world_camera.cols) + " type="
                             + std::to_string(T_world_camera.type()) + ")";
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, last_error_reason_);
        throw std::runtime_error(last_error_reason_);
    }

    if (scale_state != PhysicalLocalizationPoseScaleState::ScaledByDepthMap
        && scale_state != PhysicalLocalizationPoseScaleState::ScaledByStereo
        && scale_state != PhysicalLocalizationPoseScaleState::ScaledByImu)
    {
        // Rule 20: do NOT pretend a unit-norm pose is metric. Skip cleanly
        // and tell the caller why.
        last_error_reason_ = std::string("Pose scale_state=")
                             + DescribePhysicalLocalizationPoseScaleState(scale_state)
                             + " — grid update suppressed (mapping requires metric scale)";
        return 0;
    }

    const double cur_x = T_world_camera.at<double>(0, 3);
    const double cur_y = T_world_camera.at<double>(1, 3);

    int cur_c = 0, cur_r = 0;
    if (!WorldXyToCell(grid_, cur_x, cur_y, cur_c, cur_r)) {
        last_error_reason_ =
            "Camera world-XY (" + std::to_string(cur_x) + "," + std::to_string(cur_y)
            + ") fell outside grid bounds — extend initial_cols/rows or move world_origin_meters";
        LOG_ERROR(PHYSICAL_LOCALIZATION_LOG_TAG, last_error_reason_);
        return 0;
    }

    const float lo_min = -cfg_.max_abs_log_odds;
    const float lo_max = +cfg_.max_abs_log_odds;
    auto mark_free_at = [&](int c, int r) {
        const size_t idx = static_cast<size_t>(r) * grid_.cols + c;
        const float prev = grid_.cells_log_odds[idx];
        const float next = std::clamp(prev - cfg_.free_log_odds_increment, lo_min, lo_max);
        if (next != prev) {
            if (prev == 0.0f) ++grid_.total_cells_observed;
            grid_.cells_log_odds[idx] = next;
            ++grid_.cells_updated_this_frame;
        }
    };

    if (has_prior_pose_) {
        int prev_c = 0, prev_r = 0;
        if (WorldXyToCell(grid_, prior_world_x_meters_, prior_world_y_meters_, prev_c, prev_r)) {
            WalkLineBresenham(prev_c, prev_r, cur_c, cur_r,
                              grid_.cols, grid_.rows, mark_free_at);
        } else {
            // Prior pose was outside — just mark the current cell.
            mark_free_at(cur_c, cur_r);
        }
    } else {
        mark_free_at(cur_c, cur_r);
    }

    has_prior_pose_       = true;
    prior_world_x_meters_ = cur_x;
    prior_world_y_meters_ = cur_y;
    return grid_.cells_updated_this_frame;
}

}}} // namespace GRIM::Perception::Physical

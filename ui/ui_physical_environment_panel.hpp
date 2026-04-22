#pragma once

#include "ui_panel.hpp"
#include "ui_button.hpp"
#include "ui_dropdown.hpp"
#include "ui_inputbox.hpp"
#include "perception/physical/PhysicalCameraSource.hpp"
#include "perception/physical/PhysicalCameraCalibrator.hpp"
#include "perception/physical/PhysicalFrameConditioner.hpp"
#include "perception/physical/PhysicalFrameBus.hpp"
#include "perception/physical/PhysicalPerceptionPrimitiveBus.hpp"
#include "perception/physical/PhysicalPerceptionPrimitivesLoop.hpp"
#include "perception/physical/PhysicalSpatialGroundingBus.hpp"
#include "perception/physical/PhysicalSpatialGroundingLoop.hpp"
#include "perception/physical/PhysicalWorldStateBus.hpp"
#include "perception/physical/PhysicalWorldStateLoop.hpp"

#include <opencv2/core/mat.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Tabbed UI panel for the perception/physical/ subsystem.
//
//   ┌──────────────────────────────────────────────────┐
//   │  [ Camera ] [ Calibration ]                      │
//   ├──────────────────────────────────────────────────┤
//   │  (active tab content)                            │
//   └──────────────────────────────────────────────────┘
//
// One panel — two views — same FrameBus. Mirrors the DataHub / Training tab
// pattern. Frame rendering is shared by both tabs (raw on Camera; raw or
// undistorted on Calibration with detected-corner overlay).
//
// Rule 20: when no frame is on the bus, the panel SAYS so. No stub graphic.
class UIPhysicalEnvironmentPanel : public UIPanel {
public:
    enum class Tab : uint8_t {
        Camera      = 0,
        Calibration = 1,
        Perception  = 2,
        Spatial     = 3,
        World       = 4
    };

    UIPhysicalEnvironmentPanel();

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;

    void setActiveTab(Tab t);
    Tab  getActiveTab() const { return active_tab_; }

private:
    // ── Shared frame blit cache ──
    // Holds a pre-resized + pre-packed ARGB buffer so the per-redraw cost is a
    // row-by-row memcpy. Recomputed only on cache miss (source frame changed,
    // undistort flag flipped, or output geometry changed).
    struct PreviewBlitCache {
        uint64_t              source_id        = 0;     // 0 = empty (counters start at 1)
        bool                  source_undistort = false;
        int                   out_w            = 0;
        int                   out_h            = 0;
        std::vector<uint32_t> argb;                     // size = out_w * out_h
    };

    void DrawBgrFrameIntoOverlay(OverlayRenderer& renderer,
                                 const cv::Mat& bgr,
                                 uint64_t source_id,
                                 bool source_undistort,
                                 float frame_x, float frame_y,
                                 float frame_w, float frame_h,
                                 PreviewBlitCache& cache);

    // ── Camera tab ──
    void RebuildSourceDropdownFromDirectory();
    void HandleConnectClicked();
    void HandleDisconnectClicked();
    void HandleRefreshClicked();
    void HandleToggleCameraViewClicked();
    void HandleToggleAutoExposureClicked();
    void HandleToggleDenoiseClicked();
    void HandleToggleResizeClicked();
    void HandleToggleDeblurClicked();
    void HandleToggleStabilizationClicked();
    void HandleToggleColorModeClicked();
    void HandleToggleResizeModeClicked();
    void HandleToggleQualityGateClicked();
    void HandleApplySignalSettingsClicked();
    void HandleResetSignalSettingsClicked();
    void SyncSignalSettingsControlsFromSubsystem();
    void UpdateCameraTab(const InputState& input, float dt);
    void DrawCameraTab(OverlayRenderer& renderer);

    std::shared_ptr<UIDropdown>  source_dropdown_;
    std::shared_ptr<UIInputBox>  url_inputbox_;
    std::shared_ptr<UIButton>    refresh_button_;
    std::shared_ptr<UIButton>    connect_button_;
    std::shared_ptr<UIButton>    disconnect_button_;
    std::shared_ptr<UIButton>    signal_view_toggle_btn_;
    std::shared_ptr<UIButton>    signal_auto_exposure_btn_;
    std::shared_ptr<UIButton>    signal_denoise_btn_;
    std::shared_ptr<UIButton>    signal_resize_btn_;
    std::shared_ptr<UIButton>    signal_deblur_btn_;
    std::shared_ptr<UIButton>    signal_stabilization_btn_;
    std::shared_ptr<UIButton>    signal_color_mode_btn_;
    std::shared_ptr<UIButton>    signal_resize_mode_btn_;
    std::shared_ptr<UIButton>    signal_quality_gate_btn_;
    std::shared_ptr<UIButton>    signal_apply_btn_;
    std::shared_ptr<UIButton>    signal_reset_btn_;
    std::shared_ptr<UIInputBox>  signal_width_box_;
    std::shared_ptr<UIInputBox>  signal_height_box_;
    std::shared_ptr<UIInputBox>  signal_target_luma_box_;
    std::shared_ptr<UIInputBox>  signal_manual_gain_box_;
    std::shared_ptr<UIInputBox>  signal_denoise_strength_box_;
    std::shared_ptr<UIInputBox>  signal_deblur_amount_box_;
    std::string                  url_buffer_;
    std::string                  selection_info_;
    std::string                  signal_width_buf_;
    std::string                  signal_height_buf_;
    std::string                  signal_target_luma_buf_;
    std::string                  signal_manual_gain_buf_;
    std::string                  signal_denoise_strength_buf_;
    std::string                  signal_deblur_amount_buf_;
    std::vector<GRIM::Perception::Physical::PhysicalCameraSource> last_directory_;
    GRIM::Perception::Physical::PhysicalSignalConditioningConfig   signal_cfg_;
    GRIM::Perception::Physical::PhysicalSignalConditioningStatus   signal_status_;
    bool                        camera_show_model_signal_ = true;

    // ── Calibration tab ──
    void HandleStartCaptureClicked();
    void HandleStopCaptureClicked();
    void HandleCaptureNowClicked();
    void HandleClearSamplesClicked();
    void HandleRunCalibrationClicked();
    void HandleSaveCalibrationClicked();
    void HandleReloadCalibrationClicked();
    void HandleToggleUndistortClicked();
    void HandleApplyPatternClicked();
    void UpdateCalibrationTab(const InputState& input, float dt);
    void DrawCalibrationTab(OverlayRenderer& renderer);
    void DrawCoverageGrid(OverlayRenderer& renderer,
                          float x, float y, float w, float h,
                          const GRIM::Perception::Physical::PhysicalCalibrationStatus& st);
    void DrawBrightnessBar(OverlayRenderer& renderer,
                           float x, float y, float w, float h,
                           double brightness);
    void DrawCalibrationDataReadout(OverlayRenderer& renderer,
                                    float x, float y,
                                    const GRIM::Perception::Physical::PhysicalCalibrationStatus& st);

    std::shared_ptr<UIButton>   cal_start_btn_;
    std::shared_ptr<UIButton>   cal_stop_btn_;
    std::shared_ptr<UIButton>   cal_capture_now_btn_;
    std::shared_ptr<UIButton>   cal_clear_btn_;
    std::shared_ptr<UIButton>   cal_run_btn_;
    std::shared_ptr<UIButton>   cal_save_btn_;
    std::shared_ptr<UIButton>   cal_reload_btn_;
    std::shared_ptr<UIButton>   cal_undistort_toggle_btn_;
    std::shared_ptr<UIButton>   cal_apply_pattern_btn_;
    std::shared_ptr<UIInputBox> cal_pattern_cols_box_;
    std::shared_ptr<UIInputBox> cal_pattern_rows_box_;
    std::shared_ptr<UIInputBox> cal_square_meters_box_;
    std::string                 cal_pattern_cols_buf_   = "9";
    std::string                 cal_pattern_rows_buf_   = "6";
    std::string                 cal_square_meters_buf_  = "0.025";
    bool                        cal_show_undistorted_   = false;
    GRIM::Perception::Physical::PhysicalCalibrationStatus cal_last_status_;

    // ── Tab bar ──
    std::shared_ptr<UIButton>   tab_camera_btn_;
    std::shared_ptr<UIButton>   tab_calibration_btn_;
    std::shared_ptr<UIButton>   tab_perception_btn_;
    std::shared_ptr<UIButton>   tab_spatial_btn_;
    std::shared_ptr<UIButton>   tab_world_btn_;
    Tab                         active_tab_ = Tab::Camera;

    // ── Shared frame pull state (one bus, one cached frame) ──
    GRIM::Perception::Physical::PhysicalFrameBus::FrameView last_view_;
    uint64_t                                                last_seen_counter_ = 0;
    bool                                                    have_any_frame_ = false;

    // ── Production preview pipeline ──
    // Camera tab: raw frame → blit cache.
    PreviewBlitCache camera_blit_cache_;

    // Calibration tab: raw → [optional undistort] → [throttled chessboard
    // overlay baked into BGR Mat] → blit cache. Heavy work runs only when a
    // new source frame arrives or the undistort toggle flips. The chessboard
    // re-detection is further throttled to ~5 Hz so a 30-Hz source does not
    // pay findChessboardCornersSB on every redraw.
    cv::Mat            calib_display_frame_;                // BGR, ready to blit
    uint64_t           calib_display_source_id_      = 0;   // 0 = not yet built
    bool               calib_display_undistort_      = false;
    PreviewBlitCache   calib_blit_cache_;
    GRIM::Perception::Physical::DetectedCalibrationPattern calib_overlay_pattern_;
    bool               calib_overlay_pattern_valid_  = false;
    double             calib_overlay_seconds_since_  = 1.0e9;  // start "stale"

    // ── Perception tab ──
    void HandleTogglePerceptionObjectDetector();
    void HandleTogglePerceptionSemanticSegmenter();
    void HandleTogglePerceptionImageClassifier();
    void HandleTogglePerceptionPoseEstimator();
    void HandleTogglePerceptionSceneTextReader();
    void HandleTogglePerceptionFacialExpressionDetector();
    void HandleTogglePerceptionEntityTracker();
    void HandleTogglePerceptionInstanceSegmenter();
    void RefreshPerceptionEnableButtonLabelsFromSubsystem();
    void UpdatePerceptionTab(const InputState& input, float dt);
    void DrawPerceptionTab(OverlayRenderer& renderer);
    void DrawPerceptionDetectionsOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalObjectDetectorOutput& dets,
        int blit_x, int blit_y, int blit_w, int blit_h,
        int model_w, int model_h);
    void DrawPerceptionSegmentationOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalSemanticSegmenterOutput& seg,
        int blit_x, int blit_y, int blit_w, int blit_h);
    void DrawPerceptionPoseOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalPoseKeypointEstimatorOutput& pose,
        int blit_x, int blit_y, int blit_w, int blit_h,
        int model_w, int model_h);
    void DrawPerceptionSceneTextOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalSceneTextReaderOutput& text,
        int blit_x, int blit_y, int blit_w, int blit_h,
        int model_w, int model_h);
    void DrawPerceptionFacialExpressionOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalFacialExpressionDetectorOutput& faces,
        int blit_x, int blit_y, int blit_w, int blit_h,
        int model_w, int model_h);
    void DrawPerceptionEntityTracksOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalEntityTrackerOutput& tracker,
        int blit_x, int blit_y, int blit_w, int blit_h,
        int model_w, int model_h);
    void DrawPerceptionInstanceMasksOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalInstanceSegmenterOutput& inst,
        int blit_x, int blit_y, int blit_w, int blit_h,
        int model_w, int model_h);
    void DrawPerceptionSidebar(
        OverlayRenderer& renderer, float x, float y, float w, float h,
        const GRIM::Perception::Physical::PhysicalPerceptionPrimitiveResults& r,
        bool have_results);

    std::shared_ptr<UIButton> perc_btn_obj_;
    std::shared_ptr<UIButton> perc_btn_seg_;
    std::shared_ptr<UIButton> perc_btn_cls_;
    std::shared_ptr<UIButton> perc_btn_pose_;
    std::shared_ptr<UIButton> perc_btn_text_;
    std::shared_ptr<UIButton> perc_btn_face_;
    std::shared_ptr<UIButton> perc_btn_track_;
    std::shared_ptr<UIButton> perc_btn_inst_seg_;

    // Frame blit cache for the Perception tab (model-image view).
    PreviewBlitCache perception_blit_cache_;

    // Bus-pull state for perception results.
    GRIM::Perception::Physical::PhysicalPerceptionPrimitiveBus::ResultsView
        perc_results_view_;
    uint64_t last_perc_results_counter_ = 0;
    bool     have_any_perc_results_     = false;

    // Cached colour-mapped segmentation overlay (alpha-blended into the blit).
    std::vector<uint32_t> seg_overlay_argb_;
    int                   seg_overlay_w_         = 0;
    int                   seg_overlay_h_         = 0;
    uint64_t              seg_overlay_source_id_ = 0;

    // Per-track motion trail history. Keyed by track_id; value is a ring of
    // recent MODEL-space centres. Drawn as a fading polyline so the user can
    // see persistence over time. Capped at kMaxTrailPoints entries; tracks
    // not seen for several frames are evicted in UpdatePerceptionTab.
    static constexpr size_t kMaxTrailPoints = 32;
    struct TrackTrail {
        std::vector<std::pair<float,float>> model_centres; // back = newest
        uint64_t                            last_seen_frame_counter = 0;
    };
    std::unordered_map<uint64_t, TrackTrail> track_trails_;

    // ── Spatial tab (Stage-3) ──
    void HandleToggleSpatialDepthEstimator();
    void HandleToggleSpatialGrounder();
    void RefreshSpatialEnableButtonLabelsFromSubsystem();
    void UpdateSpatialTab(const InputState& input, float dt);
    void DrawSpatialTab(OverlayRenderer& renderer);
    void DrawSpatialDepthHeatmap(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalDepthMap& dmap,
        uint64_t source_id,
        float frame_x, float frame_y, float frame_w, float frame_h);
    void DrawSpatialGroundedEntitiesOverlay(
        OverlayRenderer& renderer,
        const std::vector<GRIM::Perception::Physical::PhysicalGroundedEntity>& entities,
        int blit_x, int blit_y, int blit_w, int blit_h,
        int model_w, int model_h);
    void DrawSpatialSidebar(
        OverlayRenderer& renderer, float x, float y, float w, float h,
        const GRIM::Perception::Physical::PhysicalSpatialGroundingResults& r,
        bool have_results);

    std::shared_ptr<UIButton> spatial_btn_depth_;
    std::shared_ptr<UIButton> spatial_btn_ground_;

    // Bus-pull state for spatial grounding results.
    GRIM::Perception::Physical::PhysicalSpatialGroundingBus::ResultsView
        spatial_results_view_;
    uint64_t spatial_last_seen_counter_ = 0;
    bool     have_any_spatial_results_  = false;

    // Heatmap blit cache (depth map → COLORMAP_INFERNO → ARGB pre-pack).
    PreviewBlitCache spatial_heatmap_blit_cache_;
    cv::Mat          spatial_heatmap_bgr_;             // built lazily from depth_map
    uint64_t         spatial_heatmap_source_id_ = 0;   // 0 = not yet built

    // ── World-state tab (Stage-4) ──
    // The model's view: identity-keyed entities with class, position,
    // velocity, visibility, depth, text-on-object, and inter-entity
    // relations. Pulls directly from PhysicalWorldStateBus — does NOT
    // re-fuse from upstream buses, because the world-state loop already
    // did that exactly once per matched frame.
    void UpdateWorldTab(const InputState& input, float dt);
    void DrawWorldTab(OverlayRenderer& renderer);
    void DrawWorldEntitiesOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalWorldStateSnapshot& snap,
        int blit_x, int blit_y, int blit_w, int blit_h);
    void DrawWorldEntitiesSidebar(
        OverlayRenderer& renderer, float x, float y, float w, float h,
        const GRIM::Perception::Physical::PhysicalWorldStateSnapshot& snap,
        bool have_results);

    GRIM::Perception::Physical::PhysicalWorldStateBus::SnapshotView
        world_snapshot_view_;
    uint64_t world_last_seen_counter_ = 0;
    bool     have_any_world_results_  = false;
    PreviewBlitCache world_blit_cache_;
};

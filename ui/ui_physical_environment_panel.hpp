#pragma once

#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_dropdown.hpp"
#include "primitives/ui_inputbox.hpp"
#include "primitives/ui_scrollbox.hpp"
#include "perception/physical/PhysicalCameraSource.hpp"
#include "perception/physical/PhysicalCameraCalibrator.hpp"
#include "perception/physical/PhysicalFrameConditioner.hpp"
#include "perception/physical/PhysicalFrameBus.hpp"
#include "perception/physical/PhysicalStereoFrameBus.hpp"
#include "perception/physical/PhysicalPerceptionPrimitiveBus.hpp"
#include "perception/physical/PhysicalPerceptionPrimitivesLoop.hpp"
#include "perception/physical/PhysicalHandGestureBus.hpp"
#include "perception/physical/PhysicalInteractionLoop.hpp"
#include "perception/physical/PhysicalGestureControlLoop.hpp"
#include "perception/physical/PhysicalSpatialGroundingBus.hpp"
#include "perception/physical/PhysicalSpatialGroundingLoop.hpp"
#include "perception/physical/PhysicalLocalizationBus.hpp"
#include "perception/physical/PhysicalLocalizationLoop.hpp"
#include "perception/physical/PhysicalKnownEntityRegistry.hpp"
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
// calibrated on Calibration with detected-corner overlay).
//
// Rule 20: when no frame is on the bus, the panel SAYS so. No stub graphic.
class UIPhysicalEnvironmentPanel : public UIPanel {
public:
    enum class Tab : uint8_t {
        Camera       = 0,
        Stereo       = 1,
        Calibration  = 2,
        Perception   = 3,
        Interaction  = 4,
        Spatial      = 5,
        Localization = 6,
        World        = 7,
        KnownEntities = 8
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
    // source variant flipped, or output geometry changed).
    struct PreviewBlitCache {
        uint64_t              source_id        = 0;     // 0 = empty (counters start at 1)
        bool                  source_is_processed = false;
        std::string           source_color_space;
        int                   out_w            = 0;
        int                   out_h            = 0;
        std::vector<uint32_t> argb;                     // size = out_w * out_h
    };

    void DrawBgrFrameIntoOverlay(OverlayRenderer& renderer,
                                 const cv::Mat& bgr,
                                 uint64_t source_id,
                                 bool source_is_processed,
                                 float frame_x, float frame_y,
                                 float frame_w, float frame_h,
                                 PreviewBlitCache& cache,
                                 const std::string& color_space_label = "BGR8_SRGB");

    // ── Camera tab ──
    void RebuildSourceDropdownFromDirectory();
    void HandleConnectClicked();
    void HandleDisconnectClicked();
    void HandleRefreshClicked();
    void HandleToggleCameraViewClicked();
    void HandleToggleAutoExposureClicked();
    void HandleToggleAntiFlickerClicked();
    void HandleToggleMotionExposureClicked();
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
    std::shared_ptr<UIButton>    signal_anti_flicker_btn_;
    std::shared_ptr<UIButton>    signal_motion_exposure_btn_;
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

    // ── Stereo capture tab ──
    void HandleStereoConnectClicked();
    void HandleStereoDisconnectClicked();
    void UpdateStereoTab(const InputState& input, float dt);
    void DrawStereoTab(OverlayRenderer& renderer);

    std::shared_ptr<UIDropdown> stereo_left_dropdown_;
    std::shared_ptr<UIDropdown> stereo_right_dropdown_;
    std::shared_ptr<UIInputBox> stereo_skew_box_;
    std::shared_ptr<UIButton>   stereo_connect_button_;
    std::shared_ptr<UIButton>   stereo_disconnect_button_;
    std::string                 stereo_skew_buffer_ = "15.0";
    std::string                 stereo_ui_status_;
    GRIM::Perception::Physical::PhysicalStereoCaptureStatus stereo_capture_status_;
    GRIM::Perception::Physical::PhysicalStereoFrameBus::FrameView stereo_frame_view_;
    uint64_t                    stereo_last_seen_pair_counter_ = 0;
    bool                        have_any_stereo_pair_ = false;
    PreviewBlitCache            stereo_left_blit_cache_;
    PreviewBlitCache            stereo_right_blit_cache_;

    // ── Calibration tab ──
    void HandleAutomaticCalibrationClicked();
    void HandleStopCaptureClicked();
    void HandleClearSamplesClicked();
    void HandleToggleCalibratedViewClicked();
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

    std::shared_ptr<UIButton>   cal_auto_btn_;
    std::shared_ptr<UIButton>   cal_stop_btn_;
    std::shared_ptr<UIButton>   cal_clear_btn_;
    std::shared_ptr<UIButton>   cal_calibrated_toggle_btn_;
    std::shared_ptr<UIButton>   cal_apply_pattern_btn_;
    std::shared_ptr<UIInputBox> cal_pattern_cols_box_;
    std::shared_ptr<UIInputBox> cal_pattern_rows_box_;
    std::shared_ptr<UIInputBox> cal_square_meters_box_;
    std::string                 cal_pattern_cols_buf_   = "9";
    std::string                 cal_pattern_rows_buf_   = "6";
    std::string                 cal_square_meters_buf_  = "0.025";
    bool                        cal_show_calibrated_    = false;
    bool                        cal_was_automatic_      = false;
    GRIM::Perception::Physical::PhysicalCalibrationStatus cal_last_status_;

    // ── Tab bar ──
    std::shared_ptr<UIButton>   tab_camera_btn_;
    std::shared_ptr<UIButton>   tab_stereo_btn_;
    std::shared_ptr<UIButton>   tab_calibration_btn_;
    std::shared_ptr<UIButton>   tab_perception_btn_;
    std::shared_ptr<UIButton>   tab_interaction_btn_;
    std::shared_ptr<UIButton>   tab_spatial_btn_;
    std::shared_ptr<UIButton>   tab_localization_btn_;
    std::shared_ptr<UIButton>   tab_world_btn_;
    std::shared_ptr<UIButton>   tab_known_entities_btn_;
    Tab                         active_tab_ = Tab::Camera;

    // ── Shared frame pull state (one bus, one cached frame) ──
    GRIM::Perception::Physical::PhysicalFrameBus::FrameView last_view_;
    uint64_t                                                last_seen_counter_ = 0;
    bool                                                    have_any_frame_ = false;

    // ── Production preview pipeline ──
    // Camera tab: raw frame → blit cache.
    PreviewBlitCache camera_blit_cache_;

    // Calibration tab: pinned analyzed raw frame → authoritative calibrator
    // corners → blit cache. Detection is asynchronous and owned by the
    // calibrator; the UI never reuses corners across different frames.
    cv::Mat            calib_display_frame_;                // BGR, ready to blit
    uint64_t           calib_display_source_id_      = 0;   // 0 = not yet built
    uint64_t           calib_display_detection_id_   = 0;
    bool               calib_display_calibrated_     = false;
    PreviewBlitCache   calib_blit_cache_;

    // ── Perception tab ──
    void HandleTogglePerceptionObjectDetector();
    void HandleTogglePerceptionSemanticSegmenter();
    void HandleTogglePerceptionImageClassifier();
    void HandleTogglePerceptionPoseEstimator();
    void HandleTogglePerceptionSceneTextReader();
    void HandleTogglePerceptionFacialExpressionDetector();
    void HandleTogglePerceptionEntityTracker();
    void HandleTogglePerceptionInstanceSegmenter();
    void HandleTogglePerceptionClassPolicy();
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
    std::shared_ptr<UIButton> perc_btn_class_policy_;

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

    // Human-interaction branch (independent Stage-2 FrameBus consumer).
    void HandleToggleHandGestures();
    void HandleReloadHandGestureBackend();
    void HandleToggleGestureController();
    void HandleToggleGestureDryRun();
    void HandleToggleGestureStudioView();
    void HandleUseLiveGesture();
    void HandleApplyGestureBinding();
    void HandleAddGestureBinding();
    void HandleDeleteGestureBinding();
    void HandleRestoreDefaultGestureBindings();
    void RefreshInteractionButtonLabels();
    void RebuildGestureBindingEditor();
    void LoadSelectedGestureBindingIntoEditor();
    void PersistGestureBindingsFromUi();
    void UpdateInteractionTab(const InputState& input, float dt);
    void DrawInteractionTab(OverlayRenderer& renderer);
    void DrawInteractionPreview(OverlayRenderer& renderer,
                                float frame_x, float frame_y,
                                float frame_w, float frame_h);
    void DrawGestureBindingsEditor(OverlayRenderer& renderer);
    void DrawHandGestureOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalHandGestureSnapshot& snapshot,
        int blit_x, int blit_y, int blit_w, int blit_h);

    std::shared_ptr<UIButton> interaction_enable_btn_;
    std::shared_ptr<UIButton> interaction_reload_btn_;
    std::shared_ptr<UIButton> interaction_controller_btn_;
    std::shared_ptr<UIButton> interaction_dry_run_btn_;
    std::shared_ptr<UIButton> interaction_view_btn_;
    std::shared_ptr<UIScrollBox> interaction_binding_list_;
    std::vector<std::shared_ptr<UIButton>> interaction_binding_rows_;
    std::shared_ptr<UIDropdown> interaction_action_select_;
    std::shared_ptr<UIDropdown> interaction_trigger_select_;
    std::shared_ptr<UIDropdown> interaction_hand_select_;
    std::shared_ptr<UIInputBox> interaction_gesture_box_;
    std::shared_ptr<UIInputBox> interaction_hold_box_;
    std::shared_ptr<UIInputBox> interaction_cooldown_box_;
    std::shared_ptr<UIInputBox> interaction_priority_box_;
    std::shared_ptr<UIButton> interaction_use_live_btn_;
    std::shared_ptr<UIButton> interaction_binding_enabled_btn_;
    std::shared_ptr<UIButton> interaction_requires_arm_btn_;
    std::shared_ptr<UIButton> interaction_apply_binding_btn_;
    std::shared_ptr<UIButton> interaction_add_binding_btn_;
    std::shared_ptr<UIButton> interaction_delete_binding_btn_;
    std::shared_ptr<UIButton> interaction_defaults_btn_;
    std::string interaction_gesture_buf_;
    std::string interaction_hold_buf_;
    std::string interaction_cooldown_buf_;
    std::string interaction_priority_buf_;
    std::string interaction_binding_status_;
    size_t interaction_selected_binding_ = 0;
    bool interaction_editor_binding_enabled_ = true;
    bool interaction_editor_requires_arm_ = false;
    bool interaction_show_bindings_ = false;
    bool interaction_binding_list_needs_rebuild_ = false;
    PreviewBlitCache interaction_blit_cache_;
    GRIM::Perception::Physical::PhysicalHandGestureBus::SnapshotView
        interaction_snapshot_view_;
    uint64_t interaction_last_seen_sequence_ = 0;
    bool     have_interaction_snapshot_ = false;

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
    void HandleToggleWorldViewport();
    void DrawWorldEntitiesOverlay(
        OverlayRenderer& renderer,
        const GRIM::Perception::Physical::PhysicalWorldStateSnapshot& snap,
        int blit_x, int blit_y, int blit_w, int blit_h,
        bool model_space);
    void DrawWorldEntitiesSidebar(
        OverlayRenderer& renderer, float x, float y, float w, float h,
        const GRIM::Perception::Physical::PhysicalWorldStateSnapshot& snap,
        bool have_results);

    GRIM::Perception::Physical::PhysicalWorldStateBus::SnapshotView
        world_snapshot_view_;
    uint64_t world_last_seen_counter_ = 0;
    bool     have_any_world_results_  = false;
    bool     world_show_model_signal_ = false;
    std::shared_ptr<UIButton> world_view_toggle_btn_;
    PreviewBlitCache world_blit_cache_;

    // ── Known Entities tab ──
    // Shows live tracks plus durable named identity profiles. Biometric
    // enrollment is always an explicit user action.
    struct KnownEntityUiRow {
        uint64_t selection_key = 0;
        uint64_t known_entity_id = 0;
        uint64_t object_id = 0;
        std::string name;
        std::string persistent_entity_id;
        GRIM::Perception::Physical::PhysicalWorldEntity entity;
        bool currently_tracked = false;
        std::vector<uint64_t> track_history;
        uint32_t automatic_relink_count = 0;
        float last_automatic_relink_score = 0.0f;
        size_t face_template_count = 0;
        float last_face_match_score = 0.0f;
        float last_face_quality = 0.0f;
        GRIM::Perception::Physical::PhysicalEntityIdentityState identity_state =
            GRIM::Perception::Physical::PhysicalEntityIdentityState::Unknown;
    };

    void UpdateKnownEntitiesTab(const InputState& input, float dt);
    void DrawKnownEntitiesTab(OverlayRenderer& renderer);
    void RebuildKnownEntityRows();
    void LoadSelectedKnownEntityName();
    void HandleApplyKnownEntityName();
    void HandleClearKnownEntityName();
    void HandleEnrollKnownEntityFace();
    void HandleForgetKnownEntityFace();
    const KnownEntityUiRow* FindSelectedKnownEntityRow() const;

    std::shared_ptr<UIScrollBox> known_entity_list_;
    std::vector<std::shared_ptr<UIButton>> known_entity_row_buttons_;
    std::shared_ptr<UIInputBox> known_entity_name_box_;
    std::shared_ptr<UIButton> known_entity_apply_btn_;
    std::shared_ptr<UIButton> known_entity_clear_btn_;
    std::shared_ptr<UIButton> known_entity_enroll_face_btn_;
    std::shared_ptr<UIButton> known_entity_forget_face_btn_;
    std::vector<KnownEntityUiRow> known_entity_rows_;
    std::string known_entity_name_buffer_;
    std::string known_entity_status_;
    uint64_t known_selected_object_id_ = 0;
    uint64_t known_registry_revision_ = 0;
    uint64_t known_last_snapshot_frame_ = 0;
    GRIM::Perception::Physical::PhysicalWorldStateBus::SnapshotView
        known_snapshot_view_;
    bool have_known_world_results_ = false;

    // ── Localization tab (Stage-5) ──
    // Pulls the latest PhysicalLocalizationSnapshot (camera pose,
    // velocity, trajectory, occupancy grid) from PhysicalLocalizationBus
    // and renders text readouts plus a top-down trajectory minimap.
    void HandleResetLocalizationClicked();
    void UpdateLocalizationTab(const InputState& input, float dt);
    void DrawLocalizationTab(OverlayRenderer& renderer);
    void DrawLocalizationTrajectoryMinimap(
        OverlayRenderer& renderer,
        float x, float y, float w, float h,
        const GRIM::Perception::Physical::PhysicalLocalizationSnapshot& snap);

    std::shared_ptr<UIButton> loc_reset_btn_;

    GRIM::Perception::Physical::PhysicalLocalizationBus::SnapshotView
        loc_snapshot_view_;
    uint64_t loc_last_seen_publish_sequence_ = 0;
    bool     have_any_loc_snapshot_          = false;
};

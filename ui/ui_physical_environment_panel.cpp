#include "ui_physical_environment_panel.hpp"

#include "logger.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "perception/physical/PhysicalCalibrationPattern.hpp"
#include "perception/physical/PhysicalEnvironmentLogTag.hpp"
#include "perception/physical/PhysicalEnvironmentLoop.hpp"
#include "perception/physical/PhysicalGestureControlConfigIO.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace PE = GRIM::Perception::Physical;

namespace {

constexpr const char* kPanelLogTag = PE::PHYSICAL_ENV_LOG_TAG;

// ── Shared formatting helpers ───────────────────────────────────────────────

std::string FormatStatus(PE::PhysicalCameraCandidateStatus s) {
    switch (s) {
        case PE::PhysicalCameraCandidateStatus::Ready:            return "Ready";
        case PE::PhysicalCameraCandidateStatus::NoNetworkAddress: return "NoNetworkAddress";
        case PE::PhysicalCameraCandidateStatus::Disabled:         return "Disabled";
    }
    return "Unknown";
}

std::string FormatOrigin(PE::PhysicalCameraOrigin o) {
    switch (o) {
        case PE::PhysicalCameraOrigin::LocalNic:    return "LocalNic";
        case PE::PhysicalCameraOrigin::HubDevice:   return "HubDevice";
        case PE::PhysicalCameraOrigin::LocalDevice: return "LocalDevice";
    }
    return "Unknown";
}

const char* FormatCameraStreamState(PE::PhysicalCameraStreamState state) {
    switch (state) {
        case PE::PhysicalCameraStreamState::Idle: return "Idle";
        case PE::PhysicalCameraStreamState::Opening: return "Opening";
        case PE::PhysicalCameraStreamState::Streaming: return "Streaming";
        case PE::PhysicalCameraStreamState::Failed: return "Failed";
        case PE::PhysicalCameraStreamState::Closed: return "Closed";
    }
    return "Invalid";
}

std::string MakeDropdownLine(const PE::PhysicalCameraSource& s) {
    return "[" + FormatOrigin(s.origin) + "/" + FormatStatus(s.status) + "] " + s.label;
}

std::string FormatStage(PE::PhysicalCalibrationStage st) {
    switch (st) {
        case PE::PhysicalCalibrationStage::Uncalibrated:   return "Uncalibrated";
        case PE::PhysicalCalibrationStage::LoadedFromDisk: return "LoadedFromDisk";
        case PE::PhysicalCalibrationStage::Capturing:      return "Capturing";
        case PE::PhysicalCalibrationStage::Calibrated:     return "Calibrated";
        case PE::PhysicalCalibrationStage::Failed:         return "Failed";
    }
    return "Unknown";
}

uint32_t MakeArgb(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    return (static_cast<uint32_t>(a) << 24)
         | (static_cast<uint32_t>(r) << 16)
         | (static_cast<uint32_t>(g) << 8)
         |  static_cast<uint32_t>(b);
}

std::string FormatDouble(double v, int prec = 3) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(prec) << v;
    return ss.str();
}

const char* FormatHandBackendState(PE::PhysicalHandGestureBackendState state) {
    switch (state) {
        case PE::PhysicalHandGestureBackendState::Disabled: return "Disabled";
        case PE::PhysicalHandGestureBackendState::BackendUnavailable: return "Backend unavailable";
        case PE::PhysicalHandGestureBackendState::ModelMissing: return "Model missing";
        case PE::PhysicalHandGestureBackendState::Initializing: return "Initializing";
        case PE::PhysicalHandGestureBackendState::Ready: return "Ready";
        case PE::PhysicalHandGestureBackendState::Failed: return "Failed";
    }
    return "Unknown";
}

const char* FormatHandedness(PE::PhysicalHandedness handedness) {
    switch (handedness) {
        case PE::PhysicalHandedness::Left: return "Left";
        case PE::PhysicalHandedness::Right: return "Right";
        case PE::PhysicalHandedness::Unknown: return "Unknown";
    }
    return "Unknown";
}

std::string CompactPath(const std::string& path, size_t max_chars = 42) {
    if (path.size() <= max_chars) return path;
    return "..." + path.substr(path.size() - (max_chars - 3));
}

bool TryParseInt(const std::string& s, int& out) {
    if (s.empty()) return false;
    try { out = std::stoi(s); return true; } catch (...) { return false; }
}

bool TryParseFloat(const std::string& s, float& out) {
    if (s.empty()) return false;
    try { out = std::stof(s); return true; } catch (...) { return false; }
}

// Layout constants — tab bar takes a single row directly under the title bar,
// content starts after that.
constexpr float kTabBarHeight = 30.0f;
constexpr float kTabBarPad    = 8.0f;

} // anonymous

// ============================================================================
//  Construction
// ============================================================================

UIPhysicalEnvironmentPanel::UIPhysicalEnvironmentPanel()
    : UIPanel("Physical Environment", true)
{
    position = { 200, 500 };
    size     = { 880, 680 };
    setBackground(UITheme::Colors::PanelBg);
    setBorder(UITheme::Colors::DividerLine);

    // ── Tab buttons ──
    tab_camera_btn_ = std::make_shared<UIButton>(" Camera ",
        [this]() { setActiveTab(Tab::Camera); });
    tab_stereo_btn_ = std::make_shared<UIButton>(" Stereo ",
        [this]() { setActiveTab(Tab::Stereo); });
    tab_calibration_btn_ = std::make_shared<UIButton>(" Calibration ",
        [this]() { setActiveTab(Tab::Calibration); });
    tab_perception_btn_ = std::make_shared<UIButton>(" Perception ",
        [this]() { setActiveTab(Tab::Perception); });
    tab_interaction_btn_ = std::make_shared<UIButton>(" Interaction ",
        [this]() { setActiveTab(Tab::Interaction); });
    tab_spatial_btn_ = std::make_shared<UIButton>(" Spatial ",
        [this]() { setActiveTab(Tab::Spatial); });
    tab_localization_btn_ = std::make_shared<UIButton>(" Localization ",
        [this]() { setActiveTab(Tab::Localization); });
    tab_world_btn_ = std::make_shared<UIButton>(" World ",
        [this]() { setActiveTab(Tab::World); });

    // ── Localization tab controls ──
    loc_reset_btn_ = std::make_shared<UIButton>(" Reset Pose ",
        [this]() { HandleResetLocalizationClicked(); });

    // ── Perception tab toggle buttons (labels refreshed at end of ctor) ──
    perc_btn_obj_  = std::make_shared<UIButton>(" Detector: on ",
        [this]{ HandleTogglePerceptionObjectDetector(); });
    perc_btn_seg_  = std::make_shared<UIButton>(" Segmenter: on ",
        [this]{ HandleTogglePerceptionSemanticSegmenter(); });
    perc_btn_cls_  = std::make_shared<UIButton>(" Classifier: on ",
        [this]{ HandleTogglePerceptionImageClassifier(); });
    perc_btn_pose_ = std::make_shared<UIButton>(" Pose: on ",
        [this]{ HandleTogglePerceptionPoseEstimator(); });
    perc_btn_text_ = std::make_shared<UIButton>(" Text: on ",
        [this]{ HandleTogglePerceptionSceneTextReader(); });
    perc_btn_face_ = std::make_shared<UIButton>(" Face: on ",
        [this]{ HandleTogglePerceptionFacialExpressionDetector(); });
    perc_btn_track_ = std::make_shared<UIButton>(" Tracks: on ",
        [this]{ HandleTogglePerceptionEntityTracker(); });
    perc_btn_inst_seg_ = std::make_shared<UIButton>(" InstSeg: on ",
        [this]{ HandleTogglePerceptionInstanceSegmenter(); });
    perc_btn_class_policy_ = std::make_shared<UIButton>(" Policy: on ",
        [this]{ HandleTogglePerceptionClassPolicy(); });
    RefreshPerceptionEnableButtonLabelsFromSubsystem();

    interaction_enable_btn_ = std::make_shared<UIButton>(" Gestures: on ",
        [this]{ HandleToggleHandGestures(); });
    interaction_reload_btn_ = std::make_shared<UIButton>(" Reinitialize local backend ",
        [this]{ HandleReloadHandGestureBackend(); });
    interaction_controller_btn_ = std::make_shared<UIButton>(" Controller: on ",
        [this]{ HandleToggleGestureController(); });
    interaction_dry_run_btn_ = std::make_shared<UIButton>(" Dry run: off ",
        [this]{ HandleToggleGestureDryRun(); });
    interaction_view_btn_ = std::make_shared<UIButton>(" View: Live ",
        [this]{ HandleToggleGestureStudioView(); });

    interaction_binding_list_ = std::make_shared<UIScrollBox>();
    interaction_binding_list_->setChildSpacing(6.0f);
    interaction_action_select_ = std::make_shared<UIDropdown>(
        "Action", std::vector<std::string>{
            "control.arm", "control.disarm", "pointer.move",
            "mouse.left_click", "mouse.right_click", "voice.wake"}, 0,
        [](int, const std::string&) {});
    interaction_trigger_select_ = std::make_shared<UIDropdown>(
        "Trigger", std::vector<std::string>{"started", "held", "released"},
        1, [](int, const std::string&) {});
    interaction_hand_select_ = std::make_shared<UIDropdown>(
        "Hand", std::vector<std::string>{"either", "left", "right"},
        0, [](int, const std::string&) {});

    interaction_gesture_box_ = std::make_shared<UIInputBox>(&interaction_gesture_buf_);
    interaction_gesture_box_->setPlaceholder("MediaPipe gesture label");
    interaction_hold_box_ = std::make_shared<UIInputBox>(&interaction_hold_buf_);
    interaction_hold_box_->setPlaceholder("hold ms");
    interaction_cooldown_box_ = std::make_shared<UIInputBox>(&interaction_cooldown_buf_);
    interaction_cooldown_box_->setPlaceholder("cooldown ms");
    interaction_priority_box_ = std::make_shared<UIInputBox>(&interaction_priority_buf_);
    interaction_priority_box_->setPlaceholder("priority");

    interaction_use_live_btn_ = std::make_shared<UIButton>(" Use live gesture ",
        [this]{ HandleUseLiveGesture(); });
    interaction_binding_enabled_btn_ = std::make_shared<UIButton>(" Binding: on ",
        [this] {
            interaction_editor_binding_enabled_ = !interaction_editor_binding_enabled_;
            interaction_binding_enabled_btn_->setText(
                interaction_editor_binding_enabled_ ? " Binding: on " : " Binding: off ");
        });
    interaction_requires_arm_btn_ = std::make_shared<UIButton>(" Requires arm: no ",
        [this] {
            interaction_editor_requires_arm_ = !interaction_editor_requires_arm_;
            interaction_requires_arm_btn_->setText(
                interaction_editor_requires_arm_ ? " Requires arm: yes "
                                                 : " Requires arm: no ");
        });
    interaction_apply_binding_btn_ = std::make_shared<UIButton>(" Apply + Save ",
        [this]{ HandleApplyGestureBinding(); });
    interaction_add_binding_btn_ = std::make_shared<UIButton>(" Add ",
        [this]{ HandleAddGestureBinding(); });
    interaction_delete_binding_btn_ = std::make_shared<UIButton>(" Delete ",
        [this]{ HandleDeleteGestureBinding(); });
    interaction_defaults_btn_ = std::make_shared<UIButton>(" Restore Defaults ",
        [this]{ HandleRestoreDefaultGestureBindings(); });
    RefreshInteractionButtonLabels();
    RebuildGestureBindingEditor();

    // ── Spatial tab toggle buttons (labels refreshed at end of ctor) ──
    spatial_btn_depth_ = std::make_shared<UIButton>(" Depth: on ",
        [this]{ HandleToggleSpatialDepthEstimator(); });
    spatial_btn_ground_ = std::make_shared<UIButton>(" Ground: on ",
        [this]{ HandleToggleSpatialGrounder(); });
    RefreshSpatialEnableButtonLabelsFromSubsystem();

    // ── Camera tab widgets ──
    url_inputbox_ = std::make_shared<UIInputBox>(&url_buffer_);
    url_inputbox_->setPlaceholder("rtsp://<host>:554/  — or HTTP MJPEG URL");

    refresh_button_    = std::make_shared<UIButton>(" Refresh ",
                            [this]() { HandleRefreshClicked(); });
    connect_button_    = std::make_shared<UIButton>(" Connect ",
                            [this]() { HandleConnectClicked(); });
    disconnect_button_ = std::make_shared<UIButton>(" Disconnect ",
                            [this]() { HandleDisconnectClicked(); });

    signal_view_toggle_btn_ = std::make_shared<UIButton>(" View: Model Signal ",
                            [this]() { HandleToggleCameraViewClicked(); });
    signal_auto_exposure_btn_ = std::make_shared<UIButton>(" Exposure: Auto ",
                            [this]() { HandleToggleAutoExposureClicked(); });
    signal_denoise_btn_ = std::make_shared<UIButton>(" Denoise: On ",
                            [this]() { HandleToggleDenoiseClicked(); });
    signal_resize_btn_ = std::make_shared<UIButton>(" Resize: On ",
                            [this]() { HandleToggleResizeClicked(); });
    signal_deblur_btn_ = std::make_shared<UIButton>(" Deblur: Off ",
                            [this]() { HandleToggleDeblurClicked(); });
    signal_stabilization_btn_ = std::make_shared<UIButton>(" Stabilize: Off ",
                            [this]() { HandleToggleStabilizationClicked(); });
    signal_color_mode_btn_ = std::make_shared<UIButton>(" Color: BGR ",
                            [this]() { HandleToggleColorModeClicked(); });
    signal_resize_mode_btn_ = std::make_shared<UIButton>(" Fit: Letterbox ",
                            [this]() { HandleToggleResizeModeClicked(); });
    signal_quality_gate_btn_ = std::make_shared<UIButton>(" QGate: On ",
                            [this]() { HandleToggleQualityGateClicked(); });
    signal_apply_btn_ = std::make_shared<UIButton>(" Apply Signal Settings ",
                            [this]() { HandleApplySignalSettingsClicked(); });
    signal_reset_btn_ = std::make_shared<UIButton>(" Reset Defaults ",
                            [this]() { HandleResetSignalSettingsClicked(); });

    signal_width_box_ = std::make_shared<UIInputBox>(&signal_width_buf_);
    signal_height_box_ = std::make_shared<UIInputBox>(&signal_height_buf_);
    signal_target_luma_box_ = std::make_shared<UIInputBox>(&signal_target_luma_buf_);
    signal_manual_gain_box_ = std::make_shared<UIInputBox>(&signal_manual_gain_buf_);
    signal_denoise_strength_box_ = std::make_shared<UIInputBox>(&signal_denoise_strength_buf_);
    signal_deblur_amount_box_ = std::make_shared<UIInputBox>(&signal_deblur_amount_buf_);

    signal_width_box_->setPlaceholder("out_w");
    signal_height_box_->setPlaceholder("out_h");
    signal_target_luma_box_->setPlaceholder("target_luma");
    signal_manual_gain_box_->setPlaceholder("manual_gain");
    signal_denoise_strength_box_->setPlaceholder("denoise_h");
    signal_deblur_amount_box_->setPlaceholder("deblur");

    source_dropdown_ = std::make_shared<UIDropdown>(
        "Source", std::vector<std::string>{"(no candidates yet)"}, 0,
        [this](int idx, const std::string& /*item*/) {
            if (idx >= 0 && idx < static_cast<int>(last_directory_.size())) {
                const auto& src = last_directory_[static_cast<size_t>(idx)];
                if (!src.url_template.empty()) {
                    url_buffer_ = src.url_template;
                    if (url_inputbox_) url_inputbox_->setText(url_buffer_);
                    selection_info_.clear();
                } else {
                    url_buffer_.clear();
                    if (url_inputbox_) url_inputbox_->clear();
                    selection_info_ = "Selected '" + src.label + "' — "
                                    + (src.status_reason.empty()
                                         ? std::string("no URL available for this candidate")
                                         : src.status_reason)
                                    + " — type a URL manually to connect.";
                }
                LOG_DEBUG(kPanelLogTag,
                          "PhysicalEnvironmentPanel: selected candidate idx="
                          + std::to_string(idx) + " label='" + src.label
                          + "' status_reason='" + src.status_reason + "'");
            }
        });

    stereo_left_dropdown_ = std::make_shared<UIDropdown>(
        "Left camera", std::vector<std::string>{"(no candidates yet)"}, 0,
        [](int, const std::string&) {});
    stereo_right_dropdown_ = std::make_shared<UIDropdown>(
        "Right camera", std::vector<std::string>{"(no candidates yet)"}, 0,
        [](int, const std::string&) {});
    stereo_skew_box_ = std::make_shared<UIInputBox>(&stereo_skew_buffer_);
    stereo_skew_box_->setPlaceholder("maximum skew ms");
    stereo_connect_button_ = std::make_shared<UIButton>(" Connect Pair ",
        [this]() { HandleStereoConnectClicked(); });
    stereo_disconnect_button_ = std::make_shared<UIButton>(" Disconnect Pair ",
        [this]() { HandleStereoDisconnectClicked(); });

    try {
        PE::RequestPhysicalCameraDirectoryRefresh();
        RebuildSourceDropdownFromDirectory();
        SyncSignalSettingsControlsFromSubsystem();
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("PhysicalEnvironmentPanel: initial directory refresh failed: ")
                  + e.what());
    }

    // ── Calibration tab widgets ──
    cal_start_btn_            = std::make_shared<UIButton>(" Start Capture ",
                                    [this]() { HandleStartCaptureClicked(); });
    cal_stop_btn_             = std::make_shared<UIButton>(" Stop Capture ",
                                    [this]() { HandleStopCaptureClicked(); });
    cal_capture_now_btn_      = std::make_shared<UIButton>(" Capture Now ",
                                    [this]() { HandleCaptureNowClicked(); });
    cal_clear_btn_            = std::make_shared<UIButton>(" Clear Samples ",
                                    [this]() { HandleClearSamplesClicked(); });
    cal_run_btn_              = std::make_shared<UIButton>(" Calibrate ",
                                    [this]() { HandleRunCalibrationClicked(); });
    cal_save_btn_             = std::make_shared<UIButton>(" Save ",
                                    [this]() { HandleSaveCalibrationClicked(); });
    cal_reload_btn_           = std::make_shared<UIButton>(" Reload ",
                                    [this]() { HandleReloadCalibrationClicked(); });
    cal_undistort_toggle_btn_ = std::make_shared<UIButton>(" View: Raw ",
                                    [this]() { HandleToggleUndistortClicked(); });
    cal_apply_pattern_btn_    = std::make_shared<UIButton>(" Apply Pattern ",
                                    [this]() { HandleApplyPatternClicked(); });

    cal_pattern_cols_box_  = std::make_shared<UIInputBox>(&cal_pattern_cols_buf_);
    cal_pattern_rows_box_  = std::make_shared<UIInputBox>(&cal_pattern_rows_buf_);
    cal_square_meters_box_ = std::make_shared<UIInputBox>(&cal_square_meters_buf_);
    cal_pattern_cols_box_->setPlaceholder("inner cols");
    cal_pattern_rows_box_->setPlaceholder("inner rows");
    cal_square_meters_box_->setPlaceholder("square (m)");

    try {
        auto st = PE::GetPhysicalCalibrationStatusSnapshot();
        cal_pattern_cols_buf_  = std::to_string(st.pattern_inner_cols);
        cal_pattern_rows_buf_  = std::to_string(st.pattern_inner_rows);
        cal_square_meters_buf_ = FormatDouble(st.pattern_square_meters, 4);
        cal_pattern_cols_box_->setText(cal_pattern_cols_buf_);
        cal_pattern_rows_box_->setText(cal_pattern_rows_buf_);
        cal_square_meters_box_->setText(cal_square_meters_buf_);
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("UIPhysicalEnvironmentPanel ctor: calib snapshot threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::setActiveTab(Tab t) {
    active_tab_ = t;
}

// ============================================================================
//  Camera-tab handlers
// ============================================================================

void UIPhysicalEnvironmentPanel::RebuildSourceDropdownFromDirectory() {
    last_directory_ = PE::GetPhysicalCameraDirectorySnapshot();
    std::vector<std::string> lines;
    lines.reserve(last_directory_.size());
    for (const auto& s : last_directory_) lines.push_back(MakeDropdownLine(s));
    if (lines.empty()) lines.push_back("(no candidates found)");
    if (source_dropdown_) {
        source_dropdown_->setItems(lines);
        for (size_t i = 0; i < last_directory_.size(); ++i) {
            if (last_directory_[i].status == PE::PhysicalCameraCandidateStatus::Ready) {
                source_dropdown_->setSelectedIndex(static_cast<int>(i));
                url_buffer_ = last_directory_[i].url_template;
                if (url_inputbox_) url_inputbox_->setText(url_buffer_);
                break;
            }
        }
    }
    if (stereo_left_dropdown_) stereo_left_dropdown_->setItems(lines);
    if (stereo_right_dropdown_) stereo_right_dropdown_->setItems(lines);
    int first_ready = -1;
    int second_ready = -1;
    for (size_t i = 0; i < last_directory_.size(); ++i) {
        if (last_directory_[i].status != PE::PhysicalCameraCandidateStatus::Ready) continue;
        if (first_ready < 0) first_ready = static_cast<int>(i);
        else if (second_ready < 0) {
            second_ready = static_cast<int>(i);
            break;
        }
    }
    if (first_ready >= 0 && stereo_left_dropdown_) {
        stereo_left_dropdown_->setSelectedIndex(first_ready);
    }
    if (second_ready >= 0 && stereo_right_dropdown_) {
        stereo_right_dropdown_->setSelectedIndex(second_ready);
    }
}

void UIPhysicalEnvironmentPanel::HandleRefreshClicked() {
    try {
        PE::RequestPhysicalCameraDirectoryRefresh();
        RebuildSourceDropdownFromDirectory();
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("HandleRefreshClicked: refresh threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::HandleConnectClicked() {
    if (url_buffer_.empty()) {
        if (selection_info_.empty()) {
            selection_info_ = "URL is empty — pick a Ready candidate from the dropdown "
                              "or type an RTSP/HTTP URL into the URL field.";
        }
        LOG_ERROR(kPanelLogTag,
                  std::string("HandleConnectClicked: URL is empty — ") + selection_info_);
        return;
    }
    std::string label = url_buffer_;
    if (source_dropdown_ && source_dropdown_->getSelectedIndex() >= 0
        && source_dropdown_->getSelectedIndex() < static_cast<int>(last_directory_.size())) {
        label = last_directory_[static_cast<size_t>(source_dropdown_->getSelectedIndex())].label;
    }
    try {
        PE::RequestOpenPhysicalCameraSource(url_buffer_, label);
        last_seen_counter_ = 0;
        have_any_frame_    = false;
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("HandleConnectClicked: open threw url='")
                  + url_buffer_ + "' what=" + e.what());
    }
}

void UIPhysicalEnvironmentPanel::HandleDisconnectClicked() {
    PE::RequestClosePhysicalCameraSource();
    last_seen_counter_ = 0;
    have_any_frame_    = false;
}

void UIPhysicalEnvironmentPanel::HandleStereoConnectClicked() {
    if (!stereo_left_dropdown_ || !stereo_right_dropdown_) {
        throw std::runtime_error(
            "HandleStereoConnectClicked: stereo camera dropdown is null");
    }
    const int left_index  = stereo_left_dropdown_->getSelectedIndex();
    const int right_index = stereo_right_dropdown_->getSelectedIndex();
    if (left_index < 0 || right_index < 0
        || left_index >= static_cast<int>(last_directory_.size())
        || right_index >= static_cast<int>(last_directory_.size())) {
        stereo_ui_status_ = "Stereo selection is outside the current camera directory.";
        LOG_ERROR(kPanelLogTag, stereo_ui_status_);
        return;
    }
    const auto& left  = last_directory_[static_cast<size_t>(left_index)];
    const auto& right = last_directory_[static_cast<size_t>(right_index)];
    if (left.status != PE::PhysicalCameraCandidateStatus::Ready
        || right.status != PE::PhysicalCameraCandidateStatus::Ready
        || left.url_template.empty() || right.url_template.empty()) {
        stereo_ui_status_ = "Both stereo selections must be Ready camera sources.";
        LOG_ERROR(kPanelLogTag, stereo_ui_status_);
        return;
    }
    float maximum_skew_ms = 0.0f;
    if (!TryParseFloat(stereo_skew_buffer_, maximum_skew_ms)) {
        stereo_ui_status_ = "Maximum stereo skew is not a valid number.";
        LOG_ERROR(kPanelLogTag, stereo_ui_status_);
        return;
    }

    PE::PhysicalStereoCaptureConfig config;
    config.left_url             = left.url_template;
    config.left_label           = left.label;
    config.right_url            = right.url_template;
    config.right_label          = right.label;
    config.maximum_pair_skew_ms = static_cast<double>(maximum_skew_ms);
    try {
        PE::RequestOpenPhysicalStereoCameraPair(config);
        stereo_last_seen_pair_counter_ = 0;
        have_any_stereo_pair_ = false;
        stereo_ui_status_ = "Stereo pair opening.";
    } catch (const std::exception& e) {
        stereo_ui_status_ = std::string("Stereo pair open failed: ") + e.what();
        LOG_ERROR(kPanelLogTag, stereo_ui_status_);
    }
}

void UIPhysicalEnvironmentPanel::HandleStereoDisconnectClicked() {
    PE::RequestClosePhysicalStereoCameraPair();
    stereo_last_seen_pair_counter_ = 0;
    have_any_stereo_pair_ = false;
    stereo_ui_status_ = "Stereo pair disconnected.";
}

void UIPhysicalEnvironmentPanel::SyncSignalSettingsControlsFromSubsystem() {
    signal_cfg_ = PE::GetPhysicalSignalConditioningConfigSnapshot();
    signal_status_ = PE::GetPhysicalSignalConditioningStatusSnapshot();

    signal_width_buf_ = std::to_string(signal_cfg_.output_width);
    signal_height_buf_ = std::to_string(signal_cfg_.output_height);
    signal_target_luma_buf_ = FormatDouble(signal_cfg_.target_luma, 1);
    signal_manual_gain_buf_ = FormatDouble(signal_cfg_.manual_exposure_gain, 2);
    signal_denoise_strength_buf_ = std::to_string(signal_cfg_.denoise_strength);
    signal_deblur_amount_buf_ = FormatDouble(signal_cfg_.deblur_amount, 2);

    if (signal_width_box_) signal_width_box_->setText(signal_width_buf_);
    if (signal_height_box_) signal_height_box_->setText(signal_height_buf_);
    if (signal_target_luma_box_) signal_target_luma_box_->setText(signal_target_luma_buf_);
    if (signal_manual_gain_box_) signal_manual_gain_box_->setText(signal_manual_gain_buf_);
    if (signal_denoise_strength_box_) signal_denoise_strength_box_->setText(signal_denoise_strength_buf_);
    if (signal_deblur_amount_box_) signal_deblur_amount_box_->setText(signal_deblur_amount_buf_);

    if (signal_view_toggle_btn_) {
        signal_view_toggle_btn_->setText(camera_show_model_signal_
            ? " View: Model Signal "
            : " View: Raw Sensor ");
    }
    if (signal_auto_exposure_btn_) {
        signal_auto_exposure_btn_->setText(signal_cfg_.exposure_auto
            ? " Exposure: Auto "
            : " Exposure: Manual ");
    }
    if (signal_denoise_btn_) {
        signal_denoise_btn_->setText(signal_cfg_.enable_denoise
            ? " Denoise: On "
            : " Denoise: Off ");
    }
    if (signal_resize_btn_) {
        signal_resize_btn_->setText(signal_cfg_.enable_resize
            ? " Resize: On "
            : " Resize: Off ");
    }
    if (signal_deblur_btn_) {
        signal_deblur_btn_->setText(signal_cfg_.enable_deblur
            ? " Deblur: On "
            : " Deblur: Off ");
    }
    if (signal_stabilization_btn_) {
        signal_stabilization_btn_->setText(signal_cfg_.enable_stabilization
            ? " Stabilize: On "
            : " Stabilize: Off ");
    }
    if (signal_color_mode_btn_) {
        signal_color_mode_btn_->setText(signal_cfg_.color_mode == PE::PhysicalSignalColorMode::Gray
            ? " Color: Gray "
            : " Color: BGR ");
    }
    if (signal_resize_mode_btn_) {
        signal_resize_mode_btn_->setText(
            signal_cfg_.resize_mode == PE::PhysicalSignalResizeMode::Letterbox
                ? " Fit: Letterbox "
                : " Fit: Stretch ");
    }
    if (signal_quality_gate_btn_) {
        signal_quality_gate_btn_->setText(signal_cfg_.quality_gate.enable
            ? " QGate: On "
            : " QGate: Off ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleCameraViewClicked() {
    camera_show_model_signal_ = !camera_show_model_signal_;
    if (signal_view_toggle_btn_) {
        signal_view_toggle_btn_->setText(camera_show_model_signal_
            ? " View: Model Signal "
            : " View: Raw Sensor ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleAutoExposureClicked() {
    signal_cfg_.exposure_auto = !signal_cfg_.exposure_auto;
    if (signal_auto_exposure_btn_) {
        signal_auto_exposure_btn_->setText(signal_cfg_.exposure_auto
            ? " Exposure: Auto "
            : " Exposure: Manual ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleDenoiseClicked() {
    signal_cfg_.enable_denoise = !signal_cfg_.enable_denoise;
    if (signal_denoise_btn_) {
        signal_denoise_btn_->setText(signal_cfg_.enable_denoise
            ? " Denoise: On "
            : " Denoise: Off ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleResizeClicked() {
    signal_cfg_.enable_resize = !signal_cfg_.enable_resize;
    if (signal_resize_btn_) {
        signal_resize_btn_->setText(signal_cfg_.enable_resize
            ? " Resize: On "
            : " Resize: Off ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleDeblurClicked() {
    signal_cfg_.enable_deblur = !signal_cfg_.enable_deblur;
    if (signal_deblur_btn_) {
        signal_deblur_btn_->setText(signal_cfg_.enable_deblur
            ? " Deblur: On "
            : " Deblur: Off ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleStabilizationClicked() {
    signal_cfg_.enable_stabilization = !signal_cfg_.enable_stabilization;
    if (signal_stabilization_btn_) {
        signal_stabilization_btn_->setText(signal_cfg_.enable_stabilization
            ? " Stabilize: On "
            : " Stabilize: Off ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleColorModeClicked() {
    signal_cfg_.color_mode = (signal_cfg_.color_mode == PE::PhysicalSignalColorMode::Bgr)
        ? PE::PhysicalSignalColorMode::Gray
        : PE::PhysicalSignalColorMode::Bgr;
    if (signal_color_mode_btn_) {
        signal_color_mode_btn_->setText(signal_cfg_.color_mode == PE::PhysicalSignalColorMode::Gray
            ? " Color: Gray "
            : " Color: BGR ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleResizeModeClicked() {
    signal_cfg_.resize_mode =
        (signal_cfg_.resize_mode == PE::PhysicalSignalResizeMode::Letterbox)
            ? PE::PhysicalSignalResizeMode::Stretch
            : PE::PhysicalSignalResizeMode::Letterbox;
    if (signal_resize_mode_btn_) {
        signal_resize_mode_btn_->setText(
            signal_cfg_.resize_mode == PE::PhysicalSignalResizeMode::Letterbox
                ? " Fit: Letterbox "
                : " Fit: Stretch ");
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleQualityGateClicked() {
    signal_cfg_.quality_gate.enable = !signal_cfg_.quality_gate.enable;
    if (signal_quality_gate_btn_) {
        signal_quality_gate_btn_->setText(signal_cfg_.quality_gate.enable
            ? " QGate: On "
            : " QGate: Off ");
    }
}

void UIPhysicalEnvironmentPanel::HandleApplySignalSettingsClicked() {
    int out_w = 0;
    int out_h = 0;
    int denoise_strength = 0;
    float target_luma = 0.0f;
    float manual_gain = 0.0f;
    float deblur_amount = 0.0f;

    if (!TryParseInt(signal_width_buf_, out_w)
        || !TryParseInt(signal_height_buf_, out_h)
        || !TryParseInt(signal_denoise_strength_buf_, denoise_strength)
        || !TryParseFloat(signal_target_luma_buf_, target_luma)
        || !TryParseFloat(signal_manual_gain_buf_, manual_gain)
        || !TryParseFloat(signal_deblur_amount_buf_, deblur_amount)) {
        LOG_ERROR(kPanelLogTag,
                  "HandleApplySignalSettingsClicked: failed to parse one or more signal config values");
        return;
    }

    signal_cfg_.output_width = out_w;
    signal_cfg_.output_height = out_h;
    signal_cfg_.denoise_strength = denoise_strength;
    signal_cfg_.target_luma = static_cast<double>(target_luma);
    signal_cfg_.manual_exposure_gain = static_cast<double>(manual_gain);
    signal_cfg_.deblur_amount = static_cast<double>(deblur_amount);

    try {
        PE::RequestConfigurePhysicalSignalConditioning(signal_cfg_);
        SyncSignalSettingsControlsFromSubsystem();
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("HandleApplySignalSettingsClicked: configure threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::HandleResetSignalSettingsClicked() {
    try {
        PE::RequestResetPhysicalSignalConditioningDefaults();
        SyncSignalSettingsControlsFromSubsystem();
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("HandleResetSignalSettingsClicked: reset threw: ") + e.what());
    }
}

// ============================================================================
//  Calibration-tab handlers
// ============================================================================

void UIPhysicalEnvironmentPanel::HandleStartCaptureClicked() {
    try { PE::RequestStartPhysicalCameraCalibrationCapture(); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleStartCaptureClicked threw: ") + e.what());
    }
}
void UIPhysicalEnvironmentPanel::HandleStopCaptureClicked() {
    try { PE::RequestStopPhysicalCameraCalibrationCapture(); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleStopCaptureClicked threw: ") + e.what());
    }
}
void UIPhysicalEnvironmentPanel::HandleCaptureNowClicked() {
    try { PE::RequestCapturePhysicalCalibrationSampleNow(); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleCaptureNowClicked threw: ") + e.what());
    }
}
void UIPhysicalEnvironmentPanel::HandleClearSamplesClicked() {
    try { PE::RequestClearPhysicalCalibrationSamples(); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleClearSamplesClicked threw: ") + e.what());
    }
}
void UIPhysicalEnvironmentPanel::HandleRunCalibrationClicked() {
    try { PE::RequestRunIntrinsicCalibrationFromSamples(); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleRunCalibrationClicked threw: ") + e.what());
    }
}
void UIPhysicalEnvironmentPanel::HandleSaveCalibrationClicked() {
    try { PE::RequestSavePhysicalCalibrationToDisk(); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleSaveCalibrationClicked threw: ") + e.what());
    }
}
void UIPhysicalEnvironmentPanel::HandleReloadCalibrationClicked() {
    try { (void)PE::RequestLoadPhysicalCalibrationFromDisk(); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleReloadCalibrationClicked threw: ") + e.what());
    }
}
void UIPhysicalEnvironmentPanel::HandleToggleUndistortClicked() {
    cal_show_undistorted_ = !cal_show_undistorted_;
    if (cal_undistort_toggle_btn_) {
        cal_undistort_toggle_btn_->setText(cal_show_undistorted_ ? " View: Undistorted "
                                                                 : " View: Raw ");
    }
}
void UIPhysicalEnvironmentPanel::HandleApplyPatternClicked() {
    int   cols = 0, rows = 0;
    float square_m = 0.0f;
    if (!TryParseInt(cal_pattern_cols_buf_, cols)
        || !TryParseInt(cal_pattern_rows_buf_, rows)
        || !TryParseFloat(cal_square_meters_buf_, square_m)) {
        LOG_ERROR(kPanelLogTag,
            std::string("HandleApplyPatternClicked: invalid input — cols='")
            + cal_pattern_cols_buf_ + "', rows='" + cal_pattern_rows_buf_
            + "', square='" + cal_square_meters_buf_ + "'");
        return;
    }
    try { PE::RequestReconfigurePhysicalCalibrationPattern(cols, rows, square_m); }
    catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag, std::string("HandleApplyPatternClicked threw: ") + e.what());
    }
}

// ============================================================================
//  Update — tab dispatch
// ============================================================================

void UIPhysicalEnvironmentPanel::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    if (!isVisible()) return;

    // One bus, one cached frame — both tabs share it.
    PE::PhysicalFrameBus::FrameView v;
    if (PE::PhysicalFrameBus::Instance().PullLatestFrameView(v, last_seen_counter_)) {
        last_view_      = std::move(v);
        have_any_frame_ = true;
    }
    signal_status_ = PE::GetPhysicalSignalConditioningStatusSnapshot();

    // ── Tab bar ──
    const float tab_y = position.y + titleBarHeight + 4.0f;
    if (tab_camera_btn_) {
        tab_camera_btn_->setSize(78.0f, kTabBarHeight - 4.0f);
        tab_camera_btn_->setPosition(position.x + kTabBarPad, tab_y);
        tab_camera_btn_->update(input, dt);
    }
    if (tab_stereo_btn_) {
        tab_stereo_btn_->setSize(78.0f, kTabBarHeight - 4.0f);
        tab_stereo_btn_->setPosition(position.x + kTabBarPad + 84.0f, tab_y);
        tab_stereo_btn_->update(input, dt);
    }
    if (tab_calibration_btn_) {
        tab_calibration_btn_->setSize(98.0f, kTabBarHeight - 4.0f);
        tab_calibration_btn_->setPosition(position.x + kTabBarPad + 168.0f, tab_y);
        tab_calibration_btn_->update(input, dt);
    }
    if (tab_perception_btn_) {
        tab_perception_btn_->setSize(96.0f, kTabBarHeight - 4.0f);
        tab_perception_btn_->setPosition(position.x + kTabBarPad + 272.0f, tab_y);
        tab_perception_btn_->update(input, dt);
    }
    if (tab_interaction_btn_) {
        tab_interaction_btn_->setSize(98.0f, kTabBarHeight - 4.0f);
        tab_interaction_btn_->setPosition(position.x + kTabBarPad + 374.0f, tab_y);
        tab_interaction_btn_->update(input, dt);
    }
    if (tab_spatial_btn_) {
        tab_spatial_btn_->setSize(76.0f, kTabBarHeight - 4.0f);
        tab_spatial_btn_->setPosition(position.x + kTabBarPad + 478.0f, tab_y);
        tab_spatial_btn_->update(input, dt);
    }
    if (tab_localization_btn_) {
        tab_localization_btn_->setSize(108.0f, kTabBarHeight - 4.0f);
        tab_localization_btn_->setPosition(position.x + kTabBarPad + 560.0f, tab_y);
        tab_localization_btn_->update(input, dt);
    }
    if (tab_world_btn_) {
        tab_world_btn_->setSize(72.0f, kTabBarHeight - 4.0f);
        tab_world_btn_->setPosition(position.x + kTabBarPad + 674.0f, tab_y);
        tab_world_btn_->update(input, dt);
    }

    // ── Tab content ──
    switch (active_tab_) {
        case Tab::Camera:       UpdateCameraTab(input, dt);       break;
        case Tab::Stereo:       UpdateStereoTab(input, dt);       break;
        case Tab::Calibration:  UpdateCalibrationTab(input, dt);  break;
        case Tab::Perception:   UpdatePerceptionTab(input, dt);   break;
        case Tab::Interaction:  UpdateInteractionTab(input, dt);  break;
        case Tab::Spatial:      UpdateSpatialTab(input, dt);      break;
        case Tab::Localization: UpdateLocalizationTab(input, dt); break;
        case Tab::World:        UpdateWorldTab(input, dt);        break;
    }
}

void UIPhysicalEnvironmentPanel::UpdateCameraTab(const InputState& input, float dt) {
    const float content_top = position.y + titleBarHeight + kTabBarHeight + 8.0f;

    if (refresh_button_) {
        refresh_button_->setSize(90, 26);
        refresh_button_->setPosition(position.x + 16, content_top);
        refresh_button_->update(input, dt);
    }
    if (source_dropdown_) {
        source_dropdown_->setPosition(position.x + 116, content_top);
        source_dropdown_->setSize(size.x - 132, 26);
        source_dropdown_->update(input, dt);
    }
    if (url_inputbox_) {
        url_inputbox_->setPosition(position.x + 16, content_top + 38);
        url_inputbox_->setSize(size.x - 240, 26);
        url_inputbox_->update(input, dt);
    }
    if (connect_button_) {
        connect_button_->setSize(100, 26);
        connect_button_->setPosition(position.x + size.x - 220, content_top + 38);
        connect_button_->update(input, dt);
    }
    if (disconnect_button_) {
        disconnect_button_->setSize(110, 26);
        disconnect_button_->setPosition(position.x + size.x - 116, content_top + 38);
        disconnect_button_->update(input, dt);
    }

    const float settings_y1 = content_top + 72.0f;
    const float settings_y2 = content_top + 104.0f;
    const float settings_y3 = content_top + 136.0f;
    const float settings_y4 = content_top + 168.0f;

    auto place_btn = [&](std::shared_ptr<UIButton>& b, float x, float y, float w) {
        if (!b) return;
        b->setPosition(x, y);
        b->setSize(w, 24.0f);
        b->update(input, dt);
    };
    auto place_box = [&](std::shared_ptr<UIInputBox>& box, float x, float y, float w) {
        if (!box) return;
        box->setPosition(x, y);
        box->setSize(w, 24.0f);
        box->update(input, dt);
    };

    const float x0 = position.x + 16.0f;
    place_btn(signal_view_toggle_btn_, x0, settings_y1, 170.0f);
    place_btn(signal_auto_exposure_btn_, x0 + 178.0f, settings_y1, 150.0f);
    place_btn(signal_denoise_btn_, x0 + 336.0f, settings_y1, 130.0f);
    place_btn(signal_resize_btn_, x0 + 474.0f, settings_y1, 120.0f);
    place_btn(signal_deblur_btn_, x0 + 602.0f, settings_y1, 120.0f);
    place_btn(signal_stabilization_btn_, x0 + 730.0f, settings_y1, 130.0f);

    place_btn(signal_color_mode_btn_, x0, settings_y2, 120.0f);
    place_box(signal_width_box_, x0 + 128.0f, settings_y2, 72.0f);
    place_box(signal_height_box_, x0 + 206.0f, settings_y2, 72.0f);
    place_box(signal_target_luma_box_, x0 + 284.0f, settings_y2, 100.0f);
    place_box(signal_manual_gain_box_, x0 + 390.0f, settings_y2, 94.0f);
    place_box(signal_denoise_strength_box_, x0 + 490.0f, settings_y2, 90.0f);
    place_box(signal_deblur_amount_box_, x0 + 586.0f, settings_y2, 90.0f);
    place_btn(signal_apply_btn_, x0 + 682.0f, settings_y2, 178.0f);

    place_btn(signal_reset_btn_, x0, settings_y3, 150.0f);
    place_btn(signal_resize_mode_btn_, x0 + 158.0f, settings_y3, 140.0f);
    place_btn(signal_quality_gate_btn_, x0 + 306.0f, settings_y3, 120.0f);

    (void)settings_y4;
}

void UIPhysicalEnvironmentPanel::UpdateStereoTab(const InputState& input, float dt) {
    stereo_capture_status_ = PE::GetPhysicalStereoCaptureStatusSnapshot();
    try {
        if (PE::PhysicalStereoFrameBus::Instance().PullLatestPhysicalStereoFrameView(
                stereo_frame_view_, stereo_last_seen_pair_counter_)) {
            have_any_stereo_pair_ = true;
        }
    } catch (const std::exception& e) {
        stereo_ui_status_ = std::string("Stereo frame pull failed: ") + e.what();
        LOG_ERROR(kPanelLogTag, stereo_ui_status_);
    }

    const float content_top = position.y + titleBarHeight + kTabBarHeight + 8.0f;
    if (stereo_left_dropdown_) {
        stereo_left_dropdown_->setPosition(position.x + 16.0f, content_top);
        stereo_left_dropdown_->setSize(size.x - 32.0f, 30.0f);
        stereo_left_dropdown_->update(input, dt);
    }
    if (stereo_right_dropdown_) {
        stereo_right_dropdown_->setPosition(position.x + 16.0f, content_top + 38.0f);
        stereo_right_dropdown_->setSize(size.x - 32.0f, 30.0f);
        stereo_right_dropdown_->update(input, dt);
    }
    if (stereo_skew_box_) {
        stereo_skew_box_->setPosition(position.x + 136.0f, content_top + 80.0f);
        stereo_skew_box_->setSize(100.0f, 26.0f);
        stereo_skew_box_->update(input, dt);
    }
    if (stereo_connect_button_) {
        stereo_connect_button_->setPosition(position.x + 250.0f, content_top + 80.0f);
        stereo_connect_button_->setSize(130.0f, 26.0f);
        stereo_connect_button_->update(input, dt);
    }
    if (stereo_disconnect_button_) {
        stereo_disconnect_button_->setPosition(position.x + 388.0f, content_top + 80.0f);
        stereo_disconnect_button_->setSize(150.0f, 26.0f);
        stereo_disconnect_button_->update(input, dt);
    }
}

void UIPhysicalEnvironmentPanel::UpdateCalibrationTab(const InputState& input, float dt) {
    cal_last_status_ = PE::GetPhysicalCalibrationStatusSnapshot();

    // ── Production preview pipeline ──
    // Heavy work (undistort + chessboard detect + corner overlay) runs here
    // ONCE per new source frame, not on every UI redraw. Re-detection is
    // further throttled to ~5 Hz so a 30-Hz source does not pay
    // findChessboardCornersSB on every incoming frame either.
    constexpr double kOverlayMinIntervalSec = 0.20;
    calib_overlay_seconds_since_ += static_cast<double>(dt);

    const bool counter_changed   = (last_seen_counter_ != calib_display_source_id_);
    const bool undistort_changed = (cal_show_undistorted_ != calib_display_undistort_);
    if (have_any_frame_ && !last_view_.raw_image.empty()
        && (counter_changed || undistort_changed)) {
        cv::Mat base;
        if (cal_show_undistorted_ && PE::IsPhysicalCalibrationDataAvailable()) {
            try {
                PE::UndistortBgrFrameUsingPhysicalCalibration(last_view_.raw_image, base);
            } catch (const std::exception& e) {
                LOG_ERROR(kPanelLogTag,
                    std::string("UpdateCalibrationTab: undistort threw: ") + e.what());
                base = last_view_.raw_image;
            }
        } else {
            base = last_view_.raw_image;
        }
        if (base.empty()) base = last_view_.raw_image;

        // Force a fresh detect when the geometry of the displayed frame just
        // changed (undistort toggled), otherwise the previously cached corners
        // are in the wrong coordinate system.
        if (undistort_changed) {
            calib_overlay_seconds_since_ = kOverlayMinIntervalSec;
            calib_overlay_pattern_valid_ = false;
        }

        if (cal_last_status_.last_pattern_found
            && calib_overlay_seconds_since_ >= kOverlayMinIntervalSec) {
            try {
                PE::DetectedCalibrationPattern p;
                PE::DetectChessboardCornersInBgrFrame(
                    base,
                    cal_last_status_.pattern_inner_cols,
                    cal_last_status_.pattern_inner_rows,
                    p);
                calib_overlay_pattern_       = p;
                calib_overlay_pattern_valid_ = p.found;
            } catch (const std::exception& e) {
                LOG_ERROR(kPanelLogTag,
                    std::string("UpdateCalibrationTab: overlay detect threw: ")
                    + e.what());
                calib_overlay_pattern_valid_ = false;
            }
            calib_overlay_seconds_since_ = 0.0;
        }

        if (calib_overlay_pattern_valid_) {
            try {
                cv::Mat with_overlay;
                PE::DrawDetectedCalibrationPatternOnBgr(
                    base, calib_overlay_pattern_,
                    cal_last_status_.pattern_inner_cols,
                    cal_last_status_.pattern_inner_rows,
                    with_overlay);
                calib_display_frame_ = std::move(with_overlay);
            } catch (const std::exception& e) {
                LOG_ERROR(kPanelLogTag,
                    std::string("UpdateCalibrationTab: overlay draw threw: ")
                    + e.what());
                calib_display_frame_ = base;
            }
        } else {
            calib_display_frame_ = base;
        }

        calib_display_source_id_  = last_seen_counter_;
        calib_display_undistort_  = cal_show_undistorted_;
    }

    const float pad = 12.0f;
    const float bx  = position.x + pad;
    const float by  = position.y + titleBarHeight + kTabBarHeight + 8.0f;
    const float bw  = 110.0f;
    const float bh  = 24.0f;

    auto place = [&](std::shared_ptr<UIButton>& b, int slot, float w = 110.0f) {
        if (!b) return;
        b->setSize(w, bh);
        b->setPosition(bx + slot * (bw + 6.0f), by);
        b->update(input, dt);
    };
    place(cal_start_btn_,       0);
    place(cal_stop_btn_,        1);
    place(cal_capture_now_btn_, 2);
    place(cal_clear_btn_,       3);
    place(cal_run_btn_,         4);
    place(cal_save_btn_,        5);
    place(cal_reload_btn_,      6);

    const float by2 = by + bh + 8.0f;
    if (cal_pattern_cols_box_) {
        cal_pattern_cols_box_->setPosition(bx, by2);
        cal_pattern_cols_box_->setSize(80, bh);
        cal_pattern_cols_box_->update(input, dt);
    }
    if (cal_pattern_rows_box_) {
        cal_pattern_rows_box_->setPosition(bx + 90, by2);
        cal_pattern_rows_box_->setSize(80, bh);
        cal_pattern_rows_box_->update(input, dt);
    }
    if (cal_square_meters_box_) {
        cal_square_meters_box_->setPosition(bx + 180, by2);
        cal_square_meters_box_->setSize(100, bh);
        cal_square_meters_box_->update(input, dt);
    }
    if (cal_apply_pattern_btn_) {
        cal_apply_pattern_btn_->setSize(120, bh);
        cal_apply_pattern_btn_->setPosition(bx + 290, by2);
        cal_apply_pattern_btn_->update(input, dt);
    }
    if (cal_undistort_toggle_btn_) {
        cal_undistort_toggle_btn_->setSize(160, bh);
        cal_undistort_toggle_btn_->setPosition(position.x + size.x - pad - 160, by2);
        cal_undistort_toggle_btn_->update(input, dt);
    }
}

// ============================================================================
//  Draw — tab dispatch
// ============================================================================

bool UIPhysicalEnvironmentPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    // ── Tab buttons ──
    if (tab_camera_btn_)       tab_camera_btn_->drawOverlay(renderer, position);
    if (tab_stereo_btn_)       tab_stereo_btn_->drawOverlay(renderer, position);
    if (tab_calibration_btn_)  tab_calibration_btn_->drawOverlay(renderer, position);
    if (tab_perception_btn_)   tab_perception_btn_->drawOverlay(renderer, position);
    if (tab_interaction_btn_)  tab_interaction_btn_->drawOverlay(renderer, position);
    if (tab_spatial_btn_)      tab_spatial_btn_->drawOverlay(renderer, position);
    if (tab_localization_btn_) tab_localization_btn_->drawOverlay(renderer, position);
    if (tab_world_btn_)        tab_world_btn_->drawOverlay(renderer, position);

    // Active-tab underline indicator (matches DataHub/Training pattern).
    {
        const float y = position.y + titleBarHeight + kTabBarHeight - 1.0f;
        float ix = 0.0f, iw = 0.0f;
        switch (active_tab_) {
            case Tab::Camera:
                ix = position.x + kTabBarPad;          iw = 78.0f; break;
            case Tab::Stereo:
                ix = position.x + kTabBarPad + 84.0f;  iw = 78.0f; break;
            case Tab::Calibration:
                ix = position.x + kTabBarPad + 168.0f; iw = 98.0f; break;
            case Tab::Perception:
                ix = position.x + kTabBarPad + 272.0f; iw = 96.0f; break;
            case Tab::Interaction:
                ix = position.x + kTabBarPad + 374.0f; iw = 98.0f; break;
            case Tab::Spatial:
                ix = position.x + kTabBarPad + 478.0f; iw = 76.0f; break;
            case Tab::Localization:
                ix = position.x + kTabBarPad + 560.0f; iw = 108.0f; break;
            case Tab::World:
                ix = position.x + kTabBarPad + 674.0f; iw = 72.0f; break;
        }
        renderer.drawRect({ix, y}, {iw, 2.0f}, UITheme::Colors::Primary);
    }

    // Divider under tab bar.
    renderer.drawRect({position.x + 4.0f,
                       position.y + titleBarHeight + kTabBarHeight + 1.0f},
                      {size.x - 8.0f, 1.0f},
                      UITheme::Colors::DividerLine);

    switch (active_tab_) {
        case Tab::Camera:       DrawCameraTab(renderer);       break;
        case Tab::Stereo:       DrawStereoTab(renderer);       break;
        case Tab::Calibration:  DrawCalibrationTab(renderer);  break;
        case Tab::Perception:   DrawPerceptionTab(renderer);   break;
        case Tab::Interaction:  DrawInteractionTab(renderer);  break;
        case Tab::Spatial:      DrawSpatialTab(renderer);      break;
        case Tab::Localization: DrawLocalizationTab(renderer); break;
        case Tab::World:        DrawWorldTab(renderer);        break;
    }

    renderer.popClipRect();
    return true;
}

void UIPhysicalEnvironmentPanel::DrawCameraTab(OverlayRenderer& renderer) {
    const float pad        = 16.0f;
    const float top_block  = 210.0f; // controls + signal settings rows
    const float status_h   = 110.0f;
    const float content_top = position.y + titleBarHeight + kTabBarHeight;

    const float frame_x = position.x + pad;
    const float frame_y = content_top + top_block;
    const float frame_w = size.x - 2 * pad;
    const float frame_h = size.y - (frame_y - position.y) - status_h - pad;

    renderer.drawRect({frame_x - 2, frame_y - 2}, {frame_w + 4, frame_h + 4},
                      UITheme::Colors::DividerLine);
    renderer.drawRect({frame_x, frame_y}, {frame_w, frame_h},
                      UITheme::Colors::Background);

    const cv::Mat& shown = camera_show_model_signal_ ? last_view_.model_image : last_view_.raw_image;
    if (have_any_frame_ && !shown.empty()) {
        DrawBgrFrameIntoOverlay(renderer, shown,
                                last_seen_counter_, camera_show_model_signal_,
                                frame_x, frame_y, frame_w, frame_h, camera_blit_cache_);
    } else {
        renderer.drawText({frame_x + 12, frame_y + 12},
                          "No frame yet — choose a source and click Connect.",
                          UITheme::Colors::TextSecondary);
    }

    // Status block under frame
    const float stat_y = frame_y + frame_h + 6.0f;
    {
        std::ostringstream ss;
        ss << "Active: " << (PE::IsPhysicalCameraStreamActive() ? "STREAMING"
                           : PE::IsPhysicalCameraStreamFailed() ? "FAILED"
                           : "idle")
           << "  |  url=" << PE::GetActivePhysicalCameraUrl()
           << "  |  label=" << PE::GetActivePhysicalCameraLabel();
        renderer.drawText({frame_x, stat_y}, ss.str(), UITheme::Colors::TextPrimary);
    }
    {
        std::ostringstream ss2;
        ss2 << "Frames=" << PE::GetActiveStreamFrameCounter()
            << "  fps=" << std::fixed << std::setprecision(1) << PE::GetActiveStreamFps();
        if (have_any_frame_) {
            ss2 << "  shown=" << shown.cols << "x" << shown.rows
                << "  view=" << (camera_show_model_signal_ ? "model" : "raw");
        }
        renderer.drawText({frame_x, stat_y + 18}, ss2.str(), UITheme::Colors::TextSecondary);
    }
    {
        std::ostringstream ss3;
        ss3 << "Signal: " << signal_status_.last_pipeline_summary
            << "| in=" << signal_status_.last_input_width << "x" << signal_status_.last_input_height
            << " out=" << signal_status_.last_output_width << "x" << signal_status_.last_output_height
            << " luma " << FormatDouble(signal_status_.last_input_luma, 1)
            << "→" << FormatDouble(signal_status_.last_output_luma, 1)
            << " gain=" << FormatDouble(signal_status_.last_applied_exposure_gain, 2);
        renderer.drawText({frame_x, stat_y + 36}, ss3.str(), UITheme::Colors::TextSecondary);
    }
    {
        std::ostringstream ss4;
        ss4 << "Flow: points=" << signal_status_.last_flow_tracked_points
            << " dx=" << FormatDouble(signal_status_.last_flow_dx, 2)
            << " dy=" << FormatDouble(signal_status_.last_flow_dy, 2)
            << " autoExp=" << (signal_status_.using_auto_exposure ? "yes" : "no");
        renderer.drawText({frame_x, stat_y + 54}, ss4.str(), UITheme::Colors::TextSecondary);
    }
    {
        // Quality gate diagnostics — gives the operator a hard reason any time
        // a frame was withheld from the model.
        std::ostringstream ss5;
        ss5 << "QGate: " << (signal_status_.last_quality_gate_passed ? "PASS" : "DROP")
            << " lapVar=" << FormatDouble(signal_status_.last_laplacian_variance, 1)
            << " clip=" << FormatDouble(signal_status_.last_clipped_pixel_ratio * 100.0, 1) << "%"
            << " dropped=" << signal_status_.total_frames_dropped_by_quality_gate;
        if (!signal_status_.last_quality_gate_passed
            && !signal_status_.last_quality_gate_reason.empty()) {
            ss5 << "  reason=" << signal_status_.last_quality_gate_reason;
        }
        renderer.drawText({frame_x, stat_y + 72}, ss5.str(),
                          signal_status_.last_quality_gate_passed
                              ? UITheme::Colors::TextSecondary
                              : UITheme::Colors::Danger);
    }
    {
        // Geometric provenance: raw->model transform. Critical for any
        // downstream consumer that wants to back-project model coords.
        const auto& t = signal_status_.last_raw_to_model;
        std::ostringstream ss6;
        ss6 << "Geom: scale=("
            << FormatDouble(t.scale_x, 3) << "," << FormatDouble(t.scale_y, 3)
            << ")  offset=("
            << FormatDouble(t.offset_x, 1) << "," << FormatDouble(t.offset_y, 1) << ")";
        renderer.drawText({frame_x, stat_y + 90}, ss6.str(), UITheme::Colors::TextSecondary);
    }
    const std::string err = PE::GetLastEnvironmentError();
    if (!err.empty()) {
        renderer.drawText({frame_x, stat_y + 108}, "Last error: " + err,
                          UITheme::Colors::Danger);
    }
    if (!selection_info_.empty()) {
        renderer.drawText({frame_x, stat_y + 126}, selection_info_,
                          UITheme::Colors::Warning);
    }

    if (refresh_button_)    refresh_button_->drawOverlay(renderer, position);
    if (url_inputbox_)      url_inputbox_->drawOverlay(renderer, position);
    if (connect_button_)    connect_button_->drawOverlay(renderer, position);
    if (disconnect_button_) disconnect_button_->drawOverlay(renderer, position);
    if (signal_view_toggle_btn_) signal_view_toggle_btn_->drawOverlay(renderer, position);
    if (signal_auto_exposure_btn_) signal_auto_exposure_btn_->drawOverlay(renderer, position);
    if (signal_denoise_btn_) signal_denoise_btn_->drawOverlay(renderer, position);
    if (signal_resize_btn_) signal_resize_btn_->drawOverlay(renderer, position);
    if (signal_deblur_btn_) signal_deblur_btn_->drawOverlay(renderer, position);
    if (signal_stabilization_btn_) signal_stabilization_btn_->drawOverlay(renderer, position);
    if (signal_color_mode_btn_) signal_color_mode_btn_->drawOverlay(renderer, position);
    if (signal_resize_mode_btn_) signal_resize_mode_btn_->drawOverlay(renderer, position);
    if (signal_quality_gate_btn_) signal_quality_gate_btn_->drawOverlay(renderer, position);
    if (signal_apply_btn_) signal_apply_btn_->drawOverlay(renderer, position);
    if (signal_reset_btn_) signal_reset_btn_->drawOverlay(renderer, position);
    if (signal_width_box_) signal_width_box_->drawOverlay(renderer, position);
    if (signal_height_box_) signal_height_box_->drawOverlay(renderer, position);
    if (signal_target_luma_box_) signal_target_luma_box_->drawOverlay(renderer, position);
    if (signal_manual_gain_box_) signal_manual_gain_box_->drawOverlay(renderer, position);
    if (signal_denoise_strength_box_) signal_denoise_strength_box_->drawOverlay(renderer, position);
    if (signal_deblur_amount_box_) signal_deblur_amount_box_->drawOverlay(renderer, position);
    if (source_dropdown_) {
        source_dropdown_->drawOverlay(renderer, position);
        if (source_dropdown_->isExpanded())
            source_dropdown_->drawExpandedList(renderer, position);
    }
}

void UIPhysicalEnvironmentPanel::DrawStereoTab(OverlayRenderer& renderer) {
    const float pad = 16.0f;
    const float gap = 10.0f;
    const float content_top = position.y + titleBarHeight + kTabBarHeight;
    const float frame_y = content_top + 126.0f;
    const float frame_w = (size.x - pad * 2.0f - gap) * 0.5f;
    const float frame_h = std::max(
        80.0f, size.y - (frame_y - position.y) - 104.0f);
    const float left_x  = position.x + pad;
    const float right_x = left_x + frame_w + gap;

    renderer.drawText({position.x + 16.0f, content_top + 88.0f},
                      "Max skew (ms)", UITheme::Colors::TextSecondary);
    renderer.drawText({left_x, frame_y - 18.0f}, "LEFT / PRIMARY",
                      UITheme::Colors::TextPrimary);
    renderer.drawText({right_x, frame_y - 18.0f}, "RIGHT",
                      UITheme::Colors::TextPrimary);

    renderer.drawRect({left_x - 1.0f, frame_y - 1.0f},
                      {frame_w + 2.0f, frame_h + 2.0f}, UITheme::Colors::DividerLine);
    renderer.drawRect({right_x - 1.0f, frame_y - 1.0f},
                      {frame_w + 2.0f, frame_h + 2.0f}, UITheme::Colors::DividerLine);
    renderer.drawRect({left_x, frame_y}, {frame_w, frame_h}, UITheme::Colors::Background);
    renderer.drawRect({right_x, frame_y}, {frame_w, frame_h}, UITheme::Colors::Background);

    if (have_any_stereo_pair_
        && !stereo_frame_view_.left_image.empty()
        && !stereo_frame_view_.right_image.empty()) {
        DrawBgrFrameIntoOverlay(renderer, stereo_frame_view_.left_image,
                                stereo_frame_view_.pair_counter, false,
                                left_x, frame_y, frame_w, frame_h,
                                stereo_left_blit_cache_);
        DrawBgrFrameIntoOverlay(renderer, stereo_frame_view_.right_image,
                                stereo_frame_view_.pair_counter, false,
                                right_x, frame_y, frame_w, frame_h,
                                stereo_right_blit_cache_);
    } else {
        renderer.drawText({left_x + 10.0f, frame_y + 10.0f},
                          "No synchronized pair published.",
                          UITheme::Colors::TextSecondary);
        renderer.drawText({right_x + 10.0f, frame_y + 10.0f},
                          "No synchronized pair published.",
                          UITheme::Colors::TextSecondary);
    }

    const float status_y = frame_y + frame_h + 8.0f;
    const bool streaming = PE::IsPhysicalStereoCaptureActive();
    renderer.drawText(
        {left_x, status_y},
        std::string("Stereo: ") + (streaming ? "STREAMING" : "not streaming")
        + "  |  left=" + FormatCameraStreamState(stereo_capture_status_.left_state)
        + "  right=" + FormatCameraStreamState(stereo_capture_status_.right_state),
        streaming ? UITheme::Colors::TextPrimary : UITheme::Colors::Warning);
    {
        std::ostringstream ss;
        ss << "Pairs=" << stereo_capture_status_.synchronized_pair_count
           << "  signed_skew=" << FormatDouble(stereo_capture_status_.last_signed_skew_ms, 3) << "ms"
           << "  limit=" << FormatDouble(stereo_capture_status_.maximum_pair_skew_ms, 1) << "ms";
        renderer.drawText({left_x, status_y + 18.0f}, ss.str(), UITheme::Colors::TextSecondary);
    }
    {
        std::ostringstream ss;
        ss << "Left: frame=" << stereo_capture_status_.left_frame_counter
           << " fps=" << FormatDouble(stereo_capture_status_.left_fps, 1)
           << " rejected=" << stereo_capture_status_.rejected_left_count
           << "  |  Right: frame=" << stereo_capture_status_.right_frame_counter
           << " fps=" << FormatDouble(stereo_capture_status_.right_fps, 1)
           << " rejected=" << stereo_capture_status_.rejected_right_count;
        renderer.drawText({left_x, status_y + 36.0f}, ss.str(), UITheme::Colors::TextSecondary);
    }
    if (!stereo_capture_status_.last_error_reason.empty()) {
        renderer.drawText({left_x, status_y + 54.0f},
                          "Error: " + stereo_capture_status_.last_error_reason,
                          UITheme::Colors::Danger);
    } else if (!stereo_ui_status_.empty()) {
        renderer.drawText({left_x, status_y + 54.0f}, stereo_ui_status_,
                          UITheme::Colors::TextSecondary);
    }

    if (stereo_skew_box_) stereo_skew_box_->drawOverlay(renderer, position);
    if (stereo_connect_button_) stereo_connect_button_->drawOverlay(renderer, position);
    if (stereo_disconnect_button_) stereo_disconnect_button_->drawOverlay(renderer, position);
    if (stereo_left_dropdown_) stereo_left_dropdown_->drawOverlay(renderer, position);
    if (stereo_right_dropdown_) stereo_right_dropdown_->drawOverlay(renderer, position);
    if (stereo_left_dropdown_ && stereo_left_dropdown_->isExpanded()) {
        stereo_left_dropdown_->drawExpandedList(renderer, position);
    }
    if (stereo_right_dropdown_ && stereo_right_dropdown_->isExpanded()) {
        stereo_right_dropdown_->drawExpandedList(renderer, position);
    }
}

void UIPhysicalEnvironmentPanel::DrawCalibrationTab(OverlayRenderer& renderer) {
    const float pad        = 12.0f;
    const float top_block  = 70.0f;   // two rows of controls
    const float right_pane = 280.0f;
    const float bottom_h   = 60.0f;
    const float content_top = position.y + titleBarHeight + kTabBarHeight;

    const float frame_x = position.x + pad;
    const float frame_y = content_top + top_block;
    const float frame_w = size.x - 2 * pad - right_pane - pad;
    const float frame_h = size.y - (frame_y - position.y) - bottom_h - pad;

    renderer.drawRect({frame_x - 2, frame_y - 2}, {frame_w + 4, frame_h + 4},
                      UITheme::Colors::DividerLine);
    renderer.drawRect({frame_x, frame_y}, {frame_w, frame_h},
                      UITheme::Colors::Background);

    // Display path is purely cached — heavy compose work happened in update().
    if (!have_any_frame_ || calib_display_frame_.empty()) {
        renderer.drawText({frame_x + 12, frame_y + 12},
                          "No frame yet — switch to the Camera tab and Connect a source.",
                          UITheme::Colors::TextSecondary);
    } else {
        DrawBgrFrameIntoOverlay(renderer, calib_display_frame_,
                                calib_display_source_id_,
                                calib_display_undistort_,
                                frame_x, frame_y, frame_w, frame_h,
                                calib_blit_cache_);
    }

    // Status under frame
    const float stat_y = frame_y + frame_h + 6.0f;
    {
        std::ostringstream ss;
        ss << "Stage: " << FormatStage(cal_last_status_.stage)
           << "  |  samples: " << cal_last_status_.accepted_sample_count
           << "  |  coverage: "  << cal_last_status_.coverage_cells_filled
                                 << "/" << (cal_last_status_.coverage_grid_cols
                                            * cal_last_status_.coverage_grid_rows)
           << "  |  pattern: " << cal_last_status_.pattern_inner_cols << "x"
                               << cal_last_status_.pattern_inner_rows
           << " @ " << FormatDouble(cal_last_status_.pattern_square_meters * 1000.0, 1) << "mm";
        renderer.drawText({frame_x, stat_y}, ss.str(), UITheme::Colors::TextPrimary);
    }
    {
        std::ostringstream ss;
        ss << "Frame: ";
        if (!cal_last_status_.last_frame_present) {
            ss << "(no frame on bus — connect a source on the Camera tab)";
        } else {
            ss << cal_last_status_.last_frame_width << "x" << cal_last_status_.last_frame_height
               << "  |  camera=" << cal_last_status_.active_source_label
               << "  |  detector=" << (cal_last_status_.last_pattern_found
                                          ? cal_last_status_.last_detector_used
                                          : std::string("(none)"))
               << "  |  preprocess_path=" << cal_last_status_.last_preprocess_path;
        }
        renderer.drawText({frame_x, stat_y + 18}, ss.str(), UITheme::Colors::TextSecondary);
    }
    if (!cal_last_status_.status_reason.empty()) {
        renderer.drawText({frame_x, stat_y + 36},
                          "Status: " + cal_last_status_.status_reason,
                          UITheme::Colors::Warning);
    }

    DrawBrightnessBar(renderer, frame_x, frame_y - 14.0f, frame_w, 8.0f,
                      cal_last_status_.last_frame_brightness);

    const float right_x = frame_x + frame_w + pad;
    const float right_y = frame_y;
    const float cov_h   = 160.0f;
    DrawCoverageGrid(renderer, right_x, right_y, right_pane, cov_h, cal_last_status_);
    DrawCalibrationDataReadout(renderer, right_x, right_y + cov_h + 12.0f, cal_last_status_);

    if (cal_start_btn_)            cal_start_btn_->drawOverlay(renderer, position);
    if (cal_stop_btn_)             cal_stop_btn_->drawOverlay(renderer, position);
    if (cal_capture_now_btn_)      cal_capture_now_btn_->drawOverlay(renderer, position);
    if (cal_clear_btn_)            cal_clear_btn_->drawOverlay(renderer, position);
    if (cal_run_btn_)              cal_run_btn_->drawOverlay(renderer, position);
    if (cal_save_btn_)             cal_save_btn_->drawOverlay(renderer, position);
    if (cal_reload_btn_)           cal_reload_btn_->drawOverlay(renderer, position);
    if (cal_undistort_toggle_btn_) cal_undistort_toggle_btn_->drawOverlay(renderer, position);
    if (cal_apply_pattern_btn_)    cal_apply_pattern_btn_->drawOverlay(renderer, position);

    if (cal_pattern_cols_box_)  cal_pattern_cols_box_->drawOverlay(renderer, position);
    if (cal_pattern_rows_box_)  cal_pattern_rows_box_->drawOverlay(renderer, position);
    if (cal_square_meters_box_) cal_square_meters_box_->drawOverlay(renderer, position);
}

// ============================================================================
//  Shared frame blit + calibration sub-draws
// ============================================================================

void UIPhysicalEnvironmentPanel::DrawBgrFrameIntoOverlay(
    OverlayRenderer& renderer, const cv::Mat& bgr,
    uint64_t source_id, bool source_undistort,
    float frame_x, float frame_y, float frame_w, float frame_h,
    PreviewBlitCache& cache)
{
    auto* pixels = static_cast<uint32_t*>(renderer.getPixels());
    if (!pixels) {
        renderer.drawText({frame_x + 12, frame_y + 12},
                          "OverlayRenderer pixel buffer is NULL — cannot blit frame",
                          UITheme::Colors::Danger);
        return;
    }
    const int dest_buf_w = renderer.getWidth();
    const int dest_buf_h = renderer.getHeight();
    if (dest_buf_w <= 0 || dest_buf_h <= 0) return;

    const int src_w = bgr.cols;
    const int src_h = bgr.rows;
    if (src_w <= 0 || src_h <= 0) return;

    const double scale_x = static_cast<double>(frame_w) / src_w;
    const double scale_y = static_cast<double>(frame_h) / src_h;
    const double scale   = std::min(scale_x, scale_y);
    const int    out_w   = std::max(1, static_cast<int>(src_w * scale));
    const int    out_h   = std::max(1, static_cast<int>(src_h * scale));
    const int    out_x   = static_cast<int>(frame_x + (frame_w - out_w) * 0.5f);
    const int    out_y   = static_cast<int>(frame_y + (frame_h - out_h) * 0.5f);

    // ── Cache hit fast path ──
    // When source frame, undistort flag, AND output geometry all match the
    // last build, skip both cv::resize and the pixel-pack loop. Production
    // UI redraws (typically 60 Hz against a 30 Hz source) take this path on
    // ~half of all frames, and EVERY frame when the panel is idle.
    const bool cache_hit =
        source_id != 0 &&
        cache.source_id        == source_id &&
        cache.source_undistort == source_undistort &&
        cache.out_w            == out_w &&
        cache.out_h            == out_h &&
        cache.argb.size() == static_cast<size_t>(out_w) * static_cast<size_t>(out_h);

    if (!cache_hit) {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(out_w, out_h), 0, 0, cv::INTER_AREA);
        if (resized.empty() || resized.type() != CV_8UC3) return;

        cache.argb.assign(static_cast<size_t>(out_w) * static_cast<size_t>(out_h), 0);
        for (int y = 0; y < out_h; ++y) {
            const auto* src = resized.ptr<uint8_t>(y);
            uint32_t*   dst = cache.argb.data() + static_cast<size_t>(y) * out_w;
            for (int x = 0; x < out_w; ++x) {
                const uint8_t b = src[x * 3 + 0];
                const uint8_t g = src[x * 3 + 1];
                const uint8_t r = src[x * 3 + 2];
                // =========================================================
                // COLOR-ORDER LANDMINE — READ BEFORE CHANGING
                // =========================================================
                // Empirically verified on macOS arm64 + OpenCV 4.12 (vcpkg)
                // AVFoundation backend (CAP=1200): the cv::Mat bytes arrive
                // in an order where the channel at offset 1 is BLUE and the
                // channel at offset 0 pairs with R to form yellow. Using the
                // "documented" BGR assumption produces a G↔B inversion.
                //
                // DO NOT "fix" this to match OverlayRenderer::drawRect's
                // ARGB = (a<<24)|(r<<16)|(g<<8)|b convention. The UI chrome
                // convention is right for synthetic UI colors. This loop
                // consumes real camera pixels that do NOT follow that
                // convention on this platform/backend.
                // =========================================================
                dst[x] = (0xFFu << 24) | (static_cast<uint32_t>(r) << 16)
                                       | (static_cast<uint32_t>(b) << 8)
                                       |  static_cast<uint32_t>(g);
            }
        }
        cache.source_id        = source_id;
        cache.source_undistort = source_undistort;
        cache.out_w            = out_w;
        cache.out_h            = out_h;
    }

    // ── Memcpy clipped rows into renderer pixels ──
    const int clip_x0 = std::max(0, out_x);
    const int clip_y0 = std::max(0, out_y);
    const int clip_x1 = std::min(dest_buf_w, out_x + out_w);
    const int clip_y1 = std::min(dest_buf_h, out_y + out_h);
    if (clip_x0 >= clip_x1 || clip_y0 >= clip_y1) return;

    const size_t row_bytes = static_cast<size_t>(clip_x1 - clip_x0) * sizeof(uint32_t);
    for (int y = clip_y0; y < clip_y1; ++y) {
        const int src_row = y - out_y;
        const uint32_t* src = cache.argb.data()
                              + static_cast<size_t>(src_row) * static_cast<size_t>(out_w)
                              + static_cast<size_t>(clip_x0 - out_x);
        uint32_t*       dst = pixels + static_cast<size_t>(y) * static_cast<size_t>(dest_buf_w)
                              + static_cast<size_t>(clip_x0);
        std::memcpy(dst, src, row_bytes);
    }
}

void UIPhysicalEnvironmentPanel::DrawCoverageGrid(
    OverlayRenderer& renderer, float x, float y, float w, float h,
    const PE::PhysicalCalibrationStatus& st)
{
    renderer.drawText({x, y}, "Sample coverage:", UITheme::Colors::TextPrimary);
    const float gx = x;
    const float gy = y + 18.0f;
    const float gw = w - 8.0f;
    const float gh = h - 28.0f;
    renderer.drawRect({gx - 1, gy - 1}, {gw + 2, gh + 2}, UITheme::Colors::DividerLine);
    renderer.drawRect({gx, gy}, {gw, gh}, UITheme::Colors::Background);

    if (st.coverage_grid_cols <= 0 || st.coverage_grid_rows <= 0) return;
    if (static_cast<int>(st.coverage_cell_counts.size())
        != st.coverage_grid_cols * st.coverage_grid_rows) {
        renderer.drawText({gx + 4, gy + 4},
                          "(coverage size mismatch — calibrator state desynced)",
                          UITheme::Colors::Danger);
        return;
    }

    const float cw = gw / static_cast<float>(st.coverage_grid_cols);
    const float ch = gh / static_cast<float>(st.coverage_grid_rows);
    for (int r = 0; r < st.coverage_grid_rows; ++r) {
        for (int c = 0; c < st.coverage_grid_cols; ++c) {
            const int idx = r * st.coverage_grid_cols + c;
            const int cnt = st.coverage_cell_counts[idx];
            const uint32_t cell_color = (cnt > 0)
                ? MakeArgb(0xFF, 0x33, 0xAA, 0x55)
                : MakeArgb(0x60, 0x88, 0x88, 0x88);
            renderer.drawRect({gx + c * cw + 1, gy + r * ch + 1},
                              {cw - 2, ch - 2}, cell_color);
        }
    }
}

void UIPhysicalEnvironmentPanel::DrawBrightnessBar(
    OverlayRenderer& renderer, float x, float y, float w, float h, double brightness)
{
    renderer.drawRect({x - 1, y - 1}, {w + 2, h + 2}, UITheme::Colors::DividerLine);
    const int bands = std::max(1, static_cast<int>(w));
    for (int i = 0; i < bands; ++i) {
        const uint8_t v = static_cast<uint8_t>(
            std::min(255, std::max(0, static_cast<int>(255.0 * i / bands))));
        renderer.drawRect({x + static_cast<float>(i), y},
                          {1.0f, h},
                          MakeArgb(0xFF, v, v, v));
    }
    const double clamped = std::max(0.0, std::min(255.0, brightness));
    const float  mx      = x + static_cast<float>(clamped / 255.0 * w);
    renderer.drawLine({mx, y - 2}, {mx, y + h + 2},
                      MakeArgb(0xFF, 0xFF, 0x55, 0x55), 2.0f);
}

void UIPhysicalEnvironmentPanel::DrawCalibrationDataReadout(
    OverlayRenderer& renderer, float x, float y, const PE::PhysicalCalibrationStatus& st)
{
    renderer.drawText({x, y}, "Calibration data:", UITheme::Colors::TextPrimary);
    if (!st.has_calibration_data
        || st.camera_matrix.empty() || st.camera_matrix.rows != 3 || st.camera_matrix.cols != 3) {
        renderer.drawText({x, y + 18}, "(none yet — collect samples and press Calibrate)",
                          UITheme::Colors::TextSecondary);
        return;
    }
    const cv::Mat& K = st.camera_matrix;
    const double fx = K.at<double>(0,0);
    const double fy = K.at<double>(1,1);
    const double cx = K.at<double>(0,2);
    const double cy = K.at<double>(1,2);
    {
        std::ostringstream ss;
        ss << "image: " << st.calibrated_image_size.width << "x"
                        << st.calibrated_image_size.height
           << "  rms: " << FormatDouble(st.rms_reprojection_error, 4) << " px";
        renderer.drawText({x, y + 18}, ss.str(), UITheme::Colors::TextSecondary);
    }
    {
        std::ostringstream ss;
        ss << "fx=" << FormatDouble(fx, 2) << "  fy=" << FormatDouble(fy, 2);
        renderer.drawText({x, y + 36}, ss.str(), UITheme::Colors::TextSecondary);
    }
    {
        std::ostringstream ss;
        ss << "cx=" << FormatDouble(cx, 2) << "  cy=" << FormatDouble(cy, 2);
        renderer.drawText({x, y + 54}, ss.str(), UITheme::Colors::TextSecondary);
    }
    {
        std::ostringstream ss;
        ss << "dist[" << st.dist_coeffs.cols * st.dist_coeffs.rows << "]: ";
        const cv::Mat& d = st.dist_coeffs;
        const int n = d.cols * d.rows;
        for (int i = 0; i < n && i < 5; ++i) {
            ss << FormatDouble(d.at<double>(i), 4);
            if (i + 1 < n && i + 1 < 5) ss << ", ";
        }
        if (n > 5) ss << ", ...";
        renderer.drawText({x, y + 72}, ss.str(), UITheme::Colors::TextSecondary);
    }
}

// ============================================================================
//  Perception tab
// ============================================================================

namespace {

uint32_t PercMakeArgb(uint8_t a, uint8_t r, uint8_t g, uint8_t b) {
    return (static_cast<uint32_t>(a) << 24)
         | (static_cast<uint32_t>(r) << 16)
         | (static_cast<uint32_t>(g) << 8)
         |  static_cast<uint32_t>(b);
}

// Deterministic class-id -> ARGB so the same id always gets the same colour.
uint32_t PercColorForClassId(int32_t id, uint8_t alpha = 0xFF) {
    if (id < 0) return PercMakeArgb(alpha, 0x80, 0x80, 0x80);
    const uint32_t k = static_cast<uint32_t>(id) * 2654435761u; // Knuth multiplicative hash
    const uint8_t r = static_cast<uint8_t>(40 + ((k >>  0) & 0xFF) * 200 / 255);
    const uint8_t g = static_cast<uint8_t>(40 + ((k >>  8) & 0xFF) * 200 / 255);
    const uint8_t b = static_cast<uint8_t>(40 + ((k >> 16) & 0xFF) * 200 / 255);
    return PercMakeArgb(alpha, r, g, b);
}

const char* OnOffStr(bool v) { return v ? "on" : "off"; }

} // anonymous

// ── Toggle handlers ─────────────────────────────────────────────────────────

void UIPhysicalEnvironmentPanel::HandleTogglePerceptionObjectDetector() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.object_detector = !f.object_detector;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionSemanticSegmenter() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.semantic_segmenter = !f.semantic_segmenter;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionImageClassifier() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.image_classifier = !f.image_classifier;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionPoseEstimator() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.pose_estimator = !f.pose_estimator;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionSceneTextReader() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.scene_text_reader = !f.scene_text_reader;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionFacialExpressionDetector() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.facial_expression_detector = !f.facial_expression_detector;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionEntityTracker() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.entity_tracker = !f.entity_tracker;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionInstanceSegmenter() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.instance_segmenter = !f.instance_segmenter;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}
void UIPhysicalEnvironmentPanel::HandleTogglePerceptionClassPolicy() {
    auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    f.class_policy = !f.class_policy;
    PE::RequestSetPhysicalPerceptionPrimitivesEnableFlags(f);
    RefreshPerceptionEnableButtonLabelsFromSubsystem();
}

void UIPhysicalEnvironmentPanel::RefreshPerceptionEnableButtonLabelsFromSubsystem() {
    const auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    if (perc_btn_obj_)  perc_btn_obj_->setText(std::string(" Detector: ")   + OnOffStr(f.object_detector)    + " ");
    if (perc_btn_seg_)  perc_btn_seg_->setText(std::string(" Segmenter: ")  + OnOffStr(f.semantic_segmenter) + " ");
    if (perc_btn_cls_)  perc_btn_cls_->setText(std::string(" Classifier: ") + OnOffStr(f.image_classifier)   + " ");
    if (perc_btn_pose_) perc_btn_pose_->setText(std::string(" Pose: ")      + OnOffStr(f.pose_estimator)     + " ");
    if (perc_btn_text_) perc_btn_text_->setText(std::string(" Text: ")      + OnOffStr(f.scene_text_reader)  + " ");
    if (perc_btn_face_) perc_btn_face_->setText(std::string(" Face: ")      + OnOffStr(f.facial_expression_detector) + " ");
    if (perc_btn_track_)perc_btn_track_->setText(std::string(" Tracks: ")    + OnOffStr(f.entity_tracker) + " ");
    if (perc_btn_inst_seg_)perc_btn_inst_seg_->setText(std::string(" InstSeg: ") + OnOffStr(f.instance_segmenter) + " ");
    if (perc_btn_class_policy_)perc_btn_class_policy_->setText(std::string(" Policy: ") + OnOffStr(f.class_policy) + " ");
}

// ── Update ──────────────────────────────────────────────────────────────────

void UIPhysicalEnvironmentPanel::UpdatePerceptionTab(const InputState& input, float dt) {
    // Pull latest perception results (frame is already in last_view_, pulled
    // by UIPhysicalEnvironmentPanel::update() above).
    try {
        if (PE::PhysicalPerceptionPrimitiveBus::Instance()
                .PullLatestPhysicalPerceptionResultsView(perc_results_view_,
                                                         last_perc_results_counter_)) {
            have_any_perc_results_ = true;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("UpdatePerceptionTab: results pull threw: ") + e.what());
    }

    // Toolbar buttons across the top of the content area.
    const float content_top = position.y + titleBarHeight + kTabBarHeight + 8.0f;
    const float bx0 = position.x + 16.0f;
    const float btn_w = 116.0f;
    const float btn_h = 24.0f;
    auto place = [&](std::shared_ptr<UIButton>& b, int slot) {
        if (!b) return;
        b->setSize(btn_w, btn_h);
        b->setPosition(bx0 + slot * (btn_w + 4.0f), content_top);
        b->update(input, dt);
    };
    place(perc_btn_obj_,  0);
    place(perc_btn_seg_,  1);
    place(perc_btn_cls_,  2);
    place(perc_btn_pose_, 3);
    place(perc_btn_text_, 4);
    place(perc_btn_face_, 5);
    place(perc_btn_track_,6);
    place(perc_btn_inst_seg_,7);
    place(perc_btn_class_policy_,8);
}

// ── Draw: detection boxes ───────────────────────────────────────────────────

void UIPhysicalEnvironmentPanel::DrawPerceptionDetectionsOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalObjectDetectorOutput& dets,
    int blit_x, int blit_y, int blit_w, int blit_h,
    int model_w, int model_h)
{
    if (dets.detections.empty() || model_w <= 0 || model_h <= 0) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(model_w);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(model_h);
    for (const auto& d : dets.detections) {
        const float x = blit_x + d.model_box.x * sx;
        const float y = blit_y + d.model_box.y * sy;
        const float w = d.model_box.width  * sx;
        const float h = d.model_box.height * sy;
        const uint32_t col = PercColorForClassId(d.class_id);
        renderer.drawRect({x,         y         }, {w, 1.0f}, col);
        renderer.drawRect({x,         y + h - 1 }, {w, 1.0f}, col);
        renderer.drawRect({x,         y         }, {1.0f, h}, col);
        renderer.drawRect({x + w - 1, y         }, {1.0f, h}, col);
        std::ostringstream ss;
        ss << d.class_label << " " << FormatDouble(d.confidence * 100.0, 0) << "%";
        const std::string label = ss.str();
        const float ly = std::max(static_cast<float>(blit_y), y - 16.0f);
        renderer.drawText({x + 3.0f, ly + 2.0f}, label, col);
    }
}

// ── Draw: segmentation alpha-blend ──────────────────────────────────────────

void UIPhysicalEnvironmentPanel::DrawPerceptionSegmentationOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalSemanticSegmenterOutput& seg,
    int blit_x, int blit_y, int blit_w, int blit_h)
{
    if (seg.segmentation.class_id_image.empty()) return;
    if (seg.segmentation.class_id_image.type() != CV_32SC1) return;
    if (blit_w <= 0 || blit_h <= 0) return;

    auto* pixels = static_cast<uint32_t*>(renderer.getPixels());
    if (!pixels) return;
    const int dest_buf_w = renderer.getWidth();
    const int dest_buf_h = renderer.getHeight();
    const cv::Mat& labels = seg.segmentation.class_id_image;

    const bool cache_hit =
        seg_overlay_source_id_ == perc_results_view_.results.source_frame_counter
        && seg_overlay_source_id_ != 0
        && seg_overlay_w_ == blit_w && seg_overlay_h_ == blit_h
        && seg_overlay_argb_.size() == static_cast<size_t>(blit_w) * static_cast<size_t>(blit_h);

    if (!cache_hit) {
        seg_overlay_argb_.assign(static_cast<size_t>(blit_w) * static_cast<size_t>(blit_h), 0);
        const float inv_sx = static_cast<float>(labels.cols) / static_cast<float>(blit_w);
        const float inv_sy = static_cast<float>(labels.rows) / static_cast<float>(blit_h);
        for (int y = 0; y < blit_h; ++y) {
            const int src_y = std::min(labels.rows - 1, static_cast<int>(y * inv_sy));
            const int32_t* src_row = labels.ptr<int32_t>(src_y);
            uint32_t* dst_row = seg_overlay_argb_.data() + static_cast<size_t>(y) * blit_w;
            for (int x = 0; x < blit_w; ++x) {
                const int src_x = std::min(labels.cols - 1, static_cast<int>(x * inv_sx));
                dst_row[x] = PercColorForClassId(src_row[src_x], /*alpha=*/0x80);
            }
        }
        seg_overlay_w_ = blit_w; seg_overlay_h_ = blit_h;
        seg_overlay_source_id_ = perc_results_view_.results.source_frame_counter;
    }

    const int x0 = std::max(0, blit_x);
    const int y0 = std::max(0, blit_y);
    const int x1 = std::min(dest_buf_w, blit_x + blit_w);
    const int y1 = std::min(dest_buf_h, blit_y + blit_h);
    for (int y = y0; y < y1; ++y) {
        const uint32_t* ov = seg_overlay_argb_.data()
                             + static_cast<size_t>(y - blit_y) * blit_w
                             + static_cast<size_t>(x0 - blit_x);
        uint32_t* dst = pixels + static_cast<size_t>(y) * dest_buf_w + static_cast<size_t>(x0);
        for (int x = x0; x < x1; ++x) {
            const uint32_t s = *ov++;
            const uint32_t d = *dst;
            const uint32_t sa = (s >> 24) & 0xFF;
            const uint32_t inv = 255 - sa;
            const uint32_t r = ((((s >> 16) & 0xFF) * sa) + (((d >> 16) & 0xFF) * inv)) / 255;
            const uint32_t g = ((((s >>  8) & 0xFF) * sa) + (((d >>  8) & 0xFF) * inv)) / 255;
            const uint32_t b = ((((s >>  0) & 0xFF) * sa) + (((d >>  0) & 0xFF) * inv)) / 255;
            *dst++ = (0xFFu << 24) | (r << 16) | (g << 8) | b;
        }
    }
}

// ── Draw: pose ─────────────────────────────────────────────────────────────

void UIPhysicalEnvironmentPanel::DrawPerceptionPoseOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalPoseKeypointEstimatorOutput& pose,
    int blit_x, int blit_y, int blit_w, int blit_h,
    int model_w, int model_h)
{
    if (pose.instances.empty() || model_w <= 0 || model_h <= 0) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(model_w);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(model_h);
    const uint32_t kp_col   = PercMakeArgb(0xFF, 0xFF, 0xC0, 0x40);
    const uint32_t bbox_col = PercMakeArgb(0xFF, 0xFF, 0xFF, 0x00);
    for (const auto& inst : pose.instances) {
        const float x = blit_x + inst.model_bbox.x * sx;
        const float y = blit_y + inst.model_bbox.y * sy;
        const float w = inst.model_bbox.width  * sx;
        const float h = inst.model_bbox.height * sy;
        renderer.drawRect({x, y         }, {w, 1.0f}, bbox_col);
        renderer.drawRect({x, y + h - 1 }, {w, 1.0f}, bbox_col);
        renderer.drawRect({x, y         }, {1.0f, h}, bbox_col);
        renderer.drawRect({x + w - 1, y }, {1.0f, h}, bbox_col);
        for (const auto& kp : inst.keypoints) {
            if (!kp.visible) continue;
            const float kx = blit_x + kp.model_xy.x * sx;
            const float ky = blit_y + kp.model_xy.y * sy;
            renderer.drawRect({kx - 2.0f, ky - 2.0f}, {4.0f, 4.0f}, kp_col);
        }
    }
}

// ── Draw: scene text quads + recognised string ─────────────────────────────

void UIPhysicalEnvironmentPanel::DrawPerceptionSceneTextOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalSceneTextReaderOutput& text,
    int blit_x, int blit_y, int blit_w, int blit_h,
    int model_w, int model_h)
{
    if (text.lines.empty() || model_w <= 0 || model_h <= 0) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(model_w);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(model_h);
    const uint32_t quad_col = PercMakeArgb(0xFF, 0x40, 0xC0, 0xFF);

    auto map = [&](const cv::Point2f& p) {
        return Vec2{ blit_x + p.x * sx, blit_y + p.y * sy };
    };
    for (const auto& ln : text.lines) {
        const Vec2 p0 = map(ln.model_quad.p0);
        const Vec2 p1 = map(ln.model_quad.p1);
        const Vec2 p2 = map(ln.model_quad.p2);
        const Vec2 p3 = map(ln.model_quad.p3);
        renderer.drawLine(p0, p1, quad_col);
        renderer.drawLine(p1, p2, quad_col);
        renderer.drawLine(p2, p3, quad_col);
        renderer.drawLine(p3, p0, quad_col);
        if (!ln.text.empty()) {
            renderer.drawText({p0.x, std::max(static_cast<float>(blit_y), p0.y - 14.0f)},
                              ln.text, quad_col);
        }
    }
}

// ── Draw: sidebar status ────────────────────────────────────────────────────

void UIPhysicalEnvironmentPanel::DrawPerceptionFacialExpressionOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalFacialExpressionDetectorOutput& faces,
    int blit_x, int blit_y, int blit_w, int blit_h,
    int model_w, int model_h)
{
    if (faces.faces.empty() || model_w <= 0 || model_h <= 0) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(model_w);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(model_h);
    const uint32_t bbox_col  = PercMakeArgb(0xFF, 0xFF, 0x60, 0xC0);
    const uint32_t label_col = PercMakeArgb(0xFF, 0xFF, 0xFF, 0xFF);
    for (const auto& fc : faces.faces) {
        const float x = blit_x + fc.model_bbox.x * sx;
        const float y = blit_y + fc.model_bbox.y * sy;
        const float w = fc.model_bbox.width  * sx;
        const float h = fc.model_bbox.height * sy;
        renderer.drawRect({x, y         }, {w,    1.0f}, bbox_col);
        renderer.drawRect({x, y + h - 1 }, {w,    1.0f}, bbox_col);
        renderer.drawRect({x, y         }, {1.0f, h   }, bbox_col);
        renderer.drawRect({x + w - 1, y }, {1.0f, h   }, bbox_col);
        if (!fc.expression_label.empty()) {
            std::string s = fc.expression_label + " "
                          + FormatDouble(fc.expression_score * 100.0, 0) + "%";
            renderer.drawText({x, std::max(static_cast<float>(blit_y), y - 14.0f)},
                              s, label_col);
        }
    }
}

// ── Draw: entity tracks (Stage-3 identity persistence) ─────────────────────
//
// Renders one box per live track in MODEL space, plus a fading polyline
// connecting the historical centres so the user can see motion / identity
// continuity. Track ID is rendered above the box so it is unmistakeable
// when two same-class objects are present (e.g. "person #4" vs "person #7").
//
// Internal validity (Rule 3): we BUMP the per-track trail history here. That
// happens exactly once per published frame because DrawPerceptionTab guards
// the overlay calls on `perc_results_view_.results.source_frame_counter ==
// last_view_.frame_counter`, and that counter advances monotonically. We
// also evict trails for tracks that no longer appear in the latest snapshot.
void UIPhysicalEnvironmentPanel::DrawPerceptionEntityTracksOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalEntityTrackerOutput& tracker,
    int blit_x, int blit_y, int blit_w, int blit_h,
    int model_w, int model_h)
{
    if (model_w <= 0 || model_h <= 0) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(model_w);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(model_h);
    const uint64_t fc = tracker.last_frame_counter;

    // Update trail history for every live track in this snapshot.
    for (const auto& t : tracker.tracks) {
        auto& tr = track_trails_[t.track_id];
        const float cx = t.smoothed_model_box.x + t.smoothed_model_box.width  * 0.5f;
        const float cy = t.smoothed_model_box.y + t.smoothed_model_box.height * 0.5f;
        tr.model_centres.emplace_back(cx, cy);
        if (tr.model_centres.size() > kMaxTrailPoints) {
            tr.model_centres.erase(tr.model_centres.begin(),
                                   tr.model_centres.begin()
                                   + (tr.model_centres.size() - kMaxTrailPoints));
        }
        tr.last_seen_frame_counter = fc;
    }
    // Evict trails for tracks not seen for a while (uses unsigned subtraction
    // safe-guarded against fc==0 or stale entries).
    for (auto it = track_trails_.begin(); it != track_trails_.end(); ) {
        const uint64_t age = (fc >= it->second.last_seen_frame_counter)
                             ? (fc - it->second.last_seen_frame_counter) : 0;
        if (age > 90) it = track_trails_.erase(it);
        else          ++it;
    }

    // Render boxes + IDs + trail.
    for (const auto& t : tracker.tracks) {
        // State-driven colour: confirmed = bright class colour, tentative =
        // dim, coasting = warning yellow. Rule 3: state visible at a glance.
        uint8_t alpha = 0xFF;
        uint32_t col = PercColorForClassId(t.class_id, alpha);
        if (t.state == PE::PhysicalEntityTrackState::Tentative) {
            col = PercColorForClassId(t.class_id, 0x70);
        } else if (t.state == PE::PhysicalEntityTrackState::Coasting) {
            col = PercMakeArgb(0xC0, 0xFF, 0xC0, 0x40); // amber, semi-transparent
        }

        const float x = blit_x + t.smoothed_model_box.x * sx;
        const float y = blit_y + t.smoothed_model_box.y * sy;
        const float w = t.smoothed_model_box.width  * sx;
        const float h = t.smoothed_model_box.height * sy;

        // 2-px box so it visibly differs from the raw detector overlay.
        for (int dx = 0; dx < 2; ++dx) {
            renderer.drawRect({x + dx,         y + dx        }, {std::max(0.0f, w - dx*2), 1.0f}, col);
            renderer.drawRect({x + dx,         y + h - 1 - dx}, {std::max(0.0f, w - dx*2), 1.0f}, col);
            renderer.drawRect({x + dx,         y + dx        }, {1.0f, std::max(0.0f, h - dx*2)}, col);
            renderer.drawRect({x + w - 1 - dx, y + dx        }, {1.0f, std::max(0.0f, h - dx*2)}, col);
        }

        // ID + class label + age, drawn ABOVE the box.
        std::ostringstream ss;
        ss << "#" << t.track_id << " " << t.class_label
           << " [" << PE::DescribePhysicalEntityTrackState(t.state) << "]"
           << " age=" << t.age_in_frames
           << " hits=" << t.total_hits;
        const std::string label = ss.str();
        const float ly = std::max(static_cast<float>(blit_y), y - 16.0f);
        renderer.drawText({x + 3.0f, ly + 2.0f}, label, col);

        // Trail polyline. Older points fade toward transparent.
        auto it = track_trails_.find(t.track_id);
        if (it != track_trails_.end() && it->second.model_centres.size() >= 2) {
            const auto& pts = it->second.model_centres;
            const size_t n = pts.size();
            for (size_t i = 1; i < n; ++i) {
                const float a = static_cast<float>(i) / static_cast<float>(n);
                const uint8_t la = static_cast<uint8_t>(40 + 215.0f * a);
                const uint32_t lcol =
                    (col & 0x00FFFFFFu) | (static_cast<uint32_t>(la) << 24);
                const float x0 = blit_x + pts[i-1].first  * sx;
                const float y0 = blit_y + pts[i-1].second * sy;
                const float x1 = blit_x + pts[i  ].first  * sx;
                const float y1 = blit_y + pts[i  ].second * sy;
                // Skip stationary segments \u2014 they degenerate to zero-length
                // lines and add no visual information.
                const float ddx = x1 - x0;
                const float ddy = y1 - y0;
                if (ddx * ddx + ddy * ddy < 0.25f) continue;
                renderer.drawLine({x0, y0}, {x1, y1}, lcol);
            }
        }
    }
}

void UIPhysicalEnvironmentPanel::DrawPerceptionInstanceMasksOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalInstanceSegmenterOutput& inst,
    int blit_x, int blit_y, int blit_w, int blit_h,
    int model_w, int model_h)
{
    if (model_w <= 0 || model_h <= 0) return;
    if (inst.segmentation.instances.empty()) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(model_w);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(model_h);

    for (const auto& m : inst.segmentation.instances) {
        if (m.mask_pixel_count <= 0 || m.mask_within_bbox.empty()) continue;
        if (m.mask_within_bbox.type() != CV_8UC1) {
            // Hard-fail visualisation rather than draw garbage \u2014 the
            // contract on PhysicalInstanceMask says CV_8UC1.
            renderer.drawText({static_cast<float>(blit_x) + 4.0f,
                               static_cast<float>(blit_y) + 4.0f},
                              std::string("InstSeg overlay: mask is not CV_8UC1 for ")
                              + m.class_label,
                              PercMakeArgb(0xFF, 0xFF, 0x40, 0x40));
            continue;
        }
        const uint32_t fill_col    = PercColorForClassId(m.class_id, /*alpha=*/0x55);
        const uint32_t outline_col = PercColorForClassId(m.class_id, /*alpha=*/0xFF);

        // Trace mask outline via cv::findContours. Outlines are far cheaper
        // to draw than per-pixel filled rectangles AND make occlusion
        // boundaries visually obvious (the whole point of segmentation).
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(m.mask_within_bbox, contours,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1);

        // Light fill: 1px squares along the outline interior so the masked
        // region is visible against busy backgrounds, without burning CPU
        // doing per-pixel blits across the whole region.
        for (const auto& contour : contours) {
            if (contour.size() < 2) continue;
            for (size_t i = 0; i < contour.size(); ++i) {
                const cv::Point& a = contour[i];
                const cv::Point& b = contour[(i + 1) % contour.size()];
                // contour points are in mask-local coords; translate to
                // model coords by adding bbox top-left, then to screen.
                const float ax = blit_x + (m.mask_model_bbox.x + a.x) * sx;
                const float ay = blit_y + (m.mask_model_bbox.y + a.y) * sy;
                const float bx = blit_x + (m.mask_model_bbox.x + b.x) * sx;
                const float by = blit_y + (m.mask_model_bbox.y + b.y) * sy;
                renderer.drawLine({ax, ay}, {bx, by}, outline_col);
            }
        }

        // Tag the mask with its class + IoU near the mask bbox top-left so
        // the user knows which detection produced this mask.
        const float lx = blit_x + m.mask_model_bbox.x * sx;
        const float ly = std::max(static_cast<float>(blit_y),
                                  blit_y + m.mask_model_bbox.y * sy - 14.0f);
        std::ostringstream label_ss;
        label_ss << m.class_label
                 << "  iou=" << FormatDouble(m.mask_confidence * 100.0, 0) << "%";
        const std::string label = label_ss.str();
        renderer.drawText({lx + 3.0f, ly + 2.0f}, label, outline_col);

        // Suppress an unused-variable warning while keeping the alpha-fill
        // colour in the API for a future per-pixel renderer mode.
        (void)fill_col;
    }
}

void UIPhysicalEnvironmentPanel::DrawPerceptionSidebar(
    OverlayRenderer& renderer, float x, float y, float w, float /*h*/,
    const PE::PhysicalPerceptionPrimitiveResults& r, bool have_results)
{
    (void)w;
    float row_y = y;
    auto line = [&](const std::string& s, uint32_t col = UITheme::Colors::TextPrimary) {
        renderer.drawText({x, row_y}, s, col);
        row_y += 16.0f;
    };
    auto sep = [&]() { row_y += 4.0f; };

    line("Stage-2 Perception Primitives", UITheme::Colors::TextPrimary);
    line(std::string("Tick=")
         + std::to_string(PE::GetPhysicalPerceptionPrimitivesTickCount())
         + "  processed="
         + std::to_string(PE::GetPhysicalPerceptionPrimitivesProcessedCount()),
         UITheme::Colors::TextSecondary);
    {
        const std::string err = PE::GetLastPhysicalPerceptionPrimitivesError();
        if (!err.empty()) line(std::string("err: ") + err, UITheme::Colors::Danger);
    }
    sep();

    if (!have_results) {
        line("No results yet on PhysicalPerceptionPrimitiveBus.",
             UITheme::Colors::TextSecondary);
        return;
    }

    line(std::string("frame#=") + std::to_string(r.source_frame_counter)
         + "  model="
         + std::to_string(r.model_image_width) + "x"
         + std::to_string(r.model_image_height));
    sep();

    auto opStatus = [&](const char* name, PE::PhysicalImageOperatorState st,
                        const std::string& err, uint64_t infcnt, double ms,
                        const std::string& extra)
    {
        const uint32_t col =
            st == PE::PhysicalImageOperatorState::ModelLoaded
                ? PercMakeArgb(0xFF, 0x70, 0xD8, 0x70)
            : st == PE::PhysicalImageOperatorState::NoModelConfigured
                ? UITheme::Colors::TextSecondary
                : UITheme::Colors::Danger;
        line(std::string(name) + ": " + PE::DescribePhysicalImageOperatorState(st), col);
        if (st == PE::PhysicalImageOperatorState::ModelLoaded) {
            line(std::string("  inf=") + std::to_string(infcnt)
                 + "  " + FormatDouble(ms, 1) + " ms"
                 + (extra.empty() ? std::string{} : ("  " + extra)),
                 UITheme::Colors::TextSecondary);
        }
        if (!err.empty()) {
            line(std::string("  ") + err, UITheme::Colors::Danger);
        }
        sep();
    };

    opStatus("Detector",
             r.object_detector.state, r.object_detector.last_error_reason,
             r.object_detector.inference_count, r.object_detector.last_inference_ms,
             std::string("dets=") + std::to_string(r.object_detector.detections.size()));
    opStatus("Segmenter",
             r.semantic_segmenter.state, r.semantic_segmenter.last_error_reason,
             r.semantic_segmenter.inference_count, r.semantic_segmenter.last_inference_ms,
             std::string("classes=") + std::to_string(r.semantic_segmenter.segmentation.num_classes));
    opStatus("Classifier",
             r.image_classifier.state, r.image_classifier.last_error_reason,
             r.image_classifier.inference_count, r.image_classifier.last_inference_ms,
             std::string("topK=") + std::to_string(r.image_classifier.top_k.size()));
    if (r.image_classifier.state == PE::PhysicalImageOperatorState::ModelLoaded) {
        for (const auto& c : r.image_classifier.top_k) {
            line(std::string("    ") + FormatDouble(c.score * 100.0, 1) + "%  "
                 + c.class_label,
                 UITheme::Colors::TextPrimary);
        }
        sep();
    }
    opStatus("Pose",
             r.pose_estimator.state, r.pose_estimator.last_error_reason,
             r.pose_estimator.inference_count, r.pose_estimator.last_inference_ms,
             std::string("instances=") + std::to_string(r.pose_estimator.instances.size()));
    opStatus("Text",
             r.scene_text_reader.state, r.scene_text_reader.last_error_reason,
             r.scene_text_reader.inference_count, r.scene_text_reader.last_inference_ms,
             std::string("lines=") + std::to_string(r.scene_text_reader.lines.size())
             + (r.scene_text_reader.recogniser_configured ? " (+OCR)" : " (det-only)"));
    opStatus("Face",
             r.facial_expression_detector.state, r.facial_expression_detector.last_error_reason,
             r.facial_expression_detector.inference_count, r.facial_expression_detector.last_inference_ms,
             std::string("faces=") + std::to_string(r.facial_expression_detector.faces.size())
             + (r.facial_expression_detector.classifier_configured ? " (+emotion)" : " (det-only)"));
    if (r.facial_expression_detector.state == PE::PhysicalImageOperatorState::ModelLoaded
        && !r.facial_expression_detector.faces.empty()) {
        for (const auto& fc : r.facial_expression_detector.faces) {
            if (fc.expression_label.empty()) continue;
            line(std::string("    ") + FormatDouble(fc.expression_score * 100.0, 1) + "%  "
                 + fc.expression_label,
                 UITheme::Colors::TextPrimary);
        }
        sep();
    }
    {
        // Stage-3 tracker. Use last_route_ms (NOT last_inference_ms) because
        // tracker has no ONNX inference \u2014 it's a pure association step.
        const auto& tk = r.entity_tracker;
        std::ostringstream extra;
        extra << "tracks=" << tk.tracks.size()
              << "  spawned=" << tk.total_tracks_spawned
              << "  confirmed=" << tk.total_tracks_confirmed
              << "  culled=" << tk.total_tracks_culled;
        opStatus("Tracker", tk.state, tk.last_error_reason,
                 tk.inference_count, tk.last_route_ms, extra.str());
        if (tk.state == PE::PhysicalImageOperatorState::ModelLoaded && !tk.tracks.empty()) {
            // Sort by id ascending (tracks vector is in spawn order already
            // because we never reorder it; but be explicit for the UI).
            std::vector<const PE::PhysicalEntityTrack*> sorted;
            sorted.reserve(tk.tracks.size());
            for (const auto& t : tk.tracks) sorted.push_back(&t);
            std::sort(sorted.begin(), sorted.end(),
                      [](const PE::PhysicalEntityTrack* a,
                         const PE::PhysicalEntityTrack* b){ return a->track_id < b->track_id; });
            const size_t shown = std::min<size_t>(6, sorted.size());
            for (size_t i = 0; i < shown; ++i) {
                const auto& t = *sorted[i];
                std::ostringstream row;
                row << "    #" << t.track_id << " " << t.class_label
                    << " [" << PE::DescribePhysicalEntityTrackState(t.state) << "]"
                    << "  hits=" << t.total_hits
                    << "  miss=" << t.miss_streak
                    << "  conf=" << FormatDouble(t.smoothed_confidence * 100.0, 0) << "%";
                line(row.str(), UITheme::Colors::TextPrimary);
            }
            if (sorted.size() > shown) {
                line(std::string("    \u2026 ") + std::to_string(sorted.size() - shown)
                     + " more", UITheme::Colors::TextSecondary);
            }
            sep();
        }
    }
    {
        // Stage-2 instance segmenter (SAM 2). Reports separate encoder /
        // decoder timings because the encoder runs once per frame while the
        // decoder loops once per prompt — extremely useful for spotting
        // "too many prompts" cost spikes.
        const auto& isg = r.instance_segmenter;
        std::ostringstream extra;
        extra << "masks=" << isg.segmentation.instances.size()
              << "  prompts=" << isg.prompt_count
              << "  enc=" << FormatDouble(isg.last_encoder_ms, 1) << "ms"
              << "  dec=" << FormatDouble(isg.last_decoder_total_ms, 1) << "ms";
        opStatus("InstSeg", isg.state, isg.last_error_reason,
                 isg.inference_count, isg.last_inference_ms, extra.str());
        if (isg.state == PE::PhysicalImageOperatorState::ModelLoaded
            && !isg.segmentation.instances.empty()) {
            const size_t shown = std::min<size_t>(6, isg.segmentation.instances.size());
            for (size_t i = 0; i < shown; ++i) {
                const auto& m = isg.segmentation.instances[i];
                std::ostringstream row;
                row << "    " << m.class_label
                    << "  iou=" << FormatDouble(m.mask_confidence * 100.0, 0) << "%"
                    << "  px=" << m.mask_pixel_count
                    << "  area=" << FormatDouble(m.mask_area_fraction * 100.0, 1) << "%";
                line(row.str(), UITheme::Colors::TextPrimary);
            }
            if (isg.segmentation.instances.size() > shown) {
                line(std::string("    \u2026 ")
                     + std::to_string(isg.segmentation.instances.size() - shown)
                     + " more", UITheme::Colors::TextSecondary);
            }
            sep();
        }
    }
    {
        // Stage-2.5 class policy. This is the section that shows what the
        // model context matrix builder will actually consume — one row per
        // canonical class, sorted by priority_rank ascending. Mutation
        // counters surface how the policy reshaped the per-operator output
        // this frame so the operator is never invisible.
        const auto& cp = r.class_policy;
        std::ostringstream extra;
        extra << "ranks=" << cp.ranked_classes.size()
              << "  rl=" << (cp.detections_relabeled
                             + cp.tracks_relabeled
                             + cp.instance_masks_relabeled
                             + cp.classifications_relabeled)
              << "  drop=" << (cp.detections_dropped
                               + cp.tracks_dropped
                               + cp.instance_masks_dropped
                               + cp.classifications_dropped);
        opStatus("Policy", cp.state, cp.last_error_reason,
                 cp.inference_count, cp.last_apply_ms, extra.str());
        if (cp.state == PE::PhysicalImageOperatorState::ModelLoaded
            && !cp.ranked_classes.empty()) {
            const size_t shown = std::min<size_t>(8, cp.ranked_classes.size());
            for (size_t i = 0; i < shown; ++i) {
                const auto& row = cp.ranked_classes[i];
                std::ostringstream rl;
                rl << "    #" << row.priority_rank << "  " << row.canonical_label
                   << "  d=" << row.detection_count
                   << " t=" << row.track_count
                   << " m=" << row.instance_mask_count
                   << " c=" << row.classification_count
                   << "  " << FormatDouble(row.max_confidence * 100.0, 0) << "%";
                line(rl.str(), UITheme::Colors::TextPrimary);
            }
            if (cp.ranked_classes.size() > shown) {
                line(std::string("    \u2026 ")
                     + std::to_string(cp.ranked_classes.size() - shown)
                     + " more", UITheme::Colors::TextSecondary);
            }
            sep();
        }
    }
}

// ── Draw: tab body ─────────────────────────────────────────────────────────

void UIPhysicalEnvironmentPanel::DrawPerceptionTab(OverlayRenderer& renderer) {
    // Toolbar buttons (already laid out + ticked in UpdatePerceptionTab).
    if (perc_btn_obj_)  perc_btn_obj_->drawOverlay(renderer, position);
    if (perc_btn_seg_)  perc_btn_seg_->drawOverlay(renderer, position);
    if (perc_btn_cls_)  perc_btn_cls_->drawOverlay(renderer, position);
    if (perc_btn_pose_) perc_btn_pose_->drawOverlay(renderer, position);
    if (perc_btn_text_) perc_btn_text_->drawOverlay(renderer, position);
    if (perc_btn_face_) perc_btn_face_->drawOverlay(renderer, position);
    if (perc_btn_track_)perc_btn_track_->drawOverlay(renderer, position);
    if (perc_btn_inst_seg_)perc_btn_inst_seg_->drawOverlay(renderer, position);
    if (perc_btn_class_policy_)perc_btn_class_policy_->drawOverlay(renderer, position);

    const float pad         = 12.0f;
    const float toolbar_h   = 32.0f;
    const float content_top = position.y + titleBarHeight + kTabBarHeight + toolbar_h + pad;

    const float sidebar_w = 290.0f;
    const float frame_x = position.x + pad;
    const float frame_y = content_top;
    const float frame_w = std::max(64.0f, size.x - sidebar_w - pad * 3.0f);
    const float frame_h = std::max(64.0f, position.y + size.y - frame_y - pad);
    const float sidebar_x = frame_x + frame_w + pad;
    const float sidebar_y = frame_y;
    const float sidebar_h = frame_h;

    // Frame area background
    renderer.drawRect({frame_x - 1, frame_y - 1}, {frame_w + 2, frame_h + 2},
                      UITheme::Colors::DividerLine);
    renderer.drawRect({frame_x, frame_y}, {frame_w, frame_h}, UITheme::Colors::Background);

    int blit_x = 0, blit_y = 0, blit_w = 0, blit_h = 0;
    int model_w = 0, model_h = 0;

    if (!have_any_frame_ || last_view_.model_image.empty()) {
        renderer.drawText({frame_x + 12, frame_y + 12},
                          "No frame on PhysicalFrameBus yet \u2014 connect a camera in the Camera tab.",
                          UITheme::Colors::TextSecondary);
    } else {
        // Reuse the shared blit helper. Tag the cache by frame_counter +
        // a constant 'true' for the model-image view (Perception always
        // shows the model-space image, NOT the raw image).
        DrawBgrFrameIntoOverlay(renderer, last_view_.model_image,
                                last_seen_counter_, /*source_undistort=*/true,
                                frame_x, frame_y, frame_w, frame_h,
                                perception_blit_cache_);
        // Recover the actual blit rect from the cache so overlays land where
        // the pixels did. DrawBgrFrameIntoOverlay computes letterboxed out_w/h.
        model_w = last_view_.model_image.cols;
        model_h = last_view_.model_image.rows;
        blit_w  = perception_blit_cache_.out_w;
        blit_h  = perception_blit_cache_.out_h;
        blit_x  = static_cast<int>(frame_x + (frame_w - blit_w) * 0.5f);
        blit_y  = static_cast<int>(frame_y + (frame_h - blit_h) * 0.5f);

        if (have_any_perc_results_
            && perc_results_view_.results.source_frame_counter == last_view_.frame_counter) {
            const auto& r = perc_results_view_.results;
            DrawPerceptionSegmentationOverlay(renderer, r.semantic_segmenter,
                                              blit_x, blit_y, blit_w, blit_h);
            DrawPerceptionDetectionsOverlay(renderer, r.object_detector,
                                            blit_x, blit_y, blit_w, blit_h, model_w, model_h);
            DrawPerceptionPoseOverlay(renderer, r.pose_estimator,
                                      blit_x, blit_y, blit_w, blit_h, model_w, model_h);
            DrawPerceptionSceneTextOverlay(renderer, r.scene_text_reader,
                                           blit_x, blit_y, blit_w, blit_h, model_w, model_h);
            DrawPerceptionFacialExpressionOverlay(renderer, r.facial_expression_detector,
                                                  blit_x, blit_y, blit_w, blit_h, model_w, model_h);
            DrawPerceptionEntityTracksOverlay(renderer, r.entity_tracker,
                                              blit_x, blit_y, blit_w, blit_h, model_w, model_h);
            DrawPerceptionInstanceMasksOverlay(renderer, r.instance_segmenter,
                                               blit_x, blit_y, blit_w, blit_h, model_w, model_h);
        } else if (have_any_perc_results_) {
            renderer.drawText({frame_x + 12, frame_y + frame_h - 22},
                              "Latest results are for an older frame "
                              "(perception is catching up)",
                              UITheme::Colors::TextSecondary);
        }
    }

    // Sidebar background
    renderer.drawRect({sidebar_x - 1, sidebar_y - 1},
                      {sidebar_w + 2, sidebar_h + 2}, UITheme::Colors::DividerLine);
    renderer.drawRect({sidebar_x, sidebar_y},
                      {sidebar_w, sidebar_h}, UITheme::Colors::PanelBg);
    DrawPerceptionSidebar(renderer, sidebar_x + 8, sidebar_y + 8,
                          sidebar_w - 16, sidebar_h - 16,
                          perc_results_view_.results, have_any_perc_results_);
}

// ============================================================================
//  Interaction tab (Stage-2 auxiliary — local controller input)
// ============================================================================

void UIPhysicalEnvironmentPanel::HandleToggleHandGestures() {
    const auto config = PE::GetPhysicalHandGestureConfig();
    PE::RequestSetPhysicalHandGesturesEnabled(!config.enabled);
    RefreshInteractionButtonLabels();
}

void UIPhysicalEnvironmentPanel::HandleReloadHandGestureBackend() {
    PE::RequestConfigurePhysicalHandGestures(
        PE::GetPhysicalHandGestureConfig());
}

void UIPhysicalEnvironmentPanel::HandleToggleGestureController() {
    auto config = PE::GetPhysicalGestureControlConfig();
    config.enabled = !config.enabled;
    PE::RequestConfigurePhysicalGestureControl(config);
    PersistGestureBindingsFromUi();
    RefreshInteractionButtonLabels();
}

void UIPhysicalEnvironmentPanel::HandleToggleGestureDryRun() {
    auto config = PE::GetPhysicalGestureControlConfig();
    config.dry_run = !config.dry_run;
    PE::RequestConfigurePhysicalGestureControl(config);
    PersistGestureBindingsFromUi();
    RefreshInteractionButtonLabels();
}

void UIPhysicalEnvironmentPanel::HandleToggleGestureStudioView() {
    interaction_show_bindings_ = !interaction_show_bindings_;
    if (interaction_show_bindings_) RebuildGestureBindingEditor();
    RefreshInteractionButtonLabels();
}

void UIPhysicalEnvironmentPanel::HandleUseLiveGesture() {
    if (!have_interaction_snapshot_) {
        interaction_binding_status_ = "No live gesture result is available.";
        return;
    }
    const auto& hands = interaction_snapshot_view_.snapshot.hands;
    const PE::PhysicalHandObservation* best = nullptr;
    for (const auto& hand : hands) {
        if (hand.gesture_label.empty() || hand.gesture_label == "None" ||
            hand.gesture_label == "none") continue;
        if (!best || hand.gesture_confidence > best->gesture_confidence)
            best = &hand;
    }
    if (!best) {
        interaction_binding_status_ = "No recognized live gesture to copy.";
        return;
    }
    interaction_gesture_buf_ = best->gesture_label;
    if (interaction_gesture_box_)
        interaction_gesture_box_->setText(interaction_gesture_buf_);
    interaction_binding_status_ = "Copied live label: " + best->gesture_label;
}

void UIPhysicalEnvironmentPanel::PersistGestureBindingsFromUi() {
    std::string error;
    if (!PE::PersistPhysicalGestureControlConfig(error)) {
        interaction_binding_status_ = "Save failed: " + error;
        LOG_ERROR(kPanelLogTag, interaction_binding_status_);
    } else {
        interaction_binding_status_ = "Gesture control saved locally.";
    }
}

void UIPhysicalEnvironmentPanel::LoadSelectedGestureBindingIntoEditor() {
    const auto config = PE::GetPhysicalGestureControlConfig();
    if (config.bindings.empty()) return;
    interaction_selected_binding_ = std::min(
        interaction_selected_binding_, config.bindings.size() - 1);
    const auto& binding = config.bindings[interaction_selected_binding_];

    interaction_gesture_buf_ = binding.gesture_label;
    interaction_hold_buf_ = std::to_string(binding.minimum_hold_ms);
    interaction_cooldown_buf_ = std::to_string(binding.cooldown_ms);
    interaction_priority_buf_ = std::to_string(binding.priority);
    if (interaction_gesture_box_) interaction_gesture_box_->setText(interaction_gesture_buf_);
    if (interaction_hold_box_) interaction_hold_box_->setText(interaction_hold_buf_);
    if (interaction_cooldown_box_) interaction_cooldown_box_->setText(interaction_cooldown_buf_);
    if (interaction_priority_box_) interaction_priority_box_->setText(interaction_priority_buf_);

    if (interaction_action_select_) {
        interaction_action_select_->setSelectedIndex(
            static_cast<int>(binding.action));
    }
    if (interaction_trigger_select_) {
        interaction_trigger_select_->setSelectedIndex(
            static_cast<int>(binding.trigger));
    }
    if (interaction_hand_select_) {
        interaction_hand_select_->setSelectedIndex(
            binding.handedness == PE::PhysicalHandedness::Left ? 1 :
            binding.handedness == PE::PhysicalHandedness::Right ? 2 : 0);
    }
    interaction_editor_binding_enabled_ = binding.enabled;
    interaction_editor_requires_arm_ = binding.requires_armed;
    if (interaction_binding_enabled_btn_) {
        interaction_binding_enabled_btn_->setText(
            binding.enabled ? " Binding: on " : " Binding: off ");
    }
    if (interaction_requires_arm_btn_) {
        interaction_requires_arm_btn_->setText(
            binding.requires_armed ? " Requires arm: yes " : " Requires arm: no ");
    }
}

void UIPhysicalEnvironmentPanel::RebuildGestureBindingEditor() {
    const auto config = PE::GetPhysicalGestureControlConfig();
    interaction_selected_binding_ = config.bindings.empty() ? 0 :
        std::min(interaction_selected_binding_, config.bindings.size() - 1);

    const float previous_scroll = interaction_binding_list_
        ? interaction_binding_list_->getScrollOffset() : 0.0f;
    interaction_binding_rows_.clear();
    if (interaction_binding_list_) interaction_binding_list_->clearChildren();

    for (size_t i = 0; i < config.bindings.size(); ++i) {
        const auto& binding = config.bindings[i];
        std::string row_text = binding.enabled ? "ON  " : "OFF ";
        row_text += binding.binding_id + " | " + binding.gesture_label;
        if (row_text.size() > 31) row_text = row_text.substr(0, 28) + "...";
        row_text = (i == interaction_selected_binding_ ? "> " : "  ") + row_text;
        const std::string binding_id = binding.binding_id;
        auto row = std::make_shared<UIButton>(row_text,
            [this, binding_id] {
                const auto current = PE::GetPhysicalGestureControlConfig();
                for (size_t index = 0; index < current.bindings.size(); ++index) {
                    if (current.bindings[index].binding_id != binding_id) continue;
                    interaction_selected_binding_ = index;
                    LoadSelectedGestureBindingIntoEditor();
                    interaction_binding_list_needs_rebuild_ = true;
                    break;
                }
            });
        row->setSize(232.0f, 34.0f);
        interaction_binding_rows_.push_back(row);
        if (interaction_binding_list_) interaction_binding_list_->addChild(row);
    }
    if (interaction_binding_list_) {
        interaction_binding_list_->autoLayoutChildren(8.0f);
        interaction_binding_list_->scrollTo(previous_scroll);
        if (interaction_selected_binding_ < interaction_binding_rows_.size() &&
            interaction_binding_list_->getSize().y > 0.0f) {
            const auto& selected_row = interaction_binding_rows_[interaction_selected_binding_];
            const float row_top = selected_row->getPosition().y;
            const float row_bottom = row_top + selected_row->getSize().y;
            const float view_top = interaction_binding_list_->getScrollOffset();
            const float view_bottom = view_top + interaction_binding_list_->getSize().y;
            if (row_top < view_top) {
                interaction_binding_list_->scrollTo(row_top);
            } else if (row_bottom > view_bottom) {
                interaction_binding_list_->scrollTo(
                    row_bottom - interaction_binding_list_->getSize().y);
            }
        }
    }
    if (!config.bindings.empty()) LoadSelectedGestureBindingIntoEditor();

    const auto issues = PE::ValidatePhysicalGestureBindings(config);
    if (!issues.empty()) interaction_binding_status_ = issues.front();
    interaction_binding_list_needs_rebuild_ = false;
}

void UIPhysicalEnvironmentPanel::HandleApplyGestureBinding() {
    auto config = PE::GetPhysicalGestureControlConfig();
    if (interaction_selected_binding_ >= config.bindings.size()) {
        interaction_binding_status_ = "Select or add a binding first.";
        return;
    }
    int hold_ms = 0, cooldown_ms = 0, priority = 0;
    if (interaction_gesture_buf_.empty() ||
        !TryParseInt(interaction_hold_buf_, hold_ms) || hold_ms < 0 ||
        !TryParseInt(interaction_cooldown_buf_, cooldown_ms) || cooldown_ms < 0 ||
        !TryParseInt(interaction_priority_buf_, priority)) {
        interaction_binding_status_ =
            "Gesture is required; hold/cooldown/priority must be valid integers.";
        return;
    }

    auto& binding = config.bindings[interaction_selected_binding_];
    const std::string selected_id = binding.binding_id;
    PE::PhysicalGestureAction action;
    PE::PhysicalGestureTrigger trigger;
    if (!PE::TryParsePhysicalGestureAction(
            interaction_action_select_->getSelectedItem(), action) ||
        !PE::TryParsePhysicalGestureTrigger(
            interaction_trigger_select_->getSelectedItem(), trigger)) {
        interaction_binding_status_ = "Invalid action or trigger selection.";
        return;
    }
    binding.gesture_label = interaction_gesture_buf_;
    binding.action = action;
    binding.trigger = trigger;
    binding.minimum_hold_ms = static_cast<uint64_t>(hold_ms);
    binding.cooldown_ms = static_cast<uint64_t>(cooldown_ms);
    binding.priority = priority;
    binding.enabled = interaction_editor_binding_enabled_;
    binding.requires_armed = interaction_editor_requires_arm_;
    const std::string hand = interaction_hand_select_->getSelectedItem();
    binding.handedness = hand == "left" ? PE::PhysicalHandedness::Left :
        hand == "right" ? PE::PhysicalHandedness::Right :
                           PE::PhysicalHandedness::Unknown;

    PE::RequestConfigurePhysicalGestureControl(config);
    const auto applied = PE::GetPhysicalGestureControlConfig();
    for (size_t i = 0; i < applied.bindings.size(); ++i) {
        if (applied.bindings[i].binding_id == selected_id) {
            interaction_selected_binding_ = i;
            break;
        }
    }
    RebuildGestureBindingEditor();
    PersistGestureBindingsFromUi();
    const auto issues = PE::ValidatePhysicalGestureBindings(applied);
    if (!issues.empty()) interaction_binding_status_ = "Saved with warning: " + issues.front();
}

void UIPhysicalEnvironmentPanel::HandleAddGestureBinding() {
    auto config = PE::GetPhysicalGestureControlConfig();
    std::unordered_set<std::string> ids;
    for (const auto& binding : config.bindings) ids.insert(binding.binding_id);
    int suffix = 1;
    std::string id;
    do { id = "custom_" + std::to_string(suffix++); }
    while (ids.count(id) != 0);

    PE::PhysicalGestureBinding binding;
    binding.binding_id = id;
    binding.gesture_label = "New_Gesture";
    binding.action = PE::PhysicalGestureAction::VoiceWake;
    binding.trigger = PE::PhysicalGestureTrigger::Held;
    binding.minimum_hold_ms = 1000;
    binding.cooldown_ms = 10000;
    binding.enabled = false;
    config.bindings.push_back(binding);
    PE::RequestConfigurePhysicalGestureControl(config);
    const auto applied = PE::GetPhysicalGestureControlConfig();
    for (size_t i = 0; i < applied.bindings.size(); ++i) {
        if (applied.bindings[i].binding_id == id) {
            interaction_selected_binding_ = i;
            break;
        }
    }
    RebuildGestureBindingEditor();
    PersistGestureBindingsFromUi();
}

void UIPhysicalEnvironmentPanel::HandleDeleteGestureBinding() {
    auto config = PE::GetPhysicalGestureControlConfig();
    if (interaction_selected_binding_ >= config.bindings.size()) return;
    config.bindings.erase(config.bindings.begin() +
        static_cast<std::ptrdiff_t>(interaction_selected_binding_));
    if (config.bindings.empty()) {
        interaction_binding_status_ =
            "At least one binding is required; restore defaults instead.";
        return;
    }
    if (interaction_selected_binding_ >= config.bindings.size())
        interaction_selected_binding_ = config.bindings.size() - 1;
    PE::RequestConfigurePhysicalGestureControl(config);
    RebuildGestureBindingEditor();
    PersistGestureBindingsFromUi();
}

void UIPhysicalEnvironmentPanel::HandleRestoreDefaultGestureBindings() {
    auto config = PE::GetPhysicalGestureControlConfig();
    config.bindings = PE::DefaultPhysicalGestureBindings();
    PE::RequestConfigurePhysicalGestureControl(config);
    interaction_selected_binding_ = 0;
    RebuildGestureBindingEditor();
    PersistGestureBindingsFromUi();
}

void UIPhysicalEnvironmentPanel::RefreshInteractionButtonLabels() {
    if (!interaction_enable_btn_) return;
    const auto config = PE::GetPhysicalHandGestureConfig();
    interaction_enable_btn_->setText(config.enabled
        ? std::string(" Gestures: on ")
        : std::string(" Gestures: off "));
    const auto control = PE::GetPhysicalGestureControlConfig();
    if (interaction_controller_btn_)
        interaction_controller_btn_->setText(control.enabled
            ? " Controller: on " : " Controller: off ");
    if (interaction_dry_run_btn_)
        interaction_dry_run_btn_->setText(control.dry_run
            ? " Dry run: on " : " Dry run: off ");
    if (interaction_view_btn_)
        interaction_view_btn_->setText(interaction_show_bindings_
            ? " View: Bindings " : " View: Live ");
}

void UIPhysicalEnvironmentPanel::UpdateInteractionTab(
    const InputState& input, float dt)
{
    try {
        if (PE::PhysicalHandGestureBus::Instance()
                .PullLatestPhysicalHandGestureSnapshot(
                    interaction_snapshot_view_,
                    interaction_last_seen_sequence_)) {
            have_interaction_snapshot_ = true;
            RefreshInteractionButtonLabels();
        }
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
            std::string("UpdateInteractionTab: snapshot pull threw: ") + e.what());
    }

    const float content_top =
        position.y + titleBarHeight + kTabBarHeight + 8.0f;
    auto place_button = [&](const std::shared_ptr<UIButton>& button,
                            float x, float y, float w) {
        if (!button) return;
        button->setSize(w, 24.0f);
        button->setPosition(x, y);
        button->update(input, dt);
    };
    const float toolbar_x = position.x + 12.0f;
    place_button(interaction_enable_btn_, toolbar_x, content_top, 112.0f);
    place_button(interaction_reload_btn_, toolbar_x + 118.0f, content_top, 190.0f);
    place_button(interaction_controller_btn_, toolbar_x + 314.0f, content_top, 126.0f);
    place_button(interaction_view_btn_, toolbar_x + 446.0f, content_top, 132.0f);
    place_button(interaction_dry_run_btn_, toolbar_x + 584.0f, content_top, 116.0f);

    if (!interaction_show_bindings_) return;

    if (interaction_binding_list_needs_rebuild_)
        RebuildGestureBindingEditor();

    const float pad = 12.0f;
    const float pane_top = position.y + titleBarHeight + kTabBarHeight + 44.0f;
    const float sidebar_w = 350.0f;
    const float sidebar_x = position.x + size.x - pad - sidebar_w;
    const float inner_x = sidebar_x + 8.0f;
    const float inner_w = sidebar_w - 16.0f;
    const float list_y = pane_top + 30.0f;
    const float list_h = 116.0f;
    if (interaction_binding_list_) {
        interaction_binding_list_->setPosition(inner_x, list_y);
        interaction_binding_list_->setSize(inner_w, list_h);
        for (const auto& row : interaction_binding_rows_)
            if (row) row->setSize(inner_w - 22.0f, 34.0f);
        interaction_binding_list_->update(input, dt);
    }
    const float list_buttons_y = list_y + list_h + 6.0f;
    place_button(interaction_add_binding_btn_, inner_x,
                 list_buttons_y, 70.0f);
    place_button(interaction_delete_binding_btn_, inner_x + 76.0f,
                 list_buttons_y, 70.0f);
    place_button(interaction_defaults_btn_, inner_x + 152.0f,
                 list_buttons_y, inner_w - 152.0f);

    const float editor_x = inner_x;
    const float editor_w = inner_w;
    const float editor_y = list_buttons_y + 52.0f;
    auto place_dropdown = [&](const std::shared_ptr<UIDropdown>& dropdown,
                              float y) {
        if (!dropdown) return;
        dropdown->setPosition(editor_x, y);
        dropdown->setSize(editor_w, 30.0f);
        dropdown->update(input, dt);
    };
    place_dropdown(interaction_action_select_, editor_y);
    place_dropdown(interaction_trigger_select_, editor_y + 34.0f);
    place_dropdown(interaction_hand_select_, editor_y + 68.0f);

    auto place_box = [&](const std::shared_ptr<UIInputBox>& box,
                         float x, float y, float w) {
        if (!box) return;
        box->setPosition(x, y);
        box->setSize(w, 28.0f);
        box->update(input, dt);
    };
    place_box(interaction_gesture_box_, editor_x, editor_y + 120.0f, 214.0f);
    place_button(interaction_use_live_btn_, editor_x + 220.0f,
                 editor_y + 122.0f, editor_w - 220.0f);
    place_box(interaction_hold_box_, editor_x, editor_y + 164.0f, 102.0f);
    place_box(interaction_cooldown_box_, editor_x + 108.0f,
              editor_y + 164.0f, 112.0f);
    place_box(interaction_priority_box_, editor_x + 226.0f,
              editor_y + 164.0f, editor_w - 226.0f);
    place_button(interaction_binding_enabled_btn_, editor_x,
                 editor_y + 204.0f, 98.0f);
    place_button(interaction_requires_arm_btn_, editor_x + 104.0f,
                 editor_y + 204.0f, 132.0f);
    place_button(interaction_apply_binding_btn_, editor_x + 242.0f,
                 editor_y + 204.0f, editor_w - 242.0f);
}

void UIPhysicalEnvironmentPanel::DrawHandGestureOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalHandGestureSnapshot& snapshot,
    int blit_x, int blit_y, int blit_w, int blit_h)
{
    static constexpr std::array<std::array<uint8_t, 2>, 21> kHandEdges{{
        {{0,1}}, {{1,2}}, {{2,3}}, {{3,4}},
        {{0,5}}, {{5,6}}, {{6,7}}, {{7,8}},
        {{5,9}}, {{9,10}}, {{10,11}}, {{11,12}},
        {{9,13}}, {{13,14}}, {{14,15}}, {{15,16}},
        {{13,17}}, {{0,17}}, {{17,18}}, {{18,19}}, {{19,20}}
    }};

    for (const auto& hand : snapshot.hands) {
        if (hand.landmark_count == 0) continue;
        const uint32_t color = hand.handedness == PE::PhysicalHandedness::Left
            ? UITheme::Colors::AccentBlue : UITheme::Colors::Success;
        auto point = [&](uint8_t index) {
            const auto& landmark = hand.landmarks[index];
            return std::pair<float, float>{
                static_cast<float>(blit_x) + landmark.normalized_x * blit_w,
                static_cast<float>(blit_y) + landmark.normalized_y * blit_h
            };
        };
        for (const auto& edge : kHandEdges) {
            if (edge[0] >= hand.landmark_count || edge[1] >= hand.landmark_count)
                continue;
            const auto a = point(edge[0]);
            const auto b = point(edge[1]);
            renderer.drawLine({a.first, a.second}, {b.first, b.second}, color);
        }
        for (uint8_t i = 0; i < std::min<uint32_t>(21, hand.landmark_count); ++i) {
            const auto p = point(i);
            renderer.drawRect({p.first - 2.0f, p.second - 2.0f},
                              {5.0f, 5.0f}, color);
        }
        const auto wrist = point(0);
        const std::string label = std::string(FormatHandedness(hand.handedness))
            + "  " + (hand.gesture_label.empty() ? "No gesture" : hand.gesture_label)
            + "  " + FormatDouble(hand.gesture_confidence * 100.0, 0) + "%";
        renderer.drawText({wrist.first + 5.0f, wrist.second + 5.0f}, label, color);
    }
}

void UIPhysicalEnvironmentPanel::DrawInteractionPreview(
    OverlayRenderer& renderer,
    float frame_x, float frame_y, float frame_w, float frame_h)
{
    renderer.drawRect({frame_x - 1, frame_y - 1}, {frame_w + 2, frame_h + 2},
                      UITheme::Colors::DividerLine);
    renderer.drawRect({frame_x, frame_y}, {frame_w, frame_h},
                      UITheme::Colors::Background);

    // MediaPipe runs asynchronously, so the newest camera frame normally
    // advances beyond the frame that produced the latest landmarks. Prefer
    // the immutable source frame pinned in the gesture snapshot; only use the
    // live latest-frame view while waiting for the first processed result.
    const PE::PhysicalFrameBus::FrameView* preview_frame = &last_view_;
    bool have_coherent_overlay = false;
    if (have_interaction_snapshot_) {
        const auto& snapshot = interaction_snapshot_view_.snapshot;
        const auto& source_frame = snapshot.source_frame;
        have_coherent_overlay =
            snapshot.source_frame_counter != 0 &&
            source_frame.packet &&
            source_frame.frame_counter == snapshot.source_frame_counter &&
            !source_frame.raw_image.empty();
        if (have_coherent_overlay) preview_frame = &source_frame;
    }

    if ((!have_any_frame_ && !have_coherent_overlay) ||
        preview_frame->raw_image.empty()) {
        renderer.drawText({frame_x + 12, frame_y + 12},
            "No camera frame yet - connect a source in Camera.",
            UITheme::Colors::TextSecondary);
        return;
    }

    DrawBgrFrameIntoOverlay(renderer, preview_frame->raw_image,
                            preview_frame->frame_counter, false,
                            frame_x, frame_y, frame_w, frame_h,
                            interaction_blit_cache_);
    const int blit_w = interaction_blit_cache_.out_w;
    const int blit_h = interaction_blit_cache_.out_h;
    const int blit_x = static_cast<int>(frame_x + (frame_w - blit_w) * 0.5f);
    const int blit_y = static_cast<int>(frame_y + (frame_h - blit_h) * 0.5f);
    if (have_coherent_overlay) {
        DrawHandGestureOverlay(renderer, interaction_snapshot_view_.snapshot,
                               blit_x, blit_y, blit_w, blit_h);
    } else if (have_interaction_snapshot_ &&
               interaction_snapshot_view_.snapshot.source_frame_counter != 0) {
        renderer.drawText({frame_x + 10, frame_y + frame_h - 22},
            "Gesture source frame is unavailable; overlay withheld.",
            UITheme::Colors::Warning);
    }
}

void UIPhysicalEnvironmentPanel::DrawInteractionTab(OverlayRenderer& renderer) {
    const std::array<std::shared_ptr<UIButton>, 5> toolbar_buttons{{
        interaction_enable_btn_, interaction_reload_btn_, interaction_controller_btn_,
        interaction_view_btn_, interaction_dry_run_btn_
    }};
    for (const auto& button : toolbar_buttons) {
        if (button) button->drawOverlay(renderer, position);
    }
    if (interaction_show_bindings_) {
        DrawGestureBindingsEditor(renderer);
        return;
    }

    const float pad = 12.0f;
    const float toolbar_h = 32.0f;
    const float content_top =
        position.y + titleBarHeight + kTabBarHeight + toolbar_h + pad;
    const float sidebar_w = 310.0f;
    const float frame_x = position.x + pad;
    const float frame_y = content_top;
    const float frame_w = std::max(64.0f, size.x - sidebar_w - pad * 3.0f);
    const float frame_h = std::max(64.0f, position.y + size.y - frame_y - pad);
    const float sidebar_x = frame_x + frame_w + pad;
    const float sidebar_y = frame_y;

    DrawInteractionPreview(renderer, frame_x, frame_y, frame_w, frame_h);

    renderer.drawRect({sidebar_x - 1, sidebar_y - 1},
                      {sidebar_w + 2, frame_h + 2}, UITheme::Colors::DividerLine);
    renderer.drawRect({sidebar_x, sidebar_y},
                      {sidebar_w, frame_h}, UITheme::Colors::PanelBg);

    float line_y = sidebar_y + 9.0f;
    auto line = [&](const std::string& text,
                    uint32_t color = UITheme::Colors::TextPrimary) {
        if (line_y <= sidebar_y + frame_h - 16.0f)
            renderer.drawText({sidebar_x + 9.0f, line_y}, text, color);
        line_y += 17.0f;
    };
    auto gap = [&] { line_y += 7.0f; };

    line("Local Hand Interaction", UITheme::Colors::TextHeader);
    line("OFFLINE ONLY - no network fallback", UITheme::Colors::Success);
    if (!have_interaction_snapshot_) {
        line("Waiting for worker status...", UITheme::Colors::Warning);
        return;
    }

    const auto& s = interaction_snapshot_view_.snapshot;
    uint32_t state_color = UITheme::Colors::Warning;
    if (s.backend_state == PE::PhysicalHandGestureBackendState::Ready)
        state_color = UITheme::Colors::Success;
    else if (s.backend_state == PE::PhysicalHandGestureBackendState::Failed)
        state_color = UITheme::Colors::Danger;
    line(std::string("State: ") + FormatHandBackendState(s.backend_state), state_color);
    line(std::string("Enabled: ") + (s.enabled ? "yes" : "no")
         + "   worker: " + (s.worker_running ? (s.worker_busy ? "busy" : "idle") : "stopped"));
    line(std::string("Metrics transport: ")
         + (s.telemetry_disabled ? "disabled" : "NOT VERIFIED"),
         s.telemetry_disabled ? UITheme::Colors::Success : UITheme::Colors::Danger);
    line("Backend: " + CompactPath(s.backend_name + " " + s.backend_version));
    line("Model: " + CompactPath(s.model_path));
    line("Detail: " + CompactPath(s.status_detail));
    if (!s.last_error.empty())
        line("Error: " + CompactPath(s.last_error), UITheme::Colors::Danger);
    gap();

    line("Frames submitted: " + std::to_string(s.frames_submitted));
    line("Frames processed: " + std::to_string(s.frames_processed));
    line("Queue replacements: " + std::to_string(s.frames_replaced));
    line("Cadence skipped: " + std::to_string(s.frames_cadence_skipped));
    line("Inference failures: " + std::to_string(s.inference_failures));
    line("Inference: " + FormatDouble(s.last_inference_ms, 1)
         + " ms  p95 " + FormatDouble(s.p95_inference_ms, 1) + " ms");
    line("Cadence: " + FormatDouble(s.effective_target_fps, 1)
         + "/" + FormatDouble(s.configured_target_fps, 1) + " FPS");
    line("Result frame: " + std::to_string(s.source_frame_counter));
    if (s.published_steady_ns != 0) {
        const uint64_t now_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        const double age_ms = now_ns >= s.published_steady_ns
            ? static_cast<double>(now_ns - s.published_steady_ns) / 1.0e6 : 0.0;
        line("Status age: " + FormatDouble(age_ms, 0) + " ms");
    }
    gap();

    const auto control = PE::GetPhysicalGestureControlStatus();
    line(std::string("Controller: ") + (control.enabled ? "enabled" : "disabled")
         + (control.dry_run ? " (DRY RUN)" : ""),
         control.enabled ? UITheme::Colors::TextPrimary : UITheme::Colors::TextMuted);
    line(std::string("Armed: ") + (control.armed ? "YES" : "no"),
         control.armed ? UITheme::Colors::Success : UITheme::Colors::Warning);
    line(std::string("Cursor: ") +
         (control.custom_cursor_active ? "armed circle" : "system arrow"),
         control.custom_cursor_active ? UITheme::Colors::Success
                                      : UITheme::Colors::TextSecondary);
    if (!control.cursor_error.empty())
        line("Cursor error: " + CompactPath(control.cursor_error),
             UITheme::Colors::Danger);
    line("Stable event: " + (control.stable_gesture.empty()
         ? std::string("none") : control.stable_gesture));
    line("Events/actions: " + std::to_string(control.events_emitted)
         + "/" + std::to_string(control.actions_executed));
    line(std::string("Pointer: ") + (control.pointer_active ? "active" : "idle")
         + "  lock " + (control.hand_lock_active ? "yes" : "no")
         + "  " + FormatDouble(control.pointer_sample_hz, 1) + " Hz",
         control.pointer_active ? UITheme::Colors::Success
                                : UITheme::Colors::TextSecondary);
    line("Raw " + FormatDouble(control.pointer_raw_x, 3) + ","
         + FormatDouble(control.pointer_raw_y, 3) + "  filtered "
         + FormatDouble(control.pointer_filtered_x, 3) + ","
         + FormatDouble(control.pointer_filtered_y, 3));
    line("Samples/moves/outliers: " + std::to_string(control.pointer_samples) + "/"
         + std::to_string(control.pointer_moves_emitted) + "/"
         + std::to_string(control.pointer_outliers_rejected));
    line("Pinch: " + (control.pinch_tracking
         ? FormatDouble(control.pinch_distance_ratio, 3)
         : std::string("not tracked"))
         + (control.pinch_closed ? " CLOSED" : "")
         + "  clicks " + std::to_string(control.pinch_clicks_emitted),
         control.pinch_closed ? UITheme::Colors::Success
                              : UITheme::Colors::TextSecondary);
    line("Classifier bypass/reacquire: "
         + std::to_string(control.pointer_classifier_bypass_frames) + "/"
         + std::to_string(control.hand_reacquisitions));
    if (control.actions_previewed != 0)
        line("Dry-run previews: " + std::to_string(control.actions_previewed),
             UITheme::Colors::Warning);
    if (!control.last_action.empty())
        line("Last action: " + CompactPath(control.last_action));
    if (!control.last_block_reason.empty())
        line("Blocked: " + CompactPath(control.last_block_reason),
             UITheme::Colors::Warning);
    if (!control.last_error.empty())
        line("Control error: " + CompactPath(control.last_error),
             UITheme::Colors::Danger);
    gap();

    line("Hands: " + std::to_string(s.hands.size()), UITheme::Colors::TextHeader);
    for (size_t i = 0; i < s.hands.size(); ++i) {
        const auto& hand = s.hands[i];
        line("#" + std::to_string(i + 1) + " " + FormatHandedness(hand.handedness)
             + " " + FormatDouble(hand.handedness_confidence * 100.0, 0) + "%");
        line("  gesture: "
             + (hand.gesture_label.empty() ? std::string("none") : hand.gesture_label)
             + " " + FormatDouble(hand.gesture_confidence * 100.0, 0) + "%",
             UITheme::Colors::TextValue);
        line("  landmarks: " + std::to_string(hand.landmark_count) + "/21");
    }
}

void UIPhysicalEnvironmentPanel::DrawGestureBindingsEditor(
    OverlayRenderer& renderer)
{
    const float pad = 12.0f;
    const float toolbar_h = 32.0f;
    const float pane_top =
        position.y + titleBarHeight + kTabBarHeight + toolbar_h + pad;
    const float sidebar_w = 350.0f;
    const float frame_x = position.x + pad;
    const float frame_y = pane_top;
    const float frame_w = std::max(64.0f, size.x - sidebar_w - pad * 3.0f);
    const float frame_h = std::max(
        64.0f, position.y + size.y - frame_y - pad);
    const float sidebar_x = frame_x + frame_w + pad;

    // Bindings customize the live result, so keep the camera and landmark
    // preview present while the right-hand diagnostics pane becomes an editor.
    DrawInteractionPreview(renderer, frame_x, frame_y, frame_w, frame_h);
    renderer.drawRect({sidebar_x - 1.0f, frame_y - 1.0f},
                      {sidebar_w + 2.0f, frame_h + 2.0f},
                      UITheme::Colors::DividerLine);
    renderer.drawRect({sidebar_x, frame_y}, {sidebar_w, frame_h},
                      UITheme::Colors::PanelBg);

    const auto config = PE::GetPhysicalGestureControlConfig();
    const float inner_x = sidebar_x + 8.0f;
    const float inner_w = sidebar_w - 16.0f;
    const float list_y = pane_top + 30.0f;
    const float list_h = 116.0f;
    const float list_buttons_y = list_y + list_h + 6.0f;
    const float editor_y = list_buttons_y + 52.0f;

    renderer.drawText({inner_x + 2.0f, pane_top + 8.0f},
        "Bindings (" + std::to_string(config.bindings.size()) + ")",
        UITheme::Colors::TextHeader);
    if (interaction_binding_list_)
        interaction_binding_list_->drawOverlay(renderer, position);
    if (interaction_add_binding_btn_)
        interaction_add_binding_btn_->drawOverlay(renderer, position);
    if (interaction_delete_binding_btn_)
        interaction_delete_binding_btn_->drawOverlay(renderer, position);
    if (interaction_defaults_btn_)
        interaction_defaults_btn_->drawOverlay(renderer, position);

    std::string selected_title = "Selected binding";
    if (interaction_selected_binding_ < config.bindings.size()) {
        const auto& binding = config.bindings[interaction_selected_binding_];
        selected_title = binding.binding_id + " - " + binding.gesture_label;
    }
    renderer.drawText({inner_x + 2.0f, editor_y - 22.0f},
                      CompactPath(selected_title, 38),
                      UITheme::Colors::TextHeader);

    if (interaction_action_select_)
        interaction_action_select_->drawOverlay(renderer, position);
    if (interaction_trigger_select_)
        interaction_trigger_select_->drawOverlay(renderer, position);
    if (interaction_hand_select_)
        interaction_hand_select_->drawOverlay(renderer, position);

    renderer.drawText({inner_x, editor_y + 104.0f}, "Gesture label",
                      UITheme::Colors::TextSecondary);
    renderer.drawText({inner_x, editor_y + 148.0f}, "hold ms",
                      UITheme::Colors::TextSecondary);
    renderer.drawText({inner_x + 108.0f, editor_y + 148.0f}, "cooldown",
                      UITheme::Colors::TextSecondary);
    renderer.drawText({inner_x + 226.0f, editor_y + 148.0f}, "priority",
                      UITheme::Colors::TextSecondary);

    const std::array<std::shared_ptr<UIInputBox>, 4> boxes{{
        interaction_gesture_box_, interaction_hold_box_,
        interaction_cooldown_box_, interaction_priority_box_
    }};
    for (const auto& box : boxes)
        if (box) box->drawOverlay(renderer, position);
    const std::array<std::shared_ptr<UIButton>, 4> editor_buttons{{
        interaction_use_live_btn_, interaction_binding_enabled_btn_,
        interaction_requires_arm_btn_, interaction_apply_binding_btn_
    }};
    for (const auto& button : editor_buttons)
        if (button) button->drawOverlay(renderer, position);

    const auto issues = PE::ValidatePhysicalGestureBindings(config);
    const float info_y = editor_y + 240.0f;
    renderer.drawText({inner_x, info_y},
        "Mouse actions require arming; priority resolves overlap.",
        UITheme::Colors::TextSecondary);
    if (!interaction_binding_status_.empty()) {
        const bool problem = interaction_binding_status_.find("failed") != std::string::npos ||
            interaction_binding_status_.find("Conflict") != std::string::npos ||
            interaction_binding_status_.find("warning") != std::string::npos ||
            interaction_binding_status_.find("required") != std::string::npos;
        renderer.drawText({inner_x, info_y + 24.0f},
            CompactPath(interaction_binding_status_, 40),
            problem ? UITheme::Colors::Warning : UITheme::Colors::Success);
    }
    if (!issues.empty()) {
        renderer.drawText({inner_x, info_y + 44.0f},
            CompactPath(issues.front(), 40), UITheme::Colors::Warning);
    }

    // Expanded dropdown lists must be emitted last so they remain above fields.
    const std::array<std::shared_ptr<UIDropdown>, 3> dropdowns{{
        interaction_action_select_, interaction_trigger_select_,
        interaction_hand_select_
    }};
    for (const auto& dropdown : dropdowns) {
        if (dropdown && dropdown->isExpanded())
            dropdown->drawExpandedList(renderer, position);
    }
}

// ============================================================================
//  Spatial tab (Stage-3 — depth + grounded entities)
// ============================================================================

void UIPhysicalEnvironmentPanel::HandleToggleSpatialDepthEstimator() {
    try {
        auto f = PE::GetPhysicalSpatialGroundingEnableFlags();
        f.depth_estimator = !f.depth_estimator;
        PE::RequestSetPhysicalSpatialGroundingEnableFlags(f);
        RefreshSpatialEnableButtonLabelsFromSubsystem();
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
            std::string("HandleToggleSpatialDepthEstimator threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::HandleToggleSpatialGrounder() {
    try {
        auto f = PE::GetPhysicalSpatialGroundingEnableFlags();
        f.spatial_grounder = !f.spatial_grounder;
        PE::RequestSetPhysicalSpatialGroundingEnableFlags(f);
        RefreshSpatialEnableButtonLabelsFromSubsystem();
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
            std::string("HandleToggleSpatialGrounder threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::RefreshSpatialEnableButtonLabelsFromSubsystem() {
    try {
        const auto f = PE::GetPhysicalSpatialGroundingEnableFlags();
        if (spatial_btn_depth_)
            spatial_btn_depth_->setText(f.depth_estimator
                                         ? std::string(" Depth: on ")
                                         : std::string(" Depth: off "));
        if (spatial_btn_ground_)
            spatial_btn_ground_->setText(f.spatial_grounder
                                          ? std::string(" Ground: on ")
                                          : std::string(" Ground: off "));
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
            std::string("RefreshSpatialEnableButtonLabelsFromSubsystem threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::UpdateSpatialTab(const InputState& input, float dt) {
    (void)dt;

    // Pull the latest grounded result; throws are surfaced to the log but
    // do NOT block the rest of the tab from rendering.
    try {
        if (PE::PhysicalSpatialGroundingBus::Instance()
                .PullLatestPhysicalSpatialGroundingResultsView(
                    spatial_results_view_, spatial_last_seen_counter_)) {
            have_any_spatial_results_ = true;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("UpdateSpatialTab: results pull threw: ") + e.what());
    }

    // Toolbar layout — two enable buttons across the top of the tab.
    const float content_top = position.y + titleBarHeight + kTabBarHeight + 8.0f;
    const float bx          = position.x + 16.0f;
    const float by          = content_top;
    const float bh          = 24.0f;
    if (spatial_btn_depth_) {
        spatial_btn_depth_->setSize(110.0f, bh);
        spatial_btn_depth_->setPosition(bx, by);
        spatial_btn_depth_->update(input, dt);
    }
    if (spatial_btn_ground_) {
        spatial_btn_ground_->setSize(120.0f, bh);
        spatial_btn_ground_->setPosition(bx + 118.0f, by);
        spatial_btn_ground_->update(input, dt);
    }

    // Build the heatmap BGR Mat ONCE per new source frame. The Bus pull
    // already gated on counter, so we further key by source_frame_counter
    // to avoid rebuilding the colormap during idle redraws.
    if (have_any_spatial_results_
        && !spatial_results_view_.results.depth_map.empty()
        && spatial_results_view_.results.source_frame_counter != spatial_heatmap_source_id_) {
        try {
            const cv::Mat& inv = spatial_results_view_.results.depth_map.inverse_depth_image;
            if (inv.type() == CV_32FC1 && inv.cols > 0 && inv.rows > 0) {
                cv::Mat u8;
                inv.convertTo(u8, CV_8UC1, 255.0);
                cv::applyColorMap(u8, spatial_heatmap_bgr_, cv::COLORMAP_INFERNO);
                spatial_heatmap_source_id_ = spatial_results_view_.results.source_frame_counter;
            }
        } catch (const std::exception& e) {
            LOG_ERROR(kPanelLogTag,
                std::string("UpdateSpatialTab: heatmap build threw: ") + e.what());
        }
    }
}

void UIPhysicalEnvironmentPanel::DrawSpatialDepthHeatmap(
    OverlayRenderer& renderer,
    const PE::PhysicalDepthMap& dmap,
    uint64_t source_id,
    float frame_x, float frame_y, float frame_w, float frame_h)
{
    if (dmap.empty() || spatial_heatmap_bgr_.empty()) {
        renderer.drawText({frame_x + 12, frame_y + 12},
            "(no depth map yet — load an ONNX MiDaS model via "
            "RequestConfigurePhysicalMonocularDepthEstimator)",
            UITheme::Colors::TextSecondary);
        return;
    }
    DrawBgrFrameIntoOverlay(renderer, spatial_heatmap_bgr_, source_id, false,
                            frame_x, frame_y, frame_w, frame_h,
                            spatial_heatmap_blit_cache_);
}

void UIPhysicalEnvironmentPanel::DrawSpatialGroundedEntitiesOverlay(
    OverlayRenderer& renderer,
    const std::vector<PE::PhysicalGroundedEntity>& entities,
    int blit_x, int blit_y, int blit_w, int blit_h,
    int model_w, int model_h)
{
    if (entities.empty() || model_w <= 0 || model_h <= 0
        || blit_w <= 0 || blit_h <= 0) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(model_w);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(model_h);
    for (const auto& g : entities) {
        const float bx = blit_x + g.model_box.x      * sx;
        const float by = blit_y + g.model_box.y      * sy;
        const float bw =          g.model_box.width  * sx;
        const float bh =          g.model_box.height * sy;
        // 2-px outline (top, bottom, left, right) — matches Detector style.
        const auto col = g.path_blocked ? UITheme::Colors::Danger
                                        : UITheme::Colors::Primary;
        renderer.drawRect({bx, by},          {bw, 2.0f}, col);
        renderer.drawRect({bx, by + bh-2.0f},{bw, 2.0f}, col);
        renderer.drawRect({bx, by},          {2.0f, bh}, col);
        renderer.drawRect({bx + bw-2.0f, by},{2.0f, bh}, col);

        // Compose label: id, class, range, support, motion, blocked.
        std::stringstream ss;
        ss << "#" << g.track_id;
        if (!g.class_label.empty()) ss << " " << g.class_label;
        if (g.units == PE::DepthUnits::Meters) {
            ss << " | " << std::fixed << std::setprecision(2)
               << g.range_value_meters << "m";
        } else {
            ss << " | rel=" << std::fixed << std::setprecision(2) << g.range_value;
        }
        ss << " conf=" << std::fixed << std::setprecision(2) << g.range_confidence;
        ss << " | " << PE::DescribePhysicalSupportSurfaceClass(g.support_surface);
        ss << " | " << PE::DescribePhysicalEntityMotionState(g.motion_state);
        if (g.path_blocked) ss << " | BLOCKED";
        renderer.drawText({bx, by - 14.0f}, ss.str(), col);
    }
}

void UIPhysicalEnvironmentPanel::DrawSpatialSidebar(
    OverlayRenderer& renderer, float x, float y, float w, float h,
    const PE::PhysicalSpatialGroundingResults& r, bool have_results)
{
    (void)w; (void)h;
    auto state_text = [](PE::PhysicalImageOperatorState s) -> const char* {
        switch (s) {
            case PE::PhysicalImageOperatorState::NoModelConfigured: return "no model";
            case PE::PhysicalImageOperatorState::ModelLoaded:       return "ready";
            case PE::PhysicalImageOperatorState::ModelLoadFailed:   return "LOAD-FAIL";
            case PE::PhysicalImageOperatorState::InferenceFailed:   return "INF-FAIL";
        }
        return "?";
    };

    float ly = y;
    renderer.drawText({x, ly}, "Stage-3: Spatial Grounding",
                      UITheme::Colors::TextPrimary); ly += 18.0f;

    if (!have_results) {
        renderer.drawText({x, ly},
            "(no result on bus yet — frame + tracker must publish)",
            UITheme::Colors::TextSecondary);
        return;
    }

    {
        std::stringstream ss;
        ss << "frame_ctr=" << r.source_frame_counter
           << "  perc_ctr=" << r.source_perception_results_counter;
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
        ly += 16.0f;
    }
    {
        std::stringstream ss;
        ss << "depth: " << state_text(r.depth_estimator_state)
           << " | " << std::fixed << std::setprecision(2)
           << r.last_depth_inference_ms << "ms"
           << " | n=" << r.depth_inference_count;
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
        ly += 16.0f;
    }
    if (!r.depth_estimator_last_error.empty()) {
        renderer.drawText({x, ly},
            "depth err: " + r.depth_estimator_last_error,
            UITheme::Colors::Warning);
        ly += 16.0f;
    }
    {
        std::stringstream ss;
        ss << "ground: " << state_text(r.grounder_state)
           << " | " << std::fixed << std::setprecision(2)
           << r.last_grounding_ms << "ms"
           << " | n=" << r.grounding_count;
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
        ly += 16.0f;
    }
    if (!r.grounder_last_error.empty()) {
        renderer.drawText({x, ly},
            "ground err: " + r.grounder_last_error,
            UITheme::Colors::Warning);
        ly += 16.0f;
    }
    {
        std::stringstream ss;
        ss << "depth_map: ";
        if (r.depth_map.empty()) {
            ss << "(empty)";
        } else {
            ss << r.depth_map.map_width << "x" << r.depth_map.map_height
               << " units=" << PE::DescribeDepthUnits(r.depth_map.units)
               << " inv=[" << std::fixed << std::setprecision(3)
               << r.depth_map.raw_inverse_depth_min << ","
               << r.depth_map.raw_inverse_depth_max << "]";
        }
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
        ly += 16.0f;
    }
    {
        std::stringstream ss;
        ss << "grounded entities: " << r.grounded_entities.size();
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextPrimary);
        ly += 18.0f;
    }
    int blocked = 0, moving = 0, on_floor = 0, on_table = 0;
    for (const auto& g : r.grounded_entities) {
        if (g.path_blocked) ++blocked;
        if (g.motion_state == PE::PhysicalEntityMotionState::Moving) ++moving;
        if (g.support_surface == PE::PhysicalSupportSurfaceClass::Floor) ++on_floor;
        if (g.support_surface == PE::PhysicalSupportSurfaceClass::Table) ++on_table;
    }
    {
        std::stringstream ss;
        ss << "  blocking path: " << blocked
           << "  moving: " << moving
           << "  on floor: " << on_floor
           << "  on table: " << on_table;
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
        ly += 18.0f;
    }
    // Per-entity detail (truncated so we don't blow past the sidebar).
    constexpr size_t kMaxRows = 12;
    const size_t n = std::min(kMaxRows, r.grounded_entities.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& g = r.grounded_entities[i];
        std::stringstream ss;
        ss << "#" << g.track_id;
        if (!g.class_label.empty()) ss << " " << g.class_label;
        if (g.units == PE::DepthUnits::Meters)
            ss << " | " << std::fixed << std::setprecision(2)
               << g.range_value_meters << "m";
        else
            ss << " | r=" << std::fixed << std::setprecision(2) << g.range_value;
        ss << " | " << PE::DescribePhysicalSupportSurfaceClass(g.support_surface);
        ss << " | " << PE::DescribePhysicalEntityMotionState(g.motion_state);
        if (g.path_blocked) ss << "  BLOCKED";
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
        ly += 14.0f;
    }
    if (r.grounded_entities.size() > n) {
        std::stringstream ss;
        ss << "  ... (+" << (r.grounded_entities.size() - n) << " more)";
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
    }
}

void UIPhysicalEnvironmentPanel::DrawSpatialTab(OverlayRenderer& renderer) {
    // Toolbar buttons (already laid out + ticked in UpdateSpatialTab).
    if (spatial_btn_depth_)  spatial_btn_depth_->drawOverlay(renderer, position);
    if (spatial_btn_ground_) spatial_btn_ground_->drawOverlay(renderer, position);

    const float pad         = 16.0f;
    const float toolbar_h   = 32.0f;
    const float content_top = position.y + titleBarHeight + kTabBarHeight + toolbar_h + 8.0f;

    // Heatmap takes left ~62% of the panel; sidebar gets the rest.
    const float total_w  = size.x - 2.0f * pad;
    const float frame_w  = total_w * 0.62f;
    const float frame_h  = size.y - (content_top - position.y) - pad;
    const float frame_x  = position.x + pad;
    const float frame_y  = content_top;

    // Sidebar.
    const float sidebar_x = frame_x + frame_w + 8.0f;
    const float sidebar_y = content_top;
    const float sidebar_w = size.x - pad - (sidebar_x - position.x);
    const float sidebar_h = frame_h;

    DrawSpatialDepthHeatmap(renderer,
                            spatial_results_view_.results.depth_map,
                            spatial_results_view_.results.source_frame_counter,
                            frame_x, frame_y, frame_w, frame_h);

    // Compute the actual blit rect (heatmap + bounding-box overlay must
    // share the same letterboxed geometry).
    if (have_any_spatial_results_
        && !spatial_results_view_.results.depth_map.empty()
        && spatial_heatmap_blit_cache_.out_w > 0
        && spatial_heatmap_blit_cache_.out_h > 0) {
        const int out_w = spatial_heatmap_blit_cache_.out_w;
        const int out_h = spatial_heatmap_blit_cache_.out_h;
        const int out_x = static_cast<int>(frame_x + (frame_w - out_w) * 0.5f);
        const int out_y = static_cast<int>(frame_y + (frame_h - out_h) * 0.5f);
        DrawSpatialGroundedEntitiesOverlay(
            renderer,
            spatial_results_view_.results.grounded_entities,
            out_x, out_y, out_w, out_h,
            spatial_results_view_.results.model_image_width,
            spatial_results_view_.results.model_image_height);
    }

    // Sidebar background + content.
    renderer.drawRect({sidebar_x - 1, sidebar_y - 1},
                      {sidebar_w + 2, sidebar_h + 2}, UITheme::Colors::DividerLine);
    renderer.drawRect({sidebar_x, sidebar_y},
                      {sidebar_w, sidebar_h}, UITheme::Colors::PanelBg);
    DrawSpatialSidebar(renderer, sidebar_x + 8, sidebar_y + 8,
                       sidebar_w - 16, sidebar_h - 16,
                       spatial_results_view_.results, have_any_spatial_results_);
}

// ============================================================================
//  World tab (Stage-4) — the model's view of the scene
// ============================================================================
//
// Displays exactly what GRIM sees when it reasons over the environment:
// identity-keyed entities (object_id), per-entity visibility/depth/text and
// inter-entity relations. Pulls from PhysicalWorldStateBus (already-fused);
// does NOT re-fuse from upstream buses.
//
// Visibility legend (matches the colour code used in the overlay):
//   green   = Visible
//   orange  = Occluded (with line back to the occluder)
//   gray    = Coasting (no fresh visual evidence; tracker extrapolation)
// ----------------------------------------------------------------------------

namespace {

uint32_t WorldVisibilityColor(PE::PhysicalEntityVisibility v) {
    switch (v) {
        case PE::PhysicalEntityVisibility::Visible:  return 0xFF22DD66u; // green
        case PE::PhysicalEntityVisibility::Occluded: return 0xFFFF9933u; // orange
        case PE::PhysicalEntityVisibility::Coasting: return 0xFF888888u; // gray
        case PE::PhysicalEntityVisibility::Unknown:  return 0xFFCCCCCCu;
    }
    return 0xFFCCCCCCu;
}

} // anonymous

void UIPhysicalEnvironmentPanel::UpdateWorldTab(const InputState& /*input*/, float /*dt*/) {
    // Pull latest published snapshot. The world-state loop publishes at most
    // once per matched (perception, grounding) pair; a Pull that returns
    // false simply means "nothing new" — keep the previous view.
    try {
        const bool advanced =
            PE::PhysicalWorldStateBus::Instance()
                .PullLatestPhysicalWorldStateSnapshotView(
                    world_snapshot_view_, world_last_seen_counter_);
        if (advanced) {
            world_last_seen_counter_ =
                world_snapshot_view_.snapshot.source_frame_counter;
            have_any_world_results_  = true;
        }
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("UpdateWorldTab: snapshot pull threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::DrawWorldEntitiesOverlay(
    OverlayRenderer& renderer,
    const PE::PhysicalWorldStateSnapshot& snap,
    int blit_x, int blit_y, int blit_w, int blit_h)
{
    if (snap.model_image_width <= 0 || snap.model_image_height <= 0) return;
    if (blit_w <= 0 || blit_h <= 0) return;

    // model → blit transform (model coords are the snapshot's authoritative
    // coordinate system; we rendered the raw frame letterboxed to the blit
    // rect, so we must transform via raw → letterbox, NOT model → letterbox.
    // The world-state snapshot also stores raw_box for exactly this reason).
    if (snap.raw_image_width <= 0 || snap.raw_image_height <= 0) return;
    const float sx = static_cast<float>(blit_w) / static_cast<float>(snap.raw_image_width);
    const float sy = static_cast<float>(blit_h) / static_cast<float>(snap.raw_image_height);

    auto raw_pt = [&](float rx, float ry) {
        return std::pair<float,float>(
            static_cast<float>(blit_x) + rx * sx,
            static_cast<float>(blit_y) + ry * sy);
    };

    // Pre-build an id → centre table for relation lines.
    std::unordered_map<uint64_t, std::pair<float,float>> id_to_screen_centre;
    id_to_screen_centre.reserve(snap.entities.size());
    for (const auto& e : snap.entities) {
        id_to_screen_centre[e.object_id] = raw_pt(e.raw_centre.x, e.raw_centre.y);
    }

    // 1. Relation lines (drawn first so boxes sit on top).
    for (const auto& e : snap.entities) {
        const auto self_pt = id_to_screen_centre[e.object_id];
        for (const auto& rel : e.relations) {
            const auto it = id_to_screen_centre.find(rel.other_object_id);
            if (it == id_to_screen_centre.end()) continue;
            uint32_t color;
            switch (rel.kind) {
                case PE::PhysicalEntityRelationKind::Contains:
                case PE::PhysicalEntityRelationKind::ContainedBy:
                case PE::PhysicalEntityRelationKind::Overlaps:
                    color = 0x66FFFFFFu; break;
                case PE::PhysicalEntityRelationKind::NearerThan:
                case PE::PhysicalEntityRelationKind::FartherThan:
                    color = 0x6633CCFFu; break;
                default:
                    color = 0x4488AAFFu; break;
            }
            renderer.drawLine({self_pt.first, self_pt.second},
                              {it->second.first, it->second.second},
                              color, 1.0f);
        }
    }

    // 2. Entity boxes (colour by visibility) + occluder linkage.
    for (const auto& e : snap.entities) {
        const uint32_t col = WorldVisibilityColor(e.visibility);
        const auto tl = raw_pt(e.raw_box.x, e.raw_box.y);
        const float w_px = e.raw_box.width  * sx;
        const float h_px = e.raw_box.height * sy;
        // Top + bottom + left + right edges as 1-px rects (drawRect = filled).
        renderer.drawRect({tl.first,            tl.second           }, {w_px, 1.0f}, col);
        renderer.drawRect({tl.first,            tl.second + h_px - 1}, {w_px, 1.0f}, col);
        renderer.drawRect({tl.first,            tl.second           }, {1.0f, h_px}, col);
        renderer.drawRect({tl.first + w_px - 1, tl.second           }, {1.0f, h_px}, col);

        // Label band above the box.
        std::stringstream ss;
        ss << "#" << e.object_id;
        if (!e.class_label.empty()) ss << " " << e.class_label;
        ss << "  " << PE::DescribePhysicalEntityVisibility(e.visibility);
        if (e.has_depth) {
            ss << "  ";
            if (e.depth_units == PE::DepthUnits::Meters)
                ss << std::fixed << std::setprecision(2) << e.range_value_meters << "m";
            else
                ss << "r=" << std::fixed << std::setprecision(2) << e.range_value;
        }
        const float label_y = std::max(static_cast<float>(blit_y),
                                       tl.second - 14.0f);
        renderer.drawText({tl.first, label_y}, ss.str(), col);

        // Occlusion linkage (orange line from this box centre to occluder).
        if (e.occluded_by_object_id != 0) {
            const auto it = id_to_screen_centre.find(e.occluded_by_object_id);
            if (it != id_to_screen_centre.end()) {
                const auto self_pt = id_to_screen_centre[e.object_id];
                renderer.drawLine({self_pt.first, self_pt.second},
                                  {it->second.first, it->second.second},
                                  0xCCFF9933u, 2.0f);
            }
        }
    }
}

void UIPhysicalEnvironmentPanel::DrawWorldEntitiesSidebar(
    OverlayRenderer& renderer, float x, float y, float w, float h,
    const PE::PhysicalWorldStateSnapshot& snap, bool have_results)
{
    (void)w; (void)h;
    float ly = y;

    if (!have_results) {
        renderer.drawText({x, ly},
                          "World-state bus: no snapshots published yet.",
                          UITheme::Colors::TextSecondary);
        ly += 16.0f;
        renderer.drawText({x, ly},
                          "Stage-4 publishes only on matched (perception, grounding) frames.",
                          UITheme::Colors::TextSecondary);
        return;
    }

    {
        std::stringstream ss;
        ss << "frame=" << snap.source_frame_counter
           << "  perc=" << snap.source_perception_results_counter
           << "  ground=" << snap.source_grounding_results_counter;
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextPrimary);
        ly += 16.0f;
    }
    {
        std::stringstream ss;
        ss << "model=" << snap.model_image_width << "x" << snap.model_image_height
           << "  raw=" << snap.raw_image_width  << "x" << snap.raw_image_height;
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
        ly += 16.0f;
    }
    {
        std::stringstream ss;
        ss << "entities: " << snap.entities.size()
           << "  visible=" << snap.num_visible_entities
           << "  occluded=" << snap.num_occluded_entities
           << "  coasting=" << snap.num_coasting_entities;
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextPrimary);
        ly += 18.0f;
    }

    // Per-entity rows (truncated so we stay inside the sidebar).
    constexpr size_t kMaxRows = 14;
    const size_t n = std::min(kMaxRows, snap.entities.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& e = snap.entities[i];
        const uint32_t col = WorldVisibilityColor(e.visibility);
        std::stringstream ss;
        ss << "#" << e.object_id;
        if (!e.class_label.empty()) ss << " " << e.class_label;
        ss << " | " << PE::DescribePhysicalEntityVisibility(e.visibility);
        ss << " | conf=" << std::fixed << std::setprecision(2) << e.confidence;
        if (e.has_depth) {
            ss << " | ";
            if (e.depth_units == PE::DepthUnits::Meters)
                ss << std::fixed << std::setprecision(2) << e.range_value_meters << "m";
            else
                ss << "r=" << std::fixed << std::setprecision(2) << e.range_value;
        }
        ss << " | rels=" << e.relations.size();
        renderer.drawText({x, ly}, ss.str(), col);
        ly += 14.0f;

        if (!e.text_on_object.empty()) {
            std::stringstream ts;
            ts << "    text:";
            for (size_t k = 0; k < e.text_on_object.size() && k < 3; ++k)
                ts << " \"" << e.text_on_object[k] << "\"";
            if (e.text_on_object.size() > 3) ts << " ...";
            renderer.drawText({x, ly}, ts.str(), UITheme::Colors::TextSecondary);
            ly += 14.0f;
        }
        if (e.occluded_by_object_id != 0) {
            std::stringstream os;
            os << "    occluded by #" << e.occluded_by_object_id
               << "  visible_frac=" << std::fixed << std::setprecision(2)
               << e.visible_area_fraction;
            renderer.drawText({x, ly}, os.str(), UITheme::Colors::TextSecondary);
            ly += 14.0f;
        }
    }
    if (snap.entities.size() > n) {
        std::stringstream ss;
        ss << "  ... (+" << (snap.entities.size() - n) << " more)";
        renderer.drawText({x, ly}, ss.str(), UITheme::Colors::TextSecondary);
    }
}

void UIPhysicalEnvironmentPanel::DrawWorldTab(OverlayRenderer& renderer) {
    const float pad         = 16.0f;
    const float content_top = position.y + titleBarHeight + kTabBarHeight + 8.0f;

    // Frame on the left (~62%), sidebar on the right.
    const float total_w = size.x - 2.0f * pad;
    const float frame_w = total_w * 0.62f;
    const float frame_h = size.y - (content_top - position.y) - pad;
    const float frame_x = position.x + pad;
    const float frame_y = content_top;

    const float sidebar_x = frame_x + frame_w + 8.0f;
    const float sidebar_y = content_top;
    const float sidebar_w = size.x - pad - (sidebar_x - position.x);
    const float sidebar_h = frame_h;

    // Draw the latest raw frame as the canvas (Rule 20: if no frame is on
    // the bus, say so — no stub graphic).
    if (have_any_frame_ && !last_view_.raw_image.empty()) {
        DrawBgrFrameIntoOverlay(renderer,
                                last_view_.raw_image,
                                last_seen_counter_,
                                /*source_undistort=*/false,
                                frame_x, frame_y, frame_w, frame_h,
                                world_blit_cache_);

        if (have_any_world_results_
            && world_blit_cache_.out_w > 0 && world_blit_cache_.out_h > 0) {
            const int out_w = world_blit_cache_.out_w;
            const int out_h = world_blit_cache_.out_h;
            const int out_x = static_cast<int>(frame_x + (frame_w - out_w) * 0.5f);
            const int out_y = static_cast<int>(frame_y + (frame_h - out_h) * 0.5f);
            DrawWorldEntitiesOverlay(renderer,
                                     world_snapshot_view_.snapshot,
                                     out_x, out_y, out_w, out_h);
        }
    } else {
        renderer.drawRect({frame_x, frame_y}, {frame_w, frame_h},
                          UITheme::Colors::PanelBg);
        renderer.drawText({frame_x + 12.0f, frame_y + 12.0f},
                          "PhysicalFrameBus: no frame published yet.",
                          UITheme::Colors::TextSecondary);
    }

    // Sidebar background + content.
    renderer.drawRect({sidebar_x - 1, sidebar_y - 1},
                      {sidebar_w + 2, sidebar_h + 2}, UITheme::Colors::DividerLine);
    renderer.drawRect({sidebar_x, sidebar_y},
                      {sidebar_w, sidebar_h}, UITheme::Colors::PanelBg);
    DrawWorldEntitiesSidebar(renderer, sidebar_x + 8, sidebar_y + 8,
                             sidebar_w - 16, sidebar_h - 16,
                             world_snapshot_view_.snapshot,
                             have_any_world_results_);
}

// ============================================================================
//  Localization tab (Stage-5)
// ============================================================================

void UIPhysicalEnvironmentPanel::HandleResetLocalizationClicked() {
    try {
        PE::RequestResetPhysicalLocalization();
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("HandleResetLocalizationClicked: threw: ") + e.what());
    }
}

void UIPhysicalEnvironmentPanel::UpdateLocalizationTab(const InputState& input, float dt) {
    // Pull latest published snapshot. The localization loop publishes once
    // per processed frame; missing publishes simply leave the cached view
    // in place so the UI does not flicker.
    try {
        const bool advanced =
            PE::PhysicalLocalizationBus::Instance()
                .PullLatestPhysicalLocalizationSnapshotView(
                    loc_snapshot_view_, loc_last_seen_publish_sequence_);
        if (advanced) have_any_loc_snapshot_ = true;
    } catch (const std::exception& e) {
        LOG_ERROR(kPanelLogTag,
                  std::string("UpdateLocalizationTab: snapshot pull threw: ") + e.what());
    }

    const float pad = 12.0f;
    const float by  = position.y + titleBarHeight + kTabBarHeight + 8.0f;
    if (loc_reset_btn_) {
        loc_reset_btn_->setSize(120.0f, 26.0f);
        loc_reset_btn_->setPosition(position.x + size.x - pad - 120.0f, by);
        loc_reset_btn_->update(input, dt);
    }
}

void UIPhysicalEnvironmentPanel::DrawLocalizationTrajectoryMinimap(
    OverlayRenderer& renderer,
    float x, float y, float w, float h,
    const PE::PhysicalLocalizationSnapshot& snap)
{
    // Border + background.
    renderer.drawRect({x - 1, y - 1}, {w + 2, h + 2}, UITheme::Colors::DividerLine);
    renderer.drawRect({x, y}, {w, h}, UITheme::Colors::PanelBg);

    if (snap.trajectory_world_meters.empty()) {
        renderer.drawText({x + 8, y + 8},
                          "Trajectory: empty (waiting for Tracking state)",
                          UITheme::Colors::TextSecondary);
        return;
    }

    // Compute X/Z (top-down) bounds, padded so the camera dot does not sit on
    // the border. We project onto X (world east) and Z (world forward) which
    // is the canonical Nav2 / SLAM-Toolbox top-down convention.
    double min_x =  1e30, max_x = -1e30;
    double min_z =  1e30, max_z = -1e30;
    for (const auto& p : snap.trajectory_world_meters) {
        min_x = std::min(min_x, p.x); max_x = std::max(max_x, p.x);
        min_z = std::min(min_z, p.z); max_z = std::max(max_z, p.z);
    }
    if (max_x - min_x < 0.5) { const double m = 0.5 - (max_x - min_x); min_x -= m * 0.5; max_x += m * 0.5; }
    if (max_z - min_z < 0.5) { const double m = 0.5 - (max_z - min_z); min_z -= m * 0.5; max_z += m * 0.5; }
    const double range_x = max_x - min_x;
    const double range_z = max_z - min_z;

    // Square aspect: pick the larger range and apply it to both axes so
    // 1 metre on screen = 1 metre on screen on both axes (Rule 20: never
    // silently distort spatial data).
    const double range = std::max(range_x, range_z);
    const double cx = 0.5 * (min_x + max_x);
    const double cz = 0.5 * (min_z + max_z);
    const double half = range * 0.5 * 1.1;   // 10 % padding

    auto project = [&](double wx, double wz) -> std::pair<float,float> {
        const double u = (wx - (cx - half)) / (2.0 * half);   // 0..1, world-X right
        const double v = ((cz + half) - wz) / (2.0 * half);   // 0..1, world-Z up
        return { static_cast<float>(x + u * w),
                 static_cast<float>(y + v * h) };
    };

    // Polyline of the trajectory.
    auto prev = project(snap.trajectory_world_meters.front().x,
                        snap.trajectory_world_meters.front().z);
    for (size_t i = 1; i < snap.trajectory_world_meters.size(); ++i) {
        const auto cur = project(snap.trajectory_world_meters[i].x,
                                 snap.trajectory_world_meters[i].z);
        renderer.drawLine({prev.first, prev.second},
                          {cur.first,  cur.second},
                          UITheme::Colors::Primary);
        prev = cur;
    }

    // Current camera position dot.
    const auto here = project(snap.pose_world_camera.position_meters[0],
                              snap.pose_world_camera.position_meters[2]);
    renderer.drawRect({here.first - 3.0f, here.second - 3.0f},
                      {6.0f, 6.0f},
                      UITheme::Colors::TextPrimary);

    // Scale-bar in metres so the user can read the minimap.
    {
        std::ostringstream ss;
        ss << "scale: full window \u2248 " << std::fixed << std::setprecision(2)
           << (2.0 * half) << " m  (top-down X / Z)";
        renderer.drawText({x + 6, y + h - 18.0f}, ss.str(),
                          UITheme::Colors::TextSecondary);
    }
}

void UIPhysicalEnvironmentPanel::DrawLocalizationTab(OverlayRenderer& renderer) {
    if (loc_reset_btn_) loc_reset_btn_->drawOverlay(renderer, position);

    const float pad        = 12.0f;
    const float content_y  = position.y + titleBarHeight + kTabBarHeight + 44.0f;
    const float content_x  = position.x + pad;
    const float content_w  = size.x - 2.0f * pad;
    const float content_h  = position.y + size.y - content_y - pad;

    if (!have_any_loc_snapshot_) {
        renderer.drawText({content_x, content_y},
                          "PhysicalLocalizationBus: no snapshot published yet.",
                          UITheme::Colors::TextSecondary);
        renderer.drawText({content_x, content_y + 18.0f},
                          "(Waiting for Stage-5 lazy init. If this persists, calibrate the camera.)",
                          UITheme::Colors::TextSecondary);
        return;
    }

    const auto& snap = loc_snapshot_view_.snapshot;

    // Left column: text readouts. Right column: trajectory minimap.
    const float minimap_w = std::min(content_w * 0.45f, 360.0f);
    const float minimap_h = std::min(content_h - 8.0f, minimap_w);
    const float minimap_x = content_x + content_w - minimap_w;
    const float minimap_y = content_y;

    const float text_x = content_x;
    float       row_y  = content_y;
    const float row_h  = 18.0f;

    auto draw_row = [&](const std::string& s, uint32_t col = UITheme::Colors::TextPrimary) {
        renderer.drawText({text_x, row_y}, s, col);
        row_y += row_h;
    };

    {
        std::ostringstream ss;
        ss << "Tracking state : "
           << PE::DescribePhysicalLocalizationTrackingState(snap.tracking_state)
           << "   |   scale: "
           << PE::DescribePhysicalLocalizationPoseScaleState(snap.pose_scale_state);
        draw_row(ss.str());
    }
    if (!snap.tracking_reason.empty()) {
        draw_row("Reason         : " + snap.tracking_reason,
                 UITheme::Colors::TextSecondary);
    }
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << "Position (m)   : x=" << snap.pose_world_camera.position_meters[0]
           << "  y="                << snap.pose_world_camera.position_meters[1]
           << "  z="                << snap.pose_world_camera.position_meters[2];
        draw_row(ss.str());
    }
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << "YPR (rad)      : yaw="   << snap.pose_world_camera.yaw_pitch_roll_radians[0]
           << "  pitch="                << snap.pose_world_camera.yaw_pitch_roll_radians[1]
           << "  roll="                 << snap.pose_world_camera.yaw_pitch_roll_radians[2];
        draw_row(ss.str());
    }
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << "Quaternion wxyz: w=" << snap.pose_world_camera.quaternion_world_camera[0]
           << "  x="                << snap.pose_world_camera.quaternion_world_camera[1]
           << "  y="                << snap.pose_world_camera.quaternion_world_camera[2]
           << "  z="                << snap.pose_world_camera.quaternion_world_camera[3];
        draw_row(ss.str());
    }
    {
        const auto& v = snap.velocity_world_camera;
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(3)
           << "Velocity       : lin=(" << v.linear_world_per_sec[0]
           << ", "                     << v.linear_world_per_sec[1]
           << ", "                     << v.linear_world_per_sec[2]
           << ")  ang=("               << v.angular_radians_per_sec[0]
           << ", "                     << v.angular_radians_per_sec[1]
           << ", "                     << v.angular_radians_per_sec[2]
           << ")  dt="                 << v.sample_interval_seconds << "s";
        draw_row(ss.str(), UITheme::Colors::TextSecondary);
    }
    row_y += 4.0f;
    {
        std::ostringstream ss;
        ss << "VO metrics     : kp=" << snap.keypoints_detected_current_frame
           << "  matches="           << snap.matches_to_prior_frame
           << "  inliers="           << snap.essential_inliers
           << std::fixed << std::setprecision(2)
           << "  median_parallax="   << snap.median_parallax_pixels << "px"
           << "  reproj_err="        << snap.mean_reprojection_error_pixels << "px";
        draw_row(ss.str());
    }
    {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(2)
           << "Timing         : last_estimation=" << snap.last_estimation_ms << "ms"
           << "  estimation_count="               << snap.estimation_count
           << "  source_frame_counter="           << snap.source_frame_counter;
        draw_row(ss.str(), UITheme::Colors::TextSecondary);
    }
    {
        const auto& g = snap.occupancy_grid;
        std::ostringstream ss;
        ss << "Occupancy grid : " << g.cols << "x" << g.rows
           << " @ " << std::fixed << std::setprecision(3) << g.resolution_meters << "m"
           << "  observed_cells=" << g.total_cells_observed
           << "  updated_this_frame=" << g.cells_updated_this_frame;
        draw_row(ss.str(), UITheme::Colors::TextSecondary);
    }

    // Right side: top-down trajectory minimap.
    DrawLocalizationTrajectoryMinimap(renderer,
                                      minimap_x, minimap_y,
                                      minimap_w, minimap_h,
                                      snap);
}

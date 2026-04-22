#include "ui_physical_environment_panel.hpp"

#include "logger.hpp"
#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "perception/physical/PhysicalCalibrationPattern.hpp"
#include "perception/physical/PhysicalEnvironmentLogTag.hpp"
#include "perception/physical/PhysicalEnvironmentLoop.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

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
    position = { 200, 200 };
    size     = { 880, 680 };
    setBackground(UITheme::Colors::PanelBg);
    setBorder(UITheme::Colors::DividerLine);

    // ── Tab buttons ──
    tab_camera_btn_ = std::make_shared<UIButton>(" Camera ",
        [this]() { setActiveTab(Tab::Camera); });
    tab_calibration_btn_ = std::make_shared<UIButton>(" Calibration ",
        [this]() { setActiveTab(Tab::Calibration); });
    tab_perception_btn_ = std::make_shared<UIButton>(" Perception ",
        [this]() { setActiveTab(Tab::Perception); });

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
    RefreshPerceptionEnableButtonLabelsFromSubsystem();

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
        tab_camera_btn_->setSize(110.0f, kTabBarHeight - 4.0f);
        tab_camera_btn_->setPosition(position.x + kTabBarPad, tab_y);
        tab_camera_btn_->update(input, dt);
    }
    if (tab_calibration_btn_) {
        tab_calibration_btn_->setSize(130.0f, kTabBarHeight - 4.0f);
        tab_calibration_btn_->setPosition(position.x + kTabBarPad + 116.0f, tab_y);
        tab_calibration_btn_->update(input, dt);
    }
    if (tab_perception_btn_) {
        tab_perception_btn_->setSize(130.0f, kTabBarHeight - 4.0f);
        tab_perception_btn_->setPosition(position.x + kTabBarPad + 252.0f, tab_y);
        tab_perception_btn_->update(input, dt);
    }

    // ── Tab content ──
    switch (active_tab_) {
        case Tab::Camera:      UpdateCameraTab(input, dt);      break;
        case Tab::Calibration: UpdateCalibrationTab(input, dt); break;
        case Tab::Perception:  UpdatePerceptionTab(input, dt);  break;
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
    if (tab_camera_btn_)      tab_camera_btn_->drawOverlay(renderer, position);
    if (tab_calibration_btn_) tab_calibration_btn_->drawOverlay(renderer, position);
    if (tab_perception_btn_)  tab_perception_btn_->drawOverlay(renderer, position);

    // Active-tab underline indicator (matches DataHub/Training pattern).
    {
        const float y = position.y + titleBarHeight + kTabBarHeight - 1.0f;
        float ix = 0.0f, iw = 0.0f;
        switch (active_tab_) {
            case Tab::Camera:
                ix = position.x + kTabBarPad;          iw = 110.0f; break;
            case Tab::Calibration:
                ix = position.x + kTabBarPad + 116.0f; iw = 130.0f; break;
            case Tab::Perception:
                ix = position.x + kTabBarPad + 252.0f; iw = 130.0f; break;
        }
        renderer.drawRect({ix, y}, {iw, 2.0f}, UITheme::Colors::Primary);
    }

    // Divider under tab bar.
    renderer.drawRect({position.x + 4.0f,
                       position.y + titleBarHeight + kTabBarHeight + 1.0f},
                      {size.x - 8.0f, 1.0f},
                      UITheme::Colors::DividerLine);

    switch (active_tab_) {
        case Tab::Camera:      DrawCameraTab(renderer);      break;
        case Tab::Calibration: DrawCalibrationTab(renderer); break;
        case Tab::Perception:  DrawPerceptionTab(renderer);  break;
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

void UIPhysicalEnvironmentPanel::RefreshPerceptionEnableButtonLabelsFromSubsystem() {
    const auto f = PE::GetPhysicalPerceptionPrimitivesEnableFlags();
    if (perc_btn_obj_)  perc_btn_obj_->setText(std::string(" Detector: ")   + OnOffStr(f.object_detector)    + " ");
    if (perc_btn_seg_)  perc_btn_seg_->setText(std::string(" Segmenter: ")  + OnOffStr(f.semantic_segmenter) + " ");
    if (perc_btn_cls_)  perc_btn_cls_->setText(std::string(" Classifier: ") + OnOffStr(f.image_classifier)   + " ");
    if (perc_btn_pose_) perc_btn_pose_->setText(std::string(" Pose: ")      + OnOffStr(f.pose_estimator)     + " ");
    if (perc_btn_text_) perc_btn_text_->setText(std::string(" Text: ")      + OnOffStr(f.scene_text_reader)  + " ");
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
    const float btn_w = 132.0f;
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
        const float tw = renderer.measureTextWidth(label) + 6.0f;
        const float ly = std::max(static_cast<float>(blit_y), y - 16.0f);
        renderer.drawRect({x, ly}, {tw, 14.0f}, PercMakeArgb(0xC0, 0, 0, 0));
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
}

// ── Draw: tab body ─────────────────────────────────────────────────────────

void UIPhysicalEnvironmentPanel::DrawPerceptionTab(OverlayRenderer& renderer) {
    // Toolbar buttons (already laid out + ticked in UpdatePerceptionTab).
    if (perc_btn_obj_)  perc_btn_obj_->drawOverlay(renderer, position);
    if (perc_btn_seg_)  perc_btn_seg_->drawOverlay(renderer, position);
    if (perc_btn_cls_)  perc_btn_cls_->drawOverlay(renderer, position);
    if (perc_btn_pose_) perc_btn_pose_->drawOverlay(renderer, position);
    if (perc_btn_text_) perc_btn_text_->drawOverlay(renderer, position);

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

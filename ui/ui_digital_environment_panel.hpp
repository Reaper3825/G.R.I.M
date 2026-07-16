#pragma once

#include "primitives/ui_panel.hpp"
#include "primitives/ui_button.hpp"
#include "primitives/ui_dropdown.hpp"

#include "perception/digital/DigitalCaptureTypes.hpp"
#include "perception/digital/DigitalEnvironmentLoop.hpp"
#include "perception/digital/DigitalFrameBus.hpp"
#include "perception/digital/DigitalPerceptionPrimitiveBus.hpp"
#include "perception/digital/DigitalPerceptionPrimitivesLoop.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class UIDigitalEnvironmentPanel : public UIPanel {
public:
    UIDigitalEnvironmentPanel();

    void update(const InputState& input, float dt) override;
    bool drawOverlay(OverlayRenderer& renderer) override;

private:
    enum class ViewMode { Capture, Ocr, Automation };

    struct SourceChoice {
        std::string label;
        GRIM::Perception::Digital::DigitalCaptureRequest request{};
    };

    struct PreviewBlitCache {
        std::uint64_t source_id = 0;
        int out_w = 0;
        int out_h = 0;
        std::vector<std::uint32_t> argb;
    };

    void RefreshSourceChoices();
    void ApplyConfig(bool requestCapture);
    void HandleCaptureNow();
    void HandlePauseResume();
    void HandleLayeredToggle();

    void DrawBgrFrame(OverlayRenderer& renderer,
                      const cv::Mat& bgr,
                      std::uint64_t source_id,
                      float frame_x,
                      float frame_y,
                      float frame_w,
                      float frame_h);
    void DrawTelemetry(OverlayRenderer& renderer,
                       float x,
                       float y,
                       float width,
                       float height) const;
    void DrawPrimitiveOverlay(OverlayRenderer& renderer,
                              float frame_x,
                              float frame_y,
                              float frame_w,
                              float frame_h) const;
    void DrawPrimitiveTelemetry(OverlayRenderer& renderer,
                                float x,
                                float y,
                                float width,
                                float height) const;

    std::shared_ptr<UIDropdown> source_dropdown_;
    std::shared_ptr<UIDropdown> interval_dropdown_;
    std::shared_ptr<UIButton> refresh_sources_button_;
    std::shared_ptr<UIButton> capture_now_button_;
    std::shared_ptr<UIButton> pause_resume_button_;
    std::shared_ptr<UIButton> layered_toggle_button_;
    std::shared_ptr<UIButton> capture_view_button_;
    std::shared_ptr<UIButton> ocr_view_button_;
    std::shared_ptr<UIButton> automation_view_button_;

    std::vector<SourceChoice> source_choices_;
    std::vector<int> interval_values_ms_{250, 500, 1000, 2000, 5000};
    GRIM::Perception::Digital::DigitalEnvironmentConfig config_{};
    GRIM::Perception::Digital::DigitalEnvironmentStatus status_{};
    GRIM::Perception::Digital::DigitalPerceptionPrimitivesStatus primitives_status_{};

    GRIM::Perception::Digital::DigitalFrameBus::FrameView latest_attempt_{};
    GRIM::Perception::Digital::DigitalFrameBus::FrameView preview_frame_{};
    std::uint64_t last_seen_counter_ = 0;
    GRIM::Perception::Digital::DigitalPerceptionPrimitiveBus::SnapshotView primitives_{};
    std::uint64_t last_seen_primitives_counter_ = 0;
    bool have_attempt_ = false;
    bool have_preview_ = false;
    bool have_primitives_ = false;
    ViewMode view_mode_ = ViewMode::Capture;
    std::string panel_message_;
    PreviewBlitCache preview_cache_{};
};

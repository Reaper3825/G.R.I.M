#include "ui_digital_environment_panel.hpp"

#include "overlay_renderer.hpp"
#include "ui_theme.hpp"
#include "logger.hpp"

#include "perception/digital/DigitalCaptureSource.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace Digital = GRIM::Perception::Digital;

namespace {

constexpr const char* kLogTag = "DigitalEnvironmentPanel";

std::string Ellipsize(const std::string& value, std::size_t max_chars) {
    if (value.size() <= max_chars) return value;
    if (max_chars <= 3) return value.substr(0, max_chars);
    return value.substr(0, max_chars - 3) + "...";
}

std::string OneLine(std::string value) {
    std::replace(value.begin(), value.end(), '\n', ' ');
    std::replace(value.begin(), value.end(), '\r', ' ');
    return value;
}

std::string FormatDouble(double value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::uint64_t SteadyNowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

UIDigitalEnvironmentPanel::UIDigitalEnvironmentPanel()
    : UIPanel("Digital Environment", true) {
    position = {180.0f, 140.0f};
    size = {1100.0f, 720.0f};
    setBackground(UITheme::Colors::PanelBg);
    setBorder(UITheme::Colors::DividerLine);

    config_ = Digital::GetDigitalEnvironmentConfigSnapshot();

    source_dropdown_ = std::make_shared<UIDropdown>(
        "Capture Source", std::vector<std::string>{"Active Monitor"}, 0,
        [this](int index, const std::string&) {
            if (index < 0 || index >= static_cast<int>(source_choices_.size())) return;
            config_.request = source_choices_[static_cast<std::size_t>(index)].request;
            ApplyConfig(true);
        });

    interval_dropdown_ = std::make_shared<UIDropdown>(
        "Cadence", std::vector<std::string>{"250 ms", "500 ms", "1 second",
                                             "2 seconds", "5 seconds"}, 2,
        [this](int index, const std::string&) {
            if (index < 0 || index >= static_cast<int>(interval_values_ms_.size())) return;
            config_.capture_interval =
                std::chrono::milliseconds(interval_values_ms_[static_cast<std::size_t>(index)]);
            ApplyConfig(true);
        });

    refresh_sources_button_ = std::make_shared<UIButton>(
        " Refresh Displays ", [this]() {
            RefreshSourceChoices();
            Digital::RequestDigitalCaptureNow();
        });
    capture_now_button_ = std::make_shared<UIButton>(
        " Capture Now ", [this]() { HandleCaptureNow(); });
    pause_resume_button_ = std::make_shared<UIButton>(
        " Pause ", [this]() { HandlePauseResume(); });
    layered_toggle_button_ = std::make_shared<UIButton>(
        " Layered: On ", [this]() { HandleLayeredToggle(); });
    capture_view_button_ = std::make_shared<UIButton>(
        " [Capture] ", [this]() { view_mode_ = ViewMode::Capture; });
    ocr_view_button_ = std::make_shared<UIButton>(
        " OCR ", [this]() { view_mode_ = ViewMode::Ocr; });
    automation_view_button_ = std::make_shared<UIButton>(
        " Windows Automation ", [this]() { view_mode_ = ViewMode::Automation; });

    int closest_interval = 0;
    int closest_distance = std::numeric_limits<int>::max();
    const int configured_ms = static_cast<int>(config_.capture_interval.count());
    for (int i = 0; i < static_cast<int>(interval_values_ms_.size()); ++i) {
        const int distance = std::abs(interval_values_ms_[static_cast<std::size_t>(i)] - configured_ms);
        if (distance < closest_distance) {
            closest_distance = distance;
            closest_interval = i;
        }
    }
    interval_dropdown_->setSelectedIndex(closest_interval);
    RefreshSourceChoices();
}

void UIDigitalEnvironmentPanel::RefreshSourceChoices() {
    source_choices_.clear();

    auto add_choice = [this](std::string label, Digital::DigitalCaptureMode mode,
                             int monitor_index) {
        SourceChoice choice;
        choice.label = std::move(label);
        choice.request.mode = mode;
        choice.request.monitor_index = monitor_index;
        choice.request.include_layered_windows = config_.request.include_layered_windows;
        source_choices_.push_back(std::move(choice));
    };

    add_choice("Active Monitor", Digital::DigitalCaptureMode::ActiveMonitor, -1);
    add_choice("Active Window", Digital::DigitalCaptureMode::ActiveWindow, -1);
    add_choice("Virtual Desktop", Digital::DigitalCaptureMode::VirtualDesktop, -1);

    auto source = Digital::CreatePlatformDigitalCaptureSource();
    if (source) {
        const auto monitors = source->EnumerateMonitors();
        for (const auto& monitor : monitors) {
            std::ostringstream label;
            label << "Monitor " << (monitor.index + 1) << ": "
                  << monitor.desktop_rect.width << 'x' << monitor.desktop_rect.height;
            if (monitor.is_primary) label << " [Primary]";
            if (!monitor.friendly_name.empty()) {
                label << " - " << Ellipsize(monitor.friendly_name, 24);
            }
            add_choice(label.str(), Digital::DigitalCaptureMode::Monitor, monitor.index);
        }
    }

    std::vector<std::string> labels;
    labels.reserve(source_choices_.size());
    int selected = 0;
    for (int i = 0; i < static_cast<int>(source_choices_.size()); ++i) {
        labels.push_back(source_choices_[static_cast<std::size_t>(i)].label);
        const auto& request = source_choices_[static_cast<std::size_t>(i)].request;
        if (request.mode == config_.request.mode &&
            (request.mode != Digital::DigitalCaptureMode::Monitor ||
             request.monitor_index == config_.request.monitor_index)) {
            selected = i;
        }
    }
    source_dropdown_->setItems(labels);
    source_dropdown_->setSelectedIndex(selected);
    panel_message_ = labels.size() > 3
        ? std::to_string(labels.size() - 3) + " display(s) available"
        : "No monitor-specific sources available";
}

void UIDigitalEnvironmentPanel::ApplyConfig(bool requestCapture) {
    for (auto& choice : source_choices_) {
        choice.request.include_layered_windows = config_.request.include_layered_windows;
    }
    Digital::UpdateDigitalEnvironmentConfig(config_);
    if (requestCapture && Digital::IsDigitalEnvironmentRunning()) {
        Digital::RequestDigitalCaptureNow();
    }
}

void UIDigitalEnvironmentPanel::HandleCaptureNow() {
    if (!Digital::IsDigitalEnvironmentRunning()) {
        panel_message_ = "Capture is paused. Resume the worker before requesting a frame.";
        return;
    }
    Digital::RequestDigitalCaptureNow();
    panel_message_ = "Immediate capture requested";
}

void UIDigitalEnvironmentPanel::HandlePauseResume() {
    if (Digital::IsDigitalEnvironmentRunning()) {
        Digital::ShutdownDigitalEnvironment();
        panel_message_ = "Digital capture paused; last successful preview retained";
    } else {
        Digital::StartDigitalEnvironment(config_);
        panel_message_ = "Digital capture resumed";
    }
}

void UIDigitalEnvironmentPanel::HandleLayeredToggle() {
    config_.request.include_layered_windows = !config_.request.include_layered_windows;
    layered_toggle_button_->setText(config_.request.include_layered_windows
        ? " Layered: On " : " Layered: Off ");
    ApplyConfig(true);
}

void UIDigitalEnvironmentPanel::update(const InputState& input, float dt) {
    UIPanel::update(input, dt);
    if (!isVisible()) return;

    Digital::DigitalFrameBus::FrameView view;
    if (Digital::DigitalFrameBus::Instance().PullLatest(view, last_seen_counter_)) {
        latest_attempt_ = view;
        have_attempt_ = true;
        if (!view.image.empty()) {
            preview_frame_ = std::move(view);
            have_preview_ = true;
        }
    }
    Digital::DigitalPerceptionPrimitiveBus::SnapshotView primitive_view;
    if (Digital::DigitalPerceptionPrimitiveBus::Instance().PullLatest(
            primitive_view, last_seen_primitives_counter_)) {
        primitives_ = std::move(primitive_view);
        have_primitives_ = static_cast<bool>(primitives_.snapshot);
    }
    status_ = Digital::GetDigitalEnvironmentStatusSnapshot();
    primitives_status_ = Digital::GetDigitalPerceptionPrimitivesStatusSnapshot();

    pause_resume_button_->setText(status_.running ? " Pause " : " Resume ");
    layered_toggle_button_->setText(config_.request.include_layered_windows
        ? " Layered: On " : " Layered: Off ");
    capture_view_button_->setText(view_mode_ == ViewMode::Capture
        ? " [Capture] " : " Capture ");
    ocr_view_button_->setText(view_mode_ == ViewMode::Ocr ? " [OCR] " : " OCR ");
    automation_view_button_->setText(view_mode_ == ViewMode::Automation
        ? " [Windows Automation] " : " Windows Automation ");

    const float top = position.y + titleBarHeight + 10.0f;
    source_dropdown_->setPosition(position.x + 16.0f, top);
    source_dropdown_->setSize(310.0f, 30.0f);
    interval_dropdown_->setPosition(position.x + 336.0f, top);
    interval_dropdown_->setSize(160.0f, 30.0f);
    refresh_sources_button_->setPosition(position.x + 506.0f, top);
    refresh_sources_button_->setSize(130.0f, 30.0f);
    capture_now_button_->setPosition(position.x + 646.0f, top);
    capture_now_button_->setSize(120.0f, 30.0f);
    pause_resume_button_->setPosition(position.x + 776.0f, top);
    pause_resume_button_->setSize(100.0f, 30.0f);
    layered_toggle_button_->setPosition(position.x + 886.0f, top);
    layered_toggle_button_->setSize(150.0f, 30.0f);
    capture_view_button_->setPosition(position.x + 16.0f, top + 38.0f);
    capture_view_button_->setSize(110.0f, 28.0f);
    ocr_view_button_->setPosition(position.x + 136.0f, top + 38.0f);
    ocr_view_button_->setSize(90.0f, 28.0f);
    automation_view_button_->setPosition(position.x + 236.0f, top + 38.0f);
    automation_view_button_->setSize(190.0f, 28.0f);

    source_dropdown_->update(input, dt);
    interval_dropdown_->update(input, dt);
    refresh_sources_button_->update(input, dt);
    capture_now_button_->update(input, dt);
    pause_resume_button_->update(input, dt);
    layered_toggle_button_->update(input, dt);
    capture_view_button_->update(input, dt);
    ocr_view_button_->update(input, dt);
    automation_view_button_->update(input, dt);
}

bool UIDigitalEnvironmentPanel::drawOverlay(OverlayRenderer& renderer) {
    if (!UIPanel::drawOverlay(renderer)) return false;

    const float pad = 16.0f;
    const float controls_h = 94.0f;
    const float sidebar_w = 310.0f;
    const float content_top = position.y + titleBarHeight + controls_h;
    const float content_h = size.y - titleBarHeight - controls_h - pad;
    const float frame_x = position.x + pad;
    const float frame_y = content_top;
    const float frame_w = std::max(160.0f, size.x - sidebar_w - pad * 3.0f);
    const float frame_h = std::max(120.0f, content_h);
    const float side_x = frame_x + frame_w + pad;

    renderer.drawRect({position.x + 8.0f, content_top - 8.0f},
                      {size.x - 16.0f, 1.0f}, UITheme::Colors::DividerLine);
    renderer.drawRoundedRect({frame_x - 2.0f, frame_y - 2.0f},
                             {frame_w + 4.0f, frame_h + 4.0f},
                             UITheme::Colors::BorderMedium, 8.0f);
    renderer.drawRect({frame_x, frame_y}, {frame_w, frame_h},
                      UITheme::Colors::Background);

    if (have_preview_ && !preview_frame_.image.empty()) {
        DrawBgrFrame(renderer, preview_frame_.image, preview_frame_.frame_counter,
                     frame_x, frame_y, frame_w, frame_h);
        if (view_mode_ != ViewMode::Capture) {
            DrawPrimitiveOverlay(renderer, frame_x, frame_y, frame_w, frame_h);
        }
        const std::string badge = status_.running ? "LIVE CAPTURE" : "PAUSED - LAST FRAME";
        const std::uint32_t badge_color = status_.running
            ? UITheme::Colors::SuccessBg : UITheme::Colors::WidgetBg;
        renderer.drawRoundedRect({frame_x + 10.0f, frame_y + 10.0f},
                                 {154.0f, 24.0f}, badge_color, 6.0f);
        renderer.drawText({frame_x + 18.0f, frame_y + 15.0f}, badge,
                          UITheme::Colors::TextPrimary);
    } else {
        renderer.drawText({frame_x + 18.0f, frame_y + 18.0f},
                          "Waiting for the first successful digital frame...",
                          UITheme::Colors::TextSecondary);
    }

    if (have_attempt_ && latest_attempt_.metadata.status != Digital::DigitalCaptureStatus::Ok) {
        renderer.drawRoundedRect({frame_x + 10.0f, frame_y + frame_h - 42.0f},
                                 {std::max(180.0f, frame_w - 20.0f), 30.0f},
                                 UITheme::Colors::DangerBg, 6.0f);
        renderer.drawText({frame_x + 18.0f, frame_y + frame_h - 35.0f},
                          "Latest capture failed: " +
                              Ellipsize(latest_attempt_.metadata.error, 72),
                          UITheme::Colors::Danger);
    }

    if (view_mode_ == ViewMode::Capture) {
        DrawTelemetry(renderer, side_x, frame_y, sidebar_w, frame_h);
    } else {
        DrawPrimitiveTelemetry(renderer, side_x, frame_y, sidebar_w, frame_h);
    }

    source_dropdown_->drawOverlay(renderer, position);
    interval_dropdown_->drawOverlay(renderer, position);
    refresh_sources_button_->drawOverlay(renderer, position);
    capture_now_button_->drawOverlay(renderer, position);
    pause_resume_button_->drawOverlay(renderer, position);
    layered_toggle_button_->drawOverlay(renderer, position);
    capture_view_button_->drawOverlay(renderer, position);
    ocr_view_button_->drawOverlay(renderer, position);
    automation_view_button_->drawOverlay(renderer, position);
    if (source_dropdown_->isExpanded()) source_dropdown_->drawExpandedList(renderer, position);
    if (interval_dropdown_->isExpanded()) interval_dropdown_->drawExpandedList(renderer, position);

    renderer.popClipRect();
    return true;
}

void UIDigitalEnvironmentPanel::DrawBgrFrame(OverlayRenderer& renderer,
                                             const cv::Mat& bgr,
                                             std::uint64_t source_id,
                                             float frame_x,
                                             float frame_y,
                                             float frame_w,
                                             float frame_h) {
    auto* pixels = static_cast<std::uint32_t*>(renderer.getPixels());
    if (!pixels || bgr.empty() || bgr.type() != CV_8UC3) return;

    const int dest_w = renderer.getWidth();
    const int dest_h = renderer.getHeight();
    if (dest_w <= 0 || dest_h <= 0) return;

    const double scale = std::min(static_cast<double>(frame_w) / bgr.cols,
                                  static_cast<double>(frame_h) / bgr.rows);
    const int out_w = std::max(1, static_cast<int>(bgr.cols * scale));
    const int out_h = std::max(1, static_cast<int>(bgr.rows * scale));
    const int out_x = static_cast<int>(frame_x + (frame_w - out_w) * 0.5f);
    const int out_y = static_cast<int>(frame_y + (frame_h - out_h) * 0.5f);

    const bool cache_hit = preview_cache_.source_id == source_id &&
                           preview_cache_.out_w == out_w &&
                           preview_cache_.out_h == out_h &&
                           preview_cache_.argb.size() ==
                               static_cast<std::size_t>(out_w) * out_h;
    if (!cache_hit) {
        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(out_w, out_h), 0.0, 0.0, cv::INTER_AREA);
        preview_cache_.argb.resize(static_cast<std::size_t>(out_w) * out_h);
        for (int y = 0; y < out_h; ++y) {
            const auto* src = resized.ptr<std::uint8_t>(y);
            auto* dst = preview_cache_.argb.data() + static_cast<std::size_t>(y) * out_w;
            for (int x = 0; x < out_w; ++x) {
                const std::uint8_t b = src[x * 3 + 0];
                const std::uint8_t g = src[x * 3 + 1];
                const std::uint8_t r = src[x * 3 + 2];
                dst[x] = 0xFF000000u | (static_cast<std::uint32_t>(r) << 16) |
                         (static_cast<std::uint32_t>(g) << 8) | b;
            }
        }
        preview_cache_.source_id = source_id;
        preview_cache_.out_w = out_w;
        preview_cache_.out_h = out_h;
    }

    const int clip_x0 = std::max(0, out_x);
    const int clip_y0 = std::max(0, out_y);
    const int clip_x1 = std::min(dest_w, out_x + out_w);
    const int clip_y1 = std::min(dest_h, out_y + out_h);
    if (clip_x0 >= clip_x1 || clip_y0 >= clip_y1) return;

    const std::size_t row_bytes = static_cast<std::size_t>(clip_x1 - clip_x0) *
                                  sizeof(std::uint32_t);
    for (int y = clip_y0; y < clip_y1; ++y) {
        const auto* src = preview_cache_.argb.data() +
            static_cast<std::size_t>(y - out_y) * out_w + (clip_x0 - out_x);
        auto* dst = pixels + static_cast<std::size_t>(y) * dest_w + clip_x0;
        std::memcpy(dst, src, row_bytes);
    }
}

void UIDigitalEnvironmentPanel::DrawTelemetry(OverlayRenderer& renderer,
                                              float x,
                                              float y,
                                              float width,
                                              float height) const {
    renderer.drawRoundedRect({x, y}, {width, height},
                             UITheme::Colors::CardSurface, 8.0f);
    renderer.drawText({x + 14.0f, y + 14.0f}, "Capture Health",
                      UITheme::Colors::TextHeader);

    float line_y = y + 42.0f;
    const float step = 20.0f;
    auto line_colored = [&](const std::string& label, const std::string& value,
                            std::uint32_t value_color) {
        renderer.drawText({x + 14.0f, line_y}, label, UITheme::Colors::TextSecondary);
        renderer.drawText({x + 112.0f, line_y}, Ellipsize(value, 27), value_color);
        line_y += step;
    };
    auto line = [&](const std::string& label, const std::string& value) {
        line_colored(label, value, UITheme::Colors::TextValue);
    };
    auto separator = [&]() {
        renderer.drawRect({x + 12.0f, line_y + 2.0f}, {width - 24.0f, 1.0f},
                          UITheme::Colors::DividerLine);
        line_y += 16.0f;
    };

    line_colored("Worker", status_.running ? "Running" : "Paused",
                 status_.running ? UITheme::Colors::Success : UITheme::Colors::Warning);
    line_colored("Latest", Digital::ToString(status_.last_status),
                 status_.last_status == Digital::DigitalCaptureStatus::Ok
                     ? UITheme::Colors::Success : UITheme::Colors::Danger);
    line("Backend", status_.backend.empty() ? "Not initialized" : status_.backend);
    if (have_attempt_) {
        line("Device", latest_attempt_.metadata.source_device_id.empty()
                           ? "Unknown" : latest_attempt_.metadata.source_device_id);
        line("Platform", latest_attempt_.metadata.source_platform.empty()
                             ? "Unknown" : latest_attempt_.metadata.source_platform);
    }
    line("Attempts", std::to_string(status_.capture_attempts) + " total");
    line("Success / Fail", std::to_string(status_.successful_captures) + " / " +
                           std::to_string(status_.failed_captures));
    line("Capture Time", FormatDouble(status_.last_capture_duration_ms, 1) + " ms");

    if (have_attempt_ && latest_attempt_.metadata.capture_steady_ns > 0) {
        const std::uint64_t now = SteadyNowNs();
        const double age_seconds = now >= latest_attempt_.metadata.capture_steady_ns
            ? static_cast<double>(now - latest_attempt_.metadata.capture_steady_ns) / 1.0e9
            : 0.0;
        line("Frame Age", FormatDouble(age_seconds, 1) + " s");
    }

    separator();
    renderer.drawText({x + 14.0f, line_y}, "Source", UITheme::Colors::TextHeader);
    line_y += 26.0f;

    if (have_attempt_) {
        const auto& metadata = latest_attempt_.metadata;
        line("Mode", Digital::ToString(metadata.mode));
        line("Monitor", metadata.monitor_id.empty() ? "Unknown" : metadata.monitor_id);
        line("Index", std::to_string(metadata.monitor_index));
        line("Source Rect", std::to_string(metadata.source_rect.x) + "," +
                            std::to_string(metadata.source_rect.y) + "  " +
                            std::to_string(metadata.source_rect.width) + "x" +
                            std::to_string(metadata.source_rect.height));
        line("DPI / Scale", std::to_string(metadata.dpi_x) + "x" +
                           std::to_string(metadata.dpi_y) + " / " +
                           FormatDouble(metadata.scale_factor, 2) + "x");
        line("Frame", std::to_string(latest_attempt_.frame_counter));
        line("Color", metadata.color_space);
    } else {
        line("Status", "No capture attempt yet");
    }

    separator();
    renderer.drawText({x + 14.0f, line_y}, "Foreground", UITheme::Colors::TextHeader);
    line_y += 26.0f;
    if (have_attempt_) {
        line("Process", latest_attempt_.metadata.active_process_name.empty()
                            ? "Unknown" : latest_attempt_.metadata.active_process_name);
        line("Window", latest_attempt_.metadata.active_window_title.empty()
                           ? "Unknown" : latest_attempt_.metadata.active_window_title);
    }

    if (!panel_message_.empty() && line_y + 32.0f < y + height) {
        separator();
        renderer.drawText({x + 14.0f, line_y}, Ellipsize(panel_message_, 38),
                          UITheme::Colors::Info);
    }
    if (!status_.last_error.empty() && line_y + 52.0f < y + height) {
        renderer.drawText({x + 14.0f, line_y + step},
                          Ellipsize("Error: " + status_.last_error, 38),
                          UITheme::Colors::Danger);
    }
}

void UIDigitalEnvironmentPanel::DrawPrimitiveOverlay(OverlayRenderer& renderer,
                                                     float frame_x,
                                                     float frame_y,
                                                     float frame_w,
                                                     float frame_h) const {
    if (!have_preview_ || preview_frame_.image.empty() || !have_primitives_ ||
        !primitives_.snapshot ||
        primitives_.snapshot->source_frame_counter != preview_frame_.frame_counter) {
        renderer.drawRoundedRect({frame_x + 10.0f, frame_y + 42.0f},
                                 {220.0f, 24.0f}, UITheme::Colors::WidgetBg, 6.0f);
        renderer.drawText({frame_x + 18.0f, frame_y + 47.0f},
                          "Analyzing current frame...", UITheme::Colors::Warning);
        return;
    }

    const auto& image = preview_frame_.image;
    const double scale = std::min(static_cast<double>(frame_w) / image.cols,
                                  static_cast<double>(frame_h) / image.rows);
    const float out_w = static_cast<float>(image.cols * scale);
    const float out_h = static_cast<float>(image.rows * scale);
    const float origin_x = frame_x + (frame_w - out_w) * 0.5f;
    const float origin_y = frame_y + (frame_h - out_h) * 0.5f;

    auto draw_box = [&](const Digital::DigitalRect& local,
                        std::uint32_t color,
                        const std::string& label) {
        const int left = std::max(0, local.x);
        const int top = std::max(0, local.y);
        const int right = std::min(image.cols, local.x + local.width);
        const int bottom = std::min(image.rows, local.y + local.height);
        if (right <= left || bottom <= top) return;
        const float x0 = origin_x + static_cast<float>(left * scale);
        const float y0 = origin_y + static_cast<float>(top * scale);
        const float x1 = origin_x + static_cast<float>(right * scale);
        const float y1 = origin_y + static_cast<float>(bottom * scale);
        renderer.drawLine({x0, y0}, {x1, y0}, color, 2.0f);
        renderer.drawLine({x1, y0}, {x1, y1}, color, 2.0f);
        renderer.drawLine({x1, y1}, {x0, y1}, color, 2.0f);
        renderer.drawLine({x0, y1}, {x0, y0}, color, 2.0f);
        if (!label.empty() && y0 > frame_y + 70.0f) {
            renderer.drawText({x0 + 3.0f, y0 + 2.0f},
                              Ellipsize(OneLine(label), 28), color);
        }
    };

    if (view_mode_ == ViewMode::Ocr) {
        const auto& regions = primitives_.snapshot->ocr.regions;
        const std::size_t limit = std::min<std::size_t>(regions.size(), 120);
        for (std::size_t i = 0; i < limit; ++i) {
            draw_box(regions[i].frame_rect, UITheme::Colors::Info, regions[i].text);
        }
        return;
    }

    const auto& elements = primitives_.snapshot->automation.elements;
    const auto& source = preview_frame_.metadata.source_rect;
    const std::size_t limit = std::min<std::size_t>(elements.size(), 160);
    for (std::size_t i = 0; i < limit; ++i) {
        const auto& element = elements[i];
        Digital::DigitalRect local{
            element.desktop_rect.x - source.x,
            element.desktop_rect.y - source.y,
            element.desktop_rect.width,
            element.desktop_rect.height
        };
        const std::string label = element.name.empty()
            ? element.role : element.role + ": " + element.name;
        draw_box(local, element.password ? UITheme::Colors::Warning
                                         : UITheme::Colors::Success,
                 label);
    }
}

void UIDigitalEnvironmentPanel::DrawPrimitiveTelemetry(OverlayRenderer& renderer,
                                                       float x,
                                                       float y,
                                                       float width,
                                                       float height) const {
    renderer.drawRoundedRect({x, y}, {width, height},
                             UITheme::Colors::CardSurface, 8.0f);
    const bool ocr_view = view_mode_ == ViewMode::Ocr;
    renderer.drawText({x + 14.0f, y + 14.0f},
                      ocr_view ? "OCR - Cross-platform Baseline"
                               : "Windows Automation - Optional",
                      UITheme::Colors::TextHeader);

    float line_y = y + 42.0f;
    const float step = 20.0f;
    auto line_colored = [&](const std::string& label, const std::string& value,
                            std::uint32_t color) {
        if (line_y + step >= y + height) return;
        renderer.drawText({x + 14.0f, line_y}, label, UITheme::Colors::TextSecondary);
        renderer.drawText({x + 112.0f, line_y}, Ellipsize(OneLine(value), 27), color);
        line_y += step;
    };
    auto line = [&](const std::string& label, const std::string& value) {
        line_colored(label, value, UITheme::Colors::TextValue);
    };
    auto separator = [&]() {
        if (line_y + 12.0f >= y + height) return;
        renderer.drawRect({x + 12.0f, line_y + 2.0f}, {width - 24.0f, 1.0f},
                          UITheme::Colors::DividerLine);
        line_y += 16.0f;
    };

    line_colored("Worker", primitives_status_.running ? "Running" : "Stopped",
                 primitives_status_.running ? UITheme::Colors::Success
                                            : UITheme::Colors::Warning);
    line("Processed", std::to_string(primitives_status_.processed_frames));
    if (!have_primitives_ || !primitives_.snapshot) {
        line("Snapshot", "Waiting for primitives");
        return;
    }

    const auto& snapshot = *primitives_.snapshot;
    const bool coherent = have_preview_ &&
        snapshot.source_frame_counter == preview_frame_.frame_counter;
    line_colored("Frame Sync", coherent ? "Exact frame" : "Analysis lagging",
                 coherent ? UITheme::Colors::Success : UITheme::Colors::Warning);
    line("Source Frame", std::to_string(snapshot.source_frame_counter));

    separator();
    if (ocr_view) {
        const auto& result = snapshot.ocr;
        line("Provider", result.provider);
        line_colored("Status", Digital::ToString(result.status),
                     result.status == Digital::DigitalPrimitiveStatus::Ok
                         ? UITheme::Colors::Success : UITheme::Colors::Warning);
        line("Duration", FormatDouble(result.duration_ms, 1) + " ms");
        line("Confidence", FormatDouble(result.mean_confidence * 100.0f, 1) + "%");
        line("Text Regions", std::to_string(result.regions.size()));
        if (!result.error.empty()) line_colored("Error", result.error, UITheme::Colors::Danger);
        separator();
        renderer.drawText({x + 14.0f, line_y}, "Recognized Text",
                          UITheme::Colors::TextHeader);
        line_y += 25.0f;
        const std::size_t limit = std::min<std::size_t>(result.regions.size(), 16);
        for (std::size_t i = 0; i < limit && line_y + step < y + height; ++i) {
            const auto& region = result.regions[i];
            renderer.drawText({x + 14.0f, line_y},
                              Ellipsize(OneLine(region.text), 39),
                              UITheme::Colors::TextValue);
            line_y += step;
        }
    } else {
        const auto& result = snapshot.automation;
        line("Provider", result.provider);
        line_colored("Status", Digital::ToString(result.status),
                     result.status == Digital::DigitalPrimitiveStatus::Ok
                         ? UITheme::Colors::Success : UITheme::Colors::Warning);
        line("Duration", FormatDouble(result.duration_ms, 1) + " ms");
        line("Elements", std::to_string(result.elements.size()) +
                         (result.truncated ? " +" : ""));
        line_colored("Target Match", result.target_matches_capture
                         ? "Matched capture" : "Changed / unknown",
                     result.target_matches_capture
                         ? UITheme::Colors::Success : UITheme::Colors::Warning);
        line("Window", result.target_window);
        if (!result.error.empty()) line_colored("Error", result.error, UITheme::Colors::Danger);
        separator();
        renderer.drawText({x + 14.0f, line_y}, "Accessible Controls",
                          UITheme::Colors::TextHeader);
        line_y += 25.0f;
        const std::size_t limit = std::min<std::size_t>(result.elements.size(), 16);
        for (std::size_t i = 0; i < limit && line_y + step < y + height; ++i) {
            const auto& element = result.elements[i];
            std::string summary = element.role;
            if (!element.name.empty()) summary += ": " + element.name;
            renderer.drawText({x + 14.0f, line_y},
                              Ellipsize(OneLine(summary), 39),
                              element.password ? UITheme::Colors::Warning
                                               : UITheme::Colors::TextValue);
            line_y += step;
        }
    }
}

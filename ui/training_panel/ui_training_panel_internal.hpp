#pragma once

#include "../ui_training_panel.hpp"
#include "../overlay_renderer.hpp"
#include "../ui_draw_helpers.hpp"
#include "../ui_theme.hpp"

#include "core/grim_platform.h"
#include "core/input_parser.hpp"
#include "logger.hpp"
#include "MMO/Core/ModelRegistry.hpp"
#include "MMO/Core/ModelLoader.hpp"
#include "MMO/Core/ResourceSignal.hpp"
#include "resources.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <thread>

extern GRIM::MMO::ModelLoader* g_modelLoader;
extern GRIM::MMO::ResourceSignal* g_resourceSignal;

namespace UITrainingPanelDetail {

inline constexpr float kTabBarY = 35.0f;
inline constexpr float kContentTopY = 68.0f;
inline constexpr float kBottomBarH = 50.0f;
inline constexpr float kStatCardH = 60.0f;
inline constexpr float kStatCardGap = 10.0f;
inline constexpr float kLogLineH = 18.0f;

inline constexpr float kParamBrowserWheelPixelsPerStep = 22.0f;
inline constexpr float kParamBrowserW = 420.0f;
inline constexpr float kParamBrowserScrollbarWidth = 12.0f;
inline constexpr float kParamBrowserScrollbarInset = 2.0f;
inline constexpr float kParamBrowserScrollbarMinThumbH = 24.0f;

struct ParamScrollbarMetrics {
    bool visible = false;
    float trackX = 0.0f;
    float trackY = 0.0f;
    float trackW = 0.0f;
    float trackH = 0.0f;
    float thumbY = 0.0f;
    float thumbH = 0.0f;
    float maxScroll = 0.0f;
    float maxThumbTravel = 0.0f;
};

inline float normalizeMouseWheelDelta(float rawDelta) {
    if (std::fabs(rawDelta) >= 120.0f) {
        return rawDelta / 120.0f;
    }
    return rawDelta;
}

inline bool pointInRect(const Vec2& point, float x, float y, float w, float h) {
    return point.x >= x && point.x <= x + w && point.y >= y && point.y <= y + h;
}

inline size_t countParamCategoryHeaderRows(
    const std::vector<const GRIM::Config::HyperparamEntry*>& params,
    const std::string& activeCategory)
{
    if (!activeCategory.empty()) {
        return 0;
    }

    size_t headerCount = 0;
    std::string lastCategory;
    for (const auto* entry : params) {
        if (!entry) continue;
        if (entry->category != lastCategory) {
            lastCategory = entry->category;
            ++headerCount;
        }
    }
    return headerCount;
}

inline float computeParamBrowserContentHeight(
    const std::vector<const GRIM::Config::HyperparamEntry*>& params,
    const std::string& activeCategory,
    float rowH)
{
    const size_t headerRows = countParamCategoryHeaderRows(params, activeCategory);
    return static_cast<float>(params.size() + headerRows) * rowH;
}

inline ParamScrollbarMetrics computeParamScrollbarMetrics(
    float x,
    float y,
    float w,
    float listH,
    float totalContentH,
    float scrollOffset)
{
    ParamScrollbarMetrics metrics;
    metrics.maxScroll = std::max(0.0f, totalContentH - listH);
    metrics.visible = metrics.maxScroll > 0.0f;
    metrics.trackW = kParamBrowserScrollbarWidth;
    metrics.trackX = x + w - metrics.trackW - kParamBrowserScrollbarInset;
    metrics.trackY = y + kParamBrowserScrollbarInset;
    metrics.trackH = std::max(0.0f, listH - 2.0f * kParamBrowserScrollbarInset);

    if (!metrics.visible || metrics.trackH <= 0.0f) {
        metrics.thumbY = metrics.trackY;
        metrics.thumbH = metrics.trackH;
        return metrics;
    }

    const float visibleRatio = listH / totalContentH;
    metrics.thumbH = std::clamp(metrics.trackH * visibleRatio,
                                kParamBrowserScrollbarMinThumbH,
                                metrics.trackH);
    metrics.maxThumbTravel = std::max(0.0f, metrics.trackH - metrics.thumbH);

    const float scrollRatio = (metrics.maxScroll > 0.0f)
        ? std::clamp(scrollOffset / metrics.maxScroll, 0.0f, 1.0f)
        : 0.0f;
    metrics.thumbY = metrics.trackY + scrollRatio * metrics.maxThumbTravel;
    return metrics;
}

inline std::filesystem::path resolvePathFromGrimRoot(const std::string& rawPath) {
    std::filesystem::path path(rawPath);
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::path(getGrimRootDir()) / path;
}

}  // namespace UITrainingPanelDetail

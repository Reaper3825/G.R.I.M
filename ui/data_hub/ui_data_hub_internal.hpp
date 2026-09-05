#pragma once

#include "../ui_data_hub.hpp"
#include "../overlay_renderer.hpp"
#include "../ui_draw_helpers.hpp"
#include "../ui_theme.hpp"

#include "core/input_parser.hpp"
#include "logger.hpp"
#include "resources.hpp"
#include "MMO/Core/ModelRegistry.hpp"
#include "DataCollection/concept_block_canonical.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace UIDataHubDetail {

inline constexpr float kPollInterval  = 0.2f;
inline constexpr float kStatsInterval = 2.0f;
inline constexpr float kTabBarY       = 35.0f;
inline constexpr float kContentTopY   = 68.0f;

inline constexpr float kCardTopBarH  = 28.0f;
inline constexpr float kCardFieldH   = 28.0f;
inline constexpr float kCardLabelGap = 3.0f;
inline constexpr float kCardLabelH   = 13.0f;
inline constexpr float kCardRowGap   = 6.0f;
inline constexpr float kCardRowH     = kCardFieldH + kCardLabelGap + kCardLabelH + kCardRowGap;
inline constexpr float kCardPadBot   = 8.0f;
inline constexpr float kCardHeight   = kCardTopBarH + 2.0f * kCardRowH + kCardPadBot;
inline constexpr float kCardGap      = 14.0f;
inline constexpr float kCardInnerPad = 12.0f;

inline constexpr float kStructRowGap     = 12.0f;
inline constexpr float kStructSectionGap = 16.0f;
inline constexpr float kStructLabelSpace = 24.0f;

inline constexpr float kSeqCardTopBarH     = 32.0f;
inline constexpr float kSeqCardEntryLabelH = 16.0f;
inline constexpr float kSeqCardEntryTextH  = 55.0f;
inline constexpr float kSeqCardEntryGapH   = 6.0f;
inline constexpr float kSeqCardEntryH      = kSeqCardEntryLabelH + kSeqCardEntryTextH + kSeqCardEntryGapH;
inline constexpr float kSeqCardBtnRowH     = 30.0f;
inline constexpr float kSeqCardPadBot      = 8.0f;
inline constexpr float kSeqCardInnerPad    = 12.0f;

inline constexpr float kPoolRowH       = 24.0f;
inline constexpr float kCBListRowH     = 44.0f;
inline constexpr float kPoolHeaderH    = 28.0f;
inline constexpr float kCurrRowH       = 28.0f;
inline constexpr float kPhaseRowH      = 22.0f;
inline constexpr float kFilterBarH     = 34.0f;
inline constexpr float kDetailDividerH = 18.0f;

inline constexpr float kColNum        = 0.06f;
inline constexpr float kColSubject    = 0.12f;
inline constexpr float kColQuality    = 0.10f;
inline constexpr float kColSource     = 0.10f;
inline constexpr float kColStructured = 0.08f;

inline std::string cbSingleLinePreview(const std::string& s, size_t maxLen) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) {
        if (c == '\n' || c == '\r' || c == '\t') {
            t.push_back(' ');
        } else {
            t.push_back(c);
        }
    }
    while (!t.empty() && t.back() == ' ') {
        t.pop_back();
    }
    if (t.size() > maxLen) {
        return t.substr(0, maxLen - 2) + "..";
    }
    return t;
}

inline std::string detectFetcherFromUrl(const std::string& url) {
    if (url.find("api.github.com") != std::string::npos || url.find("github.com") != std::string::npos) {
        return "github_api";
    }
    if (url.find("arxiv.org") != std::string::npos) {
        return "arxiv_api";
    }
    if (url.find("wikipedia.org") != std::string::npos) {
        return "wikipedia_api";
    }
    if (url.find("stackexchange.com") != std::string::npos || url.find("stackoverflow.com") != std::string::npos) {
        return "stackoverflow_api";
    }
    if (url.find("reddit.com") != std::string::npos) {
        return "reddit_api";
    }
    if (url.find("newsapi.org") != std::string::npos) {
        return "news_api";
    }
    if (url.find("huggingface.co") != std::string::npos || url.find("huggingface://") == 0) {
        return "huggingface";
    }
    return "html_crawl";
}

inline std::string getSourceConfigPath() {
    std::filesystem::path sourcePath = aiConfig.at("paths").at("grim_text").at("source_config").get<std::string>();
    if (sourcePath.is_relative()) {
        sourcePath = std::filesystem::path(getGrimRootDir()) / sourcePath;
    }
    return sourcePath.string();
}

inline std::filesystem::path resolvePathFromGrimRoot(const std::string& rawPath) {
    std::filesystem::path path(rawPath);
    if (path.is_absolute()) {
        return path;
    }
    return std::filesystem::path(getGrimRootDir()) / path;
}

struct CurriculumTabLayout {
    float listX      = 0.0f;
    float listY      = 0.0f;
    float listW      = 0.0f;
    float listH      = 0.0f;
    float editorX    = 0.0f;
    float editorW    = 0.0f;
    float bottomBarH = 36.0f;
    float statusBarH = 25.0f;
};

inline CurriculumTabLayout computeCurriculumTabLayout(const PanelRect& content) {
    CurriculumTabLayout layout;
    layout.listX = content.origin.x + 15.0f;

    float y = content.origin.y + 10.0f;
    constexpr float kToolbarRowH = 36.0f;
    y += kToolbarRowH + 6.0f;
    y += kToolbarRowH + 6.0f;
    y += kToolbarRowH + 12.0f;

    const float fullW = content.size.x - 30.0f;

    layout.listY = y;
    layout.listW = fullW * 0.38f;
    layout.listH = (content.origin.y + content.size.y) - y
                 - layout.bottomBarH - layout.statusBarH - 10.0f;
    if (layout.listH < 100.0f) {
        layout.listH = 100.0f;
    }

    layout.editorX = layout.listX + layout.listW + 10.0f;
    layout.editorW = fullW - layout.listW - 10.0f;
    return layout;
}

}  // namespace UIDataHubDetail

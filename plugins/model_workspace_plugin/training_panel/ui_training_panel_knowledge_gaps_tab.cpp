// UITrainingPanel: Knowledge Gaps tab
#include "ui_training_panel_internal.hpp"

using namespace GRIMText;
using namespace UITheme;
using namespace UITrainingPanelDetail;

// ============================================================
// Knowledge Gap Intake
// ============================================================

void UITrainingPanel::pushKnowledgeGap(KnowledgeGapEntry entry) {
    std::lock_guard<std::mutex> lock(gapMutex_);
    gapQueue_.push_back(std::move(entry));
}

size_t UITrainingPanel::pendingGapCount() const {
    std::lock_guard<std::mutex> lock(gapMutex_);
    return gapQueue_.size();
}

// ============================================================
// Knowledge Gaps Tab
// ============================================================

void UITrainingPanel::drawKnowledgeGapsTab(OverlayRenderer& renderer, const PanelRect& content) {
    std::lock_guard<std::mutex> lock(gapMutex_);

    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Knowledge Gap Queue", Colors::SectionWhisper);
    y += Sizes::HeaderHeight + Spacing::Small;

    if (gapQueue_.empty()) {
        renderer.drawText({x, y}, "No knowledge gaps queued.", Colors::TextMuted);
        renderer.drawText({x, y + 20.0f},
            "Gaps appear when the router cannot find a matching sub-model.", Colors::TextMuted);
        return;
    }

    // Column headers
    renderer.drawText({x, y}, "Subject", Colors::TextSecondary);
    renderer.drawText({x + 250.0f, y}, "Tags", Colors::TextSecondary);
    renderer.drawText({x + 600.0f, y}, "Actions", Colors::TextSecondary);
    y += 22.0f;

    UIDrawHelpers::drawDivider(renderer, {x, y}, w);
    y += 4.0f;

    for (size_t i = 0; i < gapQueue_.size(); ++i) {
        const auto& gap = gapQueue_[i];

        if (static_cast<int>(i) == hoveredGapRow_) {
            renderer.drawRoundedRect({x, y - 1.0f}, {w, kRowHeight}, Colors::ContentAreaBg, 4.0f);
        }

        renderer.drawText({x, y}, gap.subject, Colors::TextPrimary);

        std::string tagStr;
        for (size_t t = 0; t < gap.tags.size(); ++t) {
            if (t > 0) tagStr += ", ";
            tagStr += gap.tags[t];
        }
        renderer.drawText({x + 250.0f, y}, tagStr, Colors::TextSecondary);
        renderer.drawText({x + 600.0f, y}, "[Create]", Colors::AccentBlue);
        renderer.drawText({x + 670.0f, y}, "[Dismiss]", Colors::Danger);

        y += kRowHeight;
    }
}

void UITrainingPanel::processGapClicks(const InputState& input, const PanelRect& content) {
    Vec2 m = input.mousePos;
    float x = content.origin.x + Spacing::PaddingX;
    float w = content.size.x - 2.0f * Spacing::PaddingX;
    float dataY = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small + 26.0f;

    hoveredGapRow_ = -1;

    if (m.x < x || m.x > x + w || m.y < dataY) return;

    KnowledgeGapEntry gapCopy;
    int action = 0;
    {
        std::lock_guard<std::mutex> lock(gapMutex_);
        if (gapQueue_.empty()) return;

        int row = static_cast<int>((m.y - dataY) / kRowHeight);
        if (row < 0 || row >= static_cast<int>(gapQueue_.size())) return;

        hoveredGapRow_ = row;
        if (!input.mousePressed[0]) return;

        float createX  = x + 600.0f;
        float dismissX = x + 670.0f;

        if (m.x >= dismissX && m.x <= dismissX + 70.0f) {
            gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(row));
            hoveredGapRow_ = -1;
            action = 1;
        } else if (m.x >= createX && m.x <= createX + 60.0f) {
            gapCopy = gapQueue_[static_cast<size_t>(row)];
            gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(row));
            hoveredGapRow_ = -1;
            action = 2;
        }
    }

    if (action == 2) {
        prefillCreatorFromGap(gapCopy);
    }
}

void UITrainingPanel::dismissGap(size_t index) {
    std::lock_guard<std::mutex> lock(gapMutex_);
    if (index < gapQueue_.size())
        gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(index));
}

void UITrainingPanel::createFromGap(size_t index) {
    KnowledgeGapEntry gap;
    {
        std::lock_guard<std::mutex> lock(gapMutex_);
        if (index >= gapQueue_.size()) return;
        gap = gapQueue_[index];
        gapQueue_.erase(gapQueue_.begin() + static_cast<ptrdiff_t>(index));
    }
    prefillCreatorFromGap(gap);
}

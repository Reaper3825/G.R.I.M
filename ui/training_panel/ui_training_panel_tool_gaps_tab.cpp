// UITrainingPanel: Tool Gaps tab
#include "ui_training_panel_internal.hpp"

using namespace GRIMText;
using namespace UITheme;
using namespace UITrainingPanelDetail;

// ============================================================
// Tool Gap Intake
// ============================================================

void UITrainingPanel::pushToolGap(GRIM::MMO::ToolGapProposal proposal) {
    std::lock_guard<std::mutex> lock(toolGapMutex_);
    toolGapQueue_.push_back(std::move(proposal));
}

size_t UITrainingPanel::pendingToolGapCount() const {
    std::lock_guard<std::mutex> lock(toolGapMutex_);
    return toolGapQueue_.size();
}

// ============================================================
// Tool Gaps Tab
// ============================================================

void UITrainingPanel::drawToolGapsTab(OverlayRenderer& renderer, const PanelRect& content) {
    std::lock_guard<std::mutex> lock(toolGapMutex_);

    float x = content.origin.x + Spacing::PaddingX;
    float y = content.origin.y + Spacing::Small;
    float w = content.size.x - 2.0f * Spacing::PaddingX;

    UIDrawHelpers::drawSectionHeader(renderer, {x - Spacing::PaddingX, y}, content.size.x,
                                     "Tool Gap Proposals", Colors::SectionNeutral);
    y += Sizes::HeaderHeight + Spacing::Small;

    if (toolGapQueue_.empty()) {
        renderer.drawText({x, y}, "No tool gaps queued.", Colors::TextMuted);
        renderer.drawText({x, y + 20.0f},
            "Tool gaps appear when the model needs a capability not in the ToolRegistry.", Colors::TextMuted);
        return;
    }

    // Column headers
    renderer.drawText({x, y}, "Missing Capability", Colors::TextSecondary);
    renderer.drawText({x + 300.0f, y}, "Reason", Colors::TextSecondary);
    renderer.drawText({x + 500.0f, y}, "Proposed Tool", Colors::TextSecondary);
    renderer.drawText({x + 700.0f, y}, "Actions", Colors::TextSecondary);
    y += 22.0f;

    UIDrawHelpers::drawDivider(renderer, {x, y}, w);
    y += 4.0f;

    for (size_t i = 0; i < toolGapQueue_.size(); ++i) {
        const auto& proposal = toolGapQueue_[i];

        if (static_cast<int>(i) == hoveredToolGapRow_) {
            renderer.drawRoundedRect({x, y - 1.0f}, {w, kRowHeight * 2.0f}, Colors::ContentAreaBg, 4.0f);
        }

        renderer.drawText({x, y}, proposal.missing_capability, Colors::TextPrimary);

        std::string reasonStr;
        switch (proposal.reason) {
            case GRIM::MMO::ToolGapReason::NoMatchingCapability:   reasonStr = "No match"; break;
            case GRIM::MMO::ToolGapReason::CapabilityMismatch:     reasonStr = "Mismatch"; break;
            case GRIM::MMO::ToolGapReason::PermissionInsufficient: reasonStr = "Permissions"; break;
            case GRIM::MMO::ToolGapReason::PolicyBlocked:          reasonStr = "Policy"; break;
        }
        renderer.drawText({x + 300.0f, y}, reasonStr, Colors::Warning);
        renderer.drawText({x + 500.0f, y}, proposal.proposed_spec.display_name, Colors::TextSecondary);
        renderer.drawText({x + 700.0f, y}, "[Approve]", Colors::Success);
        renderer.drawText({x + 780.0f, y}, "[Dismiss]", Colors::Danger);

        // Second row: rationale
        if (!proposal.rationale.empty()) {
            std::string rationale = proposal.rationale;
            if (rationale.size() > 100) rationale = rationale.substr(0, 97) + "...";
            renderer.drawText({x + 20.0f, y + kRowHeight}, rationale, Colors::TextMuted);
        }

        y += kRowHeight * 2.0f;
    }
}

void UITrainingPanel::processToolGapClicks(const InputState& input, const PanelRect& content) {
    Vec2 m = input.mousePos;
    float x = content.origin.x + Spacing::PaddingX;
    float w = content.size.x - 2.0f * Spacing::PaddingX;
    float dataY = content.origin.y + Sizes::HeaderHeight + Spacing::Medium + Spacing::Small + 26.0f;

    hoveredToolGapRow_ = -1;

    if (m.x < x || m.x > x + w || m.y < dataY) return;

    std::lock_guard<std::mutex> lock(toolGapMutex_);
    if (toolGapQueue_.empty()) return;

    int row = static_cast<int>((m.y - dataY) / (kRowHeight * 2.0f));
    if (row < 0 || row >= static_cast<int>(toolGapQueue_.size())) return;

    hoveredToolGapRow_ = row;
    if (!input.mousePressed[0]) return;

    float approveX = x + 700.0f;
    float dismissX = x + 780.0f;

    if (m.x >= dismissX && m.x <= dismissX + 70.0f) {
        toolGapQueue_.erase(toolGapQueue_.begin() + static_cast<ptrdiff_t>(row));
        hoveredToolGapRow_ = -1;
    } else if (m.x >= approveX && m.x <= approveX + 70.0f) {
        // For now, just log approval — actual tool scaffolding is a separate pipeline
        LOG_DEBUG("UITrainingPanel", "Tool gap approved: " + toolGapQueue_[static_cast<size_t>(row)].missing_capability);
        toolGapQueue_.erase(toolGapQueue_.begin() + static_cast<ptrdiff_t>(row));
        hoveredToolGapRow_ = -1;
    }
}

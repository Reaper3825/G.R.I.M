#include "ui_training_tools_view.hpp"
#include "ui/overlay_renderer.hpp"
#include "ui/ui_draw_helpers.hpp"
#include "ui/ui_theme.hpp"
#include "core/input_parser.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

using namespace GRIM::MMO;
using namespace UITheme;
namespace {
constexpr float rowHeight = 48.0f;
std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}
const char* status(ToolSwapState state) {
    switch (state) {
        case ToolSwapState::Loaded: return "Loaded";
        case ToolSwapState::Loading: return "Loading";
        case ToolSwapState::Unloading: return "Unloading";
        case ToolSwapState::Unavailable: return "Unavailable";
    }
    return "Unknown";
}
std::string documented(const std::string& text) { return text.empty() ? "Not documented" : text; }
void strings(std::ostringstream& out, const char* label, const std::vector<std::string>& values) {
    out << '\n' << label << ":";
    if (values.empty()) out << " Not documented\n";
    else { out << '\n'; for (const auto& value : values) out << "  " << value << '\n'; }
}
std::string detailText(const ToolDescriptor& tool) {
    std::ostringstream out;
    out << tool.tool_id << "\n" << documented(tool.display_name) << "\n\n"
        << documented(tool.description) << "\n\n"
        << "Category: " << documented(tool.category) << '\n'
        << "Provider: " << (tool.provider_type == ToolProviderType::Plugin ? "Plugin" : "Built-in")
        << " / " << documented(tool.provider_name) << '\n'
        << "Source file: " << documented(tool.provider_path) << '\n'
        << "Version: " << documented(tool.version) << '\n'
        << "Status: " << status(tool.swap_state) << '\n'
        << "Type: " << (tool.is_informational ? "Informational" : "State-changing") << '\n'
        << "Confirmation required: " << (tool.needs_confirmation ? "Yes" : "No") << '\n'
        << "Permission bits: " << tool.permission_bits << "\n\n"
        << "Argument format: " << documented(tool.argument_format) << '\n'
        << "Argument schema: " << documented(tool.argument_schema_json) << "\n\n"
        << "Usage: " << documented(tool.usage) << "\n\nParameters:";
    if (tool.parameters.empty()) out << " Not documented\n";
    else for (const auto& p : tool.parameters)
        out << "\n  " << p.name << " (" << documented(p.type) << ", "
            << (p.required ? "required" : "optional") << ")\n    " << documented(p.description) << '\n';
    strings(out, "Examples", tool.examples);
    strings(out, "Aliases", tool.aliases);
    strings(out, "Keywords", tool.keywords);
    strings(out, "Preconditions", tool.preconditions);
    strings(out, "Context requirements", tool.context_requirements);
    strings(out, "Capability tags", tool.capability_tags);
    strings(out, "Affordance tokens", tool.affordance_tokens);
    out << "\nUsage count: " << tool.usage_count << '\n';
    if (tool.usage_count == 0) out << "Success rate: No recorded uses\n";
    else out << "Success rate: " << tool.success_rate * 100.0f << "%\n";
    out << "\nRegistry metadata describes registration; it does not verify readiness or enforce policy.";
    return out.str();
}
}

UITrainingToolsView::UITrainingToolsView()
    : details_("Tool details", "Select a registered tool to inspect its metadata.", nullptr) {
    search_.setPlaceholder("Search tools, descriptions, categories, providers or aliases");
    details_.setReadOnly(true);
}
void UITrainingToolsView::layout(const PanelRect& content) {
    const float x = content.origin.x + Spacing::PaddingX;
    const float y = content.origin.y + 36.0f;
    const float width = std::max(0.0f, content.size.x - 2 * Spacing::PaddingX);
    search_.setPosition(x, y);
    search_.setSize(width, 28);
    const float listWidth = std::max(0.0f, (width - 12) * 0.52f);
    const float height = std::max(0.0f, content.size.y - 100);
    list_.setPosition(x, y + 58);
    list_.setSize(listWidth, height);
    details_.setPosition(x + listWidth + 12, y + 58);
    details_.setSize(std::max(0.0f, width - listWidth - 12), height);
}
void UITrainingToolsView::filter() {
    matches_.clear();
    const auto needle = lower(query_);
    for (size_t i = 0; i < tools_.size(); ++i) {
        const auto& tool = tools_[i];
        std::string text = tool.tool_id + " " + tool.display_name + " " + tool.description + " " + tool.category + " " + tool.provider_name;
        for (const auto& alias : tool.aliases) text += " " + alias;
        for (const auto& tag : tool.capability_tags) text += " " + tag;
        if (lower(text).find(needle) != std::string::npos) matches_.push_back(i);
    }
    list_.setContentHeight(static_cast<float>(matches_.size()) * rowHeight);
}
void UITrainingToolsView::showSelection() {
    const auto selected = std::find_if(tools_.begin(), tools_.end(), [this](const auto& tool) { return tool.tool_id == selectedId_; });
    const auto text = selected == tools_.end()
        ? "Select a registered tool to inspect its metadata." : detailText(*selected);
    if (details_.getText() != text) details_.setText(text);
}
void UITrainingToolsView::refresh() {
    tools_ = ToolRegistry::instance().getAllTools();
    if (std::none_of(tools_.begin(), tools_.end(), [this](const auto& tool) { return tool.tool_id == selectedId_; })) selectedId_.clear();
    filter();
    showSelection();
}
void UITrainingToolsView::update(const InputState& input, float dt, const PanelRect& content, uint64_t panelId) {
    layout(content);
    for (Widget* widget : {static_cast<Widget*>(&search_), static_cast<Widget*>(&list_), static_cast<Widget*>(&details_)}) widget->setPanelID(panelId);
    refreshTimer_ += dt;
    if (refreshTimer_ >= 2.0f) { refreshTimer_ = 0; refresh(); }
    const auto clipped = input.clipped(content.origin, content.size);
    search_.update(clipped, dt);
    if (query_ != search_.getText()) { query_ = search_.getText(); filter(); list_.scrollTo(0); }
    list_.update(clipped, dt);
    const auto routed = list_.routeInput(clipped);
    const auto pos = list_.getPosition();
    const auto size = list_.getSize();
    const auto mouse = routed.mousePos;
    if (routed.mousePressed[0] && mouse.x >= pos.x && mouse.x < pos.x + size.x - 16 && mouse.y >= pos.y && mouse.y < pos.y + size.y) {
        const auto row = static_cast<size_t>((mouse.y - pos.y + list_.getScrollOffset()) / rowHeight);
        if (row < matches_.size()) { selectedId_ = tools_[matches_[row]].tool_id; showSelection(); details_.scrollToTop(); }
    }
    details_.update(clipped, dt);
}
void UITrainingToolsView::draw(OverlayRenderer& renderer, const PanelRect& content) {
    layout(content);
    renderer.pushClipRect(content.origin, content.size);
    UIDrawHelpers::drawSectionHeader(renderer, content.origin, content.size.x, "Registered Tools", Colors::SectionAI);
    search_.drawOverlay(renderer, {});
    const auto p = list_.getPosition();
    renderer.drawText({p.x, p.y - 23}, std::to_string(matches_.size()) + " shown / " + std::to_string(tools_.size()) + " registered", Colors::TextSecondary);
    list_.drawOverlay(renderer, {});
    renderer.pushClipRect(p, {std::max(0.0f, list_.getSize().x - 16), list_.getSize().y});
    float y = p.y - list_.getScrollOffset();
    for (const auto index : matches_) {
        const auto& tool = tools_[index];
        if (y + rowHeight > p.y && y < p.y + list_.getSize().y) {
            const float width = std::max(0.0f, list_.getSize().x - 24);
            if (tool.tool_id == selectedId_) renderer.drawRect({p.x + 2, y}, {width, rowHeight - 2}, Colors::PanelBg);
            renderer.drawText({p.x + 6, y + 3}, UIDrawHelpers::truncateText(tool.tool_id, std::max(0.0f, width - 12)), Colors::TextPrimary);
            const auto summary = documented(tool.category) + " | " + documented(tool.provider_name) + " | " + status(tool.swap_state);
            renderer.drawText({p.x + 6, y + 24}, UIDrawHelpers::truncateText(summary, std::max(0.0f, width - 12)), Colors::TextSecondary);
        }
        y += rowHeight;
    }
    if (matches_.empty()) renderer.drawText({p.x + 6, p.y + 8}, tools_.empty() ? "No tools registered." : "No matching tools.", Colors::TextMuted);
    renderer.popClipRect();
    details_.drawOverlay(renderer, {});
    renderer.popClipRect();
}
void UITrainingToolsView::hide() {
    search_.cancelInput();
    list_.cancelInput();
    details_.cancelInput();
    refreshTimer_ = 2.0f;
}

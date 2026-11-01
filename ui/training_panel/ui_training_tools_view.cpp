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
        << "Usage type: " << toolUsageName(tool.usage_type) << '\n'
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
    : usageFilter_("Usage type", {"All usage types", "Debug", "Model", "System", "Ignore"}, 0,
          [this](int, const std::string&) { filter(); list_.scrollTo(0); }),
      pluginFilter_("Plugin", {"All plugins"}, 0,
          [this](int, const std::string&) { filter(); list_.scrollTo(0); }),
      details_("Tool details", "Select a registered tool to inspect its metadata.", nullptr) {
    search_.setPlaceholder("Search tools, descriptions, categories, providers or aliases");
    details_.setReadOnly(true);
    usageFilter_.setFullWidth(true);
    pluginFilter_.setFullWidth(true);
}
void UITrainingToolsView::layout(const PanelRect& content) {
    const float x = content.origin.x + Spacing::PaddingX;
    const float y = content.origin.y + 36.0f;
    const float width = std::max(0.0f, content.size.x - 2 * Spacing::PaddingX);
    search_.setPosition(x, y);
    search_.setSize(width, 28);
    const float listWidth = std::max(0.0f, (width - 12) * 0.52f);
    const float filterWidth = std::max(0.0f, (width - 12) / 2);
    usageFilter_.setPosition(x, y + 56);
    usageFilter_.setSize(filterWidth, 28);
    pluginFilter_.setPosition(x + filterWidth + 12, y + 56);
    pluginFilter_.setSize(filterWidth, 28);
    const int visibleOptions = std::max(1, std::min(8, static_cast<int>((content.size.y - 126) / 25)));
    usageFilter_.setMaxVisibleItems(visibleOptions);
    pluginFilter_.setMaxVisibleItems(visibleOptions);
    const float height = std::max(0.0f, content.size.y - 154);
    list_.setPosition(x, y + 112);
    list_.setSize(listWidth, height);
    details_.setPosition(x + listWidth + 12, y + 112);
    details_.setSize(std::max(0.0f, width - listWidth - 12), height);
}
void UITrainingToolsView::filter() {
    matches_.clear();
    const auto needle = lower(query_);
    for (size_t i = 0; i < tools_.size(); ++i) {
        const auto& tool = tools_[i];
        const int usage = usageFilter_.getSelectedIndex();
        if (usage > 0 && tool.usage_type != static_cast<ToolUsage>(usage - 1)) continue;
        const int provider = pluginFilter_.getSelectedIndex();
        if (provider > 0 && provider <= static_cast<int>(providers_.size())) {
            const auto& selected = providers_[provider - 1];
            if (tool.provider_type != selected.first ||
                (selected.first == ToolProviderType::Plugin && tool.provider_name != selected.second)) continue;
        }
        std::string text = tool.tool_id + " " + tool.display_name + " " + tool.description + " " + tool.category + " " + tool.provider_name;
        for (const auto& alias : tool.aliases) text += " " + alias;
        for (const auto& tag : tool.capability_tags) text += " " + tag;
        if (lower(text).find(needle) != std::string::npos) matches_.push_back(i);
    }
    list_.setContentHeight(static_cast<float>(matches_.size()) * rowHeight);
    if (std::none_of(matches_.begin(), matches_.end(), [this](size_t index) { return tools_[index].tool_id == selectedId_; }))
        selectedId_.clear();
    showSelection();
}
void UITrainingToolsView::showSelection() {
    const auto selected = std::find_if(tools_.begin(), tools_.end(), [this](const auto& tool) { return tool.tool_id == selectedId_; });
    const auto text = selected == tools_.end()
        ? "Select a registered tool to inspect its metadata." : detailText(*selected);
    if (details_.getText() != text) details_.setText(text);
}
void UITrainingToolsView::refreshProviders() {
    std::vector<std::pair<ToolProviderType, std::string>> providers;
    for (const auto& tool : tools_)
        providers.emplace_back(tool.provider_type, tool.provider_type == ToolProviderType::Plugin ? tool.provider_name : "");
    std::sort(providers.begin(), providers.end());
    providers.erase(std::unique(providers.begin(), providers.end()), providers.end());
    if (providers == providers_) return; // Keep open menus stable during periodic refresh.
    const int previous = pluginFilter_.getSelectedIndex();
    int selected = 0;
    if (previous > 0 && previous <= static_cast<int>(providers_.size())) {
        const auto found = std::find(providers.begin(), providers.end(), providers_[previous - 1]);
        if (found != providers.end()) selected = 1 + static_cast<int>(found - providers.begin());
    }
    providers_ = std::move(providers);
    std::vector<std::string> labels{"All plugins"};
    for (const auto& provider : providers_)
        labels.push_back(provider.first == ToolProviderType::Plugin ? provider.second : "Built-in (host)");
    pluginFilter_.cancelInput();
    pluginFilter_.collapse();
    pluginFilter_.setItems(labels);
    pluginFilter_.setSelectedIndex(selected);
}
void UITrainingToolsView::refresh() {
    tools_ = ToolRegistry::instance().getAllTools();
    refreshProviders();
    filter();
}
void UITrainingToolsView::update(const InputState& input, float dt, const PanelRect& content, uint64_t panelId) {
    layout(content);
    for (Widget* widget : {static_cast<Widget*>(&search_), static_cast<Widget*>(&usageFilter_),
         static_cast<Widget*>(&pluginFilter_), static_cast<Widget*>(&list_), static_cast<Widget*>(&details_)}) widget->setPanelID(panelId);
    refreshTimer_ += dt;
    if (refreshTimer_ >= 2.0f) { refreshTimer_ = 0; refresh(); }
    const auto clipped = input.clipped(content.origin, content.size);
    // Popup ownership prevents menu clicks/wheel events from reaching list rows.
    if (usageFilter_.isExpanded()) { usageFilter_.update(clipped, dt); return; }
    if (pluginFilter_.isExpanded()) { pluginFilter_.update(clipped, dt); return; }
    usageFilter_.update(clipped, dt);
    pluginFilter_.update(clipped, dt);
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
    for (auto* dropdown : {&usageFilter_, &pluginFilter_}) {
        const auto position = dropdown->getPosition();
        renderer.drawText({position.x, position.y - 18}, dropdown->getLabel(), Colors::TextSecondary);
        renderer.pushClipRect(position, dropdown->getSize());
        dropdown->drawOverlay(renderer, {});
        renderer.popClipRect();
    }
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
    for (auto* dropdown : {&usageFilter_, &pluginFilter_}) {
        const auto position = dropdown->getPosition();
        renderer.pushClipRect(position, {dropdown->getSize().x, std::max(0.0f, content.origin.y + content.size.y - position.y)});
        dropdown->drawExpandedList(renderer, {});
        renderer.popClipRect();
    }
    renderer.popClipRect();
}
void UITrainingToolsView::hide() {
    search_.cancelInput();
    list_.cancelInput();
    details_.cancelInput();
    for (auto* dropdown : {&usageFilter_, &pluginFilter_}) { dropdown->cancelInput(); dropdown->collapse(); }
    refreshTimer_ = 2.0f;
}

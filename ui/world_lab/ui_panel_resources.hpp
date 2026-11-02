#pragma once
#include "ui_tool_contract.hpp"
#include <algorithm>
#include <map>

namespace GRIM::WorldLab::UITools {
struct PanelResource { std::string key, title; uint64_t runtime; };
// One observed directory per World Lab session; contains no executable handlers.
class PanelResources {
    using Panel = PanelResource;
    std::map<std::string, Panel> panels_;
    Json roster_ = Json::array();
    uint64_t revision_ = 0, registryVersion_ = 0;
    bool discovered_ = false;
    static std::string folded(std::string text) {
        for (auto& c : text) if (c >= 'A' && c <= 'Z') c += 'a' - 'A';
        return text;
    }
public:
    void clear() { panels_.clear(); roster_ = Json::array(); discovered_ = false; registryVersion_ = 0; }
    uint64_t revision() const { return revision_; }
    const Json& roster() const { return roster_; }
    void refresh(const Json& result, uint64_t version) {
        std::map<std::string, Panel> next;
        require(result.at("panels").is_array() && result.at("panels").size() <= 1000,
                "Invalid or oversized panel directory");
        for (const auto& panel : result.at("panels")) {
            Panel p{panel.at("panel_id").get<std::string>(), panel.at("title").get<std::string>(),
                    panel.at("runtime_id").get<uint64_t>()};
            require(!p.key.empty() && p.key.size() <= 256 && p.runtime &&
                    p.key.find_first_of("\r\n\0", 0, 3) == std::string::npos,
                    "Invalid panel identity");
            require(next.emplace("panel:" + p.key, p).second, "Duplicate panel identity");
        }
        auto roster = Json::array();
        for (const auto& [ref, panel] : next)
            roster.push_back({{"reference", ref}, {"name", panel.title}, {"type", "t"}});
        // Include runtime generation internally so recreation invalidates the directory.
        bool changed = roster != roster_ || next.size() != panels_.size();
        for (const auto& [ref, p] : next) {
            auto old = panels_.find(ref);
            changed |= old == panels_.end() || old->second.runtime != p.runtime;
        }
        if (!discovered_ || changed) ++revision_;
        panels_ = std::move(next); roster_ = std::move(roster);
        discovered_ = true; registryVersion_ = version;
    }
    std::string listingFeedback() const {
        // Visible observations use names, which the resolver already accepts.
        // Show references only when a name cannot identify the resource safely.
        std::map<std::string, size_t> nameCounts;
        for (const auto& [ref, panel] : panels_) ++nameCounts[folded(panel.title)];
        std::string text = panels_.empty() ? "No UI panels available." : "UI panels:";
        for (const auto& [ref, panel] : panels_) {
            const auto& name = panel.title;
            const bool usable = !name.empty() && name.size() <= 480 &&
                name.front() != ' ' && name.back() != ' ' &&
                std::none_of(name.begin(), name.end(), [](unsigned char c) { return c < 32 || c == 127; });
            text += "\n- " + (usable ? name : ref);
            if (usable && (nameCounts[folded(name)] > 1 || panels_.count(name))) text += " (" + ref + ")";
        }
        return text;
    }
    Panel resolve(const std::string& target, uint64_t version) const {
        require(discovered_ && registryVersion_ == version, "Run ui.list to discover current panel references first");
        require(!target.empty(), "A discovered panel reference or exact name is required");
        const Panel* selected = nullptr;
        const auto exact = panels_.find(target);
        if (exact != panels_.end()) selected = &exact->second;
        else {
            for (const auto& [ref, p] : panels_) if (folded(p.title) == folded(target)) {
                require(!selected, "Ambiguous panel name; use a panel: reference from ui.list");
                selected = &p;
            }
        }
        require(selected != nullptr, "Unknown panel; run ui.list and use an observed reference");
        return *selected;
    }
    static void project(const Panel& selected, const Json& panel, Json& output) {
        require(panel.at("panel_id") == selected.key && panel.at("runtime_id") == selected.runtime,
                "Panel changed during execution; inspect current state before retrying");
        output["resource"] = {{"reference", "panel:" + selected.key}, {"name", panel.at("title")},
            {"type", "t"}, {"visible", panel.at("visible")}, {"minimized", panel.at("minimized")},
            {"maximized", panel.at("maximized")}};
        std::string text = selected.title.empty() ? "panel:" + selected.key : selected.title;
        text += panel.at("visible").get<bool>() ? ": visible" : ": hidden";
        if (panel.at("minimized").get<bool>()) text += ", minimized";
        if (panel.at("maximized").get<bool>()) text += ", maximized";
        output["feedback"] = text + ".";
    }
};
}

#pragma once
#include "ui_panel_resources.hpp"
#include "ui_tool_completion.hpp"
#include <array>
#include <string_view>

namespace GRIM::WorldLab::UITools {
enum class PanelAction { Discover, Inspect, Mutate };

// Host adapter registrations carry mappings only. ToolRegistry is the sole handler
// owner and checks the validated registry revision on every invocation.
struct PanelAdapter {
    std::string_view toolId;
    PanelAction action;
    std::string_view placeholder;
    std::string_view argument;
    std::string_view generationArgument;

    void validate(const MMO::ToolDescriptor& tool) const {
        if (action == PanelAction::Discover) return;
        require(tool.bindings == std::map<std::string, std::string>{{std::string(placeholder), "t"}},
                "Unsupported UI binding template");
        require(std::any_of(tool.parameters.begin(), tool.parameters.end(), [&](const auto& p) {
            return p.name == argument && p.type == "string" && p.required;
        }), "UI adapter requires a panel_id string parameter");
    }
    Json arguments(const PanelResource& panel, bool guarded) const {
        Json args = {{std::string(argument), panel.key}};
        if (guarded && !generationArgument.empty()) args[std::string(generationArgument)] = panel.runtime;
        return args;
    }
};

inline constexpr std::array<PanelAdapter, 4> panelAdapters{{
    {"ui.list", PanelAction::Discover, {}, {}, {}},
    {"ui.inspect", PanelAction::Inspect, "t", "panel_id", {}},
    {"ui.show", PanelAction::Mutate, "t", "panel_id", "expected_runtime_id"},
    {"ui.hide", PanelAction::Mutate, "t", "panel_id", "expected_runtime_id"},
}};

inline const PanelAdapter& panelAdapter(std::string_view id) {
    const auto found = std::find_if(panelAdapters.begin(), panelAdapters.end(),
                                    [&](const auto& adapter) { return adapter.toolId == id; });
    require(found != panelAdapters.end(), "Unsupported World Lab UI action");
    return *found;
}

// UI family orchestration. Directory state belongs to this session; request
// completion is stateless and execution belongs to the loaded provider.
class PanelToolAdapter {
    PanelResources resources_;
public:
    void clear() { resources_.clear(); }
    Json execute(const PanelAdapter& mapping, const std::string& target,
                 const std::function<bool()>& cancelled) {
        const auto version = MMO::ToolRegistry::instance().version();
        const std::string id(mapping.toolId);
        const auto tool = descriptor(id, version);
        mapping.validate(tool);
        Json output = Json::object();
        if (mapping.action == PanelAction::Discover) {
            require(target.empty(), "Use ui.list without arguments");
            const auto result = PanelRequestCompletion::complete(id, Json::object(), version, tool, cancelled);
            resources_.refresh(result, version);
            output["resources"] = resources_.roster();
            output["feedback"] = resources_.listingFeedback();
        } else {
            const auto selected = resources_.resolve(target, version);
            const auto inspector = descriptor("ui.inspect", version);
            panelAdapter("ui.inspect").validate(inspector);
            require(sameProvider(inspector, tool), "UI inspect provider mismatch");
            auto observed = PanelRequestCompletion::complete("ui.inspect",
                panelAdapter("ui.inspect").arguments(selected, false), version, inspector, cancelled);
            require(observed.at("panel").at("panel_id") == selected.key &&
                    observed.at("panel").at("runtime_id") == selected.runtime,
                    "Panel was replaced; run ui.list again");
            auto result = mapping.action == PanelAction::Inspect ? std::move(observed) :
                PanelRequestCompletion::complete(id, mapping.arguments(selected, true), version, tool, cancelled);
            PanelResources::project(selected, result.at("panel"), output);
        }
        output["revision"] = resources_.revision();
        return output;
    }
};
}

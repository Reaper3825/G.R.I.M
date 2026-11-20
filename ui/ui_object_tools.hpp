#pragma once
#include "panel_tool_completion.hpp"
#include <algorithm>
#include <array>
#include <string_view>

namespace GRIM::UI {
// Maps the object-facing tool arguments to native UI operations.
class ObjectTools {
    static std::string reference(const MMO::ResourceDescriptor& resource) {
        return resource.resource_id + "@" + std::to_string(resource.runtime_id);
    }

    static Json object(const MMO::ResourceDescriptor& resource) {
        return {{"reference", reference(resource)}, {"name", resource.name},
                {"kind", "object"}, {"state", resource.state}};
    }

public:
    static constexpr std::array<std::string_view, 4> ids{
        "ui.list", "ui.inspect", "ui.show", "ui.hide"
    };

    static Json execute(const std::string& id, const std::string& target, uint64_t version,
                        const std::function<bool()>& cancelled) {
        require(std::find(ids.begin(), ids.end(), id) != ids.end(), "Unsupported object tool");
        require(!cancelled(), "Object action cancelled before submission");
        panelTool(id, version);
        const auto snapshot = MMO::ProviderRegistry::instance().snapshot();
        require(snapshot.revision == version, "Tool registry changed; retry the action");

        if (id == "ui.list") {
            require(target.empty(), "Use ui.list without arguments");
            Json resources = Json::array();
            std::string feedback = "Objects:";
            for (const auto& provider : snapshot.providers) {
                for (const auto& resource : provider.resources) {
                    if (resource.kind != "ui.panel") continue;
                    resources.push_back(object(resource));
                    feedback += "\n- " + resource.name + " (" + reference(resource) + ")";
                }
            }
            return {{"resources", std::move(resources)}, {"feedback", std::move(feedback)}};
        }

        for (const auto& provider : snapshot.providers) {
            for (const auto& resource : provider.resources) {
                if (resource.kind != "ui.panel" || reference(resource) != target) continue;
                require(resource.resource_id.starts_with("panel:"), "Invalid UI resource key");
                Json args = {{"panel_id", resource.resource_id.substr(6)}};
                if (id != "ui.inspect") args["expected_runtime_id"] = resource.runtime_id;
                const auto result = PanelRequestCompletion::complete(id, args, version, cancelled);
                const auto& panel = result.at("panel");
                require(panel.at("panel_id") == args.at("panel_id") &&
                        panel.at("runtime_id") == resource.runtime_id, "Object changed during execution");
                auto updated = resource;
                updated.name = panel.at("title").get<std::string>();
                for (const auto* field : {"visible", "minimized", "maximized", "position", "size", "z_order"}) {
                    if (panel.contains(field)) updated.state[field] = panel.at(field);
                }
                const auto feedback = updated.name + (panel.at("visible").get<bool>() ? ": visible." : ": hidden.");
                return {{"resource", object(updated)}, {"feedback", feedback}};
            }
        }
        throw std::runtime_error("Unknown or stale object reference; list current objects");
    }
};
}

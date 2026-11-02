#pragma once
#include "MMO/Core/ToolRegistry.hpp"
#include <nlohmann/json.hpp>
#include <sstream>

namespace GRIM::WorldLab {
// Shared host projection for catalog opening and entry inspection. Never invokes tools.
inline void resolveHostObservation(nlohmann::json& response) {
    if (!response.value("ok", false) || !response.contains("result")) return;
    auto& result = response.at("result");
    if (!result.contains("state")) return;
    auto& state = result.at("state");
    if (!state.contains("observation_request")) return;
    const auto request = state.at("observation_request");
    state.erase("observation_request");
    if (request.at("provider") == "tool_details") {
        const auto id = request.at("tool_id").get<std::string>();
        const auto tool = MMO::ToolRegistry::instance().getTool(id);
        std::ostringstream text;
        if (!tool || tool->tool_id != id || tool->usage_type != MMO::ToolUsage::Model ||
            tool->swap_state != MMO::ToolSwapState::Loaded) {
            text << "This tool is no longer available in the catalog.";
            state["observation_error"] = "Tool unavailable";
        } else {
            auto parameters = nlohmann::json::array();
            text << tool->tool_id << '\n' << tool->description << "\n\nTool usage: " << tool->usage;
            if (!tool->parameters.empty()) text << "\n\nArguments:";
            for (const auto& parameter : tool->parameters) {
                parameters.push_back({{"name", parameter.name}, {"type", parameter.type},
                    {"description", parameter.description}, {"required", parameter.required}});
                text << "\n" << parameter.name << (parameter.required ? " (required): " : " (optional): ")
                     << parameter.description;
            }
            state["tool_details"] = {{"entity_id", request.at("entity_id")}, {"tool_id", tool->tool_id},
                {"description", tool->description}, {"usage", tool->usage}, {"parameters", std::move(parameters)}};
        }
        if (state.contains("location_path")) text << "\n\nLOCATION: " << state.at("location_path").get<std::string>();
        state["feedback"] = text.str();
        return;
    }
    if (request.at("provider") != "tool_catalog") {
        state["feedback"] = "Observation provider unavailable.";
        state["observation_error"] = "Unknown host observation provider";
        return;
    }
    auto entries = nlohmann::json::array();
    std::ostringstream text;
    text << "Tool catalog:\n";
    for (const auto& tool : MMO::ToolRegistry::instance().getModelTools()) {
        const auto summary = tool.short_description.empty()
            ? "Short description unavailable for this registered tool" : tool.short_description;
        entries.push_back({{"tool_id", tool.tool_id},
            {"name", tool.display_name.empty() ? tool.tool_id : tool.display_name},
            {"short_description", summary}});
        text << tool.tool_id << " - " << summary << '\n';
    }
    if (entries.empty()) text << "No model tools are currently available.\n";
    state["tool_catalog"] = {{"entity_id", request.at("entity_id")}, {"tools", std::move(entries)}};
    if (state.contains("location_path")) text << "\nLOCATION: " << state.at("location_path").get<std::string>();
    state["feedback"] = text.str();
}
}

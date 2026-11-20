#pragma once
#include "RegisteredTool.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>

// Version-1 read-only registry snapshot metadata. No handlers cross this boundary.
namespace GRIM::MMO {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ToolParameter, name, type, description, required)
// Provider type retains its documented numeric values.
inline void to_json(nlohmann::json& j, const ToolDescriptor& t) {
    j = nlohmann::json::object();
    j["tool_id"] = t.tool_id;
    j["display_name"] = t.display_name;
    j["version"] = t.version;
    j["description"] = t.description;
    j["short_description"] = t.short_description;
    j["usage"] = t.usage;
    j["bindings"] = t.bindings;
    // Older version-1 readers require this key.
    j["examples"] = nlohmann::json::array();
    j["category"] = t.category;
    j["needs_confirmation"] = t.needs_confirmation;
    j["is_informational"] = t.is_informational;
    j["argument_format"] = t.argument_format;
    j["argument_schema_json"] = t.argument_schema_json;
    j["parameters"] = t.parameters;
    j["preconditions"] = t.preconditions;
    j["aliases"] = t.aliases;
    j["keywords"] = t.keywords;
    j["capability_tags"] = t.capability_tags;
    j["swap_state"] = 0; // Legacy version-1 readers require the Loaded value.
    j["usage_count"] = t.usage_count;
    j["success_count"] = t.success_count;
    j["failure_count"] = t.failure_count;
    j["success_rate"] = t.success_rate;
    j["usage_type"] = t.usage_type;
}
// A manifest version supplies defaults for authoring fields omitted from sidecars.
inline void readToolDescriptorMetadata(const nlohmann::json& j, ToolDescriptor& t,
                                       const char* manifestVersion = nullptr) {
    j.at("tool_id").get_to(t.tool_id);
    j.at("description").get_to(t.description);
    j.at("usage").get_to(t.usage);
    j.at("category").get_to(t.category);
    j.at("needs_confirmation").get_to(t.needs_confirmation);
    j.at("is_informational").get_to(t.is_informational);
    j.at("argument_format").get_to(t.argument_format);

    std::string summary;
    if (manifestVersion) {
        t.display_name = j.value("display_name", t.tool_id);
        t.version = j.value("version", std::string(manifestVersion));
        if (validToolShortDescription(t.description)) summary = t.description;
    } else {
        j.at("display_name").get_to(t.display_name);
        j.at("version").get_to(t.version);
    }
    t.short_description = j.value("short_description", std::move(summary));
    if (!validToolShortDescription(t.short_description)) throw std::invalid_argument("Invalid short description");
    t.bindings = j.value("bindings", std::map<std::string, std::string>{});
    if (!validToolBindings(t)) throw std::invalid_argument("Invalid tool bindings");

    const auto readCollection = [&](const char* key, auto& values) {
        if (manifestVersion && !j.contains(key)) values.clear();
        else j.at(key).get_to(values);
    };
    readCollection("parameters", t.parameters);
    readCollection("preconditions", t.preconditions);
    readCollection("aliases", t.aliases);
    readCollection("keywords", t.keywords);
    readCollection("capability_tags", t.capability_tags);

    const auto usage = j.value("usage_type", nlohmann::json(static_cast<uint32_t>(ToolUsage::Model)));
    if (!usage.is_number_integer() || usage < 0 || usage > 3)
        throw std::invalid_argument("Invalid tool usage type");
    t.usage_type = static_cast<ToolUsage>(usage.get<uint32_t>());
}
inline void from_json(const nlohmann::json& j, ToolDescriptor& t) {
    readToolDescriptorMetadata(j, t);
    j.at("argument_schema_json").get_to(t.argument_schema_json);
    j.at("usage_count").get_to(t.usage_count);
    j.at("success_count").get_to(t.success_count);
    j.at("failure_count").get_to(t.failure_count);
    j.at("success_rate").get_to(t.success_rate);
}
inline void to_json(nlohmann::json& j, const RegisteredTool& tool) {
    to_json(j, static_cast<const ToolDescriptor&>(tool));
    // Older version-1 inspection readers require these keys.
    j["context_requirements"] = nlohmann::json::array();
    j["affordance_tokens"] = nlohmann::json::array();
    j["provider_type"] = tool.provider.type;
    j["provider_name"] = tool.provider.name;
    j["provider_instance"] = tool.provider.instance;
    j["provider_path"] = tool.provider.path;
    j["provider_version"] = tool.provider.version;
    j["provider_api_version"] = tool.provider.api_version;
}
inline void from_json(const nlohmann::json& j, RegisteredTool& tool) {
    from_json(j, static_cast<ToolDescriptor&>(tool));
    j.at("provider_type").get_to(tool.provider.type);
    j.at("provider_name").get_to(tool.provider.name);
    j.at("provider_instance").get_to(tool.provider.instance);
    j.at("provider_path").get_to(tool.provider.path);
    tool.provider.version = j.value("provider_version", std::string{});
    tool.provider.api_version = j.value("provider_api_version", uint32_t{});
}
}

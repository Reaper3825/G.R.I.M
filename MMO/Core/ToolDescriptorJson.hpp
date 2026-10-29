#pragma once
#include "ToolDescriptor.hpp"
#include <nlohmann/json.hpp>

// Version-1 read-only registry snapshot metadata. No handlers cross this boundary.
namespace GRIM::MMO {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ToolParameter, name, type, description, required)
// Provider/swap enums retain their documented numeric values from ToolDescriptor.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ToolDescriptor,
    tool_id, display_name, provider_type, provider_name, provider_instance, provider_path, version,
    description, usage, examples, category, permission_bits, needs_confirmation, is_informational,
    argument_format, argument_schema_json, parameters, preconditions, context_requirements,
    affordance_tokens, aliases, keywords, capability_tags, swap_state,
    usage_count, success_count, failure_count, success_rate)
}

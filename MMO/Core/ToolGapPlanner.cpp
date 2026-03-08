// MMO ToolGapPlanner implementation
//======================================================//
#include "ToolGapPlanner.hpp"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <sstream>

namespace GRIM::MMO {

// =========================================================
// Constructor
// =========================================================

ToolGapPlanner::ToolGapPlanner(const ToolRegistry& registry)
    : registry_(registry) {}

// =========================================================
// evaluate — check if capability can be satisfied
// =========================================================

std::optional<ToolGapProposal> ToolGapPlanner::evaluate(
    const std::string& capability,
    const std::string& request_id) const {

    if (capability.empty())
        throw std::runtime_error("ToolGapPlanner::evaluate: capability is empty");

    // Direct tool_id match
    if (registry_.isRegistered(capability)) {
        auto tool = registry_.getTool(capability);
        if (tool && tool->swap_state == ToolSwapState::Loaded) {
            return std::nullopt;  // tool exists and is available
        }
        // Tool exists but is not loaded
        ToolGapProposal proposal;
        proposal.missing_capability = capability;
        proposal.reason = ToolGapReason::CapabilityMismatch;
        proposal.closest_tool_id = capability;
        proposal.closest_tool_mismatch = "Tool exists but swap_state is not Loaded";
        proposal.rationale = "The tool '" + capability + "' is registered but currently unavailable.";
        proposal.original_request_id = request_id;
        return proposal;
    }

    // Alias resolution
    auto resolved = registry_.resolveAlias(capability);
    if (resolved) {
        return std::nullopt;  // alias points to a valid tool
    }

    // Capability tag search
    auto by_tag = registry_.getByCapabilityTag(capability);
    if (!by_tag.empty()) {
        return std::nullopt;  // existing tool covers this capability
    }

    // No match — build a gap proposal
    ToolGapProposal proposal;
    proposal.missing_capability = capability;
    proposal.reason = ToolGapReason::NoMatchingCapability;
    proposal.original_request_id = request_id;

    // Try to find the closest tool for context
    auto closest = findClosestTool(capability);
    if (closest) {
        proposal.closest_tool_id = closest->tool_id;
        proposal.closest_tool_mismatch =
            "Closest tool '" + closest->tool_id + "' exists but does not cover capability '" + capability + "'";
    }

    proposal.rationale = formatRationale(proposal);
    return proposal;
}

// =========================================================
// findClosestTool — keyword/tag similarity search
// =========================================================

std::optional<ToolDescriptor> ToolGapPlanner::findClosestTool(
    const std::string& capability) const {

    auto all_tools = registry_.getAllTools();
    if (all_tools.empty()) return std::nullopt;

    // Simple substring match on keywords and capability_tags.
    // This is intentionally naive — the model's structured proposal
    // is the primary gap signal, not this heuristic.
    const ToolDescriptor* best = nullptr;
    int best_score = 0;

    for (const auto& tool : all_tools) {
        int score = 0;
        for (const auto& kw : tool.keywords) {
            if (capability.find(kw) != std::string::npos ||
                kw.find(capability) != std::string::npos) {
                score += 2;
            }
        }
        for (const auto& tag : tool.capability_tags) {
            if (capability.find(tag) != std::string::npos ||
                tag.find(capability) != std::string::npos) {
                score += 3;
            }
        }
        if (score > best_score) {
            best_score = score;
            best = &tool;
        }
    }

    if (best) return *best;
    return std::nullopt;
}

// =========================================================
// parseProposedSpec — from model-emitted JSON
// =========================================================

ProposedToolSpec ToolGapPlanner::parseProposedSpec(const std::string& json_text) {
    if (json_text.empty())
        throw std::runtime_error("ToolGapPlanner::parseProposedSpec: json_text is empty");

    nlohmann::json doc;
    try {
        doc = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::parse_error& e) {
        throw std::runtime_error(
            std::string("ToolGapPlanner::parseProposedSpec: invalid JSON: ") + e.what());
    }

    ProposedToolSpec spec;

    if (!doc.contains("tool_id") || !doc["tool_id"].is_string() || doc["tool_id"].get<std::string>().empty())
        throw std::runtime_error("parseProposedSpec: missing or empty 'tool_id'");
    spec.tool_id = doc["tool_id"].get<std::string>();

    if (doc.contains("display_name") && doc["display_name"].is_string())
        spec.display_name = doc["display_name"].get<std::string>();
    if (doc.contains("description") && doc["description"].is_string())
        spec.description = doc["description"].get<std::string>();
    if (doc.contains("category") && doc["category"].is_string())
        spec.category = doc["category"].get<std::string>();
    if (doc.contains("needs_confirmation") && doc["needs_confirmation"].is_boolean())
        spec.needs_confirmation = doc["needs_confirmation"].get<bool>();

    if (doc.contains("parameters") && doc["parameters"].is_array()) {
        for (const auto& p : doc["parameters"]) {
            ToolParameter param;
            if (p.contains("name") && p["name"].is_string())
                param.name = p["name"].get<std::string>();
            if (p.contains("type") && p["type"].is_string())
                param.type = p["type"].get<std::string>();
            if (p.contains("description") && p["description"].is_string())
                param.description = p["description"].get<std::string>();
            if (p.contains("required") && p["required"].is_boolean())
                param.required = p["required"].get<bool>();
            spec.parameters.push_back(std::move(param));
        }
    }

    if (doc.contains("capability_tags") && doc["capability_tags"].is_array()) {
        for (const auto& tag : doc["capability_tags"]) {
            if (tag.is_string())
                spec.capability_tags.push_back(tag.get<std::string>());
        }
    }

    return spec;
}

// =========================================================
// formatRationale — user-visible explanation
// =========================================================

std::string ToolGapPlanner::formatRationale(const ToolGapProposal& proposal) {
    std::ostringstream out;

    out << "I cannot complete this task with the currently available tools.\n\n";
    out << "Missing capability: " << proposal.missing_capability << "\n";

    switch (proposal.reason) {
        case ToolGapReason::NoMatchingCapability:
            out << "Reason: No registered tool provides this capability.\n";
            break;
        case ToolGapReason::CapabilityMismatch:
            out << "Reason: A similar tool exists but doesn't match the needed parameters.\n";
            break;
        case ToolGapReason::PermissionInsufficient:
            out << "Reason: A matching tool exists but requires additional permissions.\n";
            break;
        case ToolGapReason::PolicyBlocked:
            out << "Reason: A matching tool exists but is blocked by current policy.\n";
            break;
    }

    if (!proposal.closest_tool_id.empty()) {
        out << "Closest existing tool: " << proposal.closest_tool_id << "\n";
        if (!proposal.closest_tool_mismatch.empty())
            out << "  Mismatch: " << proposal.closest_tool_mismatch << "\n";
    }

    if (!proposal.proposed_spec.tool_id.empty()) {
        out << "\nProposed new tool: " << proposal.proposed_spec.tool_id << "\n";
        if (!proposal.proposed_spec.description.empty())
            out << "  Description: " << proposal.proposed_spec.description << "\n";
    }

    return out.str();
}

} // namespace GRIM::MMO

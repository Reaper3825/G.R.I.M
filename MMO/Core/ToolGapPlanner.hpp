// MMO ToolGapPlanner — structured tool-gap proposals
//
// When the model detects that no tool in the ToolRegistry can
// satisfy a capability request, it enters a tool-gap flow
// rather than bluffing.
//
// Flow (from plan):
//   1. Model emits structured ToolGapProposal
//   2. Body shows user-visible rationale
//   3. User explicitly confirms (or rejects)
//   4. Body-owned creation: scaffold → build → hot-load
//   5. ToolRegistry updates; task retried
//
// This file owns Step 1-3 and the data contract for Step 4.
// Actual plugin scaffolding/build is a separate pipeline;
// here we only produce the spec artifact and manage the
// proposal lifecycle.
//======================================================//
#pragma once

#include "ToolRegistry.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// ToolGapReason — why the gap exists
// =========================================================
enum class ToolGapReason : uint8_t {
    NoMatchingCapability   = 0,  // registry has nothing close
    CapabilityMismatch     = 1,  // tool exists but wrong params/scope
    PermissionInsufficient = 2,  // tool exists but needs higher perms
    PolicyBlocked          = 3,  // tool exists but policy forbids
};

// =========================================================
// ProposedToolSpec — what the new tool should look like
//
// This is NOT a full ToolDescriptor; it's the minimal spec
// the model proposes for review.  The body + user decide
// whether to scaffold from this.
// =========================================================
struct ProposedToolSpec {
    std::string    tool_id;         // proposed stable identifier
    std::string    display_name;
    std::string    description;
    std::string    category;        // "action", "information", etc.
    uint32_t       permission_bits = 0;
    bool           needs_confirmation = true;  // default: require confirm
    std::vector<ToolParameter> parameters;
    std::vector<std::string>   capability_tags;
};

// =========================================================
// ToolGapProposal — the model's structured gap declaration
// =========================================================
struct ToolGapProposal {
    // What capability is missing
    std::string      missing_capability;
    ToolGapReason    reason = ToolGapReason::NoMatchingCapability;

    // The closest existing tool (if any) and why it doesn't fit
    std::string      closest_tool_id;
    std::string      closest_tool_mismatch;

    // What the model thinks would solve the gap
    ProposedToolSpec proposed_spec;

    // Human-readable rationale for the user
    std::string      rationale;

    // The original request context that triggered the gap
    std::string      original_request_id;
};

// =========================================================
// ToolGapDecision — user's response to a proposal
// =========================================================
enum class ToolGapDecision : uint8_t {
    Pending   = 0,
    Approved  = 1,
    Rejected  = 2,
    Deferred  = 3,   // "not now" — keep proposal on record
};

// =========================================================
// ToolGapPlanner
//
// Stateless evaluator + proposal builder.  Does NOT own any
// async state; the caller manages proposal lifecycle.
//
// Usage:
//   ToolGapPlanner planner(registry);
//   auto proposal = planner.evaluate("search_github_issues", request_id);
//   if (proposal) {
//       show_to_user(proposal->rationale);
//       // ... wait for user decision ...
//   }
// =========================================================
class ToolGapPlanner {
public:
    explicit ToolGapPlanner(const ToolRegistry& registry);

    // Evaluate whether a capability can be satisfied by the
    // current registry.  Returns a proposal if no tool fits.
    //
    // capability:  the capability tag or tool_id the model requested
    // request_id:  correlation ID for the originating request
    std::optional<ToolGapProposal> evaluate(
        const std::string& capability,
        const std::string& request_id) const;

    // Build a ProposedToolSpec from model-emitted JSON.
    // Throws on malformed input (Rule 20).
    static ProposedToolSpec parseProposedSpec(const std::string& json_text);

    // Generate the user-visible explanation from a proposal.
    static std::string formatRationale(const ToolGapProposal& proposal);

private:
    const ToolRegistry& registry_;

    // Find the closest matching tool by capability tags / keywords.
    std::optional<ToolDescriptor> findClosestTool(
        const std::string& capability) const;
};

} // namespace GRIM::MMO

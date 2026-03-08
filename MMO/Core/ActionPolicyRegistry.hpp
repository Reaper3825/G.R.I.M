// Multi-Model Orchestration (MMO) - ActionPolicyRegistry
// The unified verification gate between model intent and execution.
//
// Implements the Training Wheels Protocol (Phase 4 of MMO plan):
//   1. Resolve tool_id from ToolRegistry
//   2. Validate allowlist + sandbox + preconditions
//   3. Compute GC_action_risk from policy metadata
//   4. Compute GC_action_confidence from body-side signals
//   5. Single verification gate:
//      if (risk >= threshold || confidence <= floor) → VerifyWithUser
//      else → Execute
//
// Replaces direct ActionExecutor::executeAction() / dispatchCommand()
// paths with one gated policy check.
//
// No action is ever executed without passing through this gate.
//======================================================//
#pragma once

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace GRIM::MMO {

// =========================================================
// RiskCategory — per-category threshold support
// =========================================================
enum class RiskCategory : uint8_t {
    Safe,           // read-only, informational
    Low,            // benign actions (open app, search)
    Medium,         // state changes (file write, config change)
    High,           // destructive (delete, process kill)
    Critical        // system-level (shutdown, credential, network)
};

// =========================================================
// ActionProposal — what the model wants to do
// =========================================================
struct ActionProposal {
    std::string tool_id;
    std::string args_json;           // serialized arguments
    std::string session_id;
    std::string turn_id;

    // Body-side confidence inputs
    float       parse_certainty     = 1.0f;  // how clearly the intent was parsed
    float       memory_match_quality = 1.0f;  // relevance of retrieved memories
    float       referent_resolution = 1.0f;   // did "it"/"that file" resolve?
    float       grounding_coverage  = 1.0f;   // does the model have enough facts?
    float       tool_preconditions  = 1.0f;   // are preconditions met?
    float       historical_success  = 1.0f;   // past success rate for this action+context

    // Router confidence (raw — will be calibrated internally)
    float       router_confidence   = 1.0f;
};

// =========================================================
// PolicyDecision — the gate's verdict
// =========================================================
enum class PolicyVerdict : uint8_t {
    Allow,              // Execute immediately
    VerifyRisk,         // Confirm with user (risk-dominant)
    VerifyConfidence,   // Clarify with user (confidence-gap dominant)
    VerifyBoth,         // Both risk and confidence triggered
    Deny                // Tool not in registry / policy violation
};

struct PolicyDecision {
    PolicyVerdict verdict       = PolicyVerdict::Deny;
    float         risk          = 0.0f;   // GC_action_risk
    float         confidence    = 1.0f;   // GC_action_confidence
    float         confidence_gap = 0.0f;  // 1.0 - confidence
    RiskCategory  risk_category = RiskCategory::Safe;
    std::string   reason;                 // human-readable explanation
    std::string   verification_prompt;    // what to show the user if verify
};

// =========================================================
// ActionPolicyConfig — loaded from ai_config.json → training_wheels
// =========================================================
struct ActionPolicyConfig {
    bool  enabled                   = true;

    // Default thresholds
    float risk_threshold            = 0.7f;
    float min_confidence_floor      = 0.5f;

    // Per-category risk thresholds (override default)
    std::unordered_map<std::string, float> per_category_thresholds;
    // e.g. {"destructive": 0.3, "system": 0.4, "search": 0.9}

    // Calibration
    int   calibration_min_samples   = 10;   // min labeled outcomes before trusting router conf
    float uncalibrated_router_conf  = 1.0f; // assume this when uncalibrated (trust body only)
};

// =========================================================
// ToolPolicyOverride — per-tool policy overrides
// =========================================================
struct ToolPolicyOverride {
    std::string  tool_id;
    RiskCategory risk_category = RiskCategory::Low;
    bool         always_confirm = false;    // always require user confirmation
    bool         blocked = false;           // tool is temporarily blocked
    std::string  sandbox_root;              // filesystem sandbox path (if applicable)
    std::vector<std::string> required_permissions;  // GrimPermission names
};

// =========================================================
// ActionPolicyRegistry
//
// Usage:
//   auto& policy = ActionPolicyRegistry::instance();
//   ActionProposal proposal;
//   proposal.tool_id = "delete_file";
//   proposal.parse_certainty = 0.9f;
//   auto decision = policy.evaluate(proposal);
//   if (decision.verdict == PolicyVerdict::Allow) {
//       // execute
//   } else {
//       // show decision.verification_prompt to user
//   }
//
// Thread-safe: all public methods serialized under mutex.
// =========================================================
class ActionPolicyRegistry {
public:
    static ActionPolicyRegistry& instance();

    // Load configuration from ai_config.json section.
    void configure(const ActionPolicyConfig& config);

    // ─── Policy overrides ─────────────────────────────────

    // Set a per-tool policy override.
    void setToolPolicy(const ToolPolicyOverride& override_policy);

    // Remove a per-tool policy override.
    void removeToolPolicy(const std::string& tool_id);

    // Get the policy override for a tool, if any.
    std::optional<ToolPolicyOverride> getToolPolicy(
        const std::string& tool_id) const;

    // ─── Core evaluation ──────────────────────────────────

    // Evaluate a proposed action against the policy gate.
    // This is the single point of truth for "may this action proceed?"
    PolicyDecision evaluate(const ActionProposal& proposal) const;

    // ─── Calibration ──────────────────────────────────────

    // Record an outcome for calibration (positive = success, negative = rejected).
    void recordOutcome(const std::string& tool_id,
                       float router_confidence,
                       bool success);

    // Get the calibrated router confidence for a tool bucket.
    float getCalibratedConfidence(const std::string& tool_id,
                                 float raw_router_confidence) const;

    // ─── Query ────────────────────────────────────────────

    // Check if a tool is blocked by policy.
    bool isBlocked(const std::string& tool_id) const;

    // Get the risk category assigned to a tool.
    RiskCategory getRiskCategory(const std::string& tool_id) const;

    // Get the applicable risk threshold for a risk category.
    float getThreshold(RiskCategory category) const;

    // Get current config (for diagnostics).
    const ActionPolicyConfig& config() const;

private:
    ActionPolicyRegistry() = default;

    // Compute GC_action_risk from tool metadata + policy overrides.
    float computeRisk(const ActionProposal& proposal,
                      RiskCategory& out_category) const;

    // Compute GC_action_confidence = min(body_confidence, calibrated_router).
    float computeConfidence(const ActionProposal& proposal) const;

    // Build a human-readable verification prompt.
    std::string buildVerificationPrompt(
        const ActionProposal& proposal,
        PolicyVerdict verdict,
        float risk,
        float confidence) const;

    // Calibration bucket (per tool_id for now)
    struct CalibrationBucket {
        int   total_outcomes     = 0;
        int   successful_outcomes = 0;
        float calibrated_success_rate = 0.0f;
    };

    mutable std::mutex mutex_;
    ActionPolicyConfig config_;
    std::unordered_map<std::string, ToolPolicyOverride> overrides_;
    std::unordered_map<std::string, CalibrationBucket>  calibration_;
};

} // namespace GRIM::MMO

#include "ActionPolicyRegistry.hpp"
#include "ToolRegistry.hpp"
#include "logger.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace GRIM::MMO {

// =========================================================
// Singleton
// =========================================================

ActionPolicyRegistry& ActionPolicyRegistry::instance() {
    static ActionPolicyRegistry s;
    return s;
}

// =========================================================
// Configuration
// =========================================================

void ActionPolicyRegistry::configure(const ActionPolicyConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    LOG_DEBUG("ActionPolicy", "Configured: enabled=" +
              std::string(config.enabled ? "true" : "false") +
              " risk_threshold=" + std::to_string(config.risk_threshold) +
              " min_conf_floor=" + std::to_string(config.min_confidence_floor));
}

// =========================================================
// Per-tool policy overrides
// =========================================================

void ActionPolicyRegistry::setToolPolicy(const ToolPolicyOverride& pol) {
    if (pol.tool_id.empty()) {
        throw std::runtime_error("ActionPolicyRegistry::setToolPolicy: empty tool_id");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    overrides_[pol.tool_id] = pol;
}

void ActionPolicyRegistry::removeToolPolicy(const std::string& tool_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    overrides_.erase(tool_id);
}

std::optional<ToolPolicyOverride> ActionPolicyRegistry::getToolPolicy(
    const std::string& tool_id) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = overrides_.find(tool_id);
    if (it != overrides_.end()) return it->second;
    return std::nullopt;
}

// =========================================================
// Core evaluation
// =========================================================

PolicyDecision ActionPolicyRegistry::evaluate(const ActionProposal& proposal) const {
    std::lock_guard<std::mutex> lock(mutex_);

    PolicyDecision decision;
    decision.verdict = PolicyVerdict::Deny;

    // If Training Wheels is disabled, allow everything
    if (!config_.enabled) {
        decision.verdict = PolicyVerdict::Allow;
        decision.reason = "Training Wheels disabled";
        return decision;
    }

    // Check if tool exists in ToolRegistry
    auto& toolReg = ToolRegistry::instance();
    if (!toolReg.isRegistered(proposal.tool_id)) {
        decision.reason = "Tool '" + proposal.tool_id + "' not found in ToolRegistry";
        return decision;
    }

    // Check if tool is blocked
    auto override_it = overrides_.find(proposal.tool_id);
    if (override_it != overrides_.end() && override_it->second.blocked) {
        decision.reason = "Tool '" + proposal.tool_id + "' is blocked by policy";
        return decision;
    }

    // Compute risk
    RiskCategory risk_cat;
    float risk = computeRisk(proposal, risk_cat);
    decision.risk = risk;
    decision.risk_category = risk_cat;

    // Compute confidence
    float confidence = computeConfidence(proposal);
    decision.confidence = confidence;
    decision.confidence_gap = 1.0f - confidence;

    // Check always_confirm override
    bool always_confirm = false;
    if (override_it != overrides_.end()) {
        always_confirm = override_it->second.always_confirm;
    }

    // Get applicable threshold
    float applicable_threshold = getThreshold(risk_cat);

    // Single verify rule:
    // if (risk >= threshold || confidence <= floor) → Verify
    bool risk_triggered = (risk >= applicable_threshold);
    bool conf_triggered = (confidence <= config_.min_confidence_floor);

    if (always_confirm || risk_triggered || conf_triggered) {
        if (risk_triggered && conf_triggered) {
            decision.verdict = PolicyVerdict::VerifyBoth;
        } else if (risk_triggered) {
            decision.verdict = PolicyVerdict::VerifyRisk;
        } else {
            decision.verdict = PolicyVerdict::VerifyConfidence;
        }

        decision.reason = "Verification required:";
        if (always_confirm) decision.reason += " [always_confirm]";
        if (risk_triggered) {
            decision.reason += " risk=" + std::to_string(risk) +
                              " >= threshold=" + std::to_string(applicable_threshold);
        }
        if (conf_triggered) {
            decision.reason += " confidence=" + std::to_string(confidence) +
                              " <= floor=" + std::to_string(config_.min_confidence_floor);
        }

        decision.verification_prompt = buildVerificationPrompt(
            proposal, decision.verdict, risk, confidence);
    } else {
        decision.verdict = PolicyVerdict::Allow;
        decision.reason = "Risk and confidence within acceptable bounds";
    }

    return decision;
}

// =========================================================
// Risk computation
// =========================================================

float ActionPolicyRegistry::computeRisk(
    const ActionProposal& proposal,
    RiskCategory& out_category) const
{
    // Start with per-tool override if present
    auto ov = overrides_.find(proposal.tool_id);
    if (ov != overrides_.end()) {
        out_category = ov->second.risk_category;
    } else {
        // Derive from ToolRegistry metadata
        auto tool = ToolRegistry::instance().getTool(proposal.tool_id);
        if (tool.has_value()) {
            if (tool->is_informational) {
                out_category = RiskCategory::Safe;
            } else if (tool->needs_confirmation) {
                out_category = RiskCategory::High;
            } else {
                out_category = RiskCategory::Medium;
            }
        } else {
            out_category = RiskCategory::High;  // unknown tool = high risk
        }
    }

    // Map category to base risk score
    switch (out_category) {
        case RiskCategory::Safe:     return 0.0f;
        case RiskCategory::Low:      return 0.2f;
        case RiskCategory::Medium:   return 0.5f;
        case RiskCategory::High:     return 0.8f;
        case RiskCategory::Critical: return 1.0f;
    }
    return 1.0f;
}

// =========================================================
// Confidence computation
// =========================================================

float ActionPolicyRegistry::computeConfidence(const ActionProposal& proposal) const {
    // GC_body_confidence = min of all body-side signals
    float body_conf = std::min({
        proposal.parse_certainty,
        proposal.memory_match_quality,
        proposal.referent_resolution,
        proposal.grounding_coverage,
        proposal.tool_preconditions,
        proposal.historical_success
    });

    // Get calibrated router confidence
    float calibrated_router = getCalibratedConfidence(
        proposal.tool_id, proposal.router_confidence);

    // GC_action_confidence = min(body, calibrated_router)
    return std::min(body_conf, calibrated_router);
}

// =========================================================
// Calibration
// =========================================================

void ActionPolicyRegistry::recordOutcome(
    const std::string& tool_id,
    float router_confidence,
    bool success)
{
    std::lock_guard<std::mutex> lock(mutex_);
    auto& bucket = calibration_[tool_id];
    bucket.total_outcomes++;
    if (success) bucket.successful_outcomes++;
    bucket.calibrated_success_rate =
        static_cast<float>(bucket.successful_outcomes) / bucket.total_outcomes;

    LOG_TRACE("ActionPolicy", "Calibration update for " + tool_id +
              ": " + std::to_string(bucket.successful_outcomes) + "/" +
              std::to_string(bucket.total_outcomes) +
              " (rate=" + std::to_string(bucket.calibrated_success_rate) + ")");
}

float ActionPolicyRegistry::getCalibratedConfidence(
    const std::string& tool_id,
    float raw_router_confidence) const
{
    // If uncalibrated (not enough samples), use config default
    auto it = calibration_.find(tool_id);
    if (it == calibration_.end() ||
        it->second.total_outcomes < config_.calibration_min_samples) {
        return config_.uncalibrated_router_conf;
    }

    // Bucket-calibrated: blend raw router conf with observed success rate
    // This prevents the router from gaming the gate with high raw confidence
    // when actual outcomes are poor
    float observed = it->second.calibrated_success_rate;
    return std::min(raw_router_confidence, observed);
}

// =========================================================
// Query
// =========================================================

bool ActionPolicyRegistry::isBlocked(const std::string& tool_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = overrides_.find(tool_id);
    return (it != overrides_.end() && it->second.blocked);
}

RiskCategory ActionPolicyRegistry::getRiskCategory(const std::string& tool_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = overrides_.find(tool_id);
    if (it != overrides_.end()) return it->second.risk_category;

    // Derive from tool metadata
    auto tool = ToolRegistry::instance().getTool(tool_id);
    if (tool.has_value()) {
        if (tool->is_informational) return RiskCategory::Safe;
        if (tool->needs_confirmation) return RiskCategory::High;
        return RiskCategory::Medium;
    }
    return RiskCategory::High;
}

float ActionPolicyRegistry::getThreshold(RiskCategory category) const {
    // Check per-category overrides
    std::string cat_name;
    switch (category) {
        case RiskCategory::Safe:     cat_name = "safe"; break;
        case RiskCategory::Low:      cat_name = "low"; break;
        case RiskCategory::Medium:   cat_name = "medium"; break;
        case RiskCategory::High:     cat_name = "destructive"; break;
        case RiskCategory::Critical: cat_name = "system"; break;
    }

    auto it = config_.per_category_thresholds.find(cat_name);
    if (it != config_.per_category_thresholds.end()) {
        return it->second;
    }
    return config_.risk_threshold;
}

const ActionPolicyConfig& ActionPolicyRegistry::config() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

// =========================================================
// Verification prompt builder
// =========================================================

std::string ActionPolicyRegistry::buildVerificationPrompt(
    const ActionProposal& proposal,
    PolicyVerdict verdict,
    float risk,
    float confidence) const
{
    std::ostringstream ss;

    switch (verdict) {
        case PolicyVerdict::VerifyRisk:
            ss << "I want to run '" << proposal.tool_id << "'";
            if (!proposal.args_json.empty()) {
                ss << " with arguments: " << proposal.args_json;
            }
            ss << ". This action has elevated risk. Is this what you want?";
            break;

        case PolicyVerdict::VerifyConfidence:
            ss << "I'm not confident enough to proceed with '"
               << proposal.tool_id << "' safely. "
               << "Can you confirm or provide more details?";
            break;

        case PolicyVerdict::VerifyBoth:
            ss << "'" << proposal.tool_id << "' is a risky action "
               << "and I'm not fully confident in the intent. "
               << "Can you confirm this is what you want?";
            break;

        default:
            break;
    }

    return ss.str();
}

} // namespace GRIM::MMO

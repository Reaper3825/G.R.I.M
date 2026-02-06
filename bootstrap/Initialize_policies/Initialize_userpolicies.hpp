// Initialize_userpolicies.hpp
/**
 * @file Initialize_userpolicies.hpp
 * @brief Initialization of initial user policies from probe survey 
 * 
 * This header defines the function to initialize user policies, which are loaded from configuration files or defaults.
 * User policies control various aspects of the system's behavior, such as privacy settings, feature toggles, and user preferences.
 */
#pragma once
#include <string>
#include <vector>

namespace GRIM {
    bool UserPolicyProbeUsed; // Flag to indicate if the user policy probe was used in this session

struct UserPolicy {
    float initiative_expectation;   // 0 = reactive, 1 = proactive
    float clarification_demand;      // 0 = tolerate ambiguity, 1 = demand precision
    float correction_tolerance;      // 0 = hates correction, 1 = wants blunt truth
    float exploration_bias;          // 0 = linear, 1 = branching
    float autonomy_trust;            // 0 = ask first, 1 = act freely
    float verbosity_preference;      // 0 = terse, 1 = deep
};

void initializeUserPolicies();
void ToggleUserPolicyProbe();
bool IsUserPolicyProbeActive();
} // namespace GRIM

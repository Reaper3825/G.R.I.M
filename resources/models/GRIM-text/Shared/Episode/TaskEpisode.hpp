#pragma once  

// Shared Goal-attempt episode record for environment interaction and learning.
// Live clients and generation sessions retain their existing runtime owners.

#include "Attempt.hpp"
#include "../Forward/GeneratedSequence.hpp"
#include "../Goal/GoalContract.hpp"
#include <nlohmann/json.hpp>
#include <vector>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace GRIM {
struct NamedConceptSpans;
}
 
namespace GRIM::Episodes {

inline constexpr std::size_t kMaxAttempts = 2;
enum class GoalOrigin : std::uint8_t { Undecided, Supplied, ModelGenerated };

enum class EpisodeState : std::uint8_t {
    Collecting, AttemptSubmitted, Complete, InfrastructureInvalid,
};

// Private host receipt. Never serialize a full response into model context.
struct EnvironmentAction {
    std::size_t action_close_position;
    std::string command;
    std::string status;
    nlohmann::json response;
    std::string error;
};

struct TaskEpisode {
    // Callers never supply a variable store. Local declarations and input
    // bindings originate in model output; runtime tracks them and tool results.
    std::string rollout_group_id;
    std::string task_id;
    std::string task_family;
    std::string episode_id;
    std::uint64_t policy_version = 0;
    // Resolve weights/checkpoints and model.grimcfg by model name through the
    // existing configured model-store path.
    std::string model_name;
    std::string tool_snapshot_id;
    // The prepared task's sandbox instance, isolated for this episode. Its
    // execution evidence is captured at each committed attempt boundary.
    std::string sandbox_instance_id;
    GoalOrigin goal_origin = GoalOrigin::Undecided;
    std::shared_ptr<const ConceptBlockGoal> goal;
    // Grading consumes this record's task/goal, committed attempt, and sandbox
    // snapshot. There is no separate verifier identity or parallel task payload.
    // Existing generation payload owns IDs, scores, atom channels, and tables.
    // Append-only model-visible history; private results stay with the runtime
    // until disclosed. Scoring must never replay tool side effects.
    // The inference session owns the live sequence. Transfer its final snapshot
    // here when an attempt ends; do not maintain a second mutable trajectory.
    GeneratedSequence sequence;
    nlohmann::json initial_environment_response;
    nlohmann::json latest_environment_response;
    std::vector<EnvironmentAction> actions;
    std::shared_ptr<const NamedConceptSpans> named_concept_spans;
    // Slot 0 is initial; slot 1 is an actual retry. Null means uncommitted/absent.
    // Publish with no remaining mutable aliases to the committed attempt.
    std::array<std::shared_ptr<const AttemptRecord>, kMaxAttempts> attempts{};
    // Filled after group scoring without editing committed attempts.
    std::array<std::optional<AttemptScore>, kMaxAttempts> scores{};
    EpisodeState state = EpisodeState::Collecting;
    // A failed first attempt awaits an explicit retry policy in AttemptSubmitted;
    // this record does not start a retry or reset the world on its own.
    // Group statistics, runtime state, and optimizer commits have separate owners.
};

} // namespace GRIM::Episodes

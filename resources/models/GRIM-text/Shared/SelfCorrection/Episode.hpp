#pragma once  

// Self-correction-specific episode metadata. Existing generation, goal, and
// span types retain ownership of their data; runtime/tool machinery belongs
// at its existing integration boundaries, not in this definition header.

#include "Attempt.hpp"
#include "../Forward/GeneratedSequence.hpp"
#include "../Goal/Goal.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace GRIM {
struct NamedConceptSpans;
}
 
namespace GRIM::SelfCorrection {

inline constexpr std::size_t kMaxAttempts = 2;
enum class GoalOrigin : std::uint8_t { Undecided, Supplied, ModelGenerated };

enum class EpisodeState : std::uint8_t {
    Collecting, Complete, InfrastructureInvalid,
};

struct EpisodeRecord {
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
    std::shared_ptr<const Goal> goal;
    // Grading consumes this record's task/goal, committed attempt, and sandbox
    // snapshot. There is no separate verifier identity or parallel task payload.
    // Existing generation payload owns IDs, scores, atom channels, and tables.
    // Append-only model-visible history; private results stay with the runtime
    // until disclosed. Scoring must never replay tool side effects.
    GeneratedSequence sequence;
    std::shared_ptr<const NamedConceptSpans> named_concept_spans;
    // Slot 0 is initial; slot 1 is an actual retry. Null means uncommitted/absent.
    // Publish with no remaining mutable aliases to the committed attempt.
    std::array<std::shared_ptr<const AttemptRecord>, kMaxAttempts> attempts{};
    // Filled after group scoring without editing committed attempts.
    std::array<std::optional<AttemptScore>, kMaxAttempts> scores{};
    EpisodeState state = EpisodeState::Collecting;
    // Complete after a passing initial attempt or a graded retry. Invalid
    // infrastructure episodes are excluded/replaced. Group statistics, runtime
    // state, and optimizer commits have separate owners.
};

} // namespace GRIM::SelfCorrection

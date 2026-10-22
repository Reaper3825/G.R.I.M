#pragma once

// Attempt boundaries, sampling segments, grading, and detached scoring for
// self-correction episodes. Token offsets refer to the owning episode sequence.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace GRIM::SelfCorrection {

inline constexpr std::size_t kPilotSampledTokensPerAttempt = 256;

enum class SegmentEndReason : std::uint8_t {
    ToolPause, ReadPause, ValidationPause, AttemptEnded,
};

struct SamplingSegment {
    // Half-open absolute offsets in TaskEpisode::sequence.token_ids.
    // Contains uninterrupted sampled actions, with no runtime insertions.
    std::size_t token_begin = 0;
    std::size_t token_end = 0;
    SegmentEndReason end_reason = SegmentEndReason::AttemptEnded;
};

enum class AttemptEndReason : std::uint8_t {
    AnswerClosed, Eos, SampledTokenLimit, ExecutionBudget, InfrastructureFailure,
};

enum class CheckStatus : std::uint8_t { Passed, Failed, NotApplicable };

struct CriterionOutcome {
    std::string check_id;
    CheckStatus status = CheckStatus::Failed;
    std::string private_reason;
};

enum class GradeVerdict : std::uint8_t {
    Passed, Failed, InfrastructureInvalid,
};

struct AttemptGrade {
    GradeVerdict verdict = GradeVerdict::Failed;
    std::vector<CriterionOutcome> checks;
    // Private verifier diagnostics never become model-visible feedback by copy.
};

// Published after the attempt endpoint and evidence are frozen and graded.
// Tool/read pauses do not close an attempt. A sampled answer close ends it
// independently of goal success; EOS/budget before that close is incomplete.
struct AttemptRecord {
    // Initial/retry identity comes from the owning episode's attempt slot.
    // Half-open absolute offsets; excludes earlier history, includes this
    // attempt's generation and runtime insertions. Retry feedback precedes begin.
    std::size_t token_begin = 0;
    std::size_t token_end = 0;
    std::vector<SamplingSegment> segments;
    AttemptEndReason end_reason = AttemptEndReason::SampledTokenLimit;
    // Present only for AnswerClosed: [answer_close_begin, token_end) is the
    // complete sampled close marker, including when other grammar is malformed.
    std::optional<std::size_t> answer_close_begin;
    // Frozen execution results and environment state from the episode's sandbox
    // at token_end. This is the grading evidence, not a second evidence payload.
    std::string sandbox_snapshot_id;
    AttemptGrade grade;
};

struct AttemptScore {
    float reward = 0.0f;
    float advantage = 0.0f; // Detached and attempt-local.
};

} // namespace GRIM::SelfCorrection

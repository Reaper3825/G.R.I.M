#include "EnvironmentSession.hpp"
#include "../../../../../ui/world_lab/environment_registry.hpp"

#include "../UnigramByte/SequenceLocalAtomTable.hpp"

#include <chrono>
#include <thread>
#include <utility>

namespace GRIM::Environment {

std::string validateEnvironmentCommand(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos) throw std::invalid_argument("Execute contains no command");
    auto command = text.substr(first, text.find_last_not_of(" \t\r\n\f\v") - first + 1);
    if (command.size() > 512 || command.find_first_of("\r\n\0", 0, 3) != std::string::npos)
        throw std::invalid_argument("Execute must contain one command of at most 512 bytes");
    return command;
}

nlohmann::json EnvironmentSession::submitCommand(std::size_t position, const std::string& text) {
    const auto command = validateEnvironmentCommand(text);
    for (const auto& previous : episode_.actions) {
        if (previous.action_close_position != position) continue;
        if (previous.command != command)
            throw std::logic_error("Execute boundary was already used for a different command");
        if (previous.status == "completed") return previous.response;
        throw std::runtime_error("Execute submission has unresolved completion; inspect before continuing");
    }
    if (!episode_.actions.empty() && (episode_.actions.back().status != "completed" ||
            position <= episode_.actions.back().action_close_position))
        throw std::logic_error("Cannot submit after an unresolved or older Execute boundary");
    if (episode_.state != Episodes::EpisodeState::Collecting)
        throw std::logic_error("Cannot act after the Goal attempt has ended");
    const auto& prior = episode_.latest_environment_response.at("result").at("state");
    if (prior.at("done").get<bool>()) throw std::invalid_argument("Environment is terminal; submit the Goal attempt with Answer");
    if (client_->busy()) throw std::logic_error("Environment client is busy");
    const auto identity = prior.at("episode").at("identity");
    episode_.actions.push_back({position, command, "pending", {}, {}});
    auto& action = episode_.actions.back();
    try {
        if (!client_->begin({{"op", "step"}, {"command", command}}))
            throw std::runtime_error("Environment command was not accepted by the client");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(135);
        for (;;) {
            if (auto response = client_->poll()) {
                action.response = *response;
                if (!response->value("ok", false))
                    throw std::runtime_error(response->value("error", std::string("Environment step failed")));
                const auto& state = response->at("result").at("state");
                const auto status = state.at("episode").at("status").get<std::string>();
                if ((status != "active" && status != "succeeded" && status != "failed" && status != "stopped") ||
                    state.at("done").get<bool>() != (status != "active") ||
                    state.at("episode").at("identity") != identity)
                    throw std::runtime_error("Environment step returned a mismatched episode");
                episode_.latest_environment_response = *response;
                if (state.contains("host_action") && state.at("host_action").at("status") == "unknown")
                    throw std::runtime_error("Host action completion is unknown; inspect before continuing");
                action.status = "completed";
                return *response;
            }
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Environment command response timed out");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch (const std::exception& error) {
        action.status = "unknown";
        invalidateAttempt();
        action.error = error.what();
        throw;
    }
}

void EnvironmentSession::finishAttempt(const GeneratedSequence& sequence, std::size_t prompt_tokens,
    Episodes::AttemptEndReason reason, std::optional<std::size_t> answer_close_begin) {
    using namespace Episodes;
    if (episode_.state != EpisodeState::Collecting || episode_.attempts[0])
        throw std::logic_error("Goal attempt is already submitted or invalid");
    if (client_->busy() || (!episode_.actions.empty() && episode_.actions.back().status != "completed"))
        throw std::logic_error("Cannot grade an unresolved environment action");
    if (sequence.token_origins.size() != sequence.token_ids.size() ||
        prompt_tokens > sequence.token_ids.size() ||
        (reason == AttemptEndReason::AnswerClosed) != answer_close_begin.has_value() ||
        (answer_close_begin && (*answer_close_begin < prompt_tokens ||
                               *answer_close_begin >= sequence.token_ids.size())))
        throw std::invalid_argument("Invalid Goal attempt boundaries");

    AttemptRecord attempt;
    attempt.token_begin = prompt_tokens;
    attempt.token_end = sequence.token_ids.size();
    attempt.end_reason = reason;
    attempt.answer_close_begin = answer_close_begin;
    attempt.environment_snapshot = episode_.latest_environment_response;
    const auto& world = attempt.environment_snapshot.at("result").at("state").at("episode");
    attempt.grade.verdict = reason == AttemptEndReason::AnswerClosed && world.at("status") == "succeeded"
        ? GradeVerdict::Passed : GradeVerdict::Failed;
    if (world.contains("requirements")) {
        for (const auto& check : world.at("requirements")) {
            const auto collection = check.at("collection").get<std::string>();
            const auto status = check.at("status").get<std::string>();
            const bool passed = (collection == "success_criteria" && status == "satisfied") ||
                                (collection == "constraints" && status == "unsatisfied");
            attempt.grade.checks.push_back({collection + "/" + std::to_string(check.at("index").get<int>()),
                passed ? CheckStatus::Passed : CheckStatus::Failed, status});
        }
    }
    // Sampling segments exclude observations; their offsets remain absolute.
    for (std::size_t p = prompt_tokens; p < sequence.token_origins.size();) {
        if (sequence.token_origins[p] != TokenOrigin::SampledAction) { ++p; continue; }
        const auto begin = p++;
        while (p < sequence.token_origins.size() && sequence.token_origins[p] == TokenOrigin::SampledAction) ++p;
        attempt.segments.push_back({begin, p, p == sequence.token_ids.size()
            ? SegmentEndReason::AttemptEnded : SegmentEndReason::ToolPause});
    }
    auto snapshot = sequence;
    if (snapshot.context_local_atom_table)
        snapshot.context_local_atom_table = std::make_shared<Tokenizer::SequenceLocalAtomTable>(
            *snapshot.context_local_atom_table);
    auto committed = std::make_shared<const AttemptRecord>(std::move(attempt));
    const bool passed = committed->grade.verdict == GradeVerdict::Passed;
    episode_.sequence = std::move(snapshot);
    episode_.attempts[0] = std::move(committed);
    episode_.state = passed ? EpisodeState::Complete : EpisodeState::AttemptSubmitted;
}

namespace {
Episodes::TaskEpisode makeEpisode(const ScheduledGoal& assignment) {
    if (assignment.course_id.empty() || assignment.concept_block_id.empty())
        throw std::runtime_error("Environment preparation: example identity is empty");
    Episodes::TaskEpisode episode;
    episode.task_id = assignment.concept_block_id;
    episode.task_family = assignment.course_id;
    episode.goal_origin = Episodes::GoalOrigin::Supplied;
    episode.goal = std::make_shared<const ConceptBlockGoal>(assignment.goal);
    return episode;
}

nlohmann::json startExample(WorldLab::TextWorldClient& client, const ScheduledGoal& assignment,
    const WorldLab::Document& document, bool compile) {
    nlohmann::json request{{"op", compile ? "compile" : "start_episode"},
        {"source_id", assignment.concept_block_id}, {"goal", goalToJson(assignment.goal)}};
    if (compile) request["document"] = document.gameJson();
    if (!client.begin(std::move(request)))
        throw std::runtime_error("Environment preparation: client unexpectedly busy");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(135);
    for (;;) {
        if (auto response = client.poll()) {
            if (!response->value("ok", false))
                throw std::runtime_error("Environment preparation: " +
                    response->value("error", std::string("episode startup failed")));
            const auto& result = response->at("result");
            const auto& state = result.at("state");
            const auto& episode = state.at("episode");
            const auto status = episode.at("status").get<std::string>();
            if (status != "active" && status != "succeeded" && status != "failed" && status != "stopped")
                throw std::runtime_error("Environment preparation: invalid episode status");
            const auto& identity = episode.at("identity");
            if (identity.at("source_id") != assignment.concept_block_id ||
                identity.at("environment") != result.at("environment_id") ||
                result.at("environment_id").get<std::string>().empty() ||
                state.at("done").get<bool>() != (status != "active"))
                throw std::runtime_error("Environment preparation: initial episode identity/status mismatch");
            return std::move(*response);
        }
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("Environment preparation: initial response timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}
}

EnvironmentSession::EnvironmentSession(std::vector<ScheduledGoal> schedule,
    WorldLab::Document environment, std::filesystem::path path, std::filesystem::path games_directory)
    : schedule_(std::move(schedule)), environment_(std::move(environment)), path_(std::move(path)),
      games_directory_(std::move(games_directory)), client_(std::make_unique<WorldLab::TextWorldClient>()),
      episode_(makeEpisode(currentAssignment())) {}

bool EnvironmentSession::advanceExample() {
    if (transition_failed_)
        throw std::logic_error("Example transition is unresolved; inspect before continuing");
    if (schedule_complete_) return false;
    if (!episode_.attempts[0] || episode_.state == Episodes::EpisodeState::InfrastructureInvalid || client_->busy())
        throw std::logic_error("Cannot advance an unfinished or infrastructure-invalid example");
    if (position_ + 1 == schedule_.size()) {
        schedule_complete_ = true;
        return false;
    }
    const auto& next = schedule_.at(position_ + 1);
    auto next_episode = makeEpisode(next);
    // Allocate history before touching the worker. Completed outcomes remain
    // available even if the next world cannot be resolved or started.
    if (previous_episodes_.size() == previous_episodes_.capacity())
        previous_episodes_.reserve(previous_episodes_.capacity() ? previous_episodes_.capacity() * 2 : 1);
    std::optional<WorldLab::Document> next_document;
    std::filesystem::path next_path;
    std::unique_ptr<WorldLab::TextWorldClient> next_client;
    const bool change_environment = next.environment_id != currentAssignment().environment_id;
    if (change_environment) {
        const auto entry = WorldLab::EnvironmentRegistry(games_directory_).resolve(next.environment_id);
        next_document = WorldLab::Document::parse(WorldLab::loadJsonFile(entry.path));
        if (next_document->environmentId != next.environment_id)
            throw std::runtime_error("Environment preparation: environment changed during resolution");
        next_path = entry.path;
        next_client = std::make_unique<WorldLab::TextWorldClient>();
    }
    try {
        auto response = startExample(change_environment ? *next_client : *client_, next,
            change_environment ? *next_document : environment_, change_environment);
        if (!change_environment && response.at("result").at("environment_id") !=
                episode_.initial_environment_response.at("result").at("environment_id"))
            throw std::runtime_error("Environment preparation: compiled environment changed on example reset");
        next_episode.initial_environment_response = response;
        next_episode.latest_environment_response = std::move(response);
    } catch (...) {
        // A missing/malformed reset response can leave a reused worker in an
        // unknown example. Stop it and never retry the transition automatically.
        transition_failed_ = true;
        if (!change_environment) client_->stop();
        throw;
    }
    previous_episodes_.push_back(std::move(episode_));
    episode_ = std::move(next_episode);
    if (change_environment) {
        client_.swap(next_client);
        environment_ = std::move(*next_document);
        path_ = std::move(next_path);
    }
    ++position_;
    return true;
}

std::unique_ptr<EnvironmentSession> prepareEnvironmentSession(
    std::vector<ScheduledGoal> schedule, const std::filesystem::path& games_directory) {
    if (schedule.empty()) throw std::runtime_error("Environment preparation: Goal schedule is empty");
    const auto& first = schedule.front();
    const auto entry = WorldLab::EnvironmentRegistry(games_directory).resolve(first.environment_id);
    auto document = WorldLab::Document::parse(WorldLab::loadJsonFile(entry.path));
    if (document.environmentId != first.environment_id)
        throw std::runtime_error("Environment preparation: environment changed during resolution");
    auto candidate = std::make_unique<EnvironmentSession>(
        std::move(schedule), std::move(document), entry.path, games_directory);
    // Compile starts the first example; subsequent examples reuse start_episode.
    auto response = startExample(*candidate->client_, candidate->currentAssignment(), candidate->environment_, true);
    candidate->episode_.initial_environment_response = response;
    candidate->episode_.latest_environment_response = std::move(response);
    return candidate;
}

} // namespace GRIM::Environment

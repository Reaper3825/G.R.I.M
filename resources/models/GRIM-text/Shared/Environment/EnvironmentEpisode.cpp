#include "EnvironmentEpisode.hpp"
#include "../../../../../ui/world_lab/environment_registry.hpp"

#include <chrono>
#include <thread>
#include <utility>

namespace GRIM::Environment {

std::string validateEnvironmentCommand(const std::string& answer) {
    const auto first = answer.find_first_not_of(" \t\r\n\f\v");
    if (first == std::string::npos) throw std::invalid_argument("Answer contains no command");
    auto command = answer.substr(first, answer.find_last_not_of(" \t\r\n\f\v") - first + 1);
    if (command.size() > 512 || command.find_first_of("\r\n\0", 0, 3) != std::string::npos)
        throw std::invalid_argument("Answer must contain one command of at most 512 bytes");
    return command;
}

nlohmann::json EnvironmentEpisode::submitAnswer(std::size_t position, const std::string& answer) {
    const auto command = validateEnvironmentCommand(answer);
    for (const auto& previous : actions_) {
        if (previous.answer_close_position != position) continue;
        if (previous.command != command)
            throw std::logic_error("Answer boundary was already used for a different command");
        if (previous.status == "completed") return previous.response;
        throw std::runtime_error("Answer submission has unresolved completion; inspect before continuing");
    }
    if (!actions_.empty() && (actions_.back().status != "completed" ||
            position <= actions_.back().answer_close_position))
        throw std::logic_error("Cannot submit after an unresolved or older Answer boundary");
    const auto& prior = latest_response_.at("result").at("state");
    if (prior.at("done").get<bool>()) throw std::logic_error("Environment episode is already terminal");
    if (client_.busy()) throw std::logic_error("Environment client is busy");
    const auto identity = prior.at("episode").at("identity");
    actions_.push_back({position, command, "pending", {}, {}});
    auto& action = actions_.back();
    try {
        if (!client_.begin({{"op", "step"}, {"command", command}}))
            throw std::runtime_error("Environment command was not accepted by the client");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(135);
        for (;;) {
            if (auto response = client_.poll()) {
                action.response = *response;
                if (!response->value("ok", false))
                    throw std::runtime_error(response->value("error", std::string("Environment step failed")));
                const auto& state = response->at("result").at("state");
                const auto status = state.at("episode").at("status").get<std::string>();
                if ((status != "active" && status != "succeeded" && status != "failed" && status != "stopped") ||
                    state.at("done").get<bool>() != (status != "active") ||
                    state.at("episode").at("identity") != identity)
                    throw std::runtime_error("Environment step returned a mismatched episode");
                latest_response_ = *response;
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
        action.error = error.what();
        throw;
    }
}

EnvironmentEpisode::EnvironmentEpisode(std::vector<ScheduledGoal> schedule,
    WorldLab::Document environment, std::filesystem::path path)
    : schedule_(std::move(schedule)), environment_(std::move(environment)), path_(std::move(path)) {}

std::unique_ptr<EnvironmentEpisode> prepareEnvironmentEpisode(
    std::vector<ScheduledGoal> schedule, const std::filesystem::path& games_directory) {
    if (schedule.empty()) throw std::runtime_error("Environment preparation: Goal schedule is empty");
    const auto& first = schedule.front();
    if (first.course_id.empty() || first.concept_block_id.empty())
        throw std::runtime_error("Environment preparation: assignment identity is empty");
    const WorldLab::EnvironmentRegistry registry(games_directory);
    const auto entry = registry.resolve(first.environment_id);
    auto document = WorldLab::Document::parse(WorldLab::loadJsonFile(entry.path));
    if (document.environmentId != first.environment_id)
        throw std::runtime_error("Environment preparation: environment changed during resolution");
    auto candidate = std::make_unique<EnvironmentEpisode>(
        std::move(schedule), std::move(document), entry.path);
    // The worker's compile operation starts the episode transactionally and
    // evaluates the initial state. A second start_episode would reset it again.
    const auto& assignment = candidate->currentAssignment();
    if (!candidate->client_.begin({{"op", "compile"},
            {"document", candidate->environment_.gameJson()},
            {"source_id", assignment.concept_block_id}, {"goal", goalToJson(assignment.goal)}}))
        throw std::runtime_error("Environment preparation: client unexpectedly busy");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(135);
    for (;;) {
        if (auto response = candidate->client_.poll()) {
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
            candidate->latest_response_ = std::move(*response);
            candidate->initial_response_ = candidate->latest_response_;
            return candidate; // Initial success/failure is valid; never force an action.
        }
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("Environment preparation: initial response timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

} // namespace GRIM::Environment

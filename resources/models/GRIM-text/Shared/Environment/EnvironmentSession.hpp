#pragma once

#include "../DataLoader/ConceptBlockCorpusReader.hpp"
#include "../Episode/TaskEpisode.hpp"
#include "../../../../../ui/world_lab/textworld_client.hpp"
#include "../../../../../ui/world_lab/world_document.hpp"

#include <memory>

namespace GRIM::Environment {

// Trim surrounding span whitespace, then require one nonempty command <=512 bytes.
std::string validateEnvironmentCommand(const std::string& command);

// Live runtime owner carried across the Phase1/Phase2 boundary. The shared
// TaskEpisode holds the frozen Goal, trajectory, receipts, and attempt records.
// Full responses are private host data, not a model-visible observation feed.
class EnvironmentSession {
public:
    EnvironmentSession(std::vector<ScheduledGoal> schedule,
                       WorldLab::Document environment, std::filesystem::path path,
                       std::filesystem::path games_directory);
    const ScheduledGoal& currentAssignment() const { return schedule_.at(position_); }
    const std::vector<ScheduledGoal>& schedule() const { return schedule_; }
    std::size_t position() const { return position_; }
    const WorldLab::Document& environment() const { return environment_; }
    const std::filesystem::path& environmentPath() const { return path_; }
    const Episodes::TaskEpisode& taskEpisode() const { return episode_; }
    const nlohmann::json& latestResponse() const { return episode_.latest_environment_response; }
    WorldLab::TextWorldClient& client() { return *client_; }
    const nlohmann::json& initialResponse() const { return episode_.initial_environment_response; }
    const std::vector<Episodes::EnvironmentAction>& actions() const { return episode_.actions; }
    // Boundary positions identify submissions within this episode's trajectory.
    // Repeated completed submissions return their recorded response. An uncertain
    // submission is never retried, and blocks further actions pending inspection.
    nlohmann::json submitCommand(std::size_t execute_close_position, const std::string& command);
    // Answer submits the Goal attempt, never a command. EOS/limits commit an
    // incomplete failed attempt. Grading reads the frozen host response only.
    void finishAttempt(const GeneratedSequence& sequence, std::size_t prompt_tokens,
        Episodes::AttemptEndReason reason, std::optional<std::size_t> answer_close_begin = {});
    void invalidateAttempt() { episode_.state = Episodes::EpisodeState::InfrastructureInvalid; }
    // Advance in the loader's authored/configured example order, retaining course
    // identity. A new example starts at spawn; the compiled world is reused when
    // its environment ID is unchanged. No automatic retries of failed attempts.
    bool advanceExample();
    bool scheduleComplete() const { return schedule_complete_; }
    const std::vector<Episodes::TaskEpisode>& previousEpisodes() const { return previous_episodes_; }

private:
    friend std::unique_ptr<EnvironmentSession> prepareEnvironmentSession(
        std::vector<ScheduledGoal>, const std::filesystem::path&);
    std::vector<ScheduledGoal> schedule_;
    std::size_t position_ = 0;
    WorldLab::Document environment_;
    std::filesystem::path path_;
    std::filesystem::path games_directory_;
    std::unique_ptr<WorldLab::TextWorldClient> client_;
    Episodes::TaskEpisode episode_;
    std::vector<Episodes::TaskEpisode> previous_episodes_;
    bool schedule_complete_ = false;
    bool transition_failed_ = false;
};

// Resolve the first assignment by persistent environment ID and compile/start
// its frozen Goal. Publishes no owner until a valid initial response arrives.
// Destruction of an unpublished candidate stops its worker on every failure.
std::unique_ptr<EnvironmentSession> prepareEnvironmentSession(
    std::vector<ScheduledGoal> schedule, const std::filesystem::path& games_directory);

} // namespace GRIM::Environment

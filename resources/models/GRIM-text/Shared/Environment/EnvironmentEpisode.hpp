#pragma once

#include "../DataLoader/ConceptBlockCorpusReader.hpp"
#include "../../../../../ui/world_lab/textworld_client.hpp"
#include "../../../../../ui/world_lab/world_document.hpp"

#include <memory>

namespace GRIM::Environment {

struct EnvironmentAction {
    std::size_t answer_close_position;
    std::string command;
    std::string status;
    nlohmann::json response;
    std::string error;
};

// Trim surrounding span whitespace, then require one nonempty command <=512 bytes.
std::string validateEnvironmentCommand(const std::string& answer);

// Host episode state carried across the Phase1/Phase2 boundary. The schedule
// owns frozen Goals; currentAssignment() is the sole current-Goal authority.
// Full responses are private host data, not a model-visible observation feed.
class EnvironmentEpisode {
public:
    EnvironmentEpisode(std::vector<ScheduledGoal> schedule,
                       WorldLab::Document environment, std::filesystem::path path);
    const ScheduledGoal& currentAssignment() const { return schedule_.at(position_); }
    const std::vector<ScheduledGoal>& schedule() const { return schedule_; }
    std::size_t position() const { return position_; }
    const WorldLab::Document& environment() const { return environment_; }
    const std::filesystem::path& environmentPath() const { return path_; }
    const nlohmann::json& latestResponse() const { return latest_response_; }
    WorldLab::TextWorldClient& client() { return client_; }
    const nlohmann::json& initialResponse() const { return initial_response_; }
    const std::vector<EnvironmentAction>& actions() const { return actions_; }
    // Boundary positions identify submissions within this episode's trajectory.
    // Repeated completed submissions return their recorded response. An uncertain
    // submission is never retried, and blocks further actions pending inspection.
    nlohmann::json submitAnswer(std::size_t answer_close_position, const std::string& answer);

private:
    friend std::unique_ptr<EnvironmentEpisode> prepareEnvironmentEpisode(
        std::vector<ScheduledGoal>, const std::filesystem::path&);
    std::vector<ScheduledGoal> schedule_;
    std::size_t position_ = 0;
    WorldLab::Document environment_;
    std::filesystem::path path_;
    WorldLab::TextWorldClient client_;
    nlohmann::json latest_response_;
    nlohmann::json initial_response_;
    std::vector<EnvironmentAction> actions_;
};

// Resolve the first assignment by persistent environment ID and compile/start
// its frozen Goal. Publishes no owner until a valid initial response arrives.
// Destruction of an unpublished candidate stops its worker on every failure.
std::unique_ptr<EnvironmentEpisode> prepareEnvironmentEpisode(
    std::vector<ScheduledGoal> schedule, const std::filesystem::path& games_directory);

} // namespace GRIM::Environment

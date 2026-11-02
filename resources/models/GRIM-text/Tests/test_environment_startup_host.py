"""Exercise production Phase1 environment wiring with CPU-only startup services.

No CUDA/trainer build, Docker execution, or dependency installation.
"""
from pathlib import Path

from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    root = Path(__file__).resolve().parent.parent
    startup = root / 'training/Phases'
    source = r'''
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace fs = std::filesystem;
int loads = 0, preparations = 0;
bool fail_preparation = false;
std::string metadata_stage = "rl";
namespace GRIM {
struct CurriculumMetadata { std::string training_stage; };
struct ScheduledGoal { std::string concept_block_id = "first", environment_id = "world"; };
namespace HyperParameters {
enum class ModelExecutionMode { TRAINING, INFERENCE };
enum class TrainingStage { RL, SFT };
struct Config {
    ModelExecutionMode mode = ModelExecutionMode::TRAINING;
    TrainingStage stage = TrainingStage::RL;
};
ModelExecutionMode snapshotExecutionMode(const Config& c) { return c.mode; }
struct DataHP { TrainingStage training_stage; };
DataHP dataLoadingHP(const Config& c) { return {c.stage}; }
struct TokHP { std::string data_path, training_curriculum; };
TokHP tokenizerHP(const Config&) { return {"/cache/goals.grmt", "RL Goals"}; }
struct ScheduleHP { std::string batch_strategy; };
ScheduleHP trainingScheduleHP(const Config&) { return {"PRESERVE"}; }
fs::path resolveMappedPath(const std::string& path) { return fs::path("/config") / path; }
}
namespace Batching {
enum class CurriculumOrdering { PRESERVE };
CurriculumOrdering parseCurriculumOrdering(const std::string& s) {
    assert(s == "PRESERVE"); return CurriculumOrdering::PRESERVE;
}
}
CurriculumMetadata LoadCurriculumMetadataFromRegistry(const fs::path& cache, const std::string& name) {
    assert(cache == fs::path("/cache") && name == "RL Goals");
    return {metadata_stage};
}
std::vector<ScheduledGoal> loadScheduledGoals(const fs::path& cache, const std::string& name,
    Batching::CurriculumOrdering, uint64_t seed) {
    assert(cache == fs::path("/cache") && name == "RL Goals" && seed == 123);
    ++loads; return {ScheduledGoal{}};
}
namespace Environment {
struct Response { bool is_null() const { return false; } };
struct EnvironmentEpisode {
    std::vector<ScheduledGoal> assignments;
    const ScheduledGoal& currentAssignment() const { return assignments.at(0); }
    Response latestResponse() const { return {}; }
};
std::unique_ptr<EnvironmentEpisode> prepareEnvironmentEpisode(std::vector<ScheduledGoal> schedule,
    const fs::path& directory) {
    assert(directory == fs::path("/config/resources/models/games"));
    ++preparations;
    if (fail_preparation) throw std::runtime_error("worker failed");
    return std::make_unique<EnvironmentEpisode>(EnvironmentEpisode{std::move(schedule)});
}
}
}
namespace GRIMText::Control { enum { TrainingState_Idle, TrainingState_Training }; }
namespace GRIMText::Training {
struct Logger { void log(const std::string&) {} };
struct Status {
    int last = -1;
    template<class... Args> void writeStatus(int state, Args...) { last = state; }
};
struct TrainingContext {
    GRIM::HyperParameters::Config config;
    std::unique_ptr<GRIM::Environment::EnvironmentEpisode> environment_episode;
    std::unique_ptr<int> environment_tokenizer;
    std::optional<GRIM::CurriculumMetadata> current_curriculum_metadata;
    struct { uint64_t data_seed = 123; } rng;
    struct { std::unique_ptr<Logger> logger = std::make_unique<Logger>();
             std::unique_ptr<Status> status_writer = std::make_unique<Status>(); } logging;
    std::chrono::steady_clock::time_point start_time;
    std::vector<int> epoch_batch_order;
    struct { int total_batches = 0; } epoch_plan;
};
struct Phase2HandoffInputs { TrainingContext& ctx; };
'''
    source += method((startup / 'Startup/EnvironmentReady.cu').read_text(),
                     'void EnvironmentReady(') + '\n'
    source += method((startup / 'Startup/Validation/Phase2Handoff.cu').read_text(),
                     'Phase2HandoffInputs makePhase2HandoffReady(') + '\n}\n'
    source += r'''
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
int main() {
    using namespace GRIMText::Training;
    using namespace GRIM::HyperParameters;
    TrainingContext ctx;
    rejects([&] { makePhase2HandoffReady(ctx); });
    ctx.config.mode = ModelExecutionMode::INFERENCE;
    rejects([&] { EnvironmentReady(ctx); });
    ctx.config.mode = ModelExecutionMode::TRAINING;
    ctx.config.stage = TrainingStage::SFT;
    rejects([&] { EnvironmentReady(ctx); });
    assert(loads == 0 && preparations == 0);
    ctx.config.stage = TrainingStage::RL;
    metadata_stage = "sft";
    rejects([&] { EnvironmentReady(ctx); });
    assert(loads == 0 && !ctx.current_curriculum_metadata && !ctx.environment_episode);
    metadata_stage = "rl";
    fail_preparation = true;
    rejects([&] { EnvironmentReady(ctx); });
    assert(!ctx.current_curriculum_metadata && !ctx.environment_episode);
    fail_preparation = false;
    EnvironmentReady(ctx);
    assert(ctx.current_curriculum_metadata->training_stage == "rl");
    assert(ctx.environment_episode->currentAssignment().concept_block_id == "first");
    rejects([&] { makePhase2HandoffReady(ctx); }); // Tokenizer is also required.
    ctx.environment_tokenizer = std::make_unique<int>(1);
    auto& owner = makePhase2HandoffReady(ctx).ctx;
    assert(&owner == &ctx && ctx.logging.status_writer->last == GRIMText::Control::TrainingState_Idle);
    const int before = preparations;
    rejects([&] { EnvironmentReady(ctx); });
    assert(preparations == before); // Never restart a prepared episode.
    ctx.config.stage = TrainingStage::SFT;
    makePhase2HandoffReady(ctx);
    assert(ctx.logging.status_writer->last == GRIMText::Control::TrainingState_Training);
}
'''
    compile_and_run(source)
    print('RL Phase1 schedule inputs, transactional publication and handoff checks passed')


if __name__ == '__main__':
    main()

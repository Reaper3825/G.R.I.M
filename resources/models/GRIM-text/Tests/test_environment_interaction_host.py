"""Test environment observation projection and the actual Phase2 begin body.

Tokenization/forward are controlled host services; no trainer, model, or Docker
process is built or run. The production projection and configured renderer run.
"""
import os
from pathlib import Path

from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    root = Path(__file__).resolve().parent.parent
    phase2 = root / 'training/Phases/Phase2_EnvironmentLoop.cu'
    source = r'''
#include "INPUT_IMPLEMENTATION"
#include "RENDERER_HEADER"
#include "EPISODE_HEADER"
#include <cassert>
#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
using nlohmann::json;
int prefill_calls = 0, forward_calls = 0;
bool supply_span_map = true, forward_fails = false;
std::string captured_prompt;
json captured_state;
namespace GRIM::HyperParameters {
enum class ModelExecutionMode { TRAINING, INFERENCE };
enum class TrainingStage { RL, SFT };
struct Config { ModelExecutionMode mode = ModelExecutionMode::TRAINING; TrainingStage stage = TrainingStage::RL; };
ModelExecutionMode snapshotExecutionMode(const Config& c) { return c.mode; }
struct DataHP { TrainingStage training_stage; };
DataHP dataLoadingHP(const Config& c) { return {c.stage}; }
struct GenerationHP { int max_new_tokens = 16; };
GenerationHP generationHP(const Config&) { return {}; }
}
namespace GRIM::Tokenizer { struct UniByte {}; }
namespace GRIM::Batching { struct BatchPayload { bool concept_span_delimiters; }; }
namespace GRIM::Inference {
GRIM::Batching::BatchPayload buildInferencePrefill(const HyperParameters::Config&,
    Tokenizer::UniByte&, const std::string& prompt, const json* state) {
    assert(state != nullptr); // A plain-text call cannot track phase spans.
    ++prefill_calls; captured_prompt = prompt; captured_state = *state;
    return {supply_span_map};
}
}
namespace GRIMText::Training {
struct Assignment { GRIM::ConceptBlockGoal goal; };
struct Episode {
    Assignment assignment;
    GRIM::Episodes::TaskEpisode task;
    const GRIM::Episodes::TaskEpisode& taskEpisode() const { return task; }
    json response;
    const Assignment& currentAssignment() const { return assignment; }
    const json& latestResponse() const { return response; }
};
struct GenerationState { bool active_session = false; };
struct TrainingContext {
    GRIM::HyperParameters::Config config;
    std::unique_ptr<Episode> environment_session;
    std::unique_ptr<GRIM::Tokenizer::UniByte> environment_tokenizer;
    GenerationState generation_state;
    GenerationState& requireGenerationState(const char*) { return generation_state; }
};
enum class ReturnReason { SpanBoundary, EOS, TokenLimit };
ReturnReason simulated_return = ReturnReason::SpanBoundary;
struct Phase2TextInferenceResult { int64_t encode_ms = -1; ReturnReason reason; };
RESULT_TYPE
Phase2TextInferenceResult executePhase2PayloadInference(TrainingContext& ctx,
    GRIM::Tokenizer::UniByte&, const GRIM::Batching::BatchPayload& prefill,
    const GRIM::HyperParameters::GenerationHP& hp) {
    ++forward_calls; assert(prefill.concept_span_delimiters && hp.max_new_tokens == 16);
    if (forward_fails) throw std::runtime_error("forward failed");
    ctx.generation_state.active_session = simulated_return == ReturnReason::SpanBoundary;
    return {-1, simulated_return};
}
BEGIN_INTERACTION
}
json response(const std::string& status = "active") {
    return {{"ok", true}, {"SECRET-envelope", "SECRET-transport"},
        {"result", {{"state", {{"feedback", "You are in a kitchen."},
            {"description", "There is a closed cupboard."}, {"inventory", "You carry a key."},
            {"location", "Kitchen"}, {"location_path", "/home/kitchen"},
            {"facts", {"SECRET-hidden-world"}}, {"admissible_commands", {"SECRET-command"}},
            {"tool_details", {{"internal", "SECRET-tool-envelope"}}},
            {"episode", {{"status", status}, {"requirements", {"SECRET-verifier"}}}},
            {"engine", "SECRET-engine"}, {"done", status != "active"}}}}}};
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
int main() {
    using namespace GRIMText::Training;
    GRIM::ConceptBlockGoal goal;
    goal.target_state = "Get the key to the hall";
    goal.success_criteria.push_back({"Carry the key in the hall",
        GRIM::EvidenceCondition{"SECRET-predicate", "SECRET-entity", "SECRET-target"}});
    goal.constraints.push_back({"Keep the cupboard closed",
        GRIM::EvidenceCondition{"SECRET-violation", "SECRET-cupboard", ""}});
    const auto frozen = GRIM::goalToJson(goal);
    const auto raw = response();
    const auto input = GRIM::Environment::buildEnvironmentInferenceInput(goal, raw);
    assert(input.prompt.empty());
    assert(input.reasoning_state.at("environment_context") == "Feedback:\nYou are in a kitchen.\n\nDescription:\nThere is a closed cupboard."
        "\n\nInventory:\nYou carry a key.\n\nLocation:\nKitchen\n\nLocation path:\n/home/kitchen");
    assert(input.episode_status == "active" && input.prompt.find("SECRET") == std::string::npos);
    assert(input.reasoning_state.dump().find("SECRET") == std::string::npos);
    assert(input.reasoning_state.at("knowns").empty() && input.reasoning_state.at("unknowns").empty());
    assert(GRIM::goalToJson(goal) == frozen && response() == raw);
    const std::string task = "Find the key.\nKeep the cupboard closed.";
    const auto initial = GRIM::Environment::buildEnvironmentInferenceInput(goal, raw, task);
    assert(initial.prompt == task);
    assert(initial.reasoning_state == input.reasoning_state);
    assert(initial.prompt.find("SECRET") == std::string::npos);
    assert(GRIM::Environment::buildEnvironmentInferenceInput(goal, raw, "").prompt == input.prompt);

    // Even a configured evidence leaf has nothing to render after projection.
    GRIM::NamedConceptSpanDefinition prompt;
    prompt.name = "prompt"; prompt.source_path = "/prompt";
    prompt.supervision = GRIM::ConceptSpanSupervision::Context;
    prompt.open_delimiter = "<prompt>"; prompt.close_delimiter = "</prompt>";
    auto target = prompt; target.name = "target_state"; target.source_path = "/goal/target_state";
    target.open_delimiter = "<target_state>"; target.close_delimiter = "</target_state>";
    auto evidence = prompt; evidence.name = "evidence";
    evidence.source_path = "/goal/success_criteria/0/evidence/subject";
    evidence.open_delimiter = "<evidence>"; evidence.close_delimiter = "</evidence>";
    auto environment = prompt; environment.name="environment_context";environment.source_path="/environment_context";
    environment.open_delimiter="<environment_context>";environment.close_delimiter="</environment_context>";
    GRIM::NamedConceptSpanDefinitions definitions{prompt, environment, target, evidence};
    int id = 1000;
    GRIM::resolveConceptSpanDelimiters(definitions, [&](const auto&) { return id++; });
    auto source = initial.reasoning_state; source["prompt"] = initial.prompt;
    const auto rendered = GRIM::SpanText::renderInputWithSpans(source, definitions);
    assert(rendered.text.find("<prompt>\nFind the key.") == 0);
    assert(rendered.text.find("<environment_context>\nFeedback:")!=std::string::npos);
    assert(rendered.text.find("<target_state>\nGet the key to the hall") != std::string::npos);
    assert(rendered.text.find("SECRET") == std::string::npos && rendered.text.find("<evidence>") == std::string::npos);

    auto decorated=raw;
    auto& visible=decorated["result"]["state"];
    visible["feedback"]="\n\n                    ________  ________  __    __  ________\n"
        "                   |        \\|        \\|  \\  |  \\|        \\\n"
        "                       \\$$    \\$$$$$$$$ \\$$   \\$$    \\$$\n\n"
        "-= System Hub =-\nA useful room description.\n\n>             -= System Hub =-0/1\nLOCATION: /\n";
    visible["description"]="-= System Hub =-\nA useful room description.";
    visible["location_path"]="/";
    const auto decoratedCopy=decorated;
    const auto clean=GRIM::Environment::buildEnvironmentInferenceInput(goal,decorated,task);
    const auto observation=clean.reasoning_state.at("environment_context").get<std::string>();
    assert(clean.prompt==task && decorated==decoratedCopy);
    assert(observation.find("$$$$")==std::string::npos && observation.find("-= ")==std::string::npos && observation.find("LOCATION:")==std::string::npos);
    assert(observation.find("Feedback:")==std::string::npos && observation.find("A useful room description.")!=std::string::npos);
    visible["feedback"]="You cannot go that way.";
    assert(GRIM::Environment::buildEnvironmentInferenceInput(goal,decorated).reasoning_state.at("environment_context").get<std::string>().find("You cannot go that way.")!=std::string::npos);
    assert(GRIM::Environment::cleanEnvironmentObservation("Plan:\n+---+\n| X |\n+---+")=="Plan:\n+---+\n| X |\n+---+");

    for (const auto& invalid : std::vector<json>{
            {{"ok", false}, {"error", "worker failed"}},
            {{"ok", true}, {"result", json::object()}}, response("invalid")})
        rejects([&] { GRIM::Environment::buildEnvironmentInferenceInput(goal, invalid); });
    auto invalid = raw; invalid["result"]["state"]["inventory"] = json::array();
    rejects([&] { GRIM::Environment::buildEnvironmentInferenceInput(goal, invalid); });
    invalid = raw; invalid["result"]["state"]["done"] = true;
    rejects([&] { GRIM::Environment::buildEnvironmentInferenceInput(goal, invalid); });
    invalid = raw;
    for (const auto* field : {"feedback", "description", "inventory", "location", "location_path"})
        invalid["result"]["state"].erase(field);
    rejects([&] { GRIM::Environment::buildEnvironmentInferenceInput(goal, invalid); });
    rejects([&] { GRIM::Environment::buildEnvironmentInferenceInput(goal, invalid, task); });

    TrainingContext ctx;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
    ctx.environment_session = std::make_unique<Episode>();
    ctx.environment_session->assignment.goal = goal;
    ctx.environment_session->task.goal = std::make_shared<const GRIM::ConceptBlockGoal>(goal);
    ctx.environment_session->response = raw;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
    ctx.environment_tokenizer = std::make_unique<GRIM::Tokenizer::UniByte>();
    ctx.config.stage = GRIM::HyperParameters::TrainingStage::SFT;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
    ctx.config.stage = GRIM::HyperParameters::TrainingStage::RL;
    ctx.config.mode = GRIM::HyperParameters::ModelExecutionMode::INFERENCE;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
    ctx.config.mode = GRIM::HyperParameters::ModelExecutionMode::TRAINING;
    assert(prefill_calls == 0 && forward_calls == 0);
    for (const auto* status : {"succeeded", "failed", "stopped"}) {
        ctx.environment_session->response = response(status);
        const auto terminal = beginPhase2EnvironmentInteraction(ctx);
        assert(terminal.episode_status == status && terminal.generation);
        ctx.generation_state.active_session = false;
    }
    assert(prefill_calls == 3 && forward_calls == 3);
    prefill_calls = forward_calls = 0;
    ctx.environment_session->response = raw;
    supply_span_map = false;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
    assert(forward_calls == 0 && !ctx.generation_state.active_session);
    supply_span_map = true;
    forward_fails = true;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
    assert(!ctx.generation_state.active_session && ctx.environment_session->response == raw);
    forward_fails = false;
    const auto result = beginPhase2EnvironmentInteraction(ctx);
    assert(result.episode_status == "active" && result.generation->reason == ReturnReason::SpanBoundary);
    assert(result.generation->encode_ms >= 0 && ctx.generation_state.active_session);
    assert(captured_prompt == input.prompt && captured_state == input.reasoning_state);
    const auto before = forward_calls;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
    assert(forward_calls == before); // Never silently replace a paused session.
    for (auto reason : {ReturnReason::EOS, ReturnReason::TokenLimit}) {
        ctx.generation_state.active_session = false;
        simulated_return = reason;
        const auto ended = beginPhase2EnvironmentInteraction(ctx);
        assert(ended.generation->reason == reason && ended.episode_status == "active");
        assert(!ctx.generation_state.active_session); // Model termination isn't Goal completion.
    }
    assert(ctx.environment_session->response == raw);
    ctx.environment_session->task.state = GRIM::Episodes::EpisodeState::AttemptSubmitted;
    rejects([&] { beginPhase2EnvironmentInteraction(ctx); });
}
'''
    for key, value in {
        'INPUT_IMPLEMENTATION': (root / 'Shared/Environment/EnvironmentInput.cpp').as_posix(),
        'EPISODE_HEADER': (root / 'Shared/Episode/TaskEpisode.hpp').as_posix(),
        'RENDERER_HEADER': (root / 'Shared/ConceptBlock/SpanTextRenderer.hpp').as_posix(),
        'RESULT_TYPE': method(phase2.with_suffix('.hpp').read_text(), 'struct Phase2EnvironmentResult') + ';',
        'BEGIN_INTERACTION': method(phase2.read_text(), 'Phase2EnvironmentResult beginPhase2EnvironmentInteraction('),
    }.items():
        source = source.replace(key, value)
    previous = os.environ.get('CL')
    if os.name == 'nt':
        os.environ['CL'] = (previous or '') + f' /I"{root.parent / "llama.cpp/vendor"}"'
    try:
        compile_and_run(source)
    finally:
        if previous is None:
            os.environ.pop('CL', None)
        else:
            os.environ['CL'] = previous
    print('Environment projection, structured input and Phase2 first-return routing passed')


if __name__ == '__main__':
    main()

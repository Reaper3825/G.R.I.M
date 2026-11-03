#include "resources/models/GRIM-text/Shared/Environment/EnvironmentSession.hpp"
#include "resources/models/GRIM-text/Shared/Environment/EnvironmentInput.hpp"
#include "resources/models/GRIM-text/Shared/Environment/EnvironmentBoundary.hpp"
#include "resources/models/GRIM-text/Shared/ConceptBlock/SpanTextRenderer.hpp"
#include "resources/models/GRIM-text/Shared/UnigramByte/TokenLayout.hpp"
#include "ui/world_lab/environment_registry.hpp"
#include <cassert>
#include <chrono>
#include <deque>
#include <iostream>
#include <cstdlib>

// Keep host failures visible when Windows suppresses the CRT assertion dialog.
#undef assert
#define assert(...) do { if (!(__VA_ARGS__)) { \
    std::cerr << "Assertion failed at line " << __LINE__ << ": " << #__VA_ARGS__ << '\n'; \
    std::exit(1); } } while (false)

using nlohmann::json;
namespace fs = std::filesystem;
namespace {
std::deque<json> responses;
std::vector<std::string> commands, feedback;
std::vector<json> example_starts;
bool feedback_due = false;
int resumes = 0, resets = 0;
struct Step { std::string phase, content; };
std::deque<Step> script;
json response(const std::string& status = "active", const std::string& text = "You are at the entrance.") {
    return {{"ok", true}, {"result", {{"environment_id", "fingerprint"},
        {"state", {{"feedback", text}, {"description", "A doorway is visible."},
            {"done", status != "active"}, {"facts", {"PRIVATE WORLD FACT"}},
            {"episode", {{"status", status}, {"requirements", {{{"collection", "success_criteria"},
                {"index", 0}, {"text", "PRIVATE VERIFICATION"},
                {"status", status == "succeeded" ? "satisfied" : "unsatisfied"}}}},
                {"identity", {{"source_id", "goal-a"}, {"environment", "fingerprint"}}}}}}}}}};
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
void clearPlan(std::deque<Step> steps, std::deque<json> observations) {
    script = std::move(steps); responses = std::move(observations);
    commands.clear(); feedback.clear(); example_starts.clear(); resumes = resets = 0; feedback_due = false;
}
}

namespace GRIM::WorldLab {
struct TextWorldClient::Impl { bool pending = false; };
TextWorldClient::TextWorldClient() : impl_(std::make_unique<Impl>()) {}
TextWorldClient::~TextWorldClient() = default;
bool TextWorldClient::busy() const { return impl_->pending; }
bool TextWorldClient::begin(json request) {
    assert(!impl_->pending);
    if (request.at("op") == "step") {
        commands.push_back(request.at("command").get<std::string>());
        feedback_due = true;
    } else {
        assert(request.at("op") == "compile" || request.at("op") == "start_episode");
        example_starts.push_back(request);
    }
    impl_->pending = true; return true;
}
std::optional<json> TextWorldClient::poll() {
    if (!impl_->pending) return {};
    assert(!responses.empty());
    impl_->pending = false;
    auto next = responses.front(); responses.pop_front(); return next;
}
void TextWorldClient::stop() { impl_->pending = false; }
}

namespace GRIM::Tokenizer {
struct DecodeRequest {
    const std::vector<int>& ids;
    std::size_t begin;
    std::size_t token_count, atom_entry_count, token_numeric_count, token_atom_mask_count;
    bool lenient_invalid_utf8 = true;
    DecodeRequest(const std::vector<int>& tokens, const std::vector<uint32_t>&, const AtomTable*,
        const std::vector<float>&, const std::vector<uint8_t>&, std::size_t first)
        : ids(tokens), begin(first), token_count(tokens.size() - first) {}
};
struct UniByte {
    std::string decode(const DecodeRequest& request) {
        assert(!request.lenient_invalid_utf8);
        std::string text;
        for (std::size_t p = request.begin; p < request.begin + request.token_count; ++p) {
            assert(request.ids[p] >= 500 && request.ids[p] < 756);
            text += static_cast<char>(request.ids[p] - 500);
        }
        return text;
    }
};
}

namespace GRIMText::Training {
struct Phase2TextInferenceResult {
    std::optional<GRIM::GeneratedSequence> generated_sequence;
    std::size_t prompt_token_count = 1;
};
struct Phase2EnvironmentResult {
    std::string episode_status;
    std::optional<Phase2TextInferenceResult> generation;
};
struct Phase2EnvironmentRunResult { Phase2EnvironmentResult interaction; std::string stop_reason; };
struct Phase2EnvironmentLoopResult {
    std::size_t examples_completed = 0, examples_succeeded = 0, examples_failed = 0, commands = 0;
};
struct GenerationState {
    GRIM::GeneratedSequence sequence;
    void resetSession() { ++resets; }
};
struct TrainingContext {
    std::unique_ptr<GRIM::Environment::EnvironmentSession> environment_session;
    std::unique_ptr<GRIM::Tokenizer::UniByte> environment_tokenizer = std::make_unique<GRIM::Tokenizer::UniByte>();
    GenerationState generation;
    GenerationState& requireGenerationState(const char*) { return generation; }
};
void appendToken(GRIM::GeneratedSequence& sequence, int id, GRIM::TokenOrigin origin = GRIM::TokenOrigin::SampledAction) {
    sequence.token_ids.push_back(id); sequence.token_origins.push_back(origin);
    sequence.atom_entry_ids.push_back(UINT32_MAX); sequence.token_numeric_values.push_back(0);
    sequence.token_atom_mask.push_back(0);
}
Phase2TextInferenceResult advance(TrainingContext& ctx) {
    assert(!script.empty());
    const auto step = script.front(); script.pop_front();
    auto& sequence = ctx.generation.sequence;
    if (step.phase == "repeat") return {sequence};
    if (step.phase == "eos") {
        sequence.return_reason = GRIM::Inference::InferenceReturnReason::EOS;
        sequence.finished = true; return {sequence};
    }
    const auto& definitions = sequence.concept_span_delimiters->nodes();
    std::size_t index = 0;
    while (definitions.at(index).name != (step.phase == "capped_answer" ? "answer" : step.phase)) ++index;
    const auto start = sequence.token_ids.size();
    appendToken(sequence, definitions[index].open_id);
    sequence.span_events.push_back({GRIM::GeneratedSpanEventKind::Open, static_cast<uint32_t>(index), start, start});
    if (step.content == "UNFINISHED_ATOM") appendToken(sequence, GRIM::Tokenizer::atomTypeToOpenTokenId(GRIM::Tokenizer::AtomType::ATOM_INT));
    else for (const unsigned char c : step.content) appendToken(sequence, 500 + c);
    appendToken(sequence, definitions[index].close_id);
    sequence.span_events.push_back({GRIM::GeneratedSpanEventKind::Close, static_cast<uint32_t>(index), sequence.token_ids.size() - 1, start});
    sequence.return_reason = GRIM::Inference::InferenceReturnReason::SpanBoundary;
    if (step.phase == "capped_answer") {
        sequence.return_reason = GRIM::Inference::InferenceReturnReason::TokenLimit;
        sequence.finished = true;
    }
    return {sequence};
}
Phase2EnvironmentResult beginPhase2EnvironmentInteraction(TrainingContext& ctx) {
    const auto input = GRIM::Environment::buildEnvironmentInferenceInput(
        ctx.environment_session->currentAssignment().goal, ctx.environment_session->latestResponse());
    auto& sequence = ctx.generation.sequence;
    sequence = {}; // Production begin creates a fresh generation session per example.
    GRIM::NamedConceptSpanDefinitions definitions;
    int id = 1000;
    GRIM::NamedConceptSpanDefinition context;
    context.name="environment_context";context.source_path="/environment_context";
    context.supervision=GRIM::ConceptSpanSupervision::Context;
    context.open_delimiter="<environment_context>";context.close_delimiter="</environment_context>";
    context.open_delimiter_id=id++;context.close_delimiter_id=id++;definitions.push_back(context);
    for (const auto* name : {"determine", "define", "execute", "update", "answer"}) {
        GRIM::NamedConceptSpanDefinition d;
        d.name = name; d.source_path = std::string("/") + name;
        d.supervision = GRIM::ConceptSpanSupervision::Supervised;
        d.open_delimiter = std::string("<") + name + ">"; d.close_delimiter = std::string("</") + name + ">";
        d.open_delimiter_id = id++; d.close_delimiter_id = id++;
        definitions.push_back(d);
    }
    sequence.concept_span_delimiters = std::make_shared<GRIM::ConceptSpanDelimiterMap>(definitions);
    appendToken(sequence, 99, GRIM::TokenOrigin::SuppliedContext);
    return {input.episode_status, advance(ctx)};
}
std::size_t appendPhase2RuntimeContext(TrainingContext& ctx, GRIM::Tokenizer::UniByte&, const std::string& text) {
    assert(feedback_due && text.find("PRIVATE") == std::string::npos);
    feedback.push_back(text); feedback_due = false;
    appendToken(ctx.generation.sequence, 2000, GRIM::TokenOrigin::RuntimeContext);
    return 1;
}
Phase2TextInferenceResult resumePhase2PayloadInference(TrainingContext& ctx, GRIM::Tokenizer::UniByte&) {
    assert(!feedback_due); ++resumes; return advance(ctx);
}
}

// PRODUCTION_DISPATCH

int main() try {
    const auto directory = fs::temp_directory_path() / ("grim-environment-dispatch-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directory(directory);
    auto world = GRIM::WorldLab::Document::initial(); world.environmentId = "world";
    const auto path = directory / "world.grimworld.json";
    GRIM::WorldLab::saveJsonFile(path, world.json());
    GRIM::WorldLab::EnvironmentRegistry(directory).registerFile(path);
    const auto context = [&] {
        GRIMText::Training::TrainingContext ctx;
        GRIM::ConceptBlockGoal goal; goal.target_state = "Reach the destination";
        ctx.environment_session = GRIM::Environment::prepareEnvironmentSession(
            {{"course", "world", "goal-a", goal}}, directory);
        return ctx;
    };
    using GRIMText::Training::executePhase2EnvironmentInteraction;
    clearPlan({{"determine", "Find a route"}, {"define", "A doorway"},
        {"update", "The initial observation is known"}, {"execute", "\n open door \n"},
        {"determine", "Use the new observation"}, {"execute", "go north"},
        {"answer", "I reached the destination.\nThe route is complete."}},
        {response(), response("active", "The door opens."), response("succeeded", "You arrive.")});
    {
        auto ctx = context(); const auto initial = ctx.environment_session->initialResponse();
        const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "attempt_submitted" && run.interaction.episode_status == "succeeded");
        assert(commands == std::vector<std::string>({"open door", "go north"}));
        assert(feedback.size() == 2 && feedback[0].find("The door opens.") != std::string::npos);
        assert(feedback[1].find("You arrive.") != std::string::npos);
        const auto& task = ctx.environment_session->taskEpisode();
        assert(task.state == GRIM::Episodes::EpisodeState::Complete);
        assert(task.goal->target_state == "Reach the destination");
        assert(task.attempts[0]->end_reason == GRIM::Episodes::AttemptEndReason::AnswerClosed);
        assert(task.attempts[0]->grade.verdict == GRIM::Episodes::GradeVerdict::Passed);
        assert(task.attempts[0]->grade.checks.size() == 1 &&
            task.attempts[0]->grade.checks[0].status == GRIM::Episodes::CheckStatus::Passed);
        assert(task.attempts[0]->answer_close_begin == task.sequence.token_ids.size() - 1);
        assert(task.attempts[0]->environment_snapshot == ctx.environment_session->latestResponse());
        rejects([&] { executePhase2EnvironmentInteraction(ctx); });
        assert(task.state == GRIM::Episodes::EpisodeState::Complete);
        for (const auto& segment : task.attempts[0]->segments)
            for (auto p = segment.token_begin; p < segment.token_end; ++p)
                assert(task.sequence.token_origins[p] == GRIM::TokenOrigin::SampledAction);
        const auto submitted = commands.size();
        rejects([&] { ctx.environment_session->submitCommand(task.sequence.token_ids.size() + 5, "look"); });
        assert(commands.size() == submitted);
        assert(ctx.environment_session->actions().size() == 2 && resets == 1);
        assert(ctx.environment_session->initialResponse() == initial && responses.empty());
        const auto& final = *run.interaction.generation->generated_sequence;
        auto invalid = final; invalid.token_origins.back() = GRIM::TokenOrigin::RuntimeContext;
        rejects([&] { GRIM::Environment::inspectEnvironmentBoundary(invalid, 1, {}); });
        invalid = final;
        invalid.token_origins[final.span_events.back().begin + 1] = GRIM::TokenOrigin::RuntimeContext;
        rejects([&] { GRIM::Environment::inspectEnvironmentBoundary(invalid, 1, {}); });
        rejects([&] { GRIM::Environment::inspectEnvironmentBoundary(final, 1, final.span_events.back().token_position); });
    }
    clearPlan({{"capped_answer", "go north"}}, {response()});
    {
        auto ctx = context(); const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "generation_terminal" && commands.empty());
        assert(ctx.environment_session->taskEpisode().attempts[0]->grade.verdict == GRIM::Episodes::GradeVerdict::Failed);
    }
    clearPlan({{"execute", "look"}, {"eos", ""}}, {response(), response("active", "Nothing changes.")});
    {
        auto ctx = context(); const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "generation_terminal" && run.interaction.episode_status == "active");
        assert(commands.size() == 1 && feedback.size() == 1 && ctx.environment_session->actions().size() == 1);
    }
    for (const auto* status : {"succeeded", "failed", "stopped"}) {
        clearPlan({{"answer", "Finished"}}, {response(status)});
        auto ctx = context(); const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "attempt_submitted" && run.interaction.generation && commands.empty());
        assert(ctx.environment_session->taskEpisode().attempts[0]->grade.verdict ==
            (std::string(status) == "succeeded" ? GRIM::Episodes::GradeVerdict::Passed : GRIM::Episodes::GradeVerdict::Failed));
    }
    for (const auto& steps : std::vector<std::deque<Step>>{
            {{"execute", "look\ninventory"}}, {{"execute", "   "}},
            {{"answer", "UNFINISHED_ATOM"}}, {{"determine", "D"}, {"repeat", ""}}}) {
        clearPlan(steps, {response()});
        auto ctx = context(); rejects([&] { executePhase2EnvironmentInteraction(ctx); });
        assert(commands.empty() && resets == 1);
        assert(ctx.environment_session->taskEpisode().state == GRIM::Episodes::EpisodeState::AttemptSubmitted);
        assert(ctx.environment_session->taskEpisode().attempts[0]->end_reason ==
            GRIM::Episodes::AttemptEndReason::InvalidOutput);
    }
    clearPlan({{"execute", "look"}}, {response(), {{"ok", false}, {"transport_error", true}, {"error", "timeout"}}});
    {
        auto ctx = context(); rejects([&] { executePhase2EnvironmentInteraction(ctx); });
        assert(commands.size() == 1 && ctx.environment_session->actions()[0].status == "unknown" && resets == 1);
        assert(ctx.environment_session->taskEpisode().state == GRIM::Episodes::EpisodeState::InfrastructureInvalid);
        assert(!ctx.environment_session->taskEpisode().attempts[0]);
    }
    // Answer text resembling a command never executes; unmet Goal fails at submission.
    clearPlan({{"answer", "go north"}}, {response()});
    {
        auto ctx = context(); const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "attempt_submitted" && commands.empty() && resumes == 0);
        const auto& task = ctx.environment_session->taskEpisode();
        assert(task.state == GRIM::Episodes::EpisodeState::AttemptSubmitted);
        assert(task.attempts[0]->grade.verdict == GRIM::Episodes::GradeVerdict::Failed);
        assert(!task.scores[0]);
        rejects([&] { ctx.environment_session->finishAttempt(task.sequence, 1,
            GRIM::Episodes::AttemptEndReason::AnswerClosed, task.sequence.token_ids.size() - 1); });
    }
    // Goal success without a submitted Answer remains an incomplete failed attempt.
    clearPlan({{"execute", "go north"}, {"eos", ""}}, {response(), response("succeeded")});
    {
        auto ctx = context(); const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "generation_terminal" && commands.size() == 1);
        assert(ctx.environment_session->taskEpisode().attempts[0]->grade.verdict == GRIM::Episodes::GradeVerdict::Failed);
    }
    const auto example_response = [&](const std::string& id, const std::string& status = "active") {
        auto result = response(status);
        result["result"]["state"]["episode"]["identity"]["source_id"] = id;
        return result;
    };
    auto second_world = world; second_world.environmentId = "world-b";
    const auto second_path = directory / "second.grimworld.json";
    GRIM::WorldLab::saveJsonFile(second_path, second_world.json());
    GRIM::WorldLab::EnvironmentRegistry(directory).registerFile(second_path);
    clearPlan({{"execute", "go north"}, {"answer", "First complete"}, {"eos", ""},
        {"answer", "Not complete"}, {"execute", "look\ninventory"}, {"answer", "Last complete"}},
        {example_response("a1"), example_response("a1", "succeeded"), example_response("a2"),
         example_response("b1"), example_response("b2"), example_response("b3", "succeeded")});
    {
        GRIMText::Training::TrainingContext ctx;
        GRIM::ConceptBlockGoal goal; goal.target_state = "Reach the destination";
        ctx.environment_session = GRIM::Environment::prepareEnvironmentSession({
            {"course-a", "world", "a1", goal}, {"course-a", "world", "a2", goal},
            {"course-b", "world-b", "b1", goal}, {"course-b", "world-b", "b2", goal},
            {"course-b", "world-b", "b3", goal}}, directory);
        const auto run = GRIMText::Training::executePhase2EnvironmentLoop(ctx);
        assert(run.examples_completed == 5 && run.examples_succeeded == 2 && run.examples_failed == 3);
        assert(run.commands == 1 && script.empty() && responses.empty());
        const auto& session = *ctx.environment_session;
        assert(session.scheduleComplete() && session.position() == 4);
        assert(session.environment().environmentId == "world-b" && session.environmentPath() == second_path);
        const auto& history = session.previousEpisodes();
        assert(history.size() == 4 && history[0].task_id == "a1" && history[1].task_id == "a2");
        assert(history[0].task_family == "course-a" && history[2].task_family == "course-b");
        assert(history[0].actions.size() == 1 && history[1].actions.empty());
        assert(history[1].attempts[0]->end_reason == GRIM::Episodes::AttemptEndReason::Eos);
        assert(history[3].attempts[0]->end_reason == GRIM::Episodes::AttemptEndReason::InvalidOutput);
        assert(session.taskEpisode().task_id == "b3" && session.taskEpisode().actions.empty());
        assert(session.taskEpisode().sequence.token_ids.size() < history[0].sequence.token_ids.size());
        const std::vector<std::string> expected_ops{"compile", "start_episode", "compile", "start_episode", "start_episode"};
        const std::vector<std::string> expected_ids{"a1", "a2", "b1", "b2", "b3"};
        assert(example_starts.size() == expected_ops.size());
        for (std::size_t i = 0; i < expected_ops.size(); ++i) {
            assert(example_starts[i].at("op") == expected_ops[i]);
            assert(example_starts[i].at("source_id") == expected_ids[i]);
            assert(example_starts[i].contains("document") == (expected_ops[i] == "compile"));
        }
        const auto count = example_starts.size();
        assert(!ctx.environment_session->advanceExample());
        rejects([&] { GRIMText::Training::executePhase2EnvironmentLoop(ctx); });
        assert(example_starts.size() == count);
    }
    // Infrastructure uncertainty must stop rather than consume later examples.
    clearPlan({{"execute", "look"}, {"answer", "must not run"}},
        {example_response("a1"), {{"ok", false}, {"error", "transport timeout"}}});
    {
        GRIMText::Training::TrainingContext ctx;
        GRIM::ConceptBlockGoal goal; goal.target_state = "Reach the destination";
        ctx.environment_session = GRIM::Environment::prepareEnvironmentSession({
            {"course-a", "world", "a1", goal}, {"course-a", "world", "a2", goal}}, directory);
        rejects([&] { GRIMText::Training::executePhase2EnvironmentLoop(ctx); });
        assert(ctx.environment_session->position() == 0 && example_starts.size() == 1);
        assert(ctx.environment_session->previousEpisodes().empty() && script.size() == 1);
    }
    // An unacknowledged reset retains the committed result and is never retried.
    clearPlan({{"answer", "Finished"}},
        {example_response("a1", "succeeded"), example_response("wrong-source")});
    {
        GRIMText::Training::TrainingContext ctx;
        GRIM::ConceptBlockGoal goal; goal.target_state = "Reach the destination";
        ctx.environment_session = GRIM::Environment::prepareEnvironmentSession({
            {"course-a", "world", "a1", goal}, {"course-a", "world", "a2", goal}}, directory);
        rejects([&] { GRIMText::Training::executePhase2EnvironmentLoop(ctx); });
        assert(ctx.environment_session->position() == 0 && example_starts.size() == 2);
        assert(ctx.environment_session->taskEpisode().attempts[0]->grade.verdict == GRIM::Episodes::GradeVerdict::Passed);
        rejects([&] { ctx.environment_session->advanceExample(); });
        assert(example_starts.size() == 2);
    }
    // Failed compilation in a new course preserves the prior world and result.
    clearPlan({{"answer", "Finished"}},
        {example_response("a1", "succeeded"), {{"ok", false}, {"error", "next world binding failed"}}});
    {
        GRIMText::Training::TrainingContext ctx;
        GRIM::ConceptBlockGoal goal; goal.target_state = "Reach the destination";
        ctx.environment_session = GRIM::Environment::prepareEnvironmentSession({
            {"course-a", "world", "a1", goal}, {"course-b", "world-b", "b1", goal}}, directory);
        rejects([&] { GRIMText::Training::executePhase2EnvironmentLoop(ctx); });
        assert(ctx.environment_session->position() == 0 && example_starts.size() == 2);
        assert(ctx.environment_session->environment().environmentId == "world");
        assert(ctx.environment_session->taskEpisode().task_id == "a1");
        assert(ctx.environment_session->taskEpisode().state == GRIM::Episodes::EpisodeState::Complete);
        rejects([&] { ctx.environment_session->advanceExample(); });
        assert(example_starts.size() == 2);
    }
    for (const auto& entry : fs::directory_iterator(directory)) fs::remove(entry.path());
    fs::remove(directory);
    std::cout << "Execute actions, Answer attempt submission, frozen grading and failure handling passed\n";
} catch (const std::exception& error) {
    std::cerr << "Environment controller test failed: " << error.what() << '\n';
    return 1;
}

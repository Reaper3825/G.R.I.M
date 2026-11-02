#include "resources/models/GRIM-text/Shared/Environment/EnvironmentEpisode.hpp"
#include "resources/models/GRIM-text/Shared/Environment/EnvironmentInput.hpp"
#include "resources/models/GRIM-text/Shared/Environment/EnvironmentBoundary.hpp"
#include "resources/models/GRIM-text/Shared/UnigramByte/TokenLayout.hpp"
#include "ui/world_lab/environment_registry.hpp"
#include <cassert>
#include <chrono>
#include <deque>
#include <iostream>

using nlohmann::json;
namespace fs = std::filesystem;
namespace {
std::deque<json> responses;
std::vector<std::string> commands, feedback;
bool feedback_due = false;
int resumes = 0, resets = 0;
struct Step { std::string phase, content; };
std::deque<Step> script;
json response(const std::string& status = "active", const std::string& text = "You are at the entrance.") {
    return {{"ok", true}, {"result", {{"environment_id", "fingerprint"},
        {"state", {{"feedback", text}, {"description", "A doorway is visible."},
            {"done", status != "active"}, {"facts", {"PRIVATE WORLD FACT"}},
            {"episode", {{"status", status}, {"requirements", {"PRIVATE VERIFICATION"}},
                {"identity", {{"source_id", "goal-a"}, {"environment", "fingerprint"}}}}}}}}}};
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
void clearPlan(std::deque<Step> steps, std::deque<json> observations) {
    script = std::move(steps); responses = std::move(observations);
    commands.clear(); feedback.clear(); resumes = resets = 0; feedback_due = false;
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
    } else assert(request.at("op") == "compile");
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
struct GenerationState {
    GRIM::GeneratedSequence sequence;
    void resetSession() { ++resets; }
};
struct TrainingContext {
    std::unique_ptr<GRIM::Environment::EnvironmentEpisode> environment_episode;
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
        ctx.environment_episode->currentAssignment().goal, ctx.environment_episode->latestResponse());
    if (input.episode_status != "active") return {input.episode_status, {}};
    auto& sequence = ctx.generation.sequence;
    GRIM::NamedConceptSpanDefinitions definitions;
    int id = 1000;
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
    return {"active", advance(ctx)};
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

int main() {
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
        ctx.environment_episode = GRIM::Environment::prepareEnvironmentEpisode(
            {{"course", "world", "goal-a", goal}}, directory);
        return ctx;
    };
    using GRIMText::Training::executePhase2EnvironmentInteraction;
    clearPlan({{"determine", "Find a route"}, {"define", "A doorway"}, {"execute", "Choose an action"},
        {"update", "The initial observation is known"}, {"answer", "\n open door \n"},
        {"determine", "Use the new observation"}, {"answer", "go north"}},
        {response(), response("active", "The door opens."), response("succeeded", "You arrive.")});
    {
        auto ctx = context(); const auto initial = ctx.environment_episode->initialResponse();
        const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "episode_terminal" && run.interaction.episode_status == "succeeded");
        assert(commands == std::vector<std::string>({"open door", "go north"}));
        assert(feedback.size() == 1 && feedback[0].find("The door opens.") != std::string::npos);
        assert(ctx.environment_episode->actions().size() == 2 && resets == 1);
        assert(ctx.environment_episode->initialResponse() == initial && responses.empty());
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
    }
    clearPlan({{"answer", "look"}, {"eos", ""}}, {response(), response("active", "Nothing changes.")});
    {
        auto ctx = context(); const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "generation_terminal" && run.interaction.episode_status == "active");
        assert(commands.size() == 1 && feedback.size() == 1 && ctx.environment_episode->actions().size() == 1);
    }
    for (const auto* status : {"succeeded", "failed", "stopped"}) {
        clearPlan({}, {response(status)});
        auto ctx = context(); const auto run = executePhase2EnvironmentInteraction(ctx);
        assert(run.stop_reason == "episode_terminal" && !run.interaction.generation && commands.empty());
    }
    for (const auto& steps : std::vector<std::deque<Step>>{
            {{"answer", "look\ninventory"}}, {{"answer", "   "}},
            {{"answer", "UNFINISHED_ATOM"}}, {{"determine", "D"}, {"repeat", ""}}}) {
        clearPlan(steps, {response()});
        auto ctx = context(); rejects([&] { executePhase2EnvironmentInteraction(ctx); });
        assert(commands.empty() && resets == 1);
    }
    clearPlan({{"answer", "look"}}, {response(), {{"ok", false}, {"transport_error", true}, {"error", "timeout"}}});
    {
        auto ctx = context(); rejects([&] { executePhase2EnvironmentInteraction(ctx); });
        assert(commands.size() == 1 && ctx.environment_episode->actions()[0].status == "unknown" && resets == 1);
    }
    for (const auto& entry : fs::directory_iterator(directory)) fs::remove(entry.path());
    fs::remove(directory);
    std::cout << "Environment DDEA dispatch, feedback recurrence and failure handling passed\n";
}

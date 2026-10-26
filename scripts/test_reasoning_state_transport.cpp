#include "../DataCollection/reasoning_state_json.hpp"
#include "../resources/models/GRIM-text/Shared/Forward/InferenceInput.hpp"
#include "../tests/concept_span_test_helpers.hpp"
#include <cassert>
#include <iostream>
using nlohmann::json;

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
int main() {
    const auto definitions = testSpanDefinitions();
    GRIM::ReasoningState state;
    state.knowns = {"capacity = 120 liters", "remaining = 84 liters"};
    state.unknowns = {"consumed volume"};
    state.goal = GRIM::ConceptBlockGoal{"Find consumed volume", {
        {"One assignment", GRIM::EvidenceCondition{"carrying", "key_1", ""}}}, {
        {"Use liters", GRIM::EvidenceCondition{"in", "apple_1", "box_1"}}}};
    // Client wire round trip, then the worker's payload-input adapter.
    json request{{"messages", {{{"role", "user"}, {"content", "Question?"}}}},
                 {"reasoning_state", GRIM::reasoningStateToJson(state)}};
    const auto wire = request.dump();
    state.knowns.clear(); // A queued/transmitted snapshot must not alias its source.
    const auto received = json::parse(wire);
    const auto restored = GRIM::reasoningStateFromJson(received.at("reasoning_state"));
    assert(restored.knowns.size() == 2);
    assert(restored.unknowns == state.unknowns);
    assert(restored.goal->success_criteria[0].evidence->predicate == "carrying");
    assert(restored.goal->constraints[0].evidence->target == "box_1");
    auto block = restored.withPrompt("Question?");
    assert(block.answer.empty() && block.id.empty() && block.raw.empty());
    const auto source = GRIM::Forward::inferenceInputSource("Question?", received.at("reasoning_state"));
    const auto input = GRIM::SpanText::renderInput(source, definitions);
    assert(input == GRIM::ConceptCanonical::renderReasoningPrompt(block, definitions));
    assert(input.find("<knowns>\ncapacity = 120 liters") != std::string::npos);
    assert(input.find("<unknowns>\nconsumed volume") != std::string::npos);
    assert(input.find("<target_state>\nFind consumed volume") != std::string::npos);
    assert(input.find("<criterion>\nOne assignment") != std::string::npos);
    assert(input.find("<evidence>") == std::string::npos);
    for (const auto* hidden : {"key_1", "apple_1", "box_1", "carrying"})
        assert(input.find(hidden) == std::string::npos);
    assert(input.find("<constraints>\n<constraint>\nUse liters") != std::string::npos);
    block.answer = "consumed = 36 liters";
    auto training = GRIM::ConceptCanonical::render(block, definitions);
    assert(input == training.text.substr(0, training.findNamedSpan("answer")->begin));
    assert(!GRIM::reasoningStateFromJson(json::object()).goal);
    assert(!GRIM::reasoningStateFromJson({{"goal", nullptr}}).goal);
    assert(GRIM::reasoningStateToJson(GRIM::ReasoningState{}).at("knowns").empty());
    const auto noEvidence = GRIM::goalFromJson({{"success_criteria", {{{"criterion", "Review"}}}},
                                               {"constraints", {{{"constraint", "Care"}}}}});
    assert(!noEvidence.success_criteria[0].evidence && !noEvidence.constraints[0].evidence);
    assert(!GRIM::goalToJson(noEvidence)["success_criteria"][0].contains("evidence"));
    for (const auto& evidence : std::vector<json>{"legacy prose", "", nullptr, 5, json::array(),
            {{"predicate", "carrying"}}, {{"predicate", " "}, {"subject", "key"}},
            {{"predicate", "carrying"}, {"subject", "key"}, {"target", 5}},
            {{"predicate", "carrying"}, {"subject", "key"}, {"answer", "leak"}}}) {
        for (const auto* collection : {"success_criteria", "constraints"}) {
            const auto* field = std::string(collection) == "constraints" ? "constraint" : "criterion";
            const json invalid{{"goal", {{collection, {{{field, "Requirement"}, {"evidence", evidence}}}}}}};
            rejects([&] { GRIM::reasoningStateFromJson(invalid); });
            rejects([&] { GRIM::Forward::inferenceInputSource("Question?", invalid); });
        }
    }
    for (const auto& invalid : std::vector<json>{
            nullptr, json::array(), {{"knowns", "not an array"}}, {{"unknowns", {5}}},
            {{"goal", "bad"}}, {{"goal", {{"constraints", "bad"}}}},
            {{"goal", {{"success_criteria", {"bad"}}}}}, {{"answer", "leak"}},
            {{"raw", "leak"}}, {{"prompt", "override"}}, {{"goal", {{"answer", "leak"}}}}})
    {
        rejects([&] { GRIM::reasoningStateFromJson(invalid); });
        rejects([&] { GRIM::Forward::inferenceInputSource("Question?", invalid); });
    }
    std::cout << "Reasoning-state wire round-trip, validation, and shared rendering tests passed\n";
}

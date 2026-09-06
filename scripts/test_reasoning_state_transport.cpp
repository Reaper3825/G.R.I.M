#include "../DataCollection/reasoning_state_json.hpp"
#include "../DataCollection/concept_block_canonical.hpp"
#include <cassert>
#include <iostream>
using nlohmann::json;

template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    assert(rejected);
}
int main() {
    GRIM::ReasoningState state;
    state.knowns = {"capacity = 120 liters", "remaining = 84 liters"};
    state.unknowns = {"consumed volume"};
    state.goal = GRIM::ConceptBlockGoal{"Find consumed volume", {{"One assignment", "Subtraction applies"}}, {"Use liters"}};
    // The actual client serializer and worker parser, with a full wire round trip.
    json request{{"messages", {{{"role", "user"}, {"content", "Question?"}}}},
                 {"reasoning_state", GRIM::reasoningStateToJson(state)}};
    const auto wire = request.dump();
    state.knowns.clear(); // A queued/transmitted snapshot must not alias its source.
    const auto received = json::parse(wire);
    const auto restored = GRIM::reasoningStateFromJson(received.at("reasoning_state"));
    assert(restored.knowns.size() == 2);
    assert(restored.unknowns == state.unknowns);
    assert(restored.goal->success_criteria[0].evidence == "Subtraction applies");
    auto block = restored.withPrompt("Question?");
    assert(block.answer.empty() && block.id.empty() && block.raw.empty());
    const auto input = GRIM::ConceptCanonical::renderReasoningPrompt(block);
    assert(input.find("<knowns>\ncapacity = 120 liters") != std::string::npos);
    assert(input.find("<unknowns>\nconsumed volume") != std::string::npos);
    assert(input.find("<target_state>\nFind consumed volume") != std::string::npos);
    assert(input.find("<criterion>\nOne assignment") != std::string::npos);
    assert(input.find("<evidence>\nSubtraction applies") != std::string::npos);
    assert(input.find("<constraints>\n<constraint>\nUse liters") != std::string::npos);
    block.answer = "consumed = 36 liters";
    auto training = GRIM::ConceptCanonical::render(block);
    assert(input == training.text.substr(0, training.answer.begin));
    assert(!GRIM::reasoningStateFromJson(json::object()).goal);
    assert(!GRIM::reasoningStateFromJson({{"goal", nullptr}}).goal);
    assert(GRIM::reasoningStateToJson(GRIM::ReasoningState{}).at("knowns").empty());
    for (const auto& invalid : std::vector<json>{
            nullptr, json::array(), {{"knowns", "not an array"}}, {{"unknowns", {5}}},
            {{"goal", "bad"}}, {{"goal", {{"constraints", "bad"}}}},
            {{"goal", {{"success_criteria", {"bad"}}}}}, {{"answer", "leak"}},
            {{"raw", "leak"}}, {{"prompt", "override"}}, {{"goal", {{"answer", "leak"}}}}})
        rejects([&] { GRIM::reasoningStateFromJson(invalid); });
    std::cout << "Reasoning-state wire round-trip, validation, and shared rendering tests passed\n";
}

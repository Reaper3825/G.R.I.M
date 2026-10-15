// Host-only checks of finalized config derivation and wire compatibility.
#include "../resources/models/GRIM-text/Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../control/training_control_generated.h"
#include <cassert>
#include <fstream>
#include <iostream>

using namespace GRIM::HyperParameters;

int main() try {
    std::ifstream source("ai_config.json");
    assert(source.is_open());
    auto document = nlohmann::json::parse(source);
    document["training"]["config"]["max_tokens_per_batch"] = 4096;
    document["training"]["config"]["batch_size"] = 99; // old leaf is not authoritative
    auto loaded = loadLanguageModelConfig(document);
    refreshMutableTrainingDerivedValues(loaded, 1024, "config load test");
    assert(loaded.batch_size == 4 && loaded.max_tokens_per_batch == 4096);

    LanguageModelConfig config;
    config.optimizer_kind = OptimizerKind::ADAMW;
    config.max_tokens_per_batch = 4500;
    config.batch_size = 99; // must not remain an independent capacity
    refreshMutableTrainingDerivedValues(config, 1024, "test");
    assert(config.max_tokens_per_batch == 4500 && config.batch_size == 4);
    config.max_tokens_per_batch = 10240;
    refreshMutableTrainingDerivedValues(config, 1024, "test");
    assert(config.batch_size == 10);
    config.max_tokens_per_batch = 1023;
    bool rejected = false;
    try { refreshMutableTrainingDerivedValues(config, 1024, "test"); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);

    GRIM::Config::AiConfigSnapshot snapshot;
    snapshot.document["training"]["config"] = {
        {"batch_size", 4}, {"max_seq_len", 1024},
        {"max_tokens_per_batch", 4500}, {"stability_overrides_enabled", false}};
    const auto capacity = trainingFixedShapeHP(snapshot);
    assert(capacity.batch_size == 4 && capacity.max_seq_len == 1024);
    assert(capacity.max_tokens_per_batch == 4500);
    const auto workspace = trainingStateWorkspaceHP(snapshot);
    assert(workspace.batch_size == 4 && workspace.max_tokens_per_batch == 4500);

    flatbuffers::FlatBufferBuilder builder;
    const auto wire = GRIMText::Control::CreateTrainingConfig(
        builder, 3, 4, 0.0001f, 1024, 0, true, true, 0, 0, 0, 4500);
    builder.Finish(wire);
    const auto* restored = flatbuffers::GetRoot<GRIMText::Control::TrainingConfig>(
        builder.GetBufferPointer());
    assert(restored->max_tokens_per_batch() == 4500);
    assert(restored->max_seq_len() == 1024 && restored->batch_size() == 4);
    builder.Clear();
    builder.Finish(GRIMText::Control::CreateTrainingConfig(builder));
    const auto* legacy = flatbuffers::GetRoot<GRIMText::Control::TrainingConfig>(
        builder.GetBufferPointer());
    assert(legacy->max_tokens_per_batch() == 0); // appended field is backward compatible
    std::cout << "Token-budget config and control wire tests passed\n";
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}

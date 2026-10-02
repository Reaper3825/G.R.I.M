#include "../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../training/Diagnostics/DiagnosticLens.hpp"
#include <cassert>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    assert(argc == 2);
    GRIM::Config::AiConfigSnapshot snapshot;
    std::ifstream file(argv[1]);
    assert(file);
    file >> snapshot.document;
    auto& config = snapshot.document.at("training").at("config");
    auto defaults = GRIM::HyperParameters::generationHP(snapshot);
    assert(defaults.lens_enabled == config.at("generation_lens_enabled").get<bool>());
    assert(defaults.lens_validate == config.at("generation_lens_validate").get<bool>());
    config["generation_lens_enabled"] = true;
    config["generation_lens_validate"] = true;
    config["generation_lens_top_k"] = 7;
    config["generation_lens_validation_rounds"] = 4;
    config["generation_lens_validation_tokens"] = 24;
    config["generation_lens_max_replay_rows"] = 512;
    config["generation_lens_absolute_tolerance"] = 0.002;
    const auto root = GRIM::HyperParameters::loadLanguageModelConfig(snapshot.document);
    const auto from_root = GRIM::HyperParameters::generationHP(root);
    const auto from_snapshot = GRIM::HyperParameters::generationHP(snapshot);
    for (const auto& hp : {from_root, from_snapshot}) {
        assert(hp.lens_enabled && hp.lens_validate);
        assert(hp.lens_top_k == 7 && hp.lens_validation_rounds == 4);
        assert(hp.lens_validation_tokens == 24 && hp.lens_max_replay_rows == 512);
        assert(hp.lens_absolute_tolerance == 0.002);
        const auto options = GRIMText::Training::validateDiagnosticLensOptions({
            hp.lens_enabled, hp.lens_validate, hp.lens_top_k,
            hp.lens_validation_rounds, hp.lens_validation_tokens,
            hp.lens_max_replay_rows, hp.lens_absolute_tolerance});
        assert(options.enabled && options.validate && options.top_k == 7);
    }
    std::cout << "lens generation config tests passed\n";
}

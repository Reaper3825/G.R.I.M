"""Check RL stage routing without compiling or running training/CUDA targets."""
from pathlib import Path

from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    root = Path(__file__).resolve().parent.parent
    repo = root.parents[2]
    enums = (root / 'Shared/HyperParameters/HyperparameterEnums.hpp').read_text()
    json_header = (repo / 'resources/models/llama.cpp/vendor/nlohmann/json.hpp').as_posix()
    enums = enums.replace('#include <nlohmann/json.hpp>', f'#include "{json_header}"')
    enums = enums.replace('#pragma once', '')
    checkpoint = (root / 'Common/CheckpointNaming.hpp').read_text()
    checkpoint = checkpoint.replace('#pragma once', '').replace(
        '#include "../Shared/HyperParameters/HyperparameterEnums.hpp"', '')
    startup = root / 'training/Phases/Startup'
    predecessor = method((startup / 'CheckpointLoad.cu').read_text(),
                         'std::optional<GRIM::HyperParameters::TrainingStage> predecessorTrainingStage(')
    windows = method((startup / 'SlidingWindow.cu').read_text(), 'void applySlidingWindows(')
    # Exercise the production dispatch before the first corpus mutation.
    dispatch = windows[windows.index('    switch (training_stage)'):windows.index('    // BOS must exist')]
    owner = (root / 'Shared/HyperParameters/HyperParameters_GPU.hpp').read_text()
    lora_guard = method(owner, 'if (params.execution_mode == ModelExecutionMode::TRAINING &&')
    source = '#include <cassert>\n#include <optional>\n' + enums + checkpoint + predecessor
    source += '\nnamespace GRIM::HyperParameters {\n'
    source += method(owner, 'enum class ModelExecutionMode') + ';\n}\n'
    source += '''
struct ConfigView {
    GRIM::HyperParameters::ModelExecutionMode execution_mode;
    GRIM::HyperParameters::TrainingStage training_stage;
    bool lora_model;
};
void validateLoRAConfig(const ConfigView& params) {
    using namespace GRIM::HyperParameters;
    const char* caller = "validateRootConfigDocument";
''' + lora_guard + '\n}\n'
    source += '''
void prepare(GRIM::HyperParameters::TrainingStage training_stage, int& mutations) {
    const std::string split_name = "test";
''' + dispatch + '''
    ++mutations;
}
int main() {
    using namespace GRIM::HyperParameters;
    using namespace GRIM::Checkpoint;
    static_assert(static_cast<unsigned>(TrainingStage::RLHF) == 4);
    for (auto value : {TrainingStage::PT, TrainingStage::SFT, TrainingStage::DPO,
                       TrainingStage::RLHF, TrainingStage::RL}) {
        for (auto mode : {ModelExecutionMode::TRAINING, ModelExecutionMode::INFERENCE}) {
            validateLoRAConfig({mode, value, false});
            bool rejected = false;
            try { validateLoRAConfig({mode, value, true}); }
            catch (const std::runtime_error& error) {
                rejected = true;
                assert(std::string(error.what()).find("freezes base policy parameters") != std::string::npos);
            }
            assert(rejected == (mode == ModelExecutionMode::TRAINING && value == TrainingStage::RL));
        }
    }
    assert(parseTrainingStage("rl") == TrainingStage::RL);
    assert(parseTrainingStage("RL") == TrainingStage::RL);
    assert(parseTrainingStage("reinforcement_learning") == TrainingStage::RL);
    nlohmann::json stage = TrainingStage::RL;
    assert(stage == "rl");
    assert(stage.get<TrainingStage>() == TrainingStage::RL);
    for (auto value : {TrainingStage::PT, TrainingStage::SFT, TrainingStage::DPO,
                       TrainingStage::RLHF, TrainingStage::RL}) {
        assert(parseTrainingStage(trainingStageToJsonString(value)) == value);
        auto filename = stageQualifiedCheckpointFilename(value, "_latest");
        assert(checkpointStageFromFilename(filename) == value);
    }
    assert(predecessorTrainingStage(TrainingStage::RL) == TrainingStage::SFT);
    int mutations = 0;
    prepare(TrainingStage::PT, mutations);
    prepare(TrainingStage::SFT, mutations);
    assert(mutations == 2);
    for (auto value : {TrainingStage::RL, TrainingStage::DPO, TrainingStage::RLHF,
                       TrainingStage::UNSPECIFIED}) {
        bool rejected = false;
        try { prepare(value, mutations); }
        catch (const std::runtime_error& error) {
            rejected = true;
            if (value == TrainingStage::RL)
                assert(std::string(error.what()).find("recorded episode batches") != std::string::npos);
        }
        assert(rejected);
        assert(mutations == 2);
    }
}
'''
    compile_and_run(source)
    print('RL stage, checkpoint identity, and preparation routing checks passed')


if __name__ == '__main__':
    main()

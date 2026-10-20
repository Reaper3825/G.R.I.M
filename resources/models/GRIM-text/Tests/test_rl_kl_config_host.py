"""Exercise RL config plumbing with production snippets; no trainer/CUDA build."""
import json
from pathlib import Path

from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    root = Path(__file__).resolve().parent.parent
    repo = root.parents[2]
    owner = (root / 'Shared/HyperParameters/HyperParameters_GPU.hpp').read_text()
    groups = (root / 'Shared/HyperParameters/HyperparameterGroupings.hpp').read_text()
    document = json.loads((repo / 'ai_config.json').read_text())
    assert document['training']['config']['rl_kl_coefficient'] == 0.02
    field = next(line for line in owner.splitlines() if 'float rl_kl_coefficient =' in line)
    load = next(line for line in owner.splitlines() if 'GRIM_LOAD_CONFIG_FIELD(rl_kl_coefficient)' in line)
    write = next(line for line in owner.splitlines() if 'GRIM_WRITE_FINAL_CONFIG_FIELD(rl_kl_coefficient)' in line)
    validation = next(line for line in owner.splitlines() if 'validationField("rl_kl_coefficient"' in line)
    helpers = owner[owner.index('template <typename Container, typename Value>\nstruct ValidationField'):]
    helpers = helpers[:helpers.index('template <typename Container, typename Value>\ninline void validateFiniteNonZeroFields')]
    first = method(groups, 'inline RLTrainingHP rlTrainingHP(')
    rest = groups[groups.index(first) + len(first):]
    second = method(rest, 'inline RLTrainingHP rlTrainingHP(')
    json_header = (repo / 'resources/models/llama.cpp/vendor/nlohmann/json.hpp').as_posix()
    source = f'#include "{json_header}"\n'
    source += '''
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <type_traits>
namespace GRIM::Config { struct AiConfigSnapshot { nlohmann::json document; }; }
namespace GRIM::HyperParameters {
struct LanguageModelConfig {
''' + field + '\n};\n' + helpers
    source += method(groups, 'struct RLTrainingHP') + ';\n'
    snapshot_signature = 'inline const nlohmann::json& snapshotTrainingConfig('
    source += method(owner[owner.rindex(snapshot_signature):], snapshot_signature)
    field_signature = 'inline T snapshotTrainingConfigField('
    source += '\ntemplate <typename T>\n' + method(owner[owner.rindex(field_signature):], field_signature)
    source += first + second
    source += '''
LanguageModelConfig load(const nlohmann::json& document) {
    const auto& config = document.at("training").at("config");
    LanguageModelConfig params{};
    auto load = [&](const char* name, auto member) {
        using Member = std::remove_cv_t<std::remove_reference_t<decltype(params.*member)>>;
        params.*member = config.at(name).get<Member>();
    };
#define GRIM_LOAD_CONFIG_FIELD(member) load(#member, &LanguageModelConfig::member)
''' + load + '''
#undef GRIM_LOAD_CONFIG_FIELD
    return params;
}
nlohmann::json exportConfig(const LanguageModelConfig& config) {
    nlohmann::json finalized_config = nlohmann::json::object();
#define GRIM_WRITE_FINAL_CONFIG_FIELD(member) finalized_config[#member] = config.member
''' + write + '''
#undef GRIM_WRITE_FINAL_CONFIG_FIELD
    return finalized_config;
}
void validate(const LanguageModelConfig& params) {
    validateNonNegativeFiniteFields(params, {
''' + validation + '''
    }, "test");
}
}
int main() {
    using namespace GRIM::HyperParameters;
    GRIM::Config::AiConfigSnapshot snapshot;
    for (float coefficient : {0.0f, 0.02f, 0.125f}) {
        snapshot.document["training"]["config"]["rl_kl_coefficient"] = coefficient;
        auto config = load(snapshot.document);
        validate(config);
        assert(rlTrainingHP(config).kl_coefficient == coefficient);
        assert(rlTrainingHP(snapshot).kl_coefficient == coefficient);
        assert(exportConfig(config).at("rl_kl_coefficient").get<float>() == coefficient);
    }
    for (float invalid : {-0.1f, std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        LanguageModelConfig config;
        config.rl_kl_coefficient = invalid;
        bool rejected = false;
        try { validate(config); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    snapshot.document["training"]["config"].erase("rl_kl_coefficient");
    bool missing_rejected = false;
    try { load(snapshot.document); }
    catch (const nlohmann::json::out_of_range&) { missing_rejected = true; }
    assert(missing_rejected);
}
'''
    compile_and_run(source)
    print('RL KL config load/export, grouped access, and validation checks passed')


if __name__ == '__main__':
    main()

"""Check root Environment config access without building CUDA/runtime targets."""
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
    assert document['Environment'] == 'GRIM'
    finalization = method(owner, 'inline GRIM::Config::AiConfigSnapshot finalizeAiConfigSnapshot(')
    assert finalization.index('snapshotEnvironmentName(snapshot)') < finalization.index('writeFinalizedTrainingConfigDocumentToSnapshot(')
    json_header = (repo / 'resources/models/llama.cpp/vendor/nlohmann/json.hpp').as_posix()
    source = f'#include "{json_header}"\n'
    source += '''
#include <cassert>
#include <stdexcept>
#include <string>
namespace GRIM::Config { struct AiConfigSnapshot { nlohmann::json document; }; }
namespace GRIM::HyperParameters {
'''
    source += method(owner, 'inline std::string snapshotEnvironmentName(')
    source += method(groups, 'struct EnvironmentHP') + ';\n'
    source += method(groups, 'inline EnvironmentHP environmentHP(')
    source += '''
}
int main() {
    using GRIM::HyperParameters::environmentHP;
    GRIM::Config::AiConfigSnapshot snapshot;
    snapshot.document = {{"Environment", "GRIM"},
                         {"training", {{"config", {{"Environment", "wrong scope"}}}}}};
    assert(environmentHP(snapshot).name == "GRIM");
    snapshot.document["Environment"] = "Another World";
    assert(environmentHP(snapshot).name == "Another World");
    for (const auto& invalid : {nlohmann::json(nullptr), nlohmann::json(12),
                               nlohmann::json(false), nlohmann::json::object(),
                               nlohmann::json::array(), nlohmann::json(""),
                               nlohmann::json(" \\t\\r\\n")}) {
        snapshot.document["Environment"] = invalid;
        bool rejected = false;
        try { (void)environmentHP(snapshot); }
        catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    snapshot.document.erase("Environment");
    bool rejected = false;
    try { (void)environmentHP(snapshot); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
'''
    compile_and_run(source)
    print('Environment grouped access and validation checks passed')


if __name__ == '__main__':
    main()

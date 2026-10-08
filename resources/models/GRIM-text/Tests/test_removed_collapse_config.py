"""Host-only checks for retired config fields and checkpoint compatibility.

Uses the standalone config compiler and extracted production checkpoint fact
validation/checksum functions. Does not compile CUDA or a training target.
"""
from pathlib import Path
import json
import os
import subprocess
import tempfile

from test_add_gradfn_host import compile_and_run


def main():
    tests = Path(__file__).resolve().parent
    repo = tests.parents[3]
    build = Path(os.environ.get('GRIM_CONFIG_COMPILER_BUILD', str(repo / 'build/config-compiler')))
    compiler = build / 'Release/compile_model_config.exe'
    deps = tests.parent / 'training/vcpkg_installed/x64-windows'
    out = Path(tempfile.mkdtemp(prefix='grim-removed-collapse-test-'))
    config = json.loads((repo / 'model_config.json').read_text(encoding='utf-8'))
    retired = ('center_encoder_residuals', 'lm_head_center_hidden_states',
               'center_logits', 'project_out_pc1', 'pc1_power_iters', 'lm_head_centering_enabled')
    for key in retired:
        assert key not in config
        source = out / 'invalid.json'
        source.write_text(json.dumps(dict(config, **{key: 5 if key == 'pc1_power_iters' else True})))
        result = subprocess.run([str(compiler), '--input', str(source), '--output', str(out/'invalid.grimcfg')],
                                capture_output=True, text=True)
        assert result.returncode != 0 and f'removed model config field: {key}' in result.stderr, result.stderr

    subprocess.run([str(deps/'tools/flatbuffers/flatc.exe'), '-c', '-o', str(out),
                    str(tests.parent/'training/schemas/grim_parameter_checkpoint.fbs')], check=True)
    source = (tests.parent/'Common/ParameterCheckpoint.cu').read_text(encoding='utf-8')

    def between(start, end):
        return source[source.index(start):source.index(end, source.index(start))]

    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
'''
    cpp += f'#include "{(out / "grim_parameter_checkpoint_generated.h").as_posix()}"\n'
    cpp += between('struct CompatibilityFactValue {', 'struct HostParameterEntry {')
    cpp += between('std::uint64_t rotateLeft(', 'std::string canonicalFloat(')
    cpp += between('std::uint64_t compatibilityChecksum(', 'GRIMCheckpoint::ParameterGroupType checkpointGroupType(')
    cpp += between('bool compatibilityFactsMatch(', 'bool shapeMatches(')
    cpp += r'''
bool check(std::vector<CompatibilityFactValue> facts,
           const std::vector<CompatibilityFactValue>& expected,
           bool corrupt = false, bool sort = true) {
    if (sort) std::sort(facts.begin(), facts.end(), [](const auto& a, const auto& b) { return a.name < b.name; });
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<GRIMCheckpoint::CompatibilityFact>> offsets;
    for (const auto& fact : facts) offsets.push_back(GRIMCheckpoint::CreateCompatibilityFact(
        builder, builder.CreateString(fact.name), builder.CreateString(fact.value)));
    const auto vector = builder.CreateVector(offsets);
    builder.Finish(vector);
    const auto* stored = flatbuffers::GetRoot<flatbuffers::Vector<flatbuffers::Offset<GRIMCheckpoint::CompatibilityFact>>>(builder.GetBufferPointer());
    std::string mismatch;
    return compatibilityFactsMatch(stored, compatibilityChecksum(facts) ^ (corrupt ? 1 : 0), expected, mismatch);
}
int main() {
    const std::vector<CompatibilityFactValue> current = {{"d_model", "768"}, {"tie_embeddings", "true"}};
    assert(check(current, current));
    auto legacy = current;
    const char* retired[] = {"center_encoder_residuals", "center_logits", "lm_head_center_hidden_states", "project_out_pc1"};
    for (const auto* name : retired) legacy.push_back({name, "false"});
    assert(check(legacy, current));
    assert(!check(legacy, current, true));
    for (const auto* name : retired) {
        auto enabled = current;
        enabled.push_back({name, "true"});
        assert(!check(enabled, current));
    }
    auto mismatch = legacy;
    mismatch[0].value = "1024";
    assert(!check(mismatch, current));
    auto duplicate = legacy;
    duplicate.push_back({"center_logits", "false"});
    assert(!check(duplicate, current));
    auto unknown = legacy;
    unknown.push_back({"unknown_feature", "false"});
    assert(!check(unknown, current));
    assert(!check({current[0]}, current));
    assert(!check({current[1], current[0]}, current, false, false));
    std::cout << "Removed config fields and legacy checkpoint facts passed\n";
}
'''
    previous = os.environ.get('CL')
    os.environ['CL'] = (previous or '') + f' /I"{deps / "include"}"'
    try:
        compile_and_run(cpp)
    finally:
        if previous is None:
            os.environ.pop('CL', None)
        else:
            os.environ['CL'] = previous


if __name__ == '__main__':
    main()

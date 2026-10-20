"""Exercise production token commits and result retention without CUDA/training."""
from pathlib import Path

from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    root = Path(__file__).resolve().parent.parent
    inference = root / 'training/Phases/Phase2_InferenceLoop.cu'
    source = inference.read_text(encoding='utf-8')
    header = inference.with_suffix('.hpp').read_text(encoding='utf-8')
    result_type = method(header, 'struct Phase2TextInferenceResult') + ';'
    commit = method(source, 'auto commitToken =') + ';'
    forced = method(source, 'auto commitForcedToken =') + ';'
    initialize = next(line.strip() for line in source.splitlines()
                      if 'sequence.token_origins.assign(' in line)
    handoff = source[source.index('    result.token_ids = sequence.token_ids;'):]
    handoff = handoff[:handoff.index('    return result;') + len('    return result;')]
    sequence_header = (root / 'Shared/Forward/GeneratedSequence.hpp').as_posix()
    episode_header = (root / 'Shared/SelfCorrection/Episode.hpp').as_posix()
    layout_header = (root / 'Shared/UnigramByte/TokenLayout.hpp').as_posix()
    test = r'''
#include "SEQUENCE_HEADER"
#include "EPISODE_HEADER"
#include "LAYOUT_HEADER"
#include <cassert>
#include <chrono>
#include <functional>
#include <optional>
#include <string>
#include <utility>
namespace GRIM::Sampling {
struct SampleResult { int token_id; float probability; float log_probability; };
}
RESULT_TYPE
Phase2TextInferenceResult handoff(std::vector<GRIM::GeneratedSequence> generated) {
    Phase2TextInferenceResult result;
    const auto& sequence = generated.front();
    const auto start_decode = std::chrono::high_resolution_clock::now();
HANDOFF
}
int main() {
    const std::vector<int> prompt_tokens{10, 11};
    GRIM::GeneratedSequence sequence;
    sequence.token_ids = prompt_tokens;
INITIALIZE
    sequence.token_numeric_values = {0, 0};
    sequence.token_atom_mask = {0, 0};
    sequence.atom_entry_ids = {UINT32_MAX, UINT32_MAX};
    sequence.token_local_atom_indices = {UINT32_MAX, UINT32_MAX};
    std::vector<uint32_t> sequence_atom_flags{0, 0};
    std::function<void(int, float)>* stream_callback = nullptr;
COMMIT
FORCED
    commitToken({12, 1.0f, 0.0f}); // A real sampled zero must stay an action.
    commitForcedToken(13);
    // Simulate an intervening runtime observation: it has no policy score.
    sequence.token_ids.push_back(14);
    sequence.token_origins.push_back(GRIM::TokenOrigin::RuntimeContext);
    sequence.token_numeric_values.push_back(0);
    sequence.token_atom_mask.push_back(0);
    sequence.atom_entry_ids.push_back(UINT32_MAX);
    sequence.token_local_atom_indices.push_back(UINT32_MAX);
    commitToken({3, 0.5f, -0.693147f}); // Sampled EOS is retained and scored.
    assert((sequence.score_token_positions == std::vector<std::size_t>{2, 3, 5}));
    assert(sequence.token_scores.size() == 3);
    assert(sequence.token_origins.size() == sequence.token_ids.size());
    assert(sequence.token_origins[0] == GRIM::TokenOrigin::SuppliedContext);
    assert(sequence.token_origins[2] == GRIM::TokenOrigin::SampledAction);
    assert(sequence.token_origins[3] == GRIM::TokenOrigin::ForcedToken);
    assert(sequence.token_origins[4] == GRIM::TokenOrigin::RuntimeContext);
    assert(sequence.token_origins[5] == GRIM::TokenOrigin::SampledAction);
    const auto ids = sequence.token_ids;
    const auto scores = sequence.token_scores;
    const auto positions = sequence.score_token_positions;
    const auto origins = sequence.token_origins;
    const auto numeric = sequence.token_numeric_values;
    const auto atoms = sequence.atom_entry_ids;
    const auto local = sequence.token_local_atom_indices;
    auto result = handoff({std::move(sequence)});
    assert(result.generated_sequence.has_value());
    const auto& retained = *result.generated_sequence;
    assert(result.token_ids == ids && retained.token_ids == ids);
    assert(retained.token_scores == scores);
    assert(retained.score_token_positions == positions);
    assert(retained.token_origins == origins);
    assert(retained.token_numeric_values == numeric);
    assert(retained.atom_entry_ids == atoms);
    assert(retained.token_local_atom_indices == local);
    assert(result.sequence_token_count == ids.size());
    assert(!Phase2TextInferenceResult{}.generated_sequence.has_value());
    // A prefill-only generated sequence has origins but no fabricated scores.
    GRIM::GeneratedSequence prefill;
    prefill.token_ids = prompt_tokens;
    prefill.token_origins.assign(prompt_tokens.size(), GRIM::TokenOrigin::SuppliedContext);
    auto inspected = handoff({std::move(prefill)});
    assert(inspected.generated_sequence->token_scores.empty());
    assert(inspected.generated_sequence->score_token_positions.empty());
}
'''
    for key, value in {
        'SEQUENCE_HEADER': sequence_header, 'EPISODE_HEADER': episode_header,
        'LAYOUT_HEADER': layout_header,
        'RESULT_TYPE': result_type, 'HANDOFF': handoff, 'INITIALIZE': initialize,
        'COMMIT': commit, 'FORCED': forced,
    }.items():
        test = test.replace(key, value)
    compile_and_run(test)
    print('Inference trajectory host tests passed.')


if __name__ == '__main__':
    main()

"""Exercise session initialization, sampling, completion and retention on the host.

Uses production session and sampling code with host registry/payload substitutes;
does not compile or execute CUDA, model forwards, or training targets.
"""
from pathlib import Path
import os

from test_add_gradfn_host import compile_and_run
from test_unary_gradfn_host import method


def main():
    root = Path(__file__).resolve().parent.parent
    inference = root / 'training/Phases/Phase2_InferenceLoop.cu'
    source = inference.read_text(encoding='utf-8')
    header = inference.with_suffix('.hpp').read_text(encoding='utf-8')
    result_type = method(header, 'struct Phase2TextInferenceResult') + ';'
    session = root / 'Shared/InferenceState/InferenceGenerationSession.cu'
    groups = (root / 'Shared/HyperParameters/HyperparameterGroupings.hpp').read_text(encoding='utf-8')
    def without_includes(path):
        return '\n'.join(line for line in path.read_text(encoding='utf-8').splitlines()
                         if not line.startswith('#include') and line != '#pragma once')
    session_header = without_includes(session.with_suffix('.hpp'))
    session_body = without_includes(session)
    sampling_body = without_includes(root / 'Shared/Sampling/Sampling.cu')
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
#include "SAMPLING_HEADER"
namespace GRIM::HyperParameters {
using GenerationStreamCallback = std::function<void(int, float)>;
GENERATION_HP
MODEL_HP
}
// Only CUDA-backed registry storage and the payload envelope are substituted.
// Session initialization, sampling, commits, and completion use production code.
namespace GRIM::Tokenizer {
class AtomTable {
public:
    int marker = 7;
    std::shared_ptr<AtomTable> cloneHostForGeneration() const {
        return std::make_shared<AtomTable>(*this);
    }
};
class SequenceLocalAtomTable { public: int marker = 9; };
}
namespace GRIM::Batching {
struct BatchPayload {
    int vocab_size = 32;
    std::vector<int> input_ids{10, 11};
    std::vector<float> numeric_values{0, 0};
    std::vector<uint8_t> atom_mask{0, 0};
    std::vector<uint32_t> atom_entry_ids{UINT32_MAX, UINT32_MAX};
    std::vector<uint32_t> token_local_atom_indices{UINT32_MAX, UINT32_MAX};
    std::vector<std::shared_ptr<const GRIM::Tokenizer::AtomTable>> seq_atom_tables{
        std::make_shared<GRIM::Tokenizer::AtomTable>()};
    std::vector<std::shared_ptr<const GRIM::Tokenizer::SequenceLocalAtomTable>> seq_local_atom_tables{
        std::make_shared<GRIM::Tokenizer::SequenceLocalAtomTable>()};
};
}
SESSION_HEADER
SAMPLING_BODY
SESSION_BODY
RESULT_TYPE
Phase2TextInferenceResult handoff(GRIM::GeneratedSequence sequence) {
    Phase2TextInferenceResult result;
    const auto start_decode = std::chrono::high_resolution_clock::now();
HANDOFF
}
int main() {
    const std::vector<int> prompt_tokens{10, 11};
    GRIM::Batching::BatchPayload payload;
    GRIM::HyperParameters::ModelHP model;
    model.training_max_seq_len = 32;
    GRIM::HyperParameters::GenerationHP hp;
    hp.strategy = GRIM::HyperParameters::SamplingStrategy::GREEDY;
    hp.do_sample = false;
    hp.temperature = 1;
    hp.top_p = hp.typical_p = hp.repetition_penalty = 1;
    hp.max_new_tokens = 8;
    hp.eos_token_id = 3; hp.bos_token_id = 2;
    hp.pad_token_id = 1; hp.unk_token_id = 0;
    hp.seed = 1;
    int callbacks = 0;
    GRIM::HyperParameters::GenerationStreamCallback callback = [&](int, float) { ++callbacks; };
    const auto authored_atoms = payload.seq_atom_tables[0];
    const auto authored_local = payload.seq_local_atom_tables[0];
    GRIM::Inference::InferenceGenerationSession session(payload, model, hp, &callback);
    auto& sequence = session.sequence();
    assert(sequence.token_ids == prompt_tokens);
    assert(sequence.context_atom_table != authored_atoms);
    assert(sequence.context_local_atom_table != authored_local);
    assert(session.committedNewTokens() == 0 && !session.finished());
    session.commitToken({12, 1.0f, 0.0f}); // A real sampled zero must stay an action.
    session.commitForcedToken(13);
    // Simulate an intervening runtime observation: it has no policy score.
    sequence.token_ids.push_back(14);
    sequence.token_origins.push_back(GRIM::TokenOrigin::RuntimeContext);
    sequence.token_numeric_values.push_back(0);
    sequence.token_atom_mask.push_back(0);
    sequence.atom_entry_ids.push_back(UINT32_MAX);
    sequence.token_local_atom_indices.push_back(UINT32_MAX);
    session.commitToken({3, 0.5f, -0.693147f}); // Sampled EOS is retained and scored.
    assert((sequence.score_token_positions == std::vector<std::size_t>{2, 3, 5}));
    assert(sequence.token_scores.size() == 3);
    assert(sequence.token_origins.size() == sequence.token_ids.size());
    assert(sequence.token_origins[0] == GRIM::TokenOrigin::SuppliedContext);
    assert(sequence.token_origins[2] == GRIM::TokenOrigin::SampledAction);
    assert(sequence.token_origins[3] == GRIM::TokenOrigin::ForcedToken);
    assert(sequence.token_origins[4] == GRIM::TokenOrigin::RuntimeContext);
    assert(sequence.token_origins[5] == GRIM::TokenOrigin::SampledAction);
    assert(session.finished() && callbacks == 3);
    const auto ids = sequence.token_ids;
    const auto scores = sequence.token_scores;
    const auto positions = sequence.score_token_positions;
    const auto origins = sequence.token_origins;
    const auto numeric = sequence.token_numeric_values;
    const auto atoms = sequence.atom_entry_ids;
    const auto local = sequence.token_local_atom_indices;
    auto result = handoff(session.takeSequence());
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
    hp.max_new_tokens = 1;
    GRIM::Inference::InferenceGenerationSession capped(payload, model, hp);
    capped.commitForcedToken(12);
    assert(capped.finished() && capped.remainingTokenCapacity() == 0);
    hp.max_new_tokens = 8;
    model.training_max_seq_len = 3;
    GRIM::Inference::InferenceGenerationSession context_capped(payload, model, hp);
    context_capped.commitForcedToken(12);
    assert(context_capped.finished());
    model.training_max_seq_len = 2;
    bool rejected = false;
    try { GRIM::Inference::InferenceGenerationSession invalid(payload, model, hp); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    GRIM::Inference::InferenceGenerationSession inspection(payload, model, hp, nullptr, true);
    auto inspected_session = inspection.takeSequence();
    assert(inspected_session.finished && inspected_session.token_scores.empty());
    model.training_max_seq_len = 32;
    hp.min_new_tokens = 2;
    GRIM::Inference::InferenceGenerationSession minimum(payload, model, hp);
    std::vector<float> logits(32, -10);
    logits[3] = 10; logits[12] = 5;
    auto first = minimum.selectNextToken(logits.data());
    assert(first.token_id == 12);
    minimum.commitToken(first);
    auto second = minimum.selectNextToken(logits.data());
    assert(second.token_id == 3);
    minimum.commitToken(second);
    assert(minimum.finished());
    // A prefill-only generated sequence has origins but no fabricated scores.
    GRIM::GeneratedSequence prefill;
    prefill.token_ids = prompt_tokens;
    prefill.token_origins.assign(prompt_tokens.size(), GRIM::TokenOrigin::SuppliedContext);
    auto inspected = handoff(std::move(prefill));
    assert(inspected.generated_sequence->token_scores.empty());
    assert(inspected.generated_sequence->score_token_positions.empty());
}
'''
    for key, value in {
        'SEQUENCE_HEADER': sequence_header, 'EPISODE_HEADER': episode_header,
        'LAYOUT_HEADER': layout_header,
        'RESULT_TYPE': result_type, 'HANDOFF': handoff,
        'SAMPLING_HEADER': (root / 'Shared/Sampling/Sampling.hpp').as_posix(),
        'GENERATION_HP': method(groups, 'struct GenerationHP') + ';',
        'MODEL_HP': method(groups, 'struct ModelHP') + ';',
        'SESSION_HEADER': session_header, 'SESSION_BODY': session_body,
        'SAMPLING_BODY': sampling_body,
    }.items():
        test = test.replace(key, value)
    previous = os.environ.get('CL')
    if os.name == 'nt':
        json_include = root.parent / 'llama.cpp/vendor'
        os.environ['CL'] = (previous or '') + f' /I"{json_include}"'
    try:
        compile_and_run(test)
    finally:
        if previous is None:
            os.environ.pop('CL', None)
        else:
            os.environ['CL'] = previous
    print('Inference trajectory host tests passed.')


if __name__ == '__main__':
    main()

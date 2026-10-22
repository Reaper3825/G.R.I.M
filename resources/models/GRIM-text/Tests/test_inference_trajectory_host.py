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
    advance_begin = source.index('    while (!shouldReturn(session, model_hp))')
    advance_loop = source[advance_begin:source.index('    const auto reason =', advance_begin)]
    generation_header = without_includes(root / 'Shared/InferenceState/GenerationState_GPU.hpp')
    handoff = source[source.index('    result.token_ids = sequence.token_ids;'):]
    handoff = handoff[:handoff.index('    return result;') + len('    return result;')]
    sequence_header = (root / 'Shared/Forward/GeneratedSequence.hpp').as_posix()
    episode_header = (root / 'Shared/SelfCorrection/TaskEpisode.hpp').as_posix()
    layout_header = (root / 'Shared/UnigramByte/TokenLayout.hpp').as_posix()
    test = r'''
#include "SEQUENCE_HEADER"
#include "EPISODE_HEADER"
#include "LAYOUT_HEADER"
#include "RETURN_POLICY_HEADER"
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
    std::shared_ptr<const GRIM::ConceptSpanDelimiterMap> concept_span_delimiters;
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
namespace GRIM {
struct KvCacheState { int host_seqlen = 0; void resetSession() { host_seqlen = 0; } };
struct LocalAtomRetrievalInferenceState { void resetSession() {} };
namespace Batching { struct BatchDeviceStorage {}; }
}
#define USE_CUDA
GENERATION_STATE_HEADER
#undef USE_CUDA
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
    model.training_max_seq_len = model.max_cached_seq_len = 32;
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
    auto session = GRIM::Inference::createInferenceGenerationSession(payload, model, hp, &callback);
    auto& sequence = session.sequence;
    assert(sequence.token_ids == prompt_tokens);
    assert(sequence.context_atom_table != authored_atoms);
    assert(sequence.context_local_atom_table != authored_local);
    assert(sampledTokenCount(session) == 0 && !finished(session, model));
    commitToken(session, {12, 1.0f, 0.0f}); // A real sampled zero must stay an action.
    commitForcedToken(session, 13);
    // Simulate an intervening runtime observation: it has no policy score.
    sequence.token_ids.push_back(14);
    sequence.token_origins.push_back(GRIM::TokenOrigin::RuntimeContext);
    sequence.token_numeric_values.push_back(0);
    sequence.token_atom_mask.push_back(0);
    sequence.atom_entry_ids.push_back(UINT32_MAX);
    sequence.token_local_atom_indices.push_back(UINT32_MAX);
    commitToken(session, {3, 0.5f, -0.693147f}); // Sampled EOS is retained and scored.
    assert((sequence.score_token_positions == std::vector<std::size_t>{2, 3, 5}));
    assert(sequence.token_scores.size() == 3);
    assert(sequence.token_origins.size() == sequence.token_ids.size());
    assert(sequence.token_origins[0] == GRIM::TokenOrigin::SuppliedContext);
    assert(sequence.token_origins[2] == GRIM::TokenOrigin::SampledAction);
    assert(sequence.token_origins[3] == GRIM::TokenOrigin::ForcedToken);
    assert(sequence.token_origins[4] == GRIM::TokenOrigin::RuntimeContext);
    assert(sequence.token_origins[5] == GRIM::TokenOrigin::SampledAction);
    assert(finished(session, model) && callbacks == 3);
    assert(sampledTokenCount(session) == 2);
    assert(returnReason(session, model) == GRIM::Inference::InferenceReturnReason::EOS);
    const auto ids = sequence.token_ids;
    const auto scores = sequence.token_scores;
    const auto positions = sequence.score_token_positions;
    const auto origins = sequence.token_origins;
    const auto numeric = sequence.token_numeric_values;
    const auto atoms = sequence.atom_entry_ids;
    const auto local = sequence.token_local_atom_indices;
    auto result = handoff(takeSequence(session, returnReason(session, model)));
    assert(result.generated_sequence.has_value());
    const auto& retained = *result.generated_sequence;
    assert(result.token_ids == ids && retained.token_ids == ids);
    assert(retained.token_scores == scores);
    assert(retained.score_token_positions == positions);
    assert(retained.token_origins == origins);
    assert(retained.token_numeric_values == numeric);
    assert(retained.atom_entry_ids == atoms);
    assert(retained.token_local_atom_indices == local);
    assert(retained.return_reason == GRIM::Inference::InferenceReturnReason::EOS && retained.finished);
    assert(result.sequence_token_count == ids.size());
    assert(!Phase2TextInferenceResult{}.generated_sequence.has_value());
    // Future output definitions are retained even though absent from the prompt.
    GRIM::NamedConceptSpanDefinition answer;
    answer.name = "answer"; answer.source_path = "/answer";
    answer.supervision = GRIM::ConceptSpanSupervision::Supervised;
    answer.open_delimiter = "<answer>"; answer.close_delimiter = "</answer>";
    answer.open_delimiter_id = 20; answer.close_delimiter_id = 21;
    payload.concept_span_delimiters = std::make_shared<const GRIM::ConceptSpanDelimiterMap>(
        GRIM::NamedConceptSpanDefinitions{answer});
    auto spans = GRIM::Inference::createInferenceGenerationSession(payload, model, hp);
    commitToken(spans, {20, 1.0f, 0.0f});
    assert(!shouldReturn(spans, model)); // Opening markers do not yield.
    commitToken(spans, {12, 1.0f, 0.0f});
    assert(!shouldReturn(spans, model));
    commitToken(spans, {21, 1.0f, 0.0f});
    assert(!finished(spans, model)); // A span boundary does not replace EOS.
    assert(shouldReturn(spans, model));
    assert(returnReason(spans, model) == GRIM::Inference::InferenceReturnReason::SpanBoundary);
    assert(spans.sequence.span_events.size() == 2);
    const auto& close = spans.sequence.span_events.back();
    assert(close.kind == GRIM::GeneratedSpanEventKind::Close);
    assert(close.begin == 2 && close.token_position == 4);
    auto span_result = handoff(takeSequence(spans, returnReason(spans, model)));
    assert(!span_result.generated_sequence->finished);
    assert(span_result.generated_sequence->return_reason == GRIM::Inference::InferenceReturnReason::SpanBoundary);
    assert(span_result.generated_sequence->concept_span_delimiters == payload.concept_span_delimiters);
    assert(span_result.generated_sequence->span_events.size() == 2);
    // Minimum answer length applies to EOS, not span returns; budget exhaustion
    // at the same close is terminal rather than a resumable span return.
    auto span_limits_hp = hp;
    span_limits_hp.min_new_tokens = 4;
    auto early_span = GRIM::Inference::createInferenceGenerationSession(payload, model, span_limits_hp);
    commitToken(early_span, {20, 1.0f, 0.0f});
    commitToken(early_span, {21, 1.0f, 0.0f});
    assert(returnReason(early_span, model) == GRIM::Inference::InferenceReturnReason::SpanBoundary);
    span_limits_hp.min_new_tokens = 0; span_limits_hp.max_new_tokens = 2;
    auto exhausted_span = GRIM::Inference::createInferenceGenerationSession(payload, model, span_limits_hp);
    commitToken(exhausted_span, {20, 1.0f, 0.0f});
    commitToken(exhausted_span, {21, 1.0f, 0.0f});
    assert(returnReason(exhausted_span, model) == GRIM::Inference::InferenceReturnReason::TokenLimit);
    assert(finished(exhausted_span, model));
    assert(!payload.concept_span_delimiters->find(12));
    assert(payload.concept_span_delimiters->definitions()[0].close_delimiter_id == 21);
    // Resume the same live session twice. Closing markers remain pending until
    // resume; queued runtime observations must be processed before another sample.
    GRIM::GenerationState live;
    auto live_hp = hp; live_hp.max_new_tokens = 16;
    live.prefill = payload;
    live.active_session.emplace(GRIM::Inference::createInferenceGenerationSession(*live.prefill, model, live_hp));
    live.kv_cache.host_seqlen = static_cast<int>(payload.input_ids.size());
    live.inference_device_storage = std::make_shared<GRIM::Batching::BatchDeviceStorage>();
    const auto storage = live.inference_device_storage;
    auto& running = *live.active_session;
    bool invalid_resume = false;
    try { acknowledgeSpanReturn(running); }
    catch (const std::logic_error&) { invalid_resume = true; }
    assert(invalid_resume);
    commitToken(running, {20, 1.0f, 0.0f});
    std::vector<std::size_t> processed;
    std::vector<int> predictions{12, 21, 20, 13, 21, 3};
    std::size_t prediction = 0;
    auto advance = [&] {
        auto& session = running;
        auto& kv_cache = live.kv_cache;
        auto& sequence = running.sequence;
        const auto& model_hp = model;
        const auto& prompt_payload = *live.prefill;
        const bool local_atom_retrieval_enabled = false;
        struct Retrieval {
            bool hasForcedToken() const { return false; }
            int popForcedToken() { assert(false); return 0; }
        } retrieval_state;
        struct Readback { bool last_logits, encoder_row, retrieval_logits; };
        struct Tail { std::vector<float> last_logits; };
        auto buildDecodePayload = [](const std::vector<int>& tokens) { return tokens; };
        auto runCachedForward = [&](const std::vector<int>& tokens, Readback) {
            assert(tokens.size() == 1);
            assert(tokens[0] == sequence.token_ids.at(kv_cache.host_seqlen));
            ++kv_cache.host_seqlen;
            Tail tail{std::vector<float>(32, -1000.0f)};
            tail.last_logits[predictions.at(prediction)] = 1000.0f;
            return tail;
        };
        auto processPendingLocalAtom = [&](const Tail&, std::size_t position) {
            processed.push_back(position);
            if (static_cast<std::size_t>(kv_cache.host_seqlen) == sequence.token_ids.size())
                ++prediction;
        };
        // Compile the production loop directly; only GPU forwarding is substituted.
ADVANCE_LOOP
    };
    advance();
    assert(returnReason(running, model) == GRIM::Inference::InferenceReturnReason::SpanBoundary);
    assert(live.kv_cache.host_seqlen == 4 && running.sequence.token_ids.size() == 5);
    running.sequence.return_reason = returnReason(running, model);
    const auto first_snapshot = running.sequence;
    advance(); // No acknowledgement: the same boundary returns without any work.
    assert(processed.size() == 2);
    acknowledgeSpanReturn(running);
    advance();
    assert(returnReason(running, model) == GRIM::Inference::InferenceReturnReason::SpanBoundary);
    assert(live.kv_cache.host_seqlen == 7 && running.sequence.token_ids.size() == 8);
    assert(first_snapshot.token_ids.size() == 5);
    running.sequence.return_reason = returnReason(running, model);
    const int sampled_before_observation = sampledTokenCount(running);
    const int capacity_before_observation = remainingContextTokenCapacity(running, model);
    // Simulate the validated runtime append adapter, which is a separate API.
    for (int id : {18, 19}) {
        running.sequence.token_ids.push_back(id);
        running.sequence.token_origins.push_back(GRIM::TokenOrigin::RuntimeContext);
        running.sequence.token_numeric_values.push_back(0);
        running.sequence.token_atom_mask.push_back(0);
        running.sequence.atom_entry_ids.push_back(UINT32_MAX);
        running.sequence.token_local_atom_indices.push_back(UINT32_MAX);
    }
    assert(sampledTokenCount(running) == sampled_before_observation);
    assert(remainingContextTokenCapacity(running, model) == capacity_before_observation - 2);
    acknowledgeSpanReturn(running);
    advance();
    assert((processed == std::vector<std::size_t>{2,3,4,5,6,7,8,9}));
    assert(prediction == predictions.size());
    assert(live.kv_cache.host_seqlen == 10 && running.sequence.token_ids.size() == 11);
    assert(returnReason(running, model) == GRIM::Inference::InferenceReturnReason::EOS);
    assert(sampledTokenCount(running) == 7 && running.sequence.token_scores.size() == 7);
    assert(live.inference_device_storage == storage);
    auto complete = takeSequence(running, returnReason(running, model));
    live.resetSession();
    assert(complete.finished && !live.active_session && !live.prefill);
    assert(!live.inference_device_storage && live.kv_cache.host_seqlen == 0);
    // Repeated names and shared token IDs under distinct parents resolve by scope.
    auto parent = answer;
    parent.name = "outer"; parent.open_delimiter_id = 22; parent.close_delimiter_id = 23;
    parent.children = {answer};
    auto other = parent;
    other.name = "other"; other.open_delimiter_id = 24; other.close_delimiter_id = 25;
    auto map = std::make_shared<const GRIM::ConceptSpanDelimiterMap>(
        GRIM::NamedConceptSpanDefinitions{parent, other});
    assert(map->find(20)->size() == 2);
    GRIM::ConceptSpanTracker tracker(map);
    std::vector<GRIM::GeneratedSpanEvent> events;
    tracker.observe(24, 0, events);
    tracker.observe(20, 1, events);
    tracker.observe(25, 2, events); // Wrong close cannot commit either active span.
    assert(events.back().kind == GRIM::GeneratedSpanEventKind::InvalidMarker);
    tracker.observe(21, 3, events);
    assert(events.back().definition == 3 && events.back().begin == 1);
    tracker.observe(25, 4, events);
    assert(events.back().definition == 2 && events.back().begin == 0);
    tracker.observe(21, 5, events); // Unmatched close.
    assert(events.back().kind == GRIM::GeneratedSpanEventKind::InvalidMarker);
    tracker.observe(24, 6, events); // Repeated occurrence of the same definition.
    tracker.observe(25, 7, events);
    assert(events.back().begin == 6);
    auto transparent = parent;
    transparent.open_delimiter.clear(); transparent.close_delimiter.clear();
    auto flat = std::make_shared<const GRIM::ConceptSpanDelimiterMap>(
        GRIM::NamedConceptSpanDefinitions{transparent});
    assert(flat->nodes().size() == 1);
    assert(flat->nodes()[0].parent == GRIM::kNoNamedConceptSpanEntry);
    bool bad_map = false;
    auto unresolved = answer; unresolved.close_delimiter_id = -1;
    try { GRIM::ConceptSpanDelimiterMap invalid({unresolved}); }
    catch (const std::runtime_error&) { bad_map = true; }
    assert(bad_map);
    bad_map = false;
    auto duplicate = answer; duplicate.name = "duplicate";
    try { GRIM::ConceptSpanDelimiterMap invalid({answer, duplicate}); }
    catch (const std::runtime_error&) { bad_map = true; }
    assert(bad_map);
    bad_map = false;
    auto collision = answer; collision.close_delimiter_id = collision.open_delimiter_id;
    try { GRIM::ConceptSpanDelimiterMap invalid({collision}); }
    catch (const std::runtime_error&) { bad_map = true; }
    assert(bad_map);
    auto independent = GRIM::Inference::createInferenceGenerationSession(payload, model, hp);
    commitToken(independent, {21, 1.0f, 0.0f});
    assert(independent.sequence.span_events.back().kind == GRIM::GeneratedSpanEventKind::InvalidMarker);
    assert(!shouldReturn(independent, model));
    commitToken(independent, {20, 1.0f, 0.0f});
    commitToken(independent, {21, 1.0f, 0.0f});
    assert(shouldReturn(independent, model));
    commitToken(independent, {12, 1.0f, 0.0f});
    assert(!shouldReturn(independent, model)); // An older close cannot cause another return.
    commitToken(independent, {3, 1.0f, 0.0f});
    assert(finished(independent, model));
    assert(returnReason(independent, model) == GRIM::Inference::InferenceReturnReason::EOS);
    auto non_action_tail = GRIM::Inference::createInferenceGenerationSession(payload, model, hp);
    commitToken(non_action_tail, {12, 1.0f, 0.0f});
    commitForcedToken(non_action_tail, 3);
    assert(!shouldReturn(non_action_tail, model)); // Only sampled EOS ends generation.
    non_action_tail.config.eos_token_id = -1;
    assert(!shouldReturn(non_action_tail, model)); // Disabled EOS cannot match the sentinel.
    hp.max_new_tokens = 1;
    auto capped = GRIM::Inference::createInferenceGenerationSession(payload, model, hp);
    commitForcedToken(capped, 12);
    assert(!finished(capped, model)); // Forced tokens consume context, not sampled budget.
    assert(sampledTokenCount(capped) == 0);
    commitToken(capped, {12, 1.0f, 0.0f});
    assert(finished(capped, model) && remainingTokenCapacity(capped, model) == 0);
    assert(returnReason(capped, model) == GRIM::Inference::InferenceReturnReason::TokenLimit);
    hp.max_new_tokens = 8;
    model.training_max_seq_len = 3;
    auto context_capped = GRIM::Inference::createInferenceGenerationSession(payload, model, hp);
    commitForcedToken(context_capped, 12);
    assert(finished(context_capped, model));
    assert(returnReason(context_capped, model) == GRIM::Inference::InferenceReturnReason::ContextLimit);
    model.training_max_seq_len = 2;
    bool rejected = false;
    try { auto invalid = GRIM::Inference::createInferenceGenerationSession(payload, model, hp); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    auto inspection = GRIM::Inference::createInferenceGenerationSession(payload, model, hp, nullptr, true);
    auto inspected_session = takeSequence(inspection, GRIM::Inference::InferenceReturnReason::ExplicitFinish);
    assert(inspected_session.finished && inspected_session.token_scores.empty());
    model.training_max_seq_len = model.max_cached_seq_len = 32;
    hp.min_new_tokens = 2;
    auto minimum = GRIM::Inference::createInferenceGenerationSession(payload, model, hp);
    std::vector<float> logits(32, -10);
    logits[3] = 10; logits[12] = 5;
    auto first = selectNextToken(minimum, payload, logits.data());
    assert(first.token_id == 12);
    commitToken(minimum, first);
    auto second = selectNextToken(minimum, payload, logits.data());
    assert(second.token_id == 3);
    commitToken(minimum, second);
    assert(finished(minimum, model));
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
        'RETURN_POLICY_HEADER': (root / 'Shared/InferenceState/InferenceReturnPolicy.hpp').as_posix(),
        'RESULT_TYPE': result_type, 'HANDOFF': handoff,
        'SAMPLING_HEADER': (root / 'Shared/Sampling/Sampling.hpp').as_posix(),
        'GENERATION_HP': method(groups, 'struct GenerationHP') + ';',
        'MODEL_HP': method(groups, 'struct ModelHP') + ';',
        'SESSION_HEADER': session_header, 'SESSION_BODY': session_body,
        'ADVANCE_LOOP': advance_loop, 'GENERATION_STATE_HEADER': generation_header,
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

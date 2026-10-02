#include "LensMetadata.hpp"
#include "../Batching/BatchPayload.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace GRIM::Lenses {

LensCaptureMetadata makeCaptureMetadata(
    const LensCaptureRequest& request, const Batching::BatchPayload& payload,
    const NamedConceptSpanDefinitions& definitions, int layer_count, int d_model,
    int cache_prefix_length, bool effective_causal, bool dropout_enabled,
    std::uint64_t invocation_id) {
    const auto& id = request.identity;
    if (id.execution_session_id.empty() || id.checkpoint_fingerprint.empty() ||
        id.compiled_config_fingerprint.empty() || id.tokenizer_fingerprint.empty())
        throw std::runtime_error("Lens capture requires session/checkpoint/config/tokenizer identities");
    if (payload.EnableAtomIdentification)
        throw std::runtime_error("Lens capture v1 requires token rows, not atom-insertion gaps");
    if (request.batch_row < 0 || request.batch_row >= payload.batch_size ||
        layer_count <= 0 || d_model <= 0 || cache_prefix_length < 0 ||
        request.top_k <= 0 || request.top_k > payload.vocab_size ||
        (cache_prefix_length && payload.batch_size != 1))
        throw std::runtime_error("Lens capture: invalid geometry or top_k");
    if (request.replay_identity_control &&
        (request.max_replay_rows <= 0 || payload.total_tokens > request.max_replay_rows))
        throw std::runtime_error("Lens capture: full-head replay exceeds max_replay_rows");

    const auto row = static_cast<std::size_t>(request.batch_row);
    const auto spans = payload.named_concept_spans.empty() ? nullptr : payload.named_concept_spans.at(row);
    const NamedConceptSpan* prompt_span = nullptr;
    if (!spans && request.prompt_span_entry_index)
        throw std::runtime_error("Lens capture: prompt span index supplied without a span tree");
    if (spans) {
        validateNamedConceptSpans(*spans, payload.seq_lengths.at(row));
        if (request.prompt_span_entry_index) {
            prompt_span = &spans->entries.at(*request.prompt_span_entry_index);
            if (prompt_span->name != request.prompt_span_name)
                throw std::runtime_error("Lens capture: prompt span entry/name mismatch");
        } else {
            for (const auto& entry : spans->entries) {
                if (entry.name != request.prompt_span_name) continue;
                if (prompt_span) throw std::runtime_error("Lens capture: ambiguous prompt span; select an entry index");
                prompt_span = &entry;
            }
        }
    }
    int position = request.token_position;
    if (position == -1) {
        if (!prompt_span)
            throw std::runtime_error("Lens capture: selected prompt span is unavailable; provide span metadata or an explicit token position");
        position = prompt_span->span.end - 1;
    }
    if (position < 0 || position >= payload.seq_lengths.at(row))
        throw std::runtime_error("Lens capture position is outside real tokens");
    const auto flat = row * static_cast<std::size_t>(payload.max_seq_len) + position;
    LensCaptureMetadata result;
    result.identity = id;
    result.forward_invocation_id = invocation_id;
    result.batch_row = request.batch_row;
    result.token_position = position;
    result.absolute_token_position = static_cast<std::int64_t>(cache_prefix_length) + position;
    result.layer_count = layer_count;
    result.layer_index = layer_count - 1;
    result.d_model = d_model;
    result.vocab_size = payload.vocab_size;
    result.is_training = payload.isTraining();
    result.is_prefill = payload.isInferencePrefill();
    result.dropout_enabled = dropout_enabled;
    if (!payload.seq_ids.empty()) result.sequence_id = payload.seq_ids.at(row);
    result.attention.kind = effective_causal ? VisibilityKind::CausalPrefix : VisibilityKind::FullSequence;
    result.attention.visible_keys = {0, effective_causal ? result.absolute_token_position + 1
        : static_cast<std::int64_t>(cache_prefix_length) + payload.seq_lengths.at(row)};

    auto& supervision = result.supervision;
    supervision.input_token_id = payload.input_ids.at(flat);
    if (prompt_span)
        supervision.is_prompt_token = position >= prompt_span->span.begin && position < prompt_span->span.end;
    if (payload.isTraining()) {
        supervision.target_token_id = payload.target_ids.at(flat);
        supervision.is_lm_supervised_prediction = *supervision.target_token_id >= 0;
        if (!payload.atom_aux_target_mask.empty())
            supervision.atom_aux_target_mask = payload.atom_aux_target_mask.at(flat) != 0;
    }
    result.spans = spans;
    if (result.spans) {
        validateNamedConceptSpans(*result.spans, payload.seq_lengths.at(row));
        result.resolved_span_policies = conceptSpanPolicies(result.spans->entries, definitions);
        for (std::size_t i = 0; i < result.spans->entries.size(); ++i) {
            const auto& span = result.spans->entries[i].span;
            if (position >= span.begin && position < span.end) {
                supervision.input_span_entries.push_back(static_cast<std::uint32_t>(i));
                supervision.input_span_policy = result.resolved_span_policies[i];
            }
            if (supervision.is_lm_supervised_prediction.value_or(false) &&
                position + 1 >= span.begin && position + 1 < span.end)
                supervision.target_span_entries.push_back(static_cast<std::uint32_t>(i));
        }
    }
    return result;
}

LensReadout summarizeLogits(const std::vector<float>& logits, int top_k, ReadoutKind kind) {
    if (logits.empty() || top_k <= 0 || static_cast<std::size_t>(top_k) > logits.size())
        throw std::runtime_error("Lens readout: invalid vocabulary or top_k");
    for (float value : logits)
        if (!std::isfinite(value)) throw std::runtime_error("Lens readout: nonfinite logit");
    const double maximum = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (float value : logits) sum += std::exp(static_cast<double>(value) - maximum);
    const double log_sum = std::log(sum);
    LensReadout result;
    result.kind = kind;
    for (float value : logits) {
        const double log_p = static_cast<double>(value) - maximum - log_sum;
        result.entropy_nats -= std::exp(log_p) * log_p;
    }
    std::vector<int> order(logits.size());
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    for (int i = 0; i < top_k; ++i) {
        const int token = order[i];
        result.top_tokens.push_back({token, logits[token],
            std::exp(static_cast<double>(logits[token]) - maximum - log_sum)});
    }
    return result;
}
} // namespace GRIM::Lenses

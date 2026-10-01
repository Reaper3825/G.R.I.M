#pragma once

// Host-only lens contract. No borrowed CUDA pointers or autograd tensors escape
// a forward invocation through this interface.
#include "../ConceptBlock/NamedConceptSpans.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace GRIM::Batching { struct BatchPayload; }

namespace GRIM::Lenses {

enum class ReadoutKind { ActualFinal, Direct, JacobianIdentity };
enum class VisibilityKind { CausalPrefix, FullSequence };

struct TokenRange {
    std::int64_t begin = 0;
    std::int64_t end = 0; // exclusive, absolute sequence coordinates
};

struct AttentionVisibility {
    VisibilityKind kind = VisibilityKind::CausalPrefix;
    TokenRange visible_keys;
    bool includes_self = true;
    // Describes permitted real keys, not attention weights or indirect ancestry.
    // Current implementation applies uniformly across encoder layers/heads.
};

struct LensIdentity {
    std::string execution_session_id;
    std::string checkpoint_fingerprint;
    std::string compiled_config_fingerprint;
    std::string tokenizer_fingerprint;
    std::uint64_t parameter_revision = 0;
};

// Caller-owned, borrowed only for executeModelForward's duration. Absence of
// this request disables all capture/replay work. Initial scope is token rows at
// the final encoder output; gap models require a different target boundary.
struct LensCaptureRequest {
    LensIdentity identity;
    int batch_row = 0;
    int token_position = -1; // row-local; -1 selects payload's final prompt token
    int top_k = 10;
    bool replay_identity_control = false;
    // Full-head replay retains full rectangle intermediates. Bound its row
    // count explicitly instead of silently allocating a training-sized replay.
    int max_replay_rows = 1024;
};

struct TokenSupervision {
    int input_token_id = -1;
    std::optional<bool> is_prompt_token;
    std::vector<std::uint32_t> input_span_entries;
    std::vector<std::uint32_t> target_span_entries;
    std::optional<ConceptSpanSupervision> input_span_policy;
    // Null in inference; -1 in training means a realized masked LM target.
    std::optional<int> target_token_id;
    std::optional<bool> is_lm_supervised_prediction;
    std::optional<bool> atom_aux_target_mask;
};

struct LensCaptureMetadata {
    static constexpr std::uint32_t schema_version = 1;
    LensIdentity identity;
    std::uint64_t forward_invocation_id = 0;
    std::uint64_t capture_sequence = 0;
    std::int64_t capture_timestamp_unix_ns = 0;
    std::int64_t capture_monotonic_ns = 0; // timestamp immediately before copies
    std::optional<std::uint32_t> sequence_id;
    int batch_row = 0;
    int token_position = 0;
    std::int64_t absolute_token_position = 0;
    int layer_index = 0; // zero-based post-block output
    int layer_count = 0;
    int d_model = 0;
    int vocab_size = 0;
    bool is_training = false;
    bool is_prefill = false;
    bool dropout_enabled = false;
    std::string target_boundary = "encoder_output_tensor/pre_lm_head";
    std::string readout_version = "forwardLmHead/v1";
    AttentionVisibility attention;
    TokenSupervision supervision;
    // Existing immutable row metadata; safely survives forward clear().
    std::shared_ptr<const NamedConceptSpans> spans;
    std::vector<ConceptSpanSupervision> resolved_span_policies;
};

struct RankedToken {
    int token_id = -1;
    float logit = 0;
    double probability = 0; // normalized over the FULL vocabulary at T=1
};

struct LensReadout {
    ReadoutKind kind = ReadoutKind::ActualFinal;
    std::vector<RankedToken> top_tokens;
    double entropy_nats = 0;
};

struct LensSnapshot {
    LensCaptureMetadata metadata;
    std::vector<float> readout_input; // selected row, owned host FP32
    std::vector<LensReadout> readouts;
    std::optional<double> identity_max_abs_logit_error;
    // Identity J is implicit. Direct and J-identity share one head replay;
    // neither denotes a fitted Jacobian.
};

LensCaptureMetadata makeCaptureMetadata(
    const LensCaptureRequest& request, const Batching::BatchPayload& payload,
    const NamedConceptSpanDefinitions& definitions, int layer_count, int d_model,
    int cache_prefix_length, bool effective_causal, bool dropout_enabled,
    std::uint64_t invocation_id);

LensReadout summarizeLogits(const std::vector<float>& logits, int top_k,
                           ReadoutKind kind);

} // namespace GRIM::Lenses

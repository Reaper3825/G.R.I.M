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
// this request disables all capture/replay work. Gap models require a different
// target boundary. All positions means real tokens in batch_row, never padding.
struct LensCaptureRequest {
    LensIdentity identity;
    int batch_row = 0;
    std::string prompt_span_name = "prompt";
    // Disambiguates repeated names in the existing row tree when supplied.
    std::optional<std::uint32_t> prompt_span_entry_index;
    int token_position = -1; // row-local; -1 selects the named prompt span's end
    int top_k = 10;
    bool all_layers = false;
    bool all_positions = false;
    // Bounds additional capture-owned device buffers and host reduction scratch.
    // Excludes the model/ordinary forward and the compact, persistent results.
    std::uint64_t temporary_memory_budget_bytes = 64ULL * 1024 * 1024;
    bool retain_readout_input = false;
    bool replay_identity_control = false;
    // Capture one selected row of the hidden-to-hidden Jacobian.
    bool capture_jacobian = false;
    int jacobian_target_position = -1; // -1: last real input position.
    int jacobian_output_dimension = 0; // Selected final hidden dimension.
    // Maximum rows per readout chunk (kept for existing diagnostic callers).
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
    // d(final post-block hidden[target_position,out])/d(this post-block hidden[in]).
    // Row-major [output_dimension,input_dimension] after inspection assembly.
    // A selected-row capture contains d_model entries and explicit output metadata.
    std::vector<float> hidden_jacobian;
    // Identity J is implicit. Direct and J-identity share one head replay;
    // neither denotes a fitted Jacobian.
};

// Host-owned capture from one forward, or an assembled full Jacobian inspection.
// No live device views escape.
// Ordered by layer, then row-local position; only the selected batch row.
struct LensCaptureResult {
    std::vector<LensSnapshot> snapshots;
    std::uint64_t temporary_memory_budget_bytes = 0;
    std::uint64_t planned_temporary_bytes = 0;
    int chunk_rows = 0;
    std::optional<int> jacobian_target_position;
    std::optional<int> jacobian_output_dimension; // Present for a selected row; absent for full matrices.
    int jacobian_rows = 0; // 1 for a selected row; d_model for an assembled full matrix.
};

// Conservative scratch plan for detached FP32 row-local LM-head readout.
// Includes one shared gated weight matrix when enabled and host reduction rows.
int planCaptureChunkRows(const LensCaptureRequest& request, int real_positions,
                         int d_model, int vocab_size, int mlp_d_ff,
                         bool normalized, bool bias, bool gated_weights,
                         std::uint64_t& planned_bytes);

LensCaptureMetadata makeCaptureMetadata(
    const LensCaptureRequest& request, const Batching::BatchPayload& payload,
    const NamedConceptSpanDefinitions& definitions, int layer_count, int d_model,
    int cache_prefix_length, bool effective_causal, bool dropout_enabled,
    std::uint64_t invocation_id);

LensReadout summarizeLogits(const std::vector<float>& logits, int top_k,
                           ReadoutKind kind);

} // namespace GRIM::Lenses

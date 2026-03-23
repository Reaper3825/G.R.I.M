#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "../grim_layer_gpu.hpp"
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"
#include "../../training/schemas/grim_transformer_model_generated.h"

namespace GRIM {

struct SerializationModelConfigView {
	int vocab_size = 0;
	int d_model = 0;
	int num_layers = 0;
	int num_heads = 0;
	int num_kv_heads = 0;  // GQA: number of KV heads (0 = same as num_heads for MHA)
	int d_ff = 0;
	int max_seq_len = 0;
	float dropout_rate = 0.0f;
	HyperParameters::PositionalEncodingType positional_encoding = HyperParameters::PositionalEncodingType::NONE;
	bool tie_embeddings = false;
	bool use_gpu = false;
	bool use_bias = false;
};

struct DeviceReadView {
	const float* ptr = nullptr;
	std::size_t count = 0;
};

struct DeviceWriteView {
	float* ptr = nullptr;
	std::size_t count = 0;
};

struct SerializationGpuEmbeddingReadView {
	DeviceReadView token_embeddings;
	DeviceReadView rms_gamma;
	bool has_rms_norm = false;
};

struct SerializationGpuEmbeddingWriteView {
	DeviceWriteView token_embeddings;
	DeviceWriteView rms_gamma;
	bool has_rms_norm = false;
};

struct SerializationCpuEmbeddingReadData {
	std::vector<float> token_data;
	std::vector<float> rms_gamma;
	int num_rows = 0;
	int num_cols = 0;
	bool has_rms_norm = false;
};

struct SerializationCpuEmbeddingWriteOps {
	std::function<void(const std::vector<float>& data, int rows, int cols)> set_tokens;
	std::function<void(const std::vector<float>& gamma)> set_rms_gamma;
};

struct SerializationEncoderLayerReadView {
	DeviceReadView attn_w_qkv;
	DeviceReadView attn_b_qkv;
	DeviceReadView attn_w_o;
	DeviceReadView attn_b_o;
	DeviceReadView ffn_w_gate;
	DeviceReadView ffn_w1;
	DeviceReadView ffn_w2;
	DeviceReadView ffn_b2;
	DeviceReadView rms1_gamma;
	DeviceReadView rms2_gamma;
	DeviceReadView rms_post_attn_gamma;  // Issue #148: REMOVED from model, kept for old checkpoint compat (always null)
	DeviceReadView rms_post_ffn_gamma;   // Issue #148: REMOVED from model, kept for old checkpoint compat (always null)
	DeviceReadView layer_scale1;       // [1] LayerScale attention scalar (may be empty if disabled)
	DeviceReadView layer_scale2;       // [1] LayerScale FFN scalar (may be empty if disabled)
};

struct SerializationEncoderLayerWriteView {
	DeviceWriteView attn_w_qkv;
	DeviceWriteView attn_b_qkv;
	DeviceWriteView attn_w_o;
	DeviceWriteView attn_b_o;
	DeviceWriteView ffn_w_gate;
	DeviceWriteView ffn_w1;
	DeviceWriteView ffn_w2;
	DeviceWriteView ffn_b2;
	DeviceWriteView rms1_gamma;
	DeviceWriteView rms2_gamma;
	DeviceWriteView rms_post_attn_gamma;  // Issue #148: REMOVED from model, kept for old checkpoint compat (always null)
	DeviceWriteView rms_post_ffn_gamma;   // Issue #148: REMOVED from model, kept for old checkpoint compat (always null)
	DeviceWriteView layer_scale1;      // [1] LayerScale attention scalar (may be empty if disabled)
	DeviceWriteView layer_scale2;      // [1] LayerScale FFN scalar (may be empty if disabled)
};

struct SerializationLMHeadReadView {
	DeviceReadView projection;
	DeviceReadView bias;
	bool has_projection = false;
	bool has_bias = false;
};

struct SerializationLMHeadWriteView {
	DeviceWriteView projection;
	DeviceWriteView bias;
	bool expect_bias = false;
};

struct SerializationScratchBlockReadView {
	DeviceReadView atom_type_embeddings;  // [num_atom_types, atom_embedding_dim]
	DeviceReadView atom_projection;        // [atom_embedding_dim, d_model]
	// text_feature_projection ELIMINATED — text features merged into atom embeddings (dims 48-63)
	int num_atom_types = 0;
	int atom_embedding_dim = 0;
	int d_model = 0;
	float atom_scale = 1.0f;  // Unit scale - embeddings are scaled up to match
	bool enabled = false;
};

struct SerializationScratchBlockWriteView {
	DeviceWriteView atom_type_embeddings;
	DeviceWriteView atom_projection;
	int num_atom_types = 0;
	int atom_embedding_dim = 0;
};

struct SerializationNumericHeadReadView {
	DeviceReadView weights;     // [2, d_model]
	DeviceReadView bias;        // [2]
	int d_model = 0;
	bool enabled = false;
};

struct SerializationNumericHeadWriteView {
	DeviceWriteView weights;
	DeviceWriteView bias;
	int d_model = 0;
};

struct SerializationReasoningHeadReadView {
	DeviceReadView w_op;       // [num_ops, d_total]
	DeviceReadView b_op;       // [num_ops]
	DeviceReadView w_arg1;     // [1, d_total]
	DeviceReadView w_arg2;     // [1, d_total]
	int num_ops = 0;
	int d_total = 0;
	bool enabled = false;
};

struct SerializationReasoningHeadWriteView {
	DeviceWriteView w_op;
	DeviceWriteView b_op;
	DeviceWriteView w_arg1;
	DeviceWriteView w_arg2;
	int num_ops = 0;
	int d_total = 0;
};

// ExecutionBlock v2: differentiable rewrite — no backward compat with v1 checkpoints
struct SerializationExecutionBlockReadView {
	DeviceReadView w_decode_1;      // [24, 16]
	DeviceReadView b_decode_1;      // [16]
	DeviceReadView w_decode_2;      // [16, 1]
	DeviceReadView w_arg1_select;   // [1, d_model]
	DeviceReadView w_arg2_select;   // [1, d_model]
	DeviceReadView W_op_select;     // [3*d_model, 4]
	DeviceReadView W_key_proj;      // [d_model, d_key]
	DeviceReadView W_write_query;   // [4*d_model, d_key]
	DeviceReadView W_write_key;     // [d_key, d_key]
	DeviceReadView alpha;           // [1]
	DeviceReadView beta;            // [1]
	DeviceReadView gamma;           // [1]
	DeviceReadView step_embeddings; // [K, d_model]
	DeviceReadView type_num_embed;  // [d_type]
	DeviceReadView W_value_to_emb;  // [1, d_model]
	DeviceReadView b_value_to_emb;  // [1, d_model]
	DeviceReadView w_inject_gate;   // [d_model, 1]
	DeviceReadView W_Q_read;        // [d_model, head_dim]
	DeviceReadView W_K_read;        // [d_key, head_dim]
	DeviceReadView W_V_read;        // [d_model, head_dim]
	DeviceReadView W_O_read;        // [head_dim, d_model]
	DeviceReadView W_gate_read;     // [d_model, 1]
	DeviceReadView tau;             // [1]
	bool enabled = false;
};

struct SerializationExecutionBlockWriteView {
	DeviceWriteView w_decode_1;
	DeviceWriteView b_decode_1;
	DeviceWriteView w_decode_2;
	DeviceWriteView w_arg1_select;
	DeviceWriteView w_arg2_select;
	DeviceWriteView W_op_select;
	DeviceWriteView W_key_proj;
	DeviceWriteView W_write_query;
	DeviceWriteView W_write_key;
	DeviceWriteView alpha;
	DeviceWriteView beta;
	DeviceWriteView gamma;
	DeviceWriteView step_embeddings;
	DeviceWriteView type_num_embed;
	DeviceWriteView W_value_to_emb;
	DeviceWriteView b_value_to_emb;
	DeviceWriteView w_inject_gate;
	DeviceWriteView W_Q_read;
	DeviceWriteView W_K_read;
	DeviceWriteView W_V_read;
	DeviceWriteView W_O_read;
	DeviceWriteView W_gate_read;
	DeviceWriteView tau;
};

struct SerializationSaveSources {
	SerializationModelConfigView config;
	SerializationGpuEmbeddingReadView gpu_embedding;
	SerializationCpuEmbeddingReadData cpu_embedding;
	std::vector<SerializationEncoderLayerReadView> encoder_layers;
	SerializationLMHeadReadView lm_head;
	SerializationScratchBlockReadView scratch_block;
	SerializationNumericHeadReadView numeric_head;
	SerializationReasoningHeadReadView reasoning_head;
	SerializationExecutionBlockReadView execution_block;
	DeviceReadView final_rms_gamma;
};

struct SerializationConfig {
	bool use_gpu = true;
	bool atomic_write = true;
	std::string temp_suffix = ".tmp";
};

struct SerializationSaveRequest {
	std::string path;
	std::uint32_t model_version = 0;
	SerializationSaveSources sources;
};

struct SerializationLoadRequest {
	std::string path;
	SerializationModelConfigView config;
	SerializationGpuEmbeddingWriteView gpu_embedding;
	SerializationCpuEmbeddingWriteOps cpu_embedding;
	std::vector<SerializationEncoderLayerWriteView> encoder_layers;
	SerializationLMHeadWriteView lm_head;
	SerializationScratchBlockWriteView scratch_block;
	SerializationNumericHeadWriteView numeric_head;
	SerializationReasoningHeadWriteView reasoning_head;
	SerializationExecutionBlockWriteView execution_block;
	DeviceWriteView final_rms_gamma;
};

class SerializationLayer final : public Layer<SerializationLayer, float> {
public:
	static constexpr LayerType layer_type = LayerType::kSerialization;

	SerializationLayer() = default;
	explicit SerializationLayer(SerializationConfig config);

	void setConfig(const SerializationConfig& config);
	const SerializationConfig& config() const noexcept { return config_; }

	bool save(const SerializationSaveRequest& request);
	bool load(const SerializationLoadRequest& request);

private:
	SerializationConfig config_{};
};

} // namespace GRIM

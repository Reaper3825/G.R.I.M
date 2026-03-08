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
	DeviceReadView ffn_w_gate_up;
	DeviceReadView ffn_b_gate_up;
	DeviceReadView ffn_w_down;
	DeviceReadView ffn_b_down;
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
	DeviceWriteView ffn_w_gate_up;
	DeviceWriteView ffn_b_gate_up;
	DeviceWriteView ffn_w_down;
	DeviceWriteView ffn_b_down;
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
	// text_feature_projection ELIMINATED — text features merged into atom embeddings (dims 48-63)
	int num_atom_types = 0;
	int atom_embedding_dim = 0;
};

struct SerializationSaveSources {
	SerializationModelConfigView config;
	SerializationGpuEmbeddingReadView gpu_embedding;
	SerializationCpuEmbeddingReadData cpu_embedding;
	std::vector<SerializationEncoderLayerReadView> encoder_layers;
	SerializationLMHeadReadView lm_head;
	SerializationScratchBlockReadView scratch_block;  // Optional ScratchBlock weights
	DeviceReadView final_rms_gamma;  // Issue #33: Final RMSNorm gamma [d_model]
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
	SerializationScratchBlockWriteView scratch_block;  // Optional ScratchBlock weights
	DeviceWriteView final_rms_gamma;  // Issue #33: Final RMSNorm gamma [d_model]
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

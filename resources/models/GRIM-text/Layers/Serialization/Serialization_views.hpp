#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include "../../Shared/HyperParameters/HyperParameters_GPU.hpp"

namespace GRIM {

struct SerializationModelConfigView {
	int vocab_size = 0;
	int d_model = 0;
	int num_layers = 0;
	int num_heads = 0;
	int num_kv_heads = 0;
	int d_ff = 0;
	int max_seq_len = 0;
	float dropout_rate = 0.0f;
	HyperParameters::PositionalEncodingType positional_encoding = HyperParameters::PositionalEncodingType::ALIBI_ROPE;
	bool tie_embeddings = false;
	bool use_gpu = false;
	bool use_bias = false;

	int head_dim() const { return (num_heads > 0) ? (d_model / num_heads) : 0; }
	int kv_dim()   const { return num_kv_heads * head_dim(); }
	int total_qkv_dim() const { return d_model + 2 * kv_dim(); }
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
};

struct SerializationGpuEmbeddingWriteView {
	DeviceWriteView token_embeddings;
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
	DeviceReadView rms_post_attn_gamma;
	DeviceReadView rms_post_ffn_gamma;
	DeviceReadView layer_scale1;
	DeviceReadView layer_scale2;
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
	DeviceWriteView rms_post_attn_gamma;
	DeviceWriteView rms_post_ffn_gamma;
	DeviceWriteView layer_scale1;
	DeviceWriteView layer_scale2;
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
	DeviceReadView atom_type_embeddings;
	DeviceReadView atom_projection;
	int num_atom_types = 0;
	int atom_embedding_dim = 0;
	int d_model = 0;
	float atom_scale = 1.0f;
	bool enabled = false;
};

struct SerializationScratchBlockWriteView {
	DeviceWriteView atom_type_embeddings;
	DeviceWriteView atom_projection;
	int num_atom_types = 0;
	int atom_embedding_dim = 0;
};

struct SerializationExecutionBlockReadView {
	DeviceReadView w_decode_1;
	DeviceReadView b_decode_1;
	DeviceReadView w_decode_2;
	DeviceReadView w_arg1_select;
	DeviceReadView w_arg2_select;
	DeviceReadView W_op_select;
	DeviceReadView W_key_proj;
	DeviceReadView W_write_query;
	DeviceReadView W_write_key;
	DeviceReadView alpha;
	DeviceReadView beta;
	DeviceReadView step_embeddings;
	DeviceReadView type_num_embed;
	DeviceReadView W_value_to_emb;
	DeviceReadView b_value_to_emb;
	DeviceReadView w_inject_gate;
	DeviceReadView W_Q_read;
	DeviceReadView W_K_read;
	DeviceReadView W_V_read;
	DeviceReadView W_O_read;
	DeviceReadView W_gate_read;
	DeviceReadView tau;
	DeviceReadView E_slot;
	DeviceReadView E_op;
	DeviceReadView W_scal;
	DeviceReadView b_scal;
	DeviceReadView W_trace;
	DeviceReadView b_trace;
	DeviceReadView W_reason_gate;
	DeviceReadView W_trace_gate;
	bool enabled = false;
};

struct SerializationSlotSelectorReadView {
	DeviceReadView w_q_select;
	DeviceReadView w_k_select;
	DeviceReadView null_key_select;
	DeviceReadView null_logit_bias;
	bool enabled = false;
};

struct SerializationSlotSelectorWriteView {
	DeviceWriteView w_q_select;
	DeviceWriteView w_k_select;
	DeviceWriteView null_key_select;
	DeviceWriteView null_logit_bias;
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
	DeviceWriteView E_slot;
	DeviceWriteView E_op;
	DeviceWriteView W_scal;
	DeviceWriteView b_scal;
	DeviceWriteView W_trace;
	DeviceWriteView b_trace;
	DeviceWriteView W_reason_gate;
	DeviceWriteView W_trace_gate;
};

} // namespace GRIM

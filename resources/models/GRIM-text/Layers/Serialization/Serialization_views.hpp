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
	// Head-side residual SwiGLU adapter (config.lm_head_mlp_enabled).
	// Presence-driven like the bias: populated only when the model allocated
	// the adapter tensors. mlp_d_ff is derived from the live W_gate shape.
	DeviceReadView mlp_w_gate;
	DeviceReadView mlp_w_up;
	DeviceReadView mlp_w_down;
	bool has_projection = false;
	bool has_bias = false;
	bool has_mlp = false;
	int mlp_d_ff = 0;
};

struct SerializationLMHeadWriteView {
	DeviceWriteView projection;
	DeviceWriteView bias;
	DeviceWriteView mlp_w_gate;
	DeviceWriteView mlp_w_up;
	DeviceWriteView mlp_w_down;
	bool expect_bias = false;
	bool expect_mlp = false;
};

struct SerializationExecutionBlockReadView {
	DeviceReadView w_decode_1;
	DeviceReadView b_decode_1;
	DeviceReadView w_decode_2;
	DeviceReadView w_arg1_select;
	DeviceReadView w_arg2_select;
	DeviceReadView W_arg1_to_arg2;
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
	DeviceReadView W_execute;
	DeviceReadView b_execute;
	DeviceReadView W_stop;
	DeviceReadView b_stop;
	bool enabled = false;
};

struct SerializationExecutionBlockWriteView {
	DeviceWriteView w_decode_1;
	DeviceWriteView b_decode_1;
	DeviceWriteView w_decode_2;
	DeviceWriteView w_arg1_select;
	DeviceWriteView w_arg2_select;
	DeviceWriteView W_arg1_to_arg2;
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
	DeviceWriteView W_execute;
	DeviceWriteView b_execute;
	DeviceWriteView W_stop;
	DeviceWriteView b_stop;
};

struct SerializationNumberEncoderReadView {
	DeviceReadView digit_emb;
	DeviceReadView pow10_emb;
	DeviceReadView W_c1;
	DeviceReadView b_c1;
	DeviceReadView W_c2;
	DeviceReadView W_g1;
	DeviceReadView b_g1;
	DeviceReadView W_g2;
	bool enabled = false;
};

struct SerializationNumberEncoderWriteView {
	DeviceWriteView digit_emb;
	DeviceWriteView pow10_emb;
	DeviceWriteView W_c1;
	DeviceWriteView b_c1;
	DeviceWriteView W_c2;
	DeviceWriteView W_g1;
	DeviceWriteView b_g1;
	DeviceWriteView W_g2;
};

struct SerializationArgSelectorReadView {
	DeviceReadView W_q;
	bool enabled = false;
};

struct SerializationArgSelectorWriteView {
	DeviceWriteView W_q;
};

struct SerializationSlotSeedEncoderReadView {
	DeviceReadView W_seed_in;
	DeviceReadView b_seed_in;
	DeviceReadView W_seed_out;
	DeviceReadView b_seed_out;
	DeviceReadView type_embeddings;
	bool enabled = false;
};

struct SerializationSlotSeedEncoderWriteView {
	DeviceWriteView W_seed_in;
	DeviceWriteView b_seed_in;
	DeviceWriteView W_seed_out;
	DeviceWriteView b_seed_out;
	DeviceWriteView type_embeddings;
};

} // namespace GRIM

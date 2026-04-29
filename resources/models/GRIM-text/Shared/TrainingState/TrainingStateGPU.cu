//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

// Include grim_language_model_cuda.hpp for GPUGrimEncoder, FlashAttentionBF16Scratch, etc.
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "TrainingState_GPU.hpp"
#include "../../training/Autograd/AutogradTraining.hpp"  

#include <stdexcept>
#include <string>

#ifdef USE_CUDA

namespace GRIM {

//======================================================//
//  Weight tensor accessors
//  Session 6: Embedding accessors DELETED — weights now owned by EmbeddingLayer (Pattern B).
//  Access via LanguageModel::getEmbeddingLayer()->tokenWeights().
//======================================================//

TrainingState::TrainingState() = default;

TrainingState::~TrainingState() {
	// ═══════════════════════════════════════════════════════════════
	// ALL MEMORY NOW TENSOR-MANAGED
	// ═══════════════════════════════════════════════════════════════
	// After Rule 20 migration, all float* buffers are GRIM::Tensor objects.
	// Tensor::~Tensor() calls release() to free GPU memory.
	// DO NOT manually cudaFree any Tensor member - they own their data.
	
	// Rule 20: NO encoder_layer_caches - intermediate tensors now managed via AutogradIntermediates
	// (owned and zero'd separately during training, NOT in destructor)

	// TeacherLogits has its own release function (not Tensor-based yet)
	TeacherLogits::release(teacher_logits);
	TeacherLogits::release(reference_logits);
	
	// Free optimizer states (std::vector<Tensor> clears automatically)
	freeOptimizerStates();
	
	// Free guess cache buffers (GRIM-TS - not Tensor-based, uses raw cudaMalloc)
	freeGuessCacheBuffers();

	// Free class-balanced loss weights (raw cudaMalloc)
	if (d_class_weights) { cudaFree(d_class_weights); d_class_weights = nullptr; }

	// Free cross-attention read-gate accumulator (Rule 20 Category 3 workspace)
	if (d_read_gate_accum) { cudaFree(d_read_gate_accum); d_read_gate_accum = nullptr; }
	
	// Free per-layer KV cache (raw cudaMalloc, BF16)
	for (auto& ptr : kv_cache_k) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }
	for (auto& ptr : kv_cache_v) { if (ptr) { cudaFree(ptr); ptr = nullptr; } }
	kv_cache_k.clear();
	kv_cache_v.clear();
	if (kv_cache_softmax_lse) { cudaFree(kv_cache_softmax_lse); kv_cache_softmax_lse = nullptr; }
	
	// Free decode scratch buffers (raw cudaMalloc, BF16/FP32)
	if (decode_q_bf16) { cudaFree(decode_q_bf16); decode_q_bf16 = nullptr; }
	if (decode_kv_bf16) { cudaFree(decode_kv_bf16); decode_kv_bf16 = nullptr; }
	if (decode_attn_out_bf16) { cudaFree(decode_attn_out_bf16); decode_attn_out_bf16 = nullptr; }
	if (decode_attn_out_fp32) { cudaFree(decode_attn_out_fp32); decode_attn_out_fp32 = nullptr; }
	
	// Free ScratchBlockPool (pinned memory blocks)
	if (scratch_pool) {
		delete scratch_pool;
		scratch_pool = nullptr;
	}
	
	// Free gradient norm scratch buffers
	GradNorm::freeGradNormScratch(grad_norm_scratch);
	
	// StreamController owns and destroys streams - no manual cleanup needed
	if (cublas_handle) cublasDestroy(cublas_handle);
}

//======================================================//
//  Autograd Seed Storage (Session 7: TrainingTensors deleted)
//======================================================//

void TrainingState::initializeAutogradSeed(uint64_t seed) {
	// Rule 20: Fail loud if already initialized
	if (seed_initialized_) {
		throw std::runtime_error("[TrainingState::initializeAutogradSeed] already initialized!");
	}
	
	weight_init_seed = seed;
	seed_initialized_ = true;
	
	fprintf(stdout, "[INFO] TrainingState: autograd seed stored (seed=%llu). "
	        "All weights managed by Pattern B layers.\n",
	        static_cast<unsigned long long>(seed));
}

//======================================================//
//  Intermediate Gradient Zeroing (Issue #45 FIX)
//======================================================//

void TrainingState::zeroIntermediateGrads(cudaStream_t stream) {
	// RULE 20: Fail loud on NULL stream - it causes race conditions
	StreamController::fatalIfDefaultStream(stream, "FATAL:TrainingState::zeroIntermediateGrads Default stream detected!");

	auto safe_zero = [stream](Tensor& t, const char* name) {
		if (t.data && t.numel() > 0) {
			// Zero the DATA field, not the grad field!
			cudaMemsetAsync(t.data, 0, t.numel() * sizeof(float), stream);
		}
	};
	
	// Loss backward → encoder
	safe_zero(grad_logits_tensor, "grad_logits");
	safe_zero(grad_encoder_tensor, "grad_encoder");
	
	// Encoder backward temporaries
	safe_zero(grad_ffn_input_tensor, "grad_ffn_input");
	safe_zero(grad_ffn_hidden_tensor, "grad_ffn_hidden");
	safe_zero(grad_attn_input_tensor, "grad_attn_input");
	safe_zero(grad_attn_out_proj_tensor, "grad_attn_out_proj");
	safe_zero(grad_attn_out_bhsd_tensor, "grad_attn_out_bhsd");
	safe_zero(grad_q_tensor, "grad_q");
	safe_zero(grad_k_tensor, "grad_k");
	safe_zero(grad_v_tensor, "grad_v");
	safe_zero(grad_qkv_concat_tensor, "grad_qkv_concat");
	safe_zero(grad_qkv_input_tensor, "grad_qkv_input");
	safe_zero(grad_attn_bsm_tensor, "grad_attn_bsm");
}

} // namespace GRIM

#endif  // USE_CUDA
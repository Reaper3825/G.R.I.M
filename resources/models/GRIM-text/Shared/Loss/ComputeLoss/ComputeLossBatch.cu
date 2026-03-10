#include "../../Batching/BatchPayload.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "../../GRIM/grim_language_model_cuda.hpp"
// AutogradLoss.hpp, LossContext.hpp, TeacherLogits_GPU.hpp
// removed — now handled by computeAutogradLoss() in AutogradTraining.hpp
#include "../../LogRecorder/LogRecorder.hpp"
#include "../../../training/Autograd/AutogradTraining.hpp"  // Issue #47: Full autograd forward pass
#include "../../VerboseLogging.hpp"  // Guards for expensive debug prints

namespace GRIM {

namespace {

std::atomic<unsigned long long> g_order_log_counter{0};
static bool g_order_log_enabled = false;

bool isOrderLogEnabled()
{
	return g_order_log_enabled;
}

void orderLog(const char* stage,
	size_t batch_size,
	size_t seq_len,
	size_t total_tokens,
	int valid_tokens)
{
	if (!isOrderLogEnabled()) {
		return;
	}
	const unsigned long long stamp = ++g_order_log_counter;
	fprintf(stderr, "[ORDER] %llu %s batch=%zu seq=%zu tokens=%zu valid=%d\n",
		stamp, stage, batch_size, seq_len, total_tokens, valid_tokens);
}

}  // namespace

float LanguageModel::computeLossBatch(
	const GRIM::Batching::BatchPayload& payload,
	bool is_training)
{
	// Keep intermediates for the legacy computeLossBatch() -> backward() flow on success,
	// but clear them if this function exits via exception.
	struct IntermediateExceptionGuard {
		TrainingState& ts;
		bool active = true;
		~IntermediateExceptionGuard() {
			if (active) {
				ts.autograd_intermediates.clear();
			}
		}
		void dismiss() { active = false; }
	};
	IntermediateExceptionGuard exception_guard{training_state_};

	// ═══════════════════════════════════════════════════════════════════════════
	// BatchPayload is the SINGLE SOURCE OF TRUTH for all batch metadata.
	// All padding, masking, sequence lengths, valid token counts, and cache fit
	// checks were computed ONCE in buildBatchPayload(). No recomputation here.
	// ═══════════════════════════════════════════════════════════════════════════

	orderLog("computeLossBatch.enter",
		payload.batch_size, payload.max_seq_len, payload.total_tokens, payload.valid_tokens);

	// Ensure training state is initialized
	if (!training_state_.initialized) {
		orderLog("computeLossBatch.init_start",
			payload.batch_size, 0, 0, 0);
		initTrainingState();
		if (!training_state_.initialized) {
			throw std::runtime_error("computeLossBatch: initTrainingState() completed but flag still false");
		}
		orderLog("computeLossBatch.init_done",
			payload.batch_size, 0, 0, 0);
	}

	// Rule 20: payload was already validated by buildBatchPayload, but re-validate here
	// to catch any corruption between build and use
	payload.validate("computeLossBatch");

	const auto& cfg = getConfig();

	GPUGrimEncoder* gpu_encoder = nullptr;
	try {
		gpu_encoder = &getGpuEncoder();
	} catch (const std::exception& ex) {
		throw std::runtime_error(std::string("computeLossBatch: GPU encoder not initialized: ") + ex.what());
	}
	if (!gpu_encoder) {
		throw std::runtime_error("computeLossBatch: GPU encoder is NULL after getGpuEncoder()");
	}

	const size_t batch_size = static_cast<size_t>(payload.batch_size);
	const size_t seq_len = static_cast<size_t>(payload.max_seq_len);
	const size_t total_tokens = static_cast<size_t>(payload.total_tokens);
	const int valid_tokens = payload.valid_tokens;

	const size_t logit_limit = training_state_.max_logit_tokens > 0
		? training_state_.max_logit_tokens
		: training_state_.max_cached_tokens;
	if (total_tokens > logit_limit) {
		throw std::runtime_error(
			"computeLossBatch: total_tokens=" + std::to_string(total_tokens) +
			" exceeds logit buffer capacity=" + std::to_string(logit_limit));
	}

	// ═══════════════════════════════════════════════════════════════════════════
	// GPU COPIES: Pinned-memory staged transfers via ScratchBlockPool
	// Double-buffered: submit DMA from block A while CPU fills block B.
	// ═══════════════════════════════════════════════════════════════════════════
	auto copy_start = std::chrono::high_resolution_clock::now();
	orderLog("computeLossBatch.copy_inputs",
		batch_size, seq_len, total_tokens, 0);

	cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

	// Rule 20: scratch pool MUST be available — no fallback to pageable memory
	auto* scratch_pool = training_state_.scratch_pool;
	if (!scratch_pool || !scratch_pool->isInitialized()) {
		throw std::runtime_error("computeLossBatch: scratch_pool not initialized — "
			"pinned memory staging is REQUIRED for batch transfers");
	}

	// Validate GPU destination pointers (Rule 20: crash if null)
	int* cached_token_ids_ptr = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
	if (!cached_token_ids_ptr) {
		throw std::runtime_error("computeLossBatch: cached_token_ids_tensor.data is NULL");
	}
	int* cached_targets_ptr = reinterpret_cast<int*>(training_state_.cached_targets_tensor.data);
	if (!cached_targets_ptr) {
		throw std::runtime_error("computeLossBatch: cached_targets_tensor.data is NULL");
	}
	float* cached_numeric_values_ptr = training_state_.cached_token_numeric_values.data;
	if (!cached_numeric_values_ptr) {
		throw std::runtime_error("computeLossBatch: cached_token_numeric_values.data is NULL");
	}
	float* cached_atom_mask_ptr = training_state_.cached_token_atom_mask.data;
	if (!cached_atom_mask_ptr) {
		throw std::runtime_error("computeLossBatch: cached_token_atom_mask.data is NULL");
	}

	// Compute transfer sizes
	// Compute transfer sizes using BatchPayload helpers
	const size_t input_ids_bytes    = payload.inputIdBytes();
	const size_t target_ids_bytes   = payload.targetIdBytes();
	const size_t numeric_val_bytes  = payload.numericValueBytes();
	const size_t atom_mask_bytes    = payload.atomMaskBytes();

	float* cached_text_features_ptr = training_state_.cached_token_text_features.data;
	const bool has_text_features = (cached_text_features_ptr != nullptr);
	const size_t text_feat_bytes = payload.textFeatureBytes();

	// Acquire double-buffer blocks (A and B)
	auto handleA = scratch_pool->acquire(std::max({input_ids_bytes, numeric_val_bytes, text_feat_bytes}));
	auto handleB = scratch_pool->acquire(std::max({target_ids_bytes, atom_mask_bytes}));

	// --- Round 1: input_ids (block A) + target_ids (block B) ---
	std::memcpy(handleA.data, payload.input_ids.data(), input_ids_bytes);
	CUDA_CHECK(cudaMemcpyAsync(cached_token_ids_ptr, handleA.data,
		input_ids_bytes, cudaMemcpyHostToDevice, stream));

	std::memcpy(handleB.data, payload.target_ids.data(), target_ids_bytes);
	CUDA_CHECK(cudaMemcpyAsync(cached_targets_ptr, handleB.data,
		target_ids_bytes, cudaMemcpyHostToDevice, stream));

	// Sync before reusing blocks — GPU must finish reading A and B
	CUDA_CHECK(cudaStreamSynchronize(stream));

	// --- Round 2: numeric_values (block A) + atom_mask (block B) ---
	std::memcpy(handleA.data, payload.numeric_values.data(), numeric_val_bytes);
	CUDA_CHECK(cudaMemcpyAsync(cached_numeric_values_ptr, handleA.data,
		numeric_val_bytes, cudaMemcpyHostToDevice, stream));

	std::memcpy(handleB.data, payload.atom_mask.data(), atom_mask_bytes);
	CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint8_t*>(cached_atom_mask_ptr), handleB.data,
		atom_mask_bytes, cudaMemcpyHostToDevice, stream));

	// --- Round 3: text_features (block A) ---
	if (has_text_features) {
		CUDA_CHECK(cudaStreamSynchronize(stream));

		std::memcpy(handleA.data, payload.text_features.data(), text_feat_bytes);
		CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<uint16_t*>(cached_text_features_ptr), handleA.data,
			text_feat_bytes, cudaMemcpyHostToDevice, stream));
	}

	// Final sync ensures all DMAs complete before releasing pinned blocks
	CUDA_CHECK(cudaStreamSynchronize(stream));
	scratch_pool->release(handleA);
	scratch_pool->release(handleB);

	auto copy_end = std::chrono::high_resolution_clock::now();
	auto copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
	if constexpr (VerboseLogging::ENABLE_VOCAB_TIMING_LOGS) {
		fprintf(stderr, "[VOCAB_TIMING] GPU copies complete (pinned staging): %.2f ms\n", copy_ms);
	}

	// Store dimensions in TrainingState for downstream consumers
	training_state_.cached_batch_size = static_cast<int>(batch_size);
	training_state_.cached_seq_len = static_cast<int>(seq_len);
	training_state_.cached_num_layers = cfg.num_layers;

	orderLog("computeLossBatch.inputs_copied",
		batch_size, seq_len, total_tokens, 0);

	// ═══════════════════════════════════════════════════════════════════════════
	// AUTOGRAD FORWARD PASS
	// ═══════════════════════════════════════════════════════════════════════════
	orderLog("computeLossBatch.autograd_ctx",
		batch_size, seq_len, total_tokens, 0);
	
	GPUGrimEncoder* autograd_encoder = gpu_encoder;
	ScratchBlockLayer* scratch_block = getScratchBlockLayer();
	
	static uint64_t s_autograd_forward_step = 0;
	const uint64_t autograd_forward_step = s_autograd_forward_step++;
	
	GRIM::Autograd::AutogradContext autograd_ctx = GRIM::Autograd::initAutogradContext(
		&cfg,
		&training_state_,
		autograd_encoder,
		getEmbeddingLayer(),
		getLmHeadLayer(),
		scratch_block,
		getNumericHeadLayer(),
		training_state_.cublas_handle,
		stream,
		payload,
		1.0f,
		autograd_forward_step,
		is_training
	);
	
	// ScratchBlock side-channel data is now accessed directly from BatchPayload
	// inside scratch_block_inject() — no manual wiring needed.

	orderLog("computeLossBatch.forward_start",
		batch_size, seq_len, total_tokens, 0);

	if constexpr (VerboseLogging::ENABLE_VOCAB_TIMING_LOGS) {
		fprintf(stderr, "[VOCAB_TIMING] Starting AUTOGRAD forward pass (batch=%zu, seq=%zu, vocab=%d)\n",
		        batch_size, seq_len, cfg.vocab_size);
	}
	auto fwd_start = std::chrono::high_resolution_clock::now();

	// Execute autograd forward (builds computation graph)
	GRIM::Autograd::ForwardResult fwd_result = GRIM::Autograd::executeAutogradForward(autograd_ctx);

	auto fwd_end = std::chrono::high_resolution_clock::now();
	auto fwd_ms = std::chrono::duration<double, std::milli>(fwd_end - fwd_start).count();
	if constexpr (VerboseLogging::ENABLE_VOCAB_TIMING_LOGS) {
		fprintf(stderr, "[VOCAB_TIMING] AUTOGRAD forward pass complete: %.2f ms\n", fwd_ms);
	}

	if (!fwd_result.success) {
		fprintf(stderr, "[ComputeLossBatch] FATAL: autograd forward failed: %s\n",
		        fwd_result.error_message.c_str());
		orderLog("computeLossBatch.forward_fail",
			batch_size, seq_len, total_tokens, 0);
		throw std::runtime_error("computeLossBatch: autograd forward failed - " + fwd_result.error_message);
	}

	orderLog("computeLossBatch.forward_done",
		batch_size, seq_len, total_tokens, valid_tokens);

	// Build loss config via centralized helper (Issue #142: single conversion point)
	autograd_ctx.loss_config = GRIM::Autograd::buildLossConfig(loss_options_, nullptr);

	static int loss_call_count = 0;
	++loss_call_count;
	if (loss_call_count <= 3) {
		GRIM::Logging::EmitModuleInfo("Loss", 
			"[ComputeLossBatch] Call #" + std::to_string(loss_call_count) +
			": batch=" + std::to_string(batch_size) + " seq=" + std::to_string(seq_len) +
			" valid=" + std::to_string(valid_tokens));
	}

	orderLog("computeLossBatch.loss_start",
		batch_size, seq_len, total_tokens, valid_tokens);

	// === FORWARD DIAGNOSTIC: Verify logits and targets before loss computation ===
	// This helps trace the plateau bug by showing what values reach the loss function
	// NOTE: This block causes GPU sync + D2H copies - VERY EXPENSIVE! Guard with ENABLE_FORWARD_DIAG_LOGS
	if constexpr (VerboseLogging::ENABLE_FORWARD_DIAG_LOGS) {
		cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
		
		// Sample first batch's logits to check
		const size_t sample_tokens = std::min<size_t>(5, total_tokens);
		const size_t sample_vocab = std::min<size_t>(10, static_cast<size_t>(cfg.vocab_size));
		std::vector<float> logit_sample(sample_tokens * cfg.vocab_size);
		std::vector<int> target_sample(sample_tokens);
		
		cudaMemcpy(logit_sample.data(), training_state_.autograd_intermediates.logits_tensor.data, 
		           logit_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(target_sample.data(), reinterpret_cast<int*>(training_state_.cached_targets_tensor.data), 
		           target_sample.size() * sizeof(int), cudaMemcpyDeviceToHost);
		
		fprintf(stderr, "\n[ForwardDiag] ========== PRE-LOSS CHECK ==========\n");
		fprintf(stderr, "[ForwardDiag] batch=%zu seq=%zu valid_tokens=%d\n", batch_size, seq_len, valid_tokens);
		
		// Compute and show logit stats
		float logit_min = std::numeric_limits<float>::infinity();
		float logit_max = -std::numeric_limits<float>::infinity();
		float logit_sum = 0.0f, logit_sq_sum = 0.0f;
		int logit_nan = 0, logit_inf = 0;
		for (size_t i = 0; i < logit_sample.size(); ++i) {
			float v = logit_sample[i];
			if (std::isnan(v)) { logit_nan++; continue; }
			if (std::isinf(v)) { logit_inf++; continue; }
			logit_min = std::min(logit_min, v);
			logit_max = std::max(logit_max, v);
			logit_sum += v;
			logit_sq_sum += v * v;
		}
		const size_t valid_logits = logit_sample.size() - logit_nan - logit_inf;
		const float logit_mean = valid_logits > 0 ? logit_sum / valid_logits : 0.0f;
		const float logit_var = valid_logits > 0 ? (logit_sq_sum / valid_logits) - (logit_mean * logit_mean) : 0.0f;
		const float logit_std = sqrtf(fmaxf(0.0f, logit_var));
		fprintf(stderr, "[ForwardDiag] Logits (sample): min=%.4f max=%.4f mean=%.4f std=%.4f nan=%d inf=%d\n",
		        logit_min, logit_max, logit_mean, logit_std, logit_nan, logit_inf);
		fprintf(stderr, "[ForwardDiag] Logit range: %.4f (max-min), vocab_size=%d\n", logit_max - logit_min, cfg.vocab_size);
		
		// Check for collapsed logits (all nearly the same = model not differentiating)
		if (logit_std < 0.01f) {
			fprintf(stderr, "[ForwardDiag] WARNING: Logits collapsed! std=%.6f < 0.01 → model outputs near-uniform distribution\n", logit_std);
		}
		
		// Show per-position logit distribution analysis

		for (size_t pos = 0; pos < sample_tokens; ++pos) {
			int target = target_sample[pos];
			float* pos_logits = logit_sample.data() + pos * cfg.vocab_size;
			
			// Compute per-position stats
			float pos_min = std::numeric_limits<float>::infinity();
			float pos_max = -std::numeric_limits<float>::infinity();
			float pos_sum = 0.0f, pos_sq_sum = 0.0f;
			for (int v = 0; v < cfg.vocab_size; ++v) {
				float lv = pos_logits[v];
				pos_min = std::min(pos_min, lv);
				pos_max = std::max(pos_max, lv);
				pos_sum += lv;
				pos_sq_sum += lv * lv;
			}
			float pos_mean = pos_sum / cfg.vocab_size;
			float pos_var = (pos_sq_sum / cfg.vocab_size) - (pos_mean * pos_mean);
			float pos_std = sqrtf(fmaxf(0.0f, pos_var));
			
			// Find max logit and its index
			float max_logit = -std::numeric_limits<float>::infinity();
			int max_idx = 0;
			float target_logit = 0.0f;
			for (int v = 0; v < cfg.vocab_size; ++v) {
				if (pos_logits[v] > max_logit) {
					max_logit = pos_logits[v];
					max_idx = v;
				}
				if (v == target && target >= 0) {
					target_logit = pos_logits[v];
				}
			}
			
			// Compute softmax probability for target
			float sum_exp = 0.0f;
			for (int v = 0; v < cfg.vocab_size; ++v) {
				sum_exp += expf(pos_logits[v] - max_logit);  // Numerically stable
			}
			float target_prob = (target >= 0 && target < cfg.vocab_size) 
			                   ? expf(target_logit - max_logit) / sum_exp 
			                   : 0.0f;
			float expected_loss = (target >= 0) ? -logf(target_prob + 1e-10f) : 0.0f;
			
			// z-score of target logit: how many std devs above/below mean?
			float target_zscore = (pos_std > 1e-6f && target >= 0) ? (target_logit - pos_mean) / pos_std : 0.0f;
			
			fprintf(stderr, "  pos=%zu: target=%d logit_range=[%.8f,%.8f] std=%.8f target_logit=%.8f(z=%.8f) max_logit=%.8f(tok=%d) p(target)=%.8f loss=%.8f %s\n",
			        pos, target, pos_min, pos_max, pos_std, target_logit, target_zscore, max_logit, max_idx, target_prob, expected_loss,
			        (target < 0) ? "[MASKED]" : "");
		}
		
		// Summary expectations
		fprintf(stderr, "[ForwardDiag] EXPECTATIONS:\n");
		fprintf(stderr, "  - Logit std should be 0.5-2.0 (not collapsed <0.1, not exploded >10)\n");
		fprintf(stderr, "  - Logit range (max-min) should grow during training as model differentiates\n");
		fprintf(stderr, "  - Target z-score should be POSITIVE and INCREASING during training\n");
		fprintf(stderr, "  - sum_exp ≈ vocab_size=50376 when logits uniform; < vocab_size when peaked\n");
		fprintf(stderr, "  - Random init baseline loss ≈ ln(vocab_size) = %.2f\n", logf(cfg.vocab_size));
		fprintf(stderr, "[ForwardDiag] ============================================\n\n");
	}
	// === END FORWARD DIAGNOSTIC ===

	// ═══════════════════════════════════════════════════════════════════════════
	// Compute loss via centralized autograd path (text cross-entropy)
	// ═══════════════════════════════════════════════════════════════════════════
	auto loss_result = GRIM::Autograd::computeAutogradLoss(autograd_ctx);
	if (!loss_result.success) {
		orderLog("computeLossBatch.loss_fail",
			batch_size, seq_len, total_tokens, valid_tokens);
		throw std::runtime_error("computeLossBatch: loss computation failed - " + loss_result.error_message);
	}

	orderLog("computeLossBatch.loss_done",
		batch_size, seq_len, total_tokens, valid_tokens);

	exception_guard.dismiss();
	return loss_result.loss_value;
}

}  // namespace GRIM

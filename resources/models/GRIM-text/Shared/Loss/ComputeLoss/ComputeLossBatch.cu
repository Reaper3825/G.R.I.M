#include "ComputeLossBatch.hpp"
#include "../../Batching/BatchPayload.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "AutogradLoss.hpp"  // Issue #46: Unified autograd loss system
#include "../LossContext/LossContext.hpp"
#include "../NumericLoss/NumericLoss_GPU.hpp"
#include "../../TeacherLogits/TeacherLogits_GPU.hpp"
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
	const GRIM::Batching::BatchPayload& payload)
{
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
	// GPU COPIES: Transfer padded data from payload to GPU tensors
	// ═══════════════════════════════════════════════════════════════════════════
	auto copy_start = std::chrono::high_resolution_clock::now();
	orderLog("computeLossBatch.copy_inputs",
		batch_size, seq_len, total_tokens, 0);

	cudaStream_t stream = training_state_.stream_ctrl.getPrimaryStream();

	// Token IDs
	int* cached_token_ids_ptr = reinterpret_cast<int*>(training_state_.cached_token_ids_tensor.data);
	if (!cached_token_ids_ptr) {
		throw std::runtime_error("computeLossBatch: cached_token_ids_tensor.data is NULL");
	}
	CUDA_CHECK(cudaMemcpyAsync(
		cached_token_ids_ptr,
		payload.input_ids.data(),
		total_tokens * sizeof(int),
		cudaMemcpyHostToDevice, stream));

	// Targets
	int* cached_targets_ptr = reinterpret_cast<int*>(training_state_.cached_targets_tensor.data);
	if (!cached_targets_ptr) {
		throw std::runtime_error("computeLossBatch: cached_targets_tensor.data is NULL");
	}
	CUDA_CHECK(cudaMemcpyAsync(
		cached_targets_ptr,
		payload.target_ids.data(),
		total_tokens * sizeof(int),
		cudaMemcpyHostToDevice, stream));

	// Numeric values
	float* cached_numeric_values_ptr = training_state_.cached_token_numeric_values.data;
	if (!cached_numeric_values_ptr) {
		throw std::runtime_error("computeLossBatch: cached_token_numeric_values.data is NULL");
	}
	CUDA_CHECK(cudaMemcpyAsync(
		cached_numeric_values_ptr,
		payload.numeric_values.data(),
		total_tokens * sizeof(float),
		cudaMemcpyHostToDevice, stream));

	// Numeric mask (uint8_t stored in float Tensor — cast needed)
	float* cached_numeric_mask_ptr = training_state_.cached_token_numeric_mask.data;
	if (!cached_numeric_mask_ptr) {
		throw std::runtime_error("computeLossBatch: cached_token_numeric_mask.data is NULL");
	}
	CUDA_CHECK(cudaMemcpyAsync(
		reinterpret_cast<uint8_t*>(cached_numeric_mask_ptr),
		payload.numeric_mask.data(),
		total_tokens * sizeof(uint8_t),
		cudaMemcpyHostToDevice, stream));

	// Text features (uint16_t stored in float Tensor — cast needed)
	constexpr int kTextFeatureDim = 16;
	float* cached_text_features_ptr = training_state_.cached_token_text_features.data;
	float* cached_text_mask_ptr = training_state_.cached_token_text_mask.data;
	if (cached_text_features_ptr && cached_text_mask_ptr) {
		CUDA_CHECK(cudaMemcpyAsync(
			reinterpret_cast<uint16_t*>(cached_text_features_ptr),
			payload.text_features.data(),
			total_tokens * kTextFeatureDim * sizeof(uint16_t),
			cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(
			reinterpret_cast<uint8_t*>(cached_text_mask_ptr),
			payload.text_mask.data(),
			total_tokens * sizeof(uint8_t),
			cudaMemcpyHostToDevice, stream));
	}

	auto copy_end = std::chrono::high_resolution_clock::now();
	auto copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
	if constexpr (VerboseLogging::ENABLE_VOCAB_TIMING_LOGS) {
		fprintf(stderr, "[VOCAB_TIMING] GPU copies complete: %.2f ms\n", copy_ms);
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
		scratch_block,
		training_state_.cublas_handle,
		stream,
		static_cast<int>(batch_size),
		static_cast<int>(seq_len),
		1.0f,
		autograd_forward_step,
		true
	);
	autograd_ctx.validate("computeLossBatch");
	
	// Set ScratchBlock input buffers (already on GPU from copies above)
	autograd_ctx.token_numeric_values = training_state_.cached_token_numeric_values.data;
	autograd_ctx.token_numeric_mask = reinterpret_cast<const uint8_t*>(training_state_.cached_token_numeric_mask.data);
	autograd_ctx.token_text_features = reinterpret_cast<const uint16_t*>(training_state_.cached_token_text_features.data);
	autograd_ctx.token_text_mask = reinterpret_cast<const uint8_t*>(training_state_.cached_token_text_mask.data);

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

	// LossScratch struct deleted - scratch buffers managed directly in training_state_

	LossContext::TensorViews ctx_views{};
	ctx_views.logits = training_state_.cached_logits_tensor.data;
	ctx_views.targets = reinterpret_cast<int*>(training_state_.cached_targets_tensor.data);
	ctx_views.teacher_logits = training_state_.teacher_logits.device;
	ctx_views.reference_logits = training_state_.reference_logits.device;
	ctx_views.batch_size = training_state_.cached_batch_size;
	ctx_views.seq_len = training_state_.cached_seq_len;
	ctx_views.valid_tokens = valid_tokens;
	ctx_views.vocab_size = cfg.vocab_size;
	// Only pass sequence_weights if we actually have weights set (count > 0)
	// Otherwise pass nullptr so kernel uses default sample_weight=1.0f
	ctx_views.sequence_weights = (training_state_.sequence_weight_count > 0) 
	                            ? training_state_.sequence_weights_tensor.data 
	                            : nullptr;
	ctx_views.sequence_weight_count = training_state_.sequence_weight_count;
	ctx_views.stream = training_state_.stream_ctrl.getPrimaryStream();

	// Build loss config inline (Issue #136: removed LossContext.cu module)
	Loss::LossConfig loss_config{};
	loss_config.label_smoothing.enabled = loss_options_.label_smoothing_enabled;
	loss_config.label_smoothing.epsilon = loss_options_.label_smoothing_epsilon;
	loss_config.focal.enabled = loss_options_.focal_enabled;
	loss_config.focal.gamma = loss_options_.focal_gamma;
	loss_config.focal.alpha = loss_options_.focal_alpha;
	loss_config.preference.enabled = loss_options_.preference_enabled;
	loss_config.preference.beta = loss_options_.preference_beta;
	loss_config.distillation.enabled = loss_options_.distillation_enabled;
	loss_config.distillation.temperature = loss_options_.distillation_temperature;
	loss_config.distillation.lambda = loss_options_.distillation_lambda;
	loss_config.masking.enabled = loss_options_.masking_enabled;
	loss_config.masking.tag = loss_options_.masking_tag;
	loss_config.entropy_reg.enabled = loss_options_.entropy_reg_enabled;
	loss_config.entropy_reg.lambda = loss_options_.entropy_reg_lambda;
	float* grad_logits_ptr = training_state_.grad_logits_tensor.data;  // Pass pre-allocated buffer
	// If distillation/preference are enabled and no teacher/reference logits are present,
	// mirror the current logits into the teacher/reference buffers. This keeps the call
	// sites simple; a real teacher model can overwrite these buffers before loss compute.
	if (loss_config.distillation.enabled) {
		orderLog("computeLossBatch.distill_copy",
			batch_size, seq_len, total_tokens, valid_tokens);
		if (!TeacherLogits::copyFromDevice(training_state_.teacher_logits,
		                                   training_state_.cached_logits_tensor.data,
		                                   total_tokens,
		                                   cfg.vocab_size,
		                                   training_state_.stream_ctrl.getPrimaryStream())) {
			fprintf(stderr, "[ComputeLossBatch] FATAL: distillation enabled but teacher_logits missing\n");
			throw std::runtime_error("computeLossBatch: distillation enabled without teacher logits");
		}
	}
	if (loss_config.preference.enabled) {
		orderLog("computeLossBatch.preference_copy",
			batch_size, seq_len, total_tokens, valid_tokens);
		if (!TeacherLogits::copyFromDevice(training_state_.reference_logits,
		                                   training_state_.cached_logits_tensor.data,
		                                   total_tokens,
		                                   cfg.vocab_size,
		                                   training_state_.stream_ctrl.getPrimaryStream())) {
			fprintf(stderr, "[ComputeLossBatch] FATAL: preference enabled but reference_logits missing\n");
			throw std::runtime_error("computeLossBatch: preference enabled without reference logits");
		}
	}

	// Skip stream sync/probe here for performance; errors will surface downstream.
	// Disable distillation/preference if teacher/reference logits are unavailable.
	static bool logged_teacher_warn = false;
	static bool logged_ref_warn = false;
	if (loss_config.distillation.enabled && !training_state_.teacher_logits.device) {
		if (!logged_teacher_warn) {
			GRIM::Logging::EmitModuleError("Loss", "[LossConfig] distillation enabled but teacher_logits missing; aborting");
			logged_teacher_warn = true;
		}
		throw std::runtime_error("computeLossBatch: distillation enabled without teacher logits");
	}
	if (loss_config.preference.enabled && !training_state_.reference_logits.device) {
		if (!logged_ref_warn) {
			GRIM::Logging::EmitModuleError("Loss", "[LossConfig] preference KL enabled but reference_logits missing; aborting");
			logged_ref_warn = true;
		}
		throw std::runtime_error("computeLossBatch: preference enabled without reference logits");
	}
	loss_config.limits.max_tokens = logit_limit;
	size_t valid_token_count = static_cast<size_t>(valid_tokens);

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
		
		cudaMemcpy(logit_sample.data(), training_state_.cached_logits_tensor.data, 
		           logit_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(target_sample.data(), reinterpret_cast<int*>(training_state_.cached_targets_tensor.data), 
		           target_sample.size() * sizeof(int), cudaMemcpyDeviceToHost);
		
		fprintf(stderr, "\n[ForwardDiag] ========== PRE-LOSS CHECK ==========\n");
		fprintf(stderr, "[ForwardDiag] batch=%zu seq=%zu valid_tokens=%d\n", batch_size, seq_len, valid_tokens);
		
		// Compute and show logit stats
		float logit_min = 1e30f, logit_max = -1e30f, logit_sum = 0.0f, logit_sq_sum = 0.0f;
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
		fprintf(stderr, "[ForwardDiag] Per-position analysis (first %zu positions):\n", sample_tokens);
		for (size_t pos = 0; pos < sample_tokens; ++pos) {
			int target = target_sample[pos];
			float* pos_logits = logit_sample.data() + pos * cfg.vocab_size;
			
			// Compute per-position stats
			float pos_min = 1e30f, pos_max = -1e30f, pos_sum = 0.0f, pos_sq_sum = 0.0f;
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
			float max_logit = -1e30f;
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
			
			fprintf(stderr, "  pos=%zu: target=%d logit_range=[%.3f,%.3f] std=%.4f target_logit=%.3f(z=%.2f) max_logit=%.3f(tok=%d) p(target)=%.6f loss=%.3f %s\n",
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
	// Issue #46 FIX: UNIFIED AUTOGRAD LOSS SYSTEM
	// ═══════════════════════════════════════════════════════════════════════════
	// Use autograd::cross_entropy_loss() for BOTH:
	//   1. Forward: Compute loss value (returned for logging)
	//   2. Backward: grad_fn attached to loss_tensor enables backward()
	// This replaces the OLD dual-system approach where:
	//   - UnifiedLoss computed loss in forward
	//   - Autograd computed gradients in backward (by recreating the loss computation!)
	// Now loss is computed ONCE and the grad_fn is cached for backward.
	// ═══════════════════════════════════════════════════════════════════════════

	// 'stream' already defined above when setting up autograd context
	
	// ═══════════════════════════════════════════════════════════════════════════
	// Autograd intermediates now stored in training_state_.autograd_intermediates
	// The logits_tensor there has the full grad_fn chain back to embeddings
	// ═══════════════════════════════════════════════════════════════════════════
	
	// Intermediates reference (owned by training_state_)
	auto& intermediates = training_state_.autograd_intermediates;
	
	// Setup gradient buffer for logits (reuse pre-allocated buffer)
	if (!training_state_.grad_logits_tensor.data) {
		throw std::runtime_error("[ComputeLossBatch] grad_logits_tensor.data not allocated!");
	}
	intermediates.logits_tensor.set_grad_from_buffer(
		training_state_.grad_logits_tensor.data
	);
	
	// Compute loss using AUTOGRAD - this attaches grad_fn for backward pass
	// Build autograd LossConfig from the full LossConfig
	autograd::LossConfig ag_loss_config;
	const auto& full_loss_cfg = loss_config;
	// Issue #135 FIX: MUST set _enabled booleans! Previously only set the values
	// but never the enable flags, so features could NEVER activate via config.
	ag_loss_config.focal_enabled = full_loss_cfg.focal.enabled;
	ag_loss_config.focal_alpha = full_loss_cfg.focal.enabled ? full_loss_cfg.focal.alpha : 1.0f;
	ag_loss_config.focal_gamma = full_loss_cfg.focal.enabled ? full_loss_cfg.focal.gamma : 0.0f;
	ag_loss_config.smoothing_enabled = full_loss_cfg.label_smoothing.enabled;
	ag_loss_config.smoothing_epsilon = full_loss_cfg.label_smoothing.enabled ? full_loss_cfg.label_smoothing.epsilon : 0.0f;
	ag_loss_config.entropy_reg_enabled = full_loss_cfg.entropy_reg.enabled;
	ag_loss_config.entropy_reg_lambda = full_loss_cfg.entropy_reg.enabled ? full_loss_cfg.entropy_reg.lambda : 0.0f;
	
	fprintf(stderr, "[ComputeLossBatch] STEP-A: calling autograd::unified_loss (total_tokens=%zu, vocab=%d)...\n",
	        total_tokens, cfg.vocab_size);
	fflush(stderr);
	
	// CRITICAL: Release old loss tensor BEFORE unified_loss() allocates new one.
	// unified_loss() allocates ~4 GB (log_probs + grad_buffer + LogSoftmaxGradFn saved data).
	// Without releasing first, both old and new buffers coexist in GPU memory during
	// unified_loss() execution = ~8 GB for loss alone on a 12 GB GPU → OOM at batch ~22.
	intermediates.loss_tensor.release();
	
	intermediates.loss_tensor = autograd::unified_loss(
		intermediates.logits_tensor,
		reinterpret_cast<int*>(training_state_.cached_targets_tensor.data),
		nullptr,  // valid_mask (nullptr = all valid, padding handled by target=-1)
		static_cast<int>(total_tokens),
		cfg.vocab_size,
		ag_loss_config,
		stream
	);
	
	fprintf(stderr, "[ComputeLossBatch] STEP-B: unified_loss returned, loss_tensor.data=%p grad_fn=%p owns_data=%d\n",
	        (void*)intermediates.loss_tensor.data,
	        (void*)intermediates.loss_tensor.grad_fn.get(),
	        (int)intermediates.loss_tensor.owns_data);
	fflush(stderr);
	
	// Read scalar loss value to host (needed for return value and diagnostics)
	float autograd_loss = 0.0f;
	if (intermediates.loss_tensor.data) {
		cudaMemcpyAsync(&autograd_loss, intermediates.loss_tensor.data, sizeof(float), 
		                cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
	}
	// Issue #62 DEBUG: Verify the loss value matches what unified_loss computed
	fprintf(stderr, "[ComputeLossBatch] READ BACK: autograd_loss=%.6f from loss_tensor.data=%p\n",
	        autograd_loss, (void*)intermediates.loss_tensor.data);
	fprintf(stderr, "[ComputeLossBatch] STEP-C: setting cached values...\n");
	fflush(stderr);
	training_state_.cached_loss_value = autograd_loss;
	training_state_.cached_text_loss = autograd_loss;    // Store for learned weighting backward
	// Note: cached_numeric_loss is set later after numeric_loss_avg is computed
	fprintf(stderr, "[ComputeLossBatch] STEP-D: cached values set, checking spike diag...\n");
	fflush(stderr);
	
	// Diagnostic: If loss is suspiciously high, dump detailed diagnostics
	if (autograd_loss > 20.0f) {
		cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
		fprintf(stderr, "\n[SPIKE_DIAG] ========== LOSS SPIKE DETECTED ==========\n");
		fprintf(stderr, "[SPIKE_DIAG] Loss=%.4f batch_size=%zu seq_len=%zu total_tokens=%zu valid_tokens=%d\n",
		        autograd_loss, batch_size, seq_len, total_tokens, valid_tokens);
		
		// Sample encoder outputs
		const size_t enc_sample_size = std::min<size_t>(1000, total_tokens * cfg.d_model);
		std::vector<float> enc_sample(enc_sample_size);
		cudaMemcpy(enc_sample.data(), training_state_.cached_encoder_output.data, 
		           enc_sample_size * sizeof(float), cudaMemcpyDeviceToHost);
		
		float enc_max = -1e30f, enc_min = 1e30f, enc_sum = 0.0f;
		int enc_nan = 0, enc_inf = 0;
		for (float v : enc_sample) {
			if (std::isnan(v)) { enc_nan++; continue; }
			if (std::isinf(v)) { enc_inf++; continue; }
			enc_max = std::max(enc_max, v);
			enc_min = std::min(enc_min, v);
			enc_sum += v;
		}
		fprintf(stderr, "[SPIKE_DIAG] Encoder out: min=%.4f max=%.4f mean=%.4f nan=%d inf=%d\n",
		        enc_min, enc_max, enc_sum / enc_sample_size, enc_nan, enc_inf);
		
		// Sample logits
		const size_t logit_sample_size = std::min<size_t>(50000, total_tokens * cfg.vocab_size);
		std::vector<float> logit_sample(logit_sample_size);
		cudaMemcpy(logit_sample.data(), training_state_.cached_logits_tensor.data, 
		           logit_sample_size * sizeof(float), cudaMemcpyDeviceToHost);
		
		float logit_max = -1e30f, logit_min = 1e30f, logit_sum = 0.0f;
		int logit_nan = 0, logit_inf = 0;
		for (float v : logit_sample) {
			if (std::isnan(v)) { logit_nan++; continue; }
			if (std::isinf(v)) { logit_inf++; continue; }
			logit_max = std::max(logit_max, v);
			logit_min = std::min(logit_min, v);
			logit_sum += v;
		}
		fprintf(stderr, "[SPIKE_DIAG] Logits: min=%.4f max=%.4f mean=%.4f range=%.4f nan=%d inf=%d\n",
		        logit_min, logit_max, logit_sum / logit_sample_size, logit_max - logit_min, logit_nan, logit_inf);
		
		// Check first few targets
		std::vector<int> target_sample(std::min<size_t>(20, total_tokens));
		cudaMemcpy(target_sample.data(), reinterpret_cast<int*>(training_state_.cached_targets_tensor.data), 
		           target_sample.size() * sizeof(int), cudaMemcpyDeviceToHost);
		fprintf(stderr, "[SPIKE_DIAG] First targets: ");
		for (int t : target_sample) fprintf(stderr, "%d ", t);
		fprintf(stderr, "\n");
		
		// Check logits at target positions
		fprintf(stderr, "[SPIKE_DIAG] Logit at target[i] for first 10 positions:\n");
		for (size_t i = 0; i < std::min<size_t>(10, target_sample.size()); ++i) {
			int target = target_sample[i];
			if (target >= 0 && target < cfg.vocab_size) {
				size_t logit_idx = i * cfg.vocab_size + target;
				if (logit_idx < logit_sample_size) {
					// Also get max logit at this position
					float max_logit = -1e30f;
					int max_idx = 0;
					for (int v = 0; v < std::min(cfg.vocab_size, 1000); ++v) {
						size_t idx = i * cfg.vocab_size + v;
						if (idx < logit_sample_size && logit_sample[idx] > max_logit) {
							max_logit = logit_sample[idx];
							max_idx = v;
						}
					}
					fprintf(stderr, "  pos=%zu target=%d logit[target]=%.4f max_logit=%.4f(tok=%d) diff=%.4f\n",
					        i, target, logit_sample[logit_idx], max_logit, max_idx, max_logit - logit_sample[logit_idx]);
				}
			}
		}
		fprintf(stderr, "[SPIKE_DIAG] ========================================\n\n");
	}
	
	fprintf(stderr, "[ComputeLossBatch] STEP-E: spike diag done, checking isfinite...\n");
	fflush(stderr);
	
	// Autograd loss computation doesn't have a "success" flag - if it didn't throw, it succeeded.
	// Validate the loss value is finite
	if (!std::isfinite(autograd_loss)) {
		fprintf(stderr, "ERROR: autograd loss is non-finite (%.6f). Breaking.\n", autograd_loss);
		orderLog("computeLossBatch.loss_fail",
			batch_size, seq_len, total_tokens, valid_tokens);
		throw std::runtime_error("autograd loss is non-finite");
	}

	fprintf(stderr, "[ComputeLossBatch] STEP-F: loss validated, setting cached_valid_tokens=%d\n", valid_tokens);
	fflush(stderr);
	
	// Scratch buffers remain in training_state_ (no need to update - they're already there)
	training_state_.cached_valid_tokens = valid_tokens;

	orderLog("computeLossBatch.loss_done",
		batch_size, seq_len, total_tokens, valid_tokens);

	fprintf(stderr, "[ComputeLossBatch] STEP-G: entering numeric loss section (enabled=%d)\n",
	        (int)cfg.numeric_head_enabled);
	fflush(stderr);
	
	float numeric_loss_sum = 0.0f;
	int numeric_loss_count = 0;
	if (cfg.numeric_head_enabled) {
		if (!training_state_.cached_numeric_predictions.data ||
		    !training_state_.grad_numeric_tensor.data ||
		    !training_state_.d_numeric_loss_sum.data ||
		    !training_state_.d_numeric_loss_count.data) {
			throw std::runtime_error("computeLossBatch: numeric head enabled but buffers missing");
		}
		NumericLossInputs num_inputs{};
		num_inputs.predictions = training_state_.cached_numeric_predictions.data;
		num_inputs.token_numeric_values = training_state_.cached_token_numeric_values.data;
		num_inputs.token_numeric_mask = reinterpret_cast<uint8_t*>(training_state_.cached_token_numeric_mask.data);
		num_inputs.targets = reinterpret_cast<int*>(training_state_.cached_targets_tensor.data);
		num_inputs.total_tokens = static_cast<int>(total_tokens);
		num_inputs.seq_len = static_cast<int>(seq_len);
		num_inputs.valid_text_tokens = static_cast<int>(valid_tokens);  // Issue #137: match text CE denominator
		num_inputs.huber_delta = cfg.numeric_head_huber_delta;
		num_inputs.log_scale = cfg.numeric_head_log_scale;
		num_inputs.loss_weight = cfg.numeric_head_loss_weight;

		NumericLossOutputs num_outputs{};
		num_outputs.loss_sum = training_state_.d_numeric_loss_sum.data;
		num_outputs.count = reinterpret_cast<int*>(training_state_.d_numeric_loss_count.data);
		num_outputs.grad_predictions = training_state_.grad_numeric_tensor.data;

		if (!launchNumericLoss(num_inputs, num_outputs, training_state_.stream_ctrl.getPrimaryStream())) {
			throw std::runtime_error("computeLossBatch: numeric loss kernel launch failed");
		}

		cudaMemcpyAsync(&numeric_loss_sum, training_state_.d_numeric_loss_sum.data,
		                sizeof(float), cudaMemcpyDeviceToHost,
		                training_state_.stream_ctrl.getPrimaryStream());
		cudaMemcpyAsync(&numeric_loss_count, reinterpret_cast<int*>(training_state_.d_numeric_loss_count.data),
		                sizeof(int), cudaMemcpyDeviceToHost,
		                training_state_.stream_ctrl.getPrimaryStream());
		training_state_.stream_ctrl.syncPrimaryStream();
		if (!std::isfinite(numeric_loss_sum)) {
			numeric_loss_sum = 0.0f;
			numeric_loss_count = 0;
		}
	}

	fprintf(stderr, "[ComputeLossBatch] STEP-H: numeric loss done (sum=%.6f count=%d), entering learned weighting\n",
	        numeric_loss_sum, numeric_loss_count);
	fflush(stderr);
	
	// Learned loss weighting (homoscedastic uncertainty)
	// L_total = L_text / (2*σ_text²) + L_numeric / (2*σ_numeric²) + 0.5*log(σ_text²) + 0.5*log(σ_numeric²)
	// We learn log_var = log(σ²), so: L = L / (2*exp(log_var)) + 0.5*log_var
	float avg_loss = 0.0f;
	float weight_text = 1.0f;
	float weight_numeric = cfg.numeric_head_loss_weight;
	float reg_text = 0.0f;
	float reg_numeric = 0.0f;
	float log_var_text_val = 0.0f;
	float log_var_numeric_val = 0.0f;
	
	const bool use_learned_weights = training_state_.log_var_text.data && training_state_.log_var_numeric.data;
	
	if (use_learned_weights) {
		// Read learned log-variances from GPU
		cudaMemcpyAsync(&log_var_text_val, training_state_.log_var_text.data, 
		                sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaMemcpyAsync(&log_var_numeric_val, training_state_.log_var_numeric.data,
		                sizeof(float), cudaMemcpyDeviceToHost, stream);
		cudaStreamSynchronize(stream);
		
		// Clamp log_var to prevent numerical issues
		log_var_text_val = std::clamp(log_var_text_val, -4.0f, 4.0f);      // σ² in [0.018, 54.6]
		log_var_numeric_val = std::clamp(log_var_numeric_val, -4.0f, 4.0f);
		
		// Compute weights: 1 / (2 * exp(log_var)) = 0.5 * exp(-log_var)
		weight_text = 0.5f * std::exp(-log_var_text_val);
		weight_numeric = 0.5f * std::exp(-log_var_numeric_val);
		
		// Regularization terms: 0.5 * log_var (prevents σ → ∞)
		reg_text = 0.5f * log_var_text_val;
		reg_numeric = 0.5f * log_var_numeric_val;
	}
	
	const float numeric_loss_avg = (numeric_loss_count > 0) 
		? (numeric_loss_sum / numeric_loss_count) : 0.0f;
	
	// Store for learned weighting backward pass
	training_state_.cached_numeric_loss = numeric_loss_avg;
	training_state_.cached_numeric_count = numeric_loss_count;  // Issue #137: store for weight grad normalization
	
	avg_loss = weight_text * autograd_loss + reg_text
	         + weight_numeric * numeric_loss_avg + reg_numeric;
	
	// Log both loss components separately for debugging
	fprintf(stderr, "[LossComponents] text_ce=%.4f (w=%.3f) numeric=%.4f (w=%.3f) reg=%.4f total=%.4f\n",
	        autograd_loss, weight_text, numeric_loss_avg, weight_numeric, 
	        reg_text + reg_numeric, avg_loss);
	
	if (!std::isfinite(avg_loss)) {
		fprintf(stderr, "[ComputeLossBatch] FATAL: avg_loss is non-finite (autograd=%.6f, numeric=%.6f)\n",
		        autograd_loss, numeric_loss_avg);
		throw std::runtime_error("computeLossBatch: avg_loss is non-finite");
	}
	fprintf(stderr, "[ComputeLossBatch] STEP-J: returning avg_loss=%.6f\n", avg_loss);
	fflush(stderr);
	return avg_loss;
}

}  // namespace GRIM

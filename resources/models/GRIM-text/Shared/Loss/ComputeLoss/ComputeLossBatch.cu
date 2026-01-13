#include "ComputeLossBatch.hpp"

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
#include "../../Layers/ForwardOps/ForwardOps_Orchestrator.hpp"
#include "../../Layers/ForwardOps/ForwardOps_Logging.hpp"
#include "ComputeLossHost_GPU.hpp"
#include "../LossContext/LossContext.hpp"
#include "../NumericLoss/NumericLoss_GPU.hpp"
#include "../../TeacherLogits/TeacherLogits_GPU.hpp"
#include "../../LogRecorder/LogRecorder.hpp"

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

BatchPreparationResult prepareLossBatchInputs(
	TrainingState& training_state,
	const std::vector<std::vector<int>>& batch_input_ids,
	const std::vector<std::vector<int>>& batch_target_ids,
	const std::vector<std::vector<float>>& batch_numeric_values,
	const std::vector<std::vector<uint8_t>>& batch_numeric_mask,
	const std::vector<std::vector<uint16_t>>& batch_text_features,
	const std::vector<std::vector<uint8_t>>& batch_text_mask,
	size_t max_cached_batch,
	size_t max_cached_seq_len)
{
	constexpr int kTextFeatureDim = 16;  // Must match GRIM::Tokenizer::kTextFeatureDim
	BatchPreparationResult result{};
	result.batch_size = batch_input_ids.size();

	for (const auto& seq : batch_input_ids) {
		result.max_seq_len = std::max(result.max_seq_len, seq.size());
	}

	if (result.batch_size == 0 || result.max_seq_len == 0) {
		result.fits_in_cache = false;
		return result;
	}

	if (result.batch_size > max_cached_batch ||
		result.max_seq_len > max_cached_seq_len) {
		result.fits_in_cache = false;
		return result;
	}

	const size_t total_tokens = result.batch_size * result.max_seq_len;
	
	// PERFORMANCE FIX: Reuse pre-allocated buffers from TrainingState (Rule 22)
	const size_t text_feat_size = total_tokens * kTextFeatureDim;
	
	if (training_state.batch_prep_capacity < total_tokens) {
		// Only reallocate if capacity increased
		training_state.batch_prep_input_ids.resize(total_tokens);
		training_state.batch_prep_target_ids.resize(total_tokens);
		training_state.batch_prep_numeric_values.resize(total_tokens);
		training_state.batch_prep_numeric_mask.resize(total_tokens);
		training_state.batch_prep_text_features.resize(text_feat_size);
		training_state.batch_prep_text_mask.resize(total_tokens);
		training_state.batch_prep_sequence_lengths.resize(result.batch_size);
		training_state.batch_prep_capacity = total_tokens;
	}
	
	// Zero-fill (reuse existing capacity)
	std::fill(training_state.batch_prep_input_ids.begin(), training_state.batch_prep_input_ids.begin() + total_tokens, 0);
	std::fill(training_state.batch_prep_target_ids.begin(), training_state.batch_prep_target_ids.begin() + total_tokens, -1);
	std::fill(training_state.batch_prep_numeric_values.begin(), training_state.batch_prep_numeric_values.begin() + total_tokens, 0.0f);
	std::fill(training_state.batch_prep_numeric_mask.begin(), training_state.batch_prep_numeric_mask.begin() + total_tokens, 0);
	std::fill(training_state.batch_prep_text_features.begin(), training_state.batch_prep_text_features.begin() + text_feat_size, 0);
	std::fill(training_state.batch_prep_text_mask.begin(), training_state.batch_prep_text_mask.begin() + total_tokens, 0);
	
	// Assign result to reference cached buffers
	result.padded_input_ids = training_state.batch_prep_input_ids;
	result.padded_target_ids = training_state.batch_prep_target_ids;
	result.padded_numeric_values = training_state.batch_prep_numeric_values;
	result.padded_numeric_mask = training_state.batch_prep_numeric_mask;
	result.padded_text_features = training_state.batch_prep_text_features;
	result.padded_text_mask = training_state.batch_prep_text_mask;
	
	// BUG FIX: Reuse TrainingState buffer instead of creating new local vector
	// Previously: result.sequence_lengths.resize(result.batch_size, 0);
	// This created a NEW vector, ignoring the TrainingState cache at line 147
	training_state.batch_prep_sequence_lengths.resize(result.batch_size, 0);
	result.sequence_lengths = training_state.batch_prep_sequence_lengths;

	for (size_t b = 0; b < result.batch_size; ++b) {
		const size_t seq_len = std::min(batch_input_ids[b].size(), result.max_seq_len);
		const size_t target_len = (b < batch_target_ids.size())
			                          ? std::min(batch_target_ids[b].size(), result.max_seq_len)
			                          : seq_len;
		const size_t numeric_len = (b < batch_numeric_values.size())
			                           ? std::min(batch_numeric_values[b].size(), result.max_seq_len)
			                           : 0;
		const size_t numeric_mask_len = (b < batch_numeric_mask.size())
			                                ? std::min(batch_numeric_mask[b].size(), result.max_seq_len)
			                                : 0;
		if (numeric_len < seq_len || numeric_mask_len < seq_len) {
			throw std::runtime_error("prepareLossBatchInputs: numeric side-channel length mismatch");
		}

		// IMPORTANT: valid_tokens must count ONLY unmasked targets (>=0). Many call sites
		// keep targets padded with -1; assuming target_len-1 would massively over-count
		// valid tokens, collapsing avg_loss (and gradient scale) toward 0.
		int valid_len = 0;
		if (b < batch_target_ids.size()) {
			for (size_t t = 0; t < target_len; ++t) {
				if (batch_target_ids[b][t] >= 0) {
					++valid_len;
				}
			}
			// We will mask the final position below.
			if (target_len > 0 && batch_target_ids[b][target_len - 1] >= 0) {
				--valid_len;
			}
		} else {
			valid_len = static_cast<int>(seq_len > 0 ? seq_len - 1 : 0);
		}
		result.sequence_lengths[b] = std::max(valid_len, 0);

		for (size_t t = 0; t < seq_len; ++t) {
			result.padded_input_ids[b * result.max_seq_len + t] = batch_input_ids[b][t];
		}
		for (size_t t = 0; t < seq_len; ++t) {
			const size_t offset = b * result.max_seq_len + t;
			result.padded_numeric_values[offset] = batch_numeric_values[b][t];
			result.padded_numeric_mask[offset] = batch_numeric_mask[b][t];
		}
		// GRMT v4: copy text features (kTextFeatureDim values per token)
		if (b < batch_text_features.size() && b < batch_text_mask.size()) {
			const size_t text_feat_len = batch_text_features[b].size() / kTextFeatureDim;
			const size_t copy_len = std::min(text_feat_len, seq_len);
			for (size_t t = 0; t < copy_len; ++t) {
				const size_t dst_offset = (b * result.max_seq_len + t) * kTextFeatureDim;
				const size_t src_offset = t * kTextFeatureDim;
				for (int f = 0; f < kTextFeatureDim; ++f) {
					result.padded_text_features[dst_offset + f] = batch_text_features[b][src_offset + f];
				}
				result.padded_text_mask[b * result.max_seq_len + t] = batch_text_mask[b][t];
			}
		}

		if (b < batch_target_ids.size()) {
			for (size_t t = 0; t < target_len; ++t) {
				result.padded_target_ids[b * result.max_seq_len + t] = batch_target_ids[b][t];
			}
			if (target_len > 0) {
				// Mask the final position so loss/gradients ignore the window boundary.
				result.padded_target_ids[b * result.max_seq_len + target_len - 1] = -1;
			}
		}
	}

	return result;
}

float LanguageModel::computeLossBatch(
	const std::vector<std::vector<int>>& batch_input_ids,
	const std::vector<std::vector<int>>& batch_target_ids,
	const std::vector<std::vector<float>>& batch_numeric_values,
	const std::vector<std::vector<uint8_t>>& batch_numeric_mask,
	const std::vector<std::vector<uint16_t>>& batch_text_features,
	const std::vector<std::vector<uint8_t>>& batch_text_mask)
{
	orderLog("computeLossBatch.enter",
		batch_input_ids.size(), 0, 0, 0);

	// Ensure training state is initialized before computing batch loss
	if (!training_state_.initialized) {
		orderLog("computeLossBatch.init_start",
			batch_input_ids.size(), 0, 0, 0);
		try {
			fprintf(stderr, "[computeLossBatch] Training state not initialized, attempting initTrainingState()\n");
			const_cast<LanguageModel*>(this)->initTrainingState();
			if (!training_state_.initialized) {
				fprintf(stderr, "[computeLossBatch] FATAL: initTrainingState() completed but flag still false\n");
				throw std::runtime_error("computeLossBatch: training state initialization failed");
			}
			fprintf(stderr, "[computeLossBatch] Training state initialized successfully\n");
		} catch (const std::exception& e) {
			fprintf(stderr, "[computeLossBatch] FATAL: Failed to initialize training state: %s\n", e.what());
			throw std::runtime_error(std::string("computeLossBatch: initTrainingState() failed: ") + e.what());
		} catch (...) {
			fprintf(stderr, "[computeLossBatch] FATAL: Unknown error during training state initialization\n");
			throw std::runtime_error("computeLossBatch: initTrainingState() failed with unknown error");
		}
		orderLog("computeLossBatch.init_done",
			batch_input_ids.size(), 0, 0, 0);
	}

	if (batch_input_ids.empty() || batch_target_ids.empty()) {
		fprintf(stderr,
			"[ComputeLossBatch] FATAL: empty batch (inputs=%zu, targets=%zu)\n",
			batch_input_ids.size(), batch_target_ids.size());
		throw std::runtime_error("computeLossBatch: empty batch");
	}
	if (batch_numeric_values.size() != batch_input_ids.size() ||
		batch_numeric_mask.size() != batch_input_ids.size()) {
		throw std::runtime_error("computeLossBatch: numeric side-channel batch size mismatch");
	}
	for (size_t b = 0; b < batch_input_ids.size(); ++b) {
		if (batch_numeric_values[b].size() != batch_input_ids[b].size() ||
			batch_numeric_mask[b].size() != batch_input_ids[b].size()) {
			throw std::runtime_error("computeLossBatch: numeric side-channel sequence length mismatch");
		}
	}

	const auto& cfg = getConfig();
	const size_t cache_batch_limit = static_cast<size_t>(std::max(1, cfg.max_cached_batch));
	const size_t cache_seq_limit = static_cast<size_t>(
		std::max(1, std::min(cfg.max_seq_len, cfg.max_cached_seq_len)));

	orderLog("computeLossBatch.prep_start",
		batch_input_ids.size(), 0, 0, 0);
	auto prep_start = std::chrono::high_resolution_clock::now();
	const auto prep = prepareLossBatchInputs(
		training_state_,
		batch_input_ids,
		batch_target_ids,
		batch_numeric_values,
		batch_numeric_mask,
		batch_text_features,
		batch_text_mask,
		cache_batch_limit,
		cache_seq_limit);
	auto prep_end = std::chrono::high_resolution_clock::now();
	auto prep_ms = std::chrono::duration<double, std::milli>(prep_end - prep_start).count();
	fprintf(stderr, "[VOCAB_TIMING] prepareLossBatchInputs: %.2f ms\n", prep_ms);

	orderLog("computeLossBatch.prep",
		prep.batch_size, prep.max_seq_len,
		prep.batch_size * prep.max_seq_len,
		std::accumulate(prep.sequence_lengths.begin(), prep.sequence_lengths.end(), 0));

	if (!prep.fits_in_cache) {
		const char* reason = (batch_input_ids.size() > cache_batch_limit) 
			? "BATCH_SIZE" 
			: (prep.max_seq_len > cache_seq_limit) ? "SEQ_LEN" : "UNKNOWN";
		fprintf(stderr,
			"[ComputeLossBatch] FATAL: batch does not fit cache (%s): batch=%zu [limit=%zu], seq_len=%zu [limit=%zu]\n",
			reason,
			batch_input_ids.size(), cache_batch_limit,
			prep.max_seq_len, cache_seq_limit);
		orderLog("computeLossBatch.prep_fail",
			prep.batch_size, prep.max_seq_len,
			prep.batch_size * prep.max_seq_len, 0);
		throw std::runtime_error("computeLossBatch: batch does not fit cache; fallback disabled");
	}

	if (!training_state_.initialized) {
		initTrainingState();
	}

	GRIM::ForwardOps::LogUnexpectedGradState(training_state_, "computeLossBatch");

	GPUGrimEncoder* gpu_encoder = nullptr;
	EmbeddingRuntime* embedding_runtime = nullptr;
	try {
		gpu_encoder = &getGpuEncoder();
		embedding_runtime = &getGpuEmbedder();
	} catch (const std::exception& ex) {
		fprintf(stderr, "[ComputeLossBatch] FATAL: GPU components not initialized: %s\n", ex.what());
		throw std::runtime_error("computeLossBatch: GPU components not initialized");
	}
	if (!gpu_encoder || !embedding_runtime) {
		fprintf(stderr, "[ComputeLossBatch] FATAL: GPU components not initialized\n");
		throw std::runtime_error("computeLossBatch: GPU components not initialized");
	}

	const size_t batch_size = prep.batch_size;
	const size_t seq_len = prep.max_seq_len;
	const size_t total_tokens = batch_size * seq_len;
	const size_t logit_limit = training_state_.max_logit_tokens > 0
		? training_state_.max_logit_tokens
		: training_state_.max_cached_tokens;
	if (total_tokens > logit_limit) {
		fprintf(stderr,
		        "[ComputeLossBatch] FATAL: total_tokens=%zu exceeds logit buffer capacity=%zu\n",
		        total_tokens,
		        logit_limit);
		throw std::runtime_error("computeLossBatch: token count exceeds logit buffer capacity");
	}

	// PRE-VALIDATE: Calculate valid_tokens BEFORE forward pass (CPU data only)
	const int valid_tokens = std::accumulate(
		prep.sequence_lengths.begin(),
		prep.sequence_lengths.end(),
		0);
	
	if (valid_tokens <= 0) {
		std::cerr << "[ComputeLossBatch] FATAL: valid_tokens=" << valid_tokens 
		          << " (must be > 0 before forward pass)" << std::endl << std::flush;
		orderLog("computeLossBatch.valid_tokens_fail",
			batch_size, seq_len, total_tokens, valid_tokens);
		throw std::runtime_error("computeLossBatch: valid_tokens <= 0");
	}

	// Copy input/target data to GPU
	auto copy_start = std::chrono::high_resolution_clock::now();
	orderLog("computeLossBatch.copy_inputs",
		batch_size, seq_len, total_tokens, 0);
	fprintf(stderr, "[GPU_COPY] Copying token_ids: dst=%p src=%p size=%zu bytes\n",
	        training_state_.cached_token_ids,
	        prep.padded_input_ids.data(),
	        total_tokens * sizeof(int));
	cudaMemcpyAsync(
		training_state_.cached_token_ids,
		prep.padded_input_ids.data(),
		total_tokens * sizeof(int),
		cudaMemcpyHostToDevice,
		training_state_.stream_ctrl.getPrimaryStream());
	fprintf(stderr, "[GPU_COPY] token_ids copy initiated\n");

	orderLog("computeLossBatch.copy_targets",
		batch_size, seq_len, total_tokens, 0);
	fprintf(stderr, "[GPU_COPY] Copying targets: dst=%p src=%p size=%zu bytes\n",
	        training_state_.cached_targets,
	        prep.padded_target_ids.data(),
	        total_tokens * sizeof(int));
	cudaMemcpyAsync(
		training_state_.cached_targets,
		prep.padded_target_ids.data(),
		total_tokens * sizeof(int),
		cudaMemcpyHostToDevice,
		training_state_.stream_ctrl.getPrimaryStream());
	fprintf(stderr, "[GPU_COPY] targets copy initiated\n");

	orderLog("computeLossBatch.copy_numeric",
		batch_size, seq_len, total_tokens, 0);
	fprintf(stderr, "[GPU_COPY] Checking numeric buffers: values=%p mask=%p\n",
	        training_state_.cached_token_numeric_values,
	        training_state_.cached_token_numeric_mask);
	if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
		throw std::runtime_error("computeLossBatch: numeric side-channel buffers not initialized");
	}
	fprintf(stderr, "[GPU_COPY] Copying numeric_values: dst=%p src=%p size=%zu bytes\n",
	        training_state_.cached_token_numeric_values,
	        prep.padded_numeric_values.data(),
	        total_tokens * sizeof(float));
	cudaMemcpyAsync(
		training_state_.cached_token_numeric_values,
		prep.padded_numeric_values.data(),
		total_tokens * sizeof(float),
		cudaMemcpyHostToDevice,
		training_state_.stream_ctrl.getPrimaryStream());
	fprintf(stderr, "[GPU_COPY] numeric_values copy initiated\n");
	
	fprintf(stderr, "[GPU_COPY] Copying numeric_mask: dst=%p src=%p size=%zu bytes\n",
	        training_state_.cached_token_numeric_mask,
	        prep.padded_numeric_mask.data(),
	        total_tokens * sizeof(uint8_t));
	cudaMemcpyAsync(
		training_state_.cached_token_numeric_mask,
		prep.padded_numeric_mask.data(),
		total_tokens * sizeof(uint8_t),
		cudaMemcpyHostToDevice,
		training_state_.stream_ctrl.getPrimaryStream());
	fprintf(stderr, "[GPU_COPY] numeric_mask copy initiated\n");

	// GRMT v4: copy text features
	constexpr int kTextFeatureDim = 16;  // Must match GRIM::Tokenizer::kTextFeatureDim
	fprintf(stderr, "[GPU_COPY] Checking text feature buffers: features=%p mask=%p\n",
	        training_state_.cached_token_text_features,
	        training_state_.cached_token_text_mask);
	if (training_state_.cached_token_text_features && training_state_.cached_token_text_mask) {
		fprintf(stderr, "[GPU_COPY] Copying text_features: dst=%p src=%p size=%zu bytes\n",
		        training_state_.cached_token_text_features,
		        prep.padded_text_features.data(),
		        total_tokens * kTextFeatureDim * sizeof(uint16_t));
		cudaMemcpyAsync(
			training_state_.cached_token_text_features,
			prep.padded_text_features.data(),
			total_tokens * kTextFeatureDim * sizeof(uint16_t),
			cudaMemcpyHostToDevice,
			training_state_.stream_ctrl.getPrimaryStream());
		fprintf(stderr, "[GPU_COPY] text_features copy initiated\n");
		
		fprintf(stderr, "[GPU_COPY] Copying text_mask: dst=%p src=%p size=%zu bytes\n",
		        training_state_.cached_token_text_mask,
		        prep.padded_text_mask.data(),
		        total_tokens * sizeof(uint8_t));
		cudaMemcpyAsync(
			training_state_.cached_token_text_mask,
			prep.padded_text_mask.data(),
			total_tokens * sizeof(uint8_t),
			cudaMemcpyHostToDevice,
			training_state_.stream_ctrl.getPrimaryStream());
		fprintf(stderr, "[GPU_COPY] text_mask copy initiated\n");
	}
	auto copy_end = std::chrono::high_resolution_clock::now();
	auto copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
	fprintf(stderr, "[VOCAB_TIMING] GPU copies complete: %.2f ms\n", copy_ms);

	training_state_.cached_batch_size = static_cast<int>(batch_size);
	training_state_.cached_seq_len = static_cast<int>(seq_len);
	training_state_.cached_num_layers = cfg.num_layers;

	orderLog("computeLossBatch.inputs_copied",
		batch_size, seq_len, total_tokens, 0);

	orderLog("computeLossBatch.forward_ctx",
		batch_size, seq_len, total_tokens, 0);
	auto fwd_ctx = GRIM::Forward::initForwardContext(
		*this,
		GRIM::Forward::ForwardMode::TrainingFull,
		static_cast<int>(batch_size),
		static_cast<int>(seq_len),
		GRIM::Forward::ForwardLogitsTarget::FullSequence,
		nullptr,
		true,
		-1,
		-1,
		scratch_block_layer_ && scratch_block_layer_->isEnabled(),
		cfg.activation_quantization.enabled,
		true);

	orderLog("computeLossBatch.forward_start",
		batch_size, seq_len, total_tokens, 0);

	fprintf(stderr, "[VOCAB_TIMING] Starting forward pass (batch=%zu, seq=%zu, vocab=%d)\n",
	        batch_size, seq_len, cfg.vocab_size);
	auto fwd_start = std::chrono::high_resolution_clock::now();
	const auto fwd_status = GRIM::Forward::executeForward(fwd_ctx);
	

	auto fwd_end = std::chrono::high_resolution_clock::now();
	auto fwd_ms = std::chrono::duration<double, std::milli>(fwd_end - fwd_start).count();
	fprintf(stderr, "[VOCAB_TIMING] Forward pass complete: %.2f ms\n", fwd_ms);
	if (fwd_status != GRIM::Forward::ForwardStatus::SUCCESS) {
		fprintf(stderr, "[ComputeLossBatch] FATAL: forward failed: %s (%s)\n",
		        GRIM::Forward::statusToString(fwd_status),
		        fwd_ctx.error_message.c_str());
		orderLog("computeLossBatch.forward_fail",
			batch_size, seq_len, total_tokens, 0);
		throw std::runtime_error("computeLossBatch: forward failed");
	}

	orderLog("computeLossBatch.forward_done",
		batch_size, seq_len, total_tokens, valid_tokens);

	LossScratch scratch{
		training_state_.d_loss_scratch,
		training_state_.d_loss_sum_scratch,
		training_state_.loss_scratch_capacity};

	LossContext::TensorViews ctx_views{};
	ctx_views.logits = training_state_.cached_logits;
	ctx_views.targets = training_state_.cached_targets;
	ctx_views.teacher_logits = training_state_.teacher_logits.device;
	ctx_views.reference_logits = training_state_.reference_logits.device;
	ctx_views.batch_size = training_state_.cached_batch_size;
	ctx_views.seq_len = training_state_.cached_seq_len;
	ctx_views.valid_tokens = valid_tokens;
	ctx_views.vocab_size = cfg.vocab_size;
	// Only pass sequence_weights if we actually have weights set (count > 0)
	// Otherwise pass nullptr so kernel uses default sample_weight=1.0f
	ctx_views.sequence_weights = (training_state_.sequence_weight_count > 0) 
	                            ? training_state_.sequence_weights 
	                            : nullptr;
	ctx_views.sequence_weight_count = training_state_.sequence_weight_count;
	ctx_views.stream = training_state_.stream_ctrl.getPrimaryStream();

	LossComputationInputs loss_inputs{};
	loss_inputs.context = LossContext::MakeContext(ctx_views);
	loss_inputs.config = LossContext::BuildLossConfig(loss_options_, false);
	loss_inputs.grad_logits = training_state_.grad_logits;  // Pass pre-allocated buffer
	// If distillation/preference are enabled and no teacher/reference logits are present,
	// mirror the current logits into the teacher/reference buffers. This keeps the call
	// sites simple; a real teacher model can overwrite these buffers before loss compute.
	if (loss_inputs.config.distillation.enabled) {
		orderLog("computeLossBatch.distill_copy",
			batch_size, seq_len, total_tokens, valid_tokens);
		if (!TeacherLogits::copyFromDevice(training_state_.teacher_logits,
		                                   training_state_.cached_logits,
		                                   total_tokens,
		                                   cfg.vocab_size,
		                                   training_state_.stream_ctrl.getPrimaryStream())) {
			fprintf(stderr, "[ComputeLossBatch] FATAL: distillation enabled but teacher_logits missing\n");
			throw std::runtime_error("computeLossBatch: distillation enabled without teacher logits");
		}
	}
	if (loss_inputs.config.preference.enabled) {
		orderLog("computeLossBatch.preference_copy",
			batch_size, seq_len, total_tokens, valid_tokens);
		if (!TeacherLogits::copyFromDevice(training_state_.reference_logits,
		                                   training_state_.cached_logits,
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
	if (loss_inputs.config.distillation.enabled && !training_state_.teacher_logits.device) {
		if (!logged_teacher_warn) {
			GRIM::Logging::EmitModuleError("Loss", "[LossConfig] distillation enabled but teacher_logits missing; aborting");
			logged_teacher_warn = true;
		}
		throw std::runtime_error("computeLossBatch: distillation enabled without teacher logits");
	}
	if (loss_inputs.config.preference.enabled && !training_state_.reference_logits.device) {
		if (!logged_ref_warn) {
			GRIM::Logging::EmitModuleError("Loss", "[LossConfig] preference KL enabled but reference_logits missing; aborting");
			logged_ref_warn = true;
		}
		throw std::runtime_error("computeLossBatch: preference enabled without reference logits");
	}
	loss_inputs.config.limits.max_tokens = logit_limit;
	loss_inputs.valid_token_count = static_cast<size_t>(valid_tokens);

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
	{
		cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
		
		// Sample first batch's logits to check
		const size_t sample_tokens = std::min<size_t>(5, total_tokens);
		const size_t sample_vocab = std::min<size_t>(10, static_cast<size_t>(cfg.vocab_size));
		std::vector<float> logit_sample(sample_tokens * cfg.vocab_size);
		std::vector<int> target_sample(sample_tokens);
		
		cudaMemcpy(logit_sample.data(), training_state_.cached_logits, 
		           logit_sample.size() * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(target_sample.data(), training_state_.cached_targets, 
		           target_sample.size() * sizeof(int), cudaMemcpyDeviceToHost);
		
		fprintf(stderr, "\n[ForwardDiag] ========== PRE-LOSS CHECK ==========\n");
		fprintf(stderr, "[ForwardDiag] batch=%zu seq=%zu valid_tokens=%d\n", batch_size, seq_len, valid_tokens);
		
		// Compute and show logit stats
		float logit_min = 1e30f, logit_max = -1e30f, logit_sum = 0.0f;
		int logit_nan = 0, logit_inf = 0;
		for (size_t i = 0; i < logit_sample.size(); ++i) {
			float v = logit_sample[i];
			if (std::isnan(v)) { logit_nan++; continue; }
			if (std::isinf(v)) { logit_inf++; continue; }
			logit_min = std::min(logit_min, v);
			logit_max = std::max(logit_max, v);
			logit_sum += v;
		}
		const size_t valid_logits = logit_sample.size() - logit_nan - logit_inf;
		fprintf(stderr, "[ForwardDiag] Logits (sample): min=%.4f max=%.4f mean=%.4f nan=%d inf=%d\n",
		        logit_min, logit_max, valid_logits > 0 ? logit_sum / valid_logits : 0.0f, logit_nan, logit_inf);
		fprintf(stderr, "[ForwardDiag] EXPECTED: Logits should be in range [-20, 20], mean near 0\n");
		
		// Show per-position details
		fprintf(stderr, "[ForwardDiag] First %zu positions:\n", sample_tokens);
		for (size_t pos = 0; pos < sample_tokens; ++pos) {
			int target = target_sample[pos];
			float* pos_logits = logit_sample.data() + pos * cfg.vocab_size;
			
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
			
			fprintf(stderr, "  pos=%zu: target=%d target_logit=%.3f max_logit=%.3f(tok=%d) p(target)=%.6f expected_loss=%.3f %s\n",
			        pos, target, target_logit, max_logit, max_idx, target_prob, expected_loss,
			        (target < 0) ? "[MASKED]" : "");
		}
		fprintf(stderr, "[ForwardDiag] EXPECTED: p(target) should INCREASE during training (loss decrease)\n");
		fprintf(stderr, "[ForwardDiag] EXPECTED: Random init baseline loss ≈ ln(vocab_size) = %.2f\n", logf(cfg.vocab_size));
		fprintf(stderr, "[ForwardDiag] ============================================\n\n");
	}
	// === END FORWARD DIAGNOSTIC ===

	const auto loss_result = computeLossHost(loss_inputs, scratch);
	
	// Loss pipeline already syncs for host-visible results.

	// DIAGNOSTIC: If loss is suspiciously high, dump detailed diagnostics
	if (loss_result.average_loss > 20.0f) {
		cudaStreamSynchronize(training_state_.stream_ctrl.getPrimaryStream());
		fprintf(stderr, "\n[SPIKE_DIAG] ========== LOSS SPIKE DETECTED ==========\n");
		fprintf(stderr, "[SPIKE_DIAG] Loss=%.4f batch_size=%zu seq_len=%zu total_tokens=%zu valid_tokens=%d\n",
		        loss_result.average_loss, batch_size, seq_len, total_tokens, valid_tokens);
		
		// Sample encoder outputs
		const size_t enc_sample_size = std::min<size_t>(1000, total_tokens * cfg.d_model);
		std::vector<float> enc_sample(enc_sample_size);
		cudaMemcpy(enc_sample.data(), training_state_.cached_encoder_outputs, 
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
		cudaMemcpy(logit_sample.data(), training_state_.cached_logits, 
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
		cudaMemcpy(target_sample.data(), training_state_.cached_targets, 
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
	
	if (!loss_result.success) {
		fprintf(stderr, "ERROR: computeLossHost failed. Breaking instead of fallback.\n");
		orderLog("computeLossBatch.loss_fail",
			batch_size, seq_len, total_tokens, valid_tokens);
		throw std::runtime_error("computeLossHost failed: fallback disabled");
	}

	training_state_.d_loss_scratch = scratch.loss_values;
	training_state_.d_loss_sum_scratch = scratch.loss_accumulator;
	training_state_.loss_scratch_capacity = scratch.capacity;
	training_state_.cached_valid_tokens = valid_tokens;

	orderLog("computeLossBatch.loss_done",
		batch_size, seq_len, total_tokens, valid_tokens);

	float numeric_loss_sum = 0.0f;
	int numeric_loss_count = 0;
	if (cfg.numeric_head_enabled) {
		if (!training_state_.cached_numeric_predictions ||
		    !training_state_.grad_numeric_predictions ||
		    !training_state_.d_numeric_loss_sum ||
		    !training_state_.d_numeric_loss_count) {
			throw std::runtime_error("computeLossBatch: numeric head enabled but buffers missing");
		}
		NumericLossInputs num_inputs{};
		num_inputs.predictions = training_state_.cached_numeric_predictions;
		num_inputs.token_numeric_values = training_state_.cached_token_numeric_values;
		num_inputs.token_numeric_mask = training_state_.cached_token_numeric_mask;
		num_inputs.targets = training_state_.cached_targets;
		num_inputs.total_tokens = static_cast<int>(total_tokens);
		num_inputs.seq_len = static_cast<int>(seq_len);
		num_inputs.huber_delta = cfg.numeric_head_huber_delta;
		num_inputs.log_scale = cfg.numeric_head_log_scale;
		num_inputs.loss_weight = cfg.numeric_head_loss_weight;

		NumericLossOutputs num_outputs{};
		num_outputs.loss_sum = training_state_.d_numeric_loss_sum;
		num_outputs.count = training_state_.d_numeric_loss_count;
		num_outputs.grad_predictions = training_state_.grad_numeric_predictions;

		if (!launchNumericLoss(num_inputs, num_outputs, training_state_.stream_ctrl.getPrimaryStream())) {
			throw std::runtime_error("computeLossBatch: numeric loss kernel launch failed");
		}

		cudaMemcpyAsync(&numeric_loss_sum, training_state_.d_numeric_loss_sum,
		                sizeof(float), cudaMemcpyDeviceToHost,
		                training_state_.stream_ctrl.getPrimaryStream());
		cudaMemcpyAsync(&numeric_loss_count, training_state_.d_numeric_loss_count,
		                sizeof(int), cudaMemcpyDeviceToHost,
		                training_state_.stream_ctrl.getPrimaryStream());
		training_state_.stream_ctrl.syncPrimaryStream();
		if (!std::isfinite(numeric_loss_sum)) {
			numeric_loss_sum = 0.0f;
			numeric_loss_count = 0;
		}
	}

	const float weighted_numeric_loss = (numeric_loss_count > 0)
		? cfg.numeric_head_loss_weight * numeric_loss_sum
		: 0.0f;
	const float avg_loss = (loss_result.total_loss + weighted_numeric_loss) /
		static_cast<float>(valid_tokens);
	if (!std::isfinite(avg_loss)) {
		fprintf(stderr, "[ComputeLossBatch] FATAL: avg_loss is non-finite\n");
		throw std::runtime_error("computeLossBatch: avg_loss is non-finite");
	}
	return avg_loss;
}

}  // namespace GRIM

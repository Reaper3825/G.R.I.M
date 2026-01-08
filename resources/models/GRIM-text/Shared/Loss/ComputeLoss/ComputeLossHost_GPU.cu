#include "ComputeLossHost_GPU.hpp"

#include "ComputeLoss_GPU.hpp"
#include "HyperParameters/HyperParameters_GPU.hpp"
#include "../../LogRecorder/LogRecorder.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <sstream>
#include <string_view>

namespace GRIM {

namespace {

constexpr size_t kFallbackMaxLossTokens = HyperParameters::CUDA_FALLBACK_MAX_LOSS_TOKENS;
constexpr const char* kLossLogModule = "Loss";

inline void logInfo(std::string_view message)
{
	GRIM::Logging::EmitModuleInfo(kLossLogModule, message);
}

inline void logError(std::string_view message)
{
	GRIM::Logging::EmitModuleError(kLossLogModule, message);
}

size_t resolveTokenLimit(const LossComputationInputs& inputs)
{
	const size_t configured_limit = inputs.config.limits.max_tokens;
	return configured_limit > 0 ? configured_limit : kFallbackMaxLossTokens;
}

size_t resolveGradientTokenLimit(const LossGradientInputs& inputs)
{
	return inputs.max_tokens > 0 ? inputs.max_tokens : kFallbackMaxLossTokens;
}

bool validateInputs(const LossComputationInputs& inputs)
{
	const auto& ctx = inputs.context;
	if (!ctx.logits || !ctx.targets) {
		logError("ComputeLossHost: invalid null input buffers");
		return false;
	}
	if (!ctx.stream) {
		logError("ComputeLossHost: missing CUDA stream (use StreamController primary stream)");
		return false;
	}

	if (ctx.batch_size <= 0 || ctx.seq_len <= 0 || ctx.vocab_size <= 0) {
		std::ostringstream msg;
		msg << "ComputeLossHost: invalid dimensions (batch="
		    << ctx.batch_size << ", seq=" << ctx.seq_len
		    << ", vocab=" << ctx.vocab_size << ")";
		logError(msg.str());
		return false;
	}

	const size_t total_tokens = static_cast<size_t>(ctx.batch_size) * ctx.seq_len;
	const size_t max_tokens = resolveTokenLimit(inputs);
	if (total_tokens > max_tokens) {
		std::ostringstream msg;
		msg << "ComputeLossHost: token count " << total_tokens
		    << " exceeds max supported " << max_tokens;
		logError(msg.str());
		return false;
	}

	return true;
}

bool validateGradientInputs(const LossGradientInputs& inputs)
{
	const auto& ctx = inputs.context;
	if (!ctx.logits || !ctx.targets || !inputs.grad_logits) {
		logError("ComputeLossHost: invalid gradient input buffers");
		return false;
	}
	if (!ctx.stream) {
		logError("ComputeLossHost: missing CUDA stream for gradient compute");
		return false;
	}

	if (ctx.batch_size <= 0 || ctx.seq_len <= 0 || ctx.vocab_size <= 0) {
		std::ostringstream msg;
		msg << "ComputeLossHost: invalid gradient dimensions (batch="
		    << ctx.batch_size << ", seq=" << ctx.seq_len
		    << ", vocab=" << ctx.vocab_size << ")";
		logError(msg.str());
		return false;
	}

	const size_t total_tokens = static_cast<size_t>(ctx.batch_size) * ctx.seq_len;
	const size_t max_tokens = resolveGradientTokenLimit(inputs);
	if (total_tokens > max_tokens) {
		std::ostringstream msg;
		msg << "ComputeLossHost: gradient token count " << total_tokens
		    << " exceeds max supported " << max_tokens;
		logError(msg.str());
		return false;
	}

	return true;
}

bool reportCudaFailure(std::string_view where, cudaError_t err)
{
	if (err == cudaSuccess) {
		return false;
	}
	std::ostringstream msg;
	msg << "ComputeLossHost: CUDA failure at " << where << " - "
	    << cudaGetErrorString(err);
	logError(msg.str());
	return true;
}

} // namespace

bool ensureLossScratchCapacity(LossScratch& scratch,
	size_t required_tokens,
	cudaStream_t stream)
{
	if (required_tokens == 0) {
		return false;
	}
	if (!stream) {
		logError("ComputeLossHost: missing CUDA stream for loss scratch allocation");
		return false;
	}

	if (scratch.capacity >= required_tokens &&
		scratch.loss_values && scratch.loss_accumulator) {
		return true;
	}

	if (scratch.loss_values) {
		cudaFreeAsync(scratch.loss_values, stream);
		scratch.loss_values = nullptr;
	}
	if (scratch.loss_accumulator) {
		cudaFreeAsync(scratch.loss_accumulator, stream);
		scratch.loss_accumulator = nullptr;
	}

	const size_t bytes = required_tokens * sizeof(float);
	if (cudaMallocAsync(&scratch.loss_values, bytes, stream) != cudaSuccess) {
		logError("ComputeLossHost: cudaMallocAsync loss_values failed");
		scratch.capacity = 0;
		return false;
	}

	if (cudaMallocAsync(&scratch.loss_accumulator, sizeof(float), stream) != cudaSuccess) {
		logError("ComputeLossHost: cudaMallocAsync loss_accumulator failed");
		cudaFreeAsync(scratch.loss_values, stream);
		scratch.loss_values = nullptr;
		scratch.capacity = 0;
		return false;
	}

	scratch.capacity = required_tokens;
	return true;
}

LossComputationResult computeLossHost(
	const LossComputationInputs& inputs,
	LossScratch& scratch)
{
	LossComputationResult result{};
	if (!validateInputs(inputs)) {
		return result;
	}

	const auto& ctx = inputs.context;
	const size_t total_tokens = static_cast<size_t>(ctx.batch_size) * ctx.seq_len;
	const size_t valid_tokens = (inputs.valid_token_count > 0 && inputs.valid_token_count <= total_tokens)
		                            ? inputs.valid_token_count
		                            : total_tokens;
	if (valid_tokens == 0) {
		return result;
	}
	if (!ensureLossScratchCapacity(scratch, total_tokens, ctx.stream)) {
		return result;
	}

	static bool logged_verification = false;
	if (!logged_verification) {
		std::ostringstream msg;
		msg << "[ComputeLossHost] loss pipeline invoked (batch="
		    << ctx.batch_size << ", seq=" << ctx.seq_len
		    << ", tokens=" << total_tokens << ")";
		logInfo(msg.str());
		logged_verification = true;
	}

	Loss::DeviceBuffers buffers{};
	buffers.token_losses = scratch.loss_values;
	buffers.scratch = scratch.loss_accumulator;
	buffers.grad_logits = inputs.grad_logits;  // From training_state

	const auto breakdown = Loss::launchLossPipeline(ctx, inputs.config, buffers);
	// Catch device-side issues (illegal memory access, launch failures) deterministically.
	// Without this, the process can fast-fail later during unrelated cleanup.
	reportCudaFailure("launchLossPipeline (post-launch)", cudaGetLastError());
	result.breakdown = breakdown;
	result.total_loss = breakdown.total;
	result.average_loss = result.total_loss / static_cast<float>(valid_tokens);

	if (!std::isfinite(result.total_loss) || !std::isfinite(result.average_loss)) {
		logError("ComputeLossHost: non-finite loss detected");
		return result;
	}

	result.success = true;

	// One-time breakdown log to see which term dominates
	static bool logged_breakdown = false;
	if (!logged_breakdown) {
		logged_breakdown = true;
		std::ostringstream msg;
		msg << "[ComputeLossHost] breakdown ce=" << result.breakdown.cross_entropy
		    << " (ls_delta=" << result.breakdown.label_smoothing
		    << " focal_delta=" << result.breakdown.focal << ")"
		    << " distill=" << result.breakdown.distillation_kl
		    << " pref=" << result.breakdown.preference_kl
		    << " total=" << result.total_loss
		    << " avg=" << result.average_loss;
		logInfo(msg.str());
	}

	return result;
}

bool computeCrossEntropyGradientHost(const LossGradientInputs& inputs)
{
	if (!validateGradientInputs(inputs)) {
		return false;
	}

	Loss::DeviceBuffers buffers{};
	buffers.grad_logits = inputs.grad_logits;

	Loss::launchCrossEntropyGradient(inputs.context, buffers);

	const cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::ostringstream msg;
		msg << "ComputeLossHost: cross-entropy gradient kernel failed - "
		    << cudaGetErrorString(err);
		logError(msg.str());
		return false;
	}

	return true;
}

} // namespace GRIM


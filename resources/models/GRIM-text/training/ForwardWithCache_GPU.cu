#define USE_CUDA

#include <cstddef>
#include <exception>
#include <memory>
#include <vector>
#include <cstdint>
#include <cuda_runtime.h>
#include "../GRIM/grim_language_model_cuda.hpp"
#include "../Layers/ForwardOps/ForwardOps_Logging.hpp"
#include "../Layers/ForwardOps/ForwardOps_Orchestrator.hpp"
#include "../Layers/Quantization/Quantization_GPU.hpp"

namespace GRIM {

Vector LanguageModel::forwardWithCache(const std::vector<int>& token_ids,
                                       const std::vector<float>& token_numeric_values,
                                       const std::vector<uint8_t>& token_numeric_mask,
                                       bool tokens_on_device,
                                       const std::vector<uint16_t>& token_text_features,
                                       const std::vector<uint8_t>& token_text_mask) {
    GRIM::ForwardOps::LogUnexpectedGradState(training_state_, "forwardWithCache");
    const auto prefix = GRIM::ForwardOps::BuildForwardPrefix(training_state_, "forwardWithCache");

    if (!training_state_.initialized) {
        FWD_ERROR(prefix << " Training state not initialized");
        return Vector();
    }

    const auto& cfg = getConfig();

    try {
        (void)getGpuEncoder();
        (void)getGpuEmbedder();
    } catch (const std::exception& ex) {
        FWD_ERROR(prefix << " GPU runtime not initialized: " << ex.what());
        return Vector();
    }

    const int seq_len = static_cast<int>(token_ids.size());
    if (seq_len <= 0) {
        return Vector();
    }
    if (token_numeric_values.size() != static_cast<size_t>(seq_len) ||
        token_numeric_mask.size() != static_cast<size_t>(seq_len)) {
        FWD_ERROR(prefix << " numeric side-channel length mismatch");
        return Vector();
    }
    const size_t logit_limit = training_state_.max_logit_tokens > 0
        ? training_state_.max_logit_tokens
        : training_state_.max_cached_tokens;
    if (static_cast<size_t>(seq_len) > logit_limit) {
        FWD_ERROR(prefix << " token count " << seq_len
                         << " exceeds logit buffer capacity " << logit_limit);
        return Vector();
    }

    training_state_.cached_batch_size = 1;
    training_state_.cached_seq_len = seq_len;
    training_state_.cached_num_layers = cfg.num_layers;

    bool use_device_tokens = tokens_on_device;
    if (!use_device_tokens && staged_prompt_ready_) {
        if (staged_prompt_len_ == seq_len) {
            use_device_tokens = true;
        } else {
            FWD_WARN(prefix << " staged prompt length mismatch expected=" << seq_len
                            << " got=" << staged_prompt_len_);
        }
        staged_prompt_ready_ = false;
        staged_prompt_len_ = 0;
    }
    if (use_device_tokens) {
        staged_prompt_ready_ = false;
        staged_prompt_len_ = 0;
    }

    if (!training_state_.cached_token_numeric_values || !training_state_.cached_token_numeric_mask) {
        FWD_ERROR(prefix << " numeric side-channel buffers not initialized");
        return Vector();
    }
    cudaMemcpyAsync(training_state_.cached_token_numeric_values,
                    token_numeric_values.data(),
                    static_cast<size_t>(seq_len) * sizeof(float),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());
    cudaMemcpyAsync(training_state_.cached_token_numeric_mask,
                    token_numeric_mask.data(),
                    static_cast<size_t>(seq_len) * sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    training_state_.stream_ctrl.getPrimaryStream());

    // GRMT v4: copy text features if provided
    constexpr int kTextFeatureDim = 16;  // Must match GRIM::Tokenizer::kTextFeatureDim
    if (!token_text_features.empty() && !token_text_mask.empty()) {
        cudaMemcpyAsync(training_state_.cached_token_text_features,
                        token_text_features.data(),
                        static_cast<size_t>(seq_len) * kTextFeatureDim * sizeof(uint16_t),
                        cudaMemcpyHostToDevice,
                        training_state_.stream_ctrl.getPrimaryStream());
        cudaMemcpyAsync(training_state_.cached_token_text_mask,
                        token_text_mask.data(),
                        static_cast<size_t>(seq_len) * sizeof(uint8_t),
                        cudaMemcpyHostToDevice,
                        training_state_.stream_ctrl.getPrimaryStream());
    } else {
        // Zero out text feature buffers if not provided
        cudaMemsetAsync(training_state_.cached_token_text_features, 0,
                        static_cast<size_t>(seq_len) * kTextFeatureDim * sizeof(uint16_t),
                        training_state_.stream_ctrl.getPrimaryStream());
        cudaMemsetAsync(training_state_.cached_token_text_mask, 0,
                        static_cast<size_t>(seq_len) * sizeof(uint8_t),
                        training_state_.stream_ctrl.getPrimaryStream());
    }

    auto ctx = GRIM::Forward::initForwardContext(
        *this,
        GRIM::Forward::ForwardMode::TrainingFull,
        1,
        seq_len,
        GRIM::Forward::ForwardLogitsTarget::FullSequence,
        token_ids.data(),
        use_device_tokens,
        -1,
        -1,
        scratch_block_layer_ && scratch_block_layer_->isEnabled(),
        cfg.activation_quantization.enabled,
        true);

    const auto status = GRIM::Forward::executeForward(ctx);
    if (status != GRIM::Forward::ForwardStatus::SUCCESS) {
        FWD_ERROR(prefix << " Forward failed: " << GRIM::Forward::statusToString(status)
                         << " (" << ctx.error_message << ")");
        return Vector();
    }

    Vector last_hidden(cfg.d_model);
    if (seq_len > 0) {
        const size_t last_token_offset = static_cast<size_t>(seq_len - 1) * cfg.d_model;
        cudaMemcpyAsync(last_hidden.data.data(),
                        training_state_.cached_encoder_outputs + last_token_offset,
                        cfg.d_model * sizeof(float),
                        cudaMemcpyDeviceToHost,
                        training_state_.stream_ctrl.getPrimaryStream());
        training_state_.stream_ctrl.syncPrimaryStream();
    }
    return last_hidden;
}

void LanguageModel::applyActivationQuantization(float* device_buffer, std::size_t elements) {
    const auto& quant_cfg = config_.activation_quantization;
    if (!quant_cfg.enabled || !device_buffer || elements == 0) {
        return;
    }

    auto buildConfig = [&](cudaStream_t stream) {
        Quantization::QuantizationConfig layer_cfg{};
        layer_cfg.scale = quant_cfg.scale;
        layer_cfg.clip_min = quant_cfg.clip_min;
        layer_cfg.clip_max = quant_cfg.clip_max;
        layer_cfg.zero_point = quant_cfg.zero_point;
        layer_cfg.symmetric = quant_cfg.symmetric;
        layer_cfg.stream = stream;
        return layer_cfg;
    };

    try {
        if (!activation_quantizer_) {
            activation_quantizer_ = std::make_unique<Quantization::QuantizationLayer>(buildConfig(training_state_.stream_ctrl.getPrimaryStream()));
        } else {
            const auto& current = activation_quantizer_->config();
            const auto desired = buildConfig(training_state_.stream_ctrl.getPrimaryStream());
            const bool needs_update =
                current.scale != desired.scale ||
                current.clip_min != desired.clip_min ||
                current.clip_max != desired.clip_max ||
                current.zero_point != desired.zero_point ||
                current.symmetric != desired.symmetric ||
                current.stream != desired.stream;
            if (needs_update) {
                activation_quantizer_->setConfig(desired);
            }
        }

        activation_quantizer_->quantizeAndDequantize(
            device_buffer,
            device_buffer,
            elements,
            training_state_.stream_ctrl.getPrimaryStream());
    } catch (const std::exception& ex) {
        FWD_WARN("[ActivationQuantization] failed: " << ex.what());
    }
}

} // namespace GRIM

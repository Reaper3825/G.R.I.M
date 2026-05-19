//======================================================//
//  MTP_GPU.cu
//  Multi-Token Prediction (MTP) kernels & loss orchestration
//
//  Kernels moved from AutogradLoss.cu.
//  Loss orchestration extracted from AutogradTraining.cu::computeAutogradLoss.
//======================================================//

#include "MTP_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include <iostream>
#include <cmath>
#include <stdexcept>

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {
namespace MTP {

// MTP accuracy: per-token argmax(logits[t]) == targets[t], count valid (targets[t] != -1)
__global__ void kernelMTPAccuracy(
    const float* __restrict__ logits,
    const int* __restrict__ targets,
    int total_tokens,
    int vocab_size,
    int* __restrict__ d_correct,
    int* __restrict__ d_valid
) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    int my_correct = 0, my_valid = 0;
    // Masking: target == -1 means masked/padding — must match CrossEntropyNLL exactly
    if (t < total_tokens && targets[t] != -1) {
        my_valid = 1;
        const float* row = logits + t * static_cast<size_t>(vocab_size);
        int best = 0;
        float best_val = row[0];
        for (int v = 1; v < vocab_size; ++v) {
            if (row[v] > best_val) {
                best_val = row[v];
                best = v;
            }
        }
        if (best == targets[t]) my_correct = 1;
    }
    atomicAdd(d_valid, my_valid);
    atomicAdd(d_correct, my_correct);
}

void launchMTPAccuracyKernel(
    const float* logits,
    const int* targets,
    int total_tokens,
    int vocab_size,
    int* d_correct,
    int* d_valid,
    cudaStream_t stream
) {
    if (!logits || !targets || !d_correct || !d_valid || total_tokens <= 0 || vocab_size <= 0) return;
    cudaMemsetAsync(d_correct, 0, sizeof(int), stream);
    cudaMemsetAsync(d_valid, 0, sizeof(int), stream);
    const int block = 256;
    const int grid = (total_tokens + block - 1) / block;
    kernelMTPAccuracy<<<grid, block, 0, stream>>>(logits, targets, total_tokens, vocab_size, d_correct, d_valid);
}

//========================================================================
// MTP loss computation — extracted from AutogradTraining.cu
//========================================================================

void computeMTPAuxiliaryLosses(
    Autograd::AutogradContext& ctx,
    Autograd::AutogradIntermediates& intermediates,
    MTPDiagnostics& diagnostics,
    float mtp_alpha_effective
) {
    const auto* cfg = ctx.config;
    diagnostics.clear();

    if (!ctx.payload) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: ctx.payload is NULL — caller MUST provide valid BatchPayload");
    }
    const auto& payload = *ctx.payload;
    const int total_tokens = payload.total_tokens;
    const int vocab_size = payload.vocab_size;

    if (!ctx.model || !cfg->mtp_enabled || cfg->mtp_k <= 0 ||
        !intermediates.encoder_output_tensor.data) {
        return;
    }

    const int K = cfg->mtp_k;

    // Rule 20: payload must have MTP shifted targets computed by buildBatchPayload
    if (static_cast<int>(payload.mtp_shifted_targets.size()) != K) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: payload.mtp_shifted_targets.size()=" +
            std::to_string(payload.mtp_shifted_targets.size()) + " != mtp_k=" + std::to_string(K) +
            " — buildBatchPayload must be called with mtp_k=" + std::to_string(K));
    }
    if (static_cast<int>(payload.mtp_valid_counts.size()) != K) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: payload.mtp_valid_counts.size()=" +
            std::to_string(payload.mtp_valid_counts.size()) + " != mtp_k=" + std::to_string(K));
    }
    if (!ctx.device_bindings) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: ctx.device_bindings is NULL — caller MUST upload BatchPayload before MTP loss");
    }
    if (!ctx.training_state) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: ctx.training_state is NULL — CE workspace is required for MTP loss");
    }
    if (!ctx.device_bindings->d_mtp_shifted_targets) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: BatchDeviceBindings.d_mtp_shifted_targets is NULL for MTP payload");
    }

    float L0_main = 0.0f;
    cudaMemcpyAsync(&L0_main, intermediates.loss_tensor.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
    cudaStreamSynchronize(ctx.stream);
    diagnostics.L0_main = L0_main;
    if (!std::isfinite(L0_main)) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: main CE loss (L0_main) is non-finite (" + std::to_string(L0_main) +
            ") — unified_loss failed before MTP. num_tokens=" + std::to_string(total_tokens) + " vocab=" + std::to_string(vocab_size));
    }

    if (!std::isfinite(mtp_alpha_effective) || mtp_alpha_effective < 0.0f) {
        throw std::runtime_error("computeMTPAuxiliaryLosses: mtp_alpha_effective must be finite and >= 0, got " +
            std::to_string(mtp_alpha_effective));
    }
    const float alpha_effective = mtp_alpha_effective;
    float scale = 0.0f;
    if (alpha_effective > 0.0f) {
        scale = alpha_effective / static_cast<float>(K);
    }
    intermediates.mtp_logits_tensors.clear();
    diagnostics.head_loss.clear();
    diagnostics.head_acc.clear();
    diagnostics.alpha_effective = alpha_effective;

    // Resolve mtp_input: same representation as LM head matmul input (A1 fix)
    // MTP IS an auxiliary training objective — gradients flow through encoder.
    // DeepSeek-V3 design: encoder learns future-token structure from MTP signal.
    const Tensor* mtp_input = nullptr;
    if (intermediates.centered_encoder_output.data) {
        mtp_input = &intermediates.centered_encoder_output;
    } else if (ctx.lm_head->finalRmsGamma().data) {
        intermediates.mtp_input_tensor = autograd::rms_norm(
            intermediates.encoder_output_tensor,
            ctx.lm_head->finalRmsGamma(),
            ctx.lm_head->hp().rms_epsilon,
            ctx.stream
        );
        mtp_input = &intermediates.mtp_input_tensor;
    } else {
        mtp_input = &intermediates.encoder_output_tensor;
    }

    for (int k = 0; k < K && scale > 0.0f; ++k) {
        LanguageModel::MTPHead* head = ctx.model->getMtpHead(k);
        if (!head || !head->weight.data || !head->bias.data) continue;

        // Soft-skip heads with zero valid shifted targets. Short or heavily
        // masked batches may legitimately have no valid horizon-(k+1) targets
        // while still having valid LM targets. The head contributes 0 to total
        // loss and receives no gradient; we still emit zero diagnostics so the
        // head_loss / head_acc vectors stay aligned with K.
        if (payload.mtp_valid_counts[k] == 0) {
            diagnostics.head_loss.push_back(0.0f);
            diagnostics.head_acc.push_back(0.0f);
            continue;
        }

        const int* d_targets_k = ctx.device_bindings->d_mtp_shifted_targets +
            static_cast<size_t>(k) * total_tokens;

        Tensor logits_k = autograd::matmul(
            *mtp_input,
            head->weight,
            ctx.stream,
            mtp_input->data,
            nullptr,
            true
        );
        logits_k = autograd::broadcast_add(logits_k, head->bias, ctx.stream);
        intermediates.mtp_logits_tensors.push_back(std::move(logits_k));
        Tensor loss_k = autograd::unified_loss_for_mtp_head(
            intermediates.mtp_logits_tensors.back(),
            payload,
            *ctx.device_bindings,
            k,
            ctx.loss_config,
            ctx.d_class_weights,
            ctx.stream
        );
        float h_loss_k = 0.0f;
        cudaMemcpyAsync(&h_loss_k, loss_k.data, sizeof(float), cudaMemcpyDeviceToHost, ctx.stream);
        int* d_correct = nullptr;
        int* d_valid = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_correct), sizeof(int), "mtp_d_correct");
        cudaMallocOrThrow(reinterpret_cast<void**>(&d_valid), sizeof(int), "mtp_d_valid");
        launchMTPAccuracyKernel(
            intermediates.mtp_logits_tensors.back().data,
            d_targets_k,
            total_tokens,
            vocab_size,
            d_correct,
            d_valid,
            ctx.stream
        );
        cudaStreamSynchronize(ctx.stream);
        if (!std::isfinite(h_loss_k)) {
            throw std::runtime_error("computeMTPAuxiliaryLosses: MTP head k=" + std::to_string(k) +
                " loss is non-finite (" + std::to_string(h_loss_k) + ") — shift=" + std::to_string(k + 1));
        }
        diagnostics.head_loss.push_back(h_loss_k * scale);  // Report ACTUAL contribution to total loss
        int h_correct = 0, h_valid = 0;
        cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_valid, d_valid, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_correct);
        cudaFree(d_valid);

        // Cross-validate: GPU-counted valid must match CPU-counted payload.mtp_valid_counts[k].
        // A mismatch means buildBatchPayload and the GPU kernel disagree on masking — data corruption.
        const int expected_valid = payload.mtp_valid_counts[k];
        if (h_valid != expected_valid) {
            throw std::runtime_error("computeMTPAuxiliaryLosses: MTP head k=" + std::to_string(k) +
                " valid count mismatch: GPU=" + std::to_string(h_valid) +
                " vs payload.mtp_valid_counts[k]=" + std::to_string(expected_valid) +
                " — buildBatchPayload and accuracy kernel disagree on masking");
        }
        float acc_k = (h_valid > 0) ? (static_cast<float>(h_correct) / static_cast<float>(h_valid)) * 100.0f : 0.0f;
        diagnostics.head_acc.push_back(acc_k);
        Tensor scaled_k = autograd::scale_scalar(loss_k, scale, ctx.stream);
        intermediates.loss_tensor = autograd::add(intermediates.loss_tensor, scaled_k, ctx.stream);
    }
    diagnostics.valid = !diagnostics.head_loss.empty();
}

bool verifyMTPGradients(const LanguageModel& model) {
    if (model.getMtpK() <= 0) return true;
    bool ok = true;
    for (int k = 0; k < model.getMtpK(); ++k) {
        const LanguageModel::MTPHead* head = model.getMtpHead(k);
        if (head) {
            if (head->weight.data && !head->weight.has_grad()) {
                std::cerr << "[AutogradTraining] WARN: MTP head " << k << " weight.grad is NULL\n";
                ok = false;
            }
            if (head->bias.data && !head->bias.has_grad()) {
                std::cerr << "[AutogradTraining] WARN: MTP head " << k << " bias.grad is NULL\n";
                ok = false;
            }
        }
    }
    return ok;
}

}  // namespace MTP
}  // namespace GRIM

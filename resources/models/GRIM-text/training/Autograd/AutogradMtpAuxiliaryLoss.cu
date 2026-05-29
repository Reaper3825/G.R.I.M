//======================================================//
//  AutogradMtpAuxiliaryLoss.cu
//  Autograd-owned Multi-Token Prediction (MTP) loss assembly
//======================================================//

#include "AutogradMtpAuxiliaryLoss.hpp"

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Shared/Batching/BatchDeviceBindings.hpp"
#include "../../Shared/Batching/BatchPayload.hpp"
#include "../../Shared/HyperParameters/HyperparameterGroupings.hpp"
#include "../../Shared/Loss/ComputeLoss/AutogradLoss.hpp"
#include "../../Shared/MTP/MTP_GPU.hpp"
#include "../../Shared/MTP/MTPDiagnostics.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"
#include "../../Shared/CudaAllocUtils.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

using GRIM::CudaAlloc::cudaMallocOrThrow;

namespace GRIM {
namespace Autograd {

namespace {

void checkCuda(cudaError_t err, const char* call) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("computeAutogradMtpAuxiliaryLosses: ") +
                                 call + " failed: " + cudaGetErrorString(err));
    }
}

const TensorContract::Shape2D& require2DShape(const Tensor& tensor, const char* label) {
    tensor.require(label);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string("computeAutogradMtpAuxiliaryLosses: ") +
                                 label + " must be a 2D tensor");
    }
    return tensor.shape.as_2d();
}

struct DeviceIntScalar {
    int* ptr = nullptr;
    const char* name = nullptr;

    DeviceIntScalar(const char* allocation_name, cudaStream_t stream) : name(allocation_name) {
        cudaMallocOrThrow(reinterpret_cast<void**>(&ptr), sizeof(int), allocation_name);
        checkCuda(cudaMemsetAsync(ptr, 0, sizeof(int), stream),
                  (std::string("cudaMemsetAsync(") + allocation_name + ")").c_str());
    }

    ~DeviceIntScalar() {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess) {
                std::cerr << "[AutogradMtpAuxiliaryLoss] cudaFree(" << name
                          << ") failed: " << cudaGetErrorString(err) << std::endl;
            }
        }
    }

    DeviceIntScalar(const DeviceIntScalar&) = delete;
    DeviceIntScalar& operator=(const DeviceIntScalar&) = delete;
};

}  // namespace

float computeAutogradMtpAuxiliaryLosses(
    LanguageModel& model,
    Tensor& loss_tensor,
    std::vector<Tensor>& mtp_logits_tensors,
    MTP::MTPDiagnostics& diagnostics,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const HyperParameters::LossConfigHP& loss_config,
    const float* d_class_weights,
    cudaStream_t stream,
    float mtp_alpha_effective,
    float text_ce_loss
) {
    diagnostics.clear();

    if (!stream) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: stream is NULL — caller MUST provide valid CUDA stream");
    }
    if (!std::isfinite(text_ce_loss)) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: text_ce_loss is non-finite (" +
                                 std::to_string(text_ce_loss) + ")");
    }
    if (!std::isfinite(mtp_alpha_effective) || mtp_alpha_effective < 0.0f) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: mtp_alpha_effective must be finite and >= 0, got " +
                                 std::to_string(mtp_alpha_effective));
    }

    payload.validate("computeAutogradMtpAuxiliaryLosses");
    diagnostics.L0_main = text_ce_loss;
    diagnostics.alpha_effective = mtp_alpha_effective;
    diagnostics.L_total = text_ce_loss;

    if (model.getMtpK() <= 0) {
        return 0.0f;
    }

    if (!loss_tensor.data) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: loss_tensor.data is NULL — text CE must be assembled before MTP loss");
    }
    if (static_cast<int>(payload.mtp_shifted_targets.size()) != model.getMtpK()) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: payload.mtp_shifted_targets.size()=" +
            std::to_string(payload.mtp_shifted_targets.size()) + " != model.getMtpK()=" + std::to_string(model.getMtpK()) +
            " — buildBatchPayload must author shifted targets for every MTP head");
    }
    if (static_cast<int>(payload.mtp_valid_counts.size()) != model.getMtpK()) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: payload.mtp_valid_counts.size()=" +
            std::to_string(payload.mtp_valid_counts.size()) + " != model.getMtpK()=" + std::to_string(model.getMtpK()));
    }
    if (!bindings.d_mtp_shifted_targets) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: BatchDeviceBindings.d_mtp_shifted_targets is NULL for MTP payload");
    }
    if (mtp_alpha_effective == 0.0f) {
        return 0.0f;
    }
    if (static_cast<int>(mtp_logits_tensors.size()) != model.getMtpK()) {
        throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: mtp_logits_tensors.size()=" +
            std::to_string(mtp_logits_tensors.size()) + " != model.getMtpK()=" + std::to_string(model.getMtpK()) +
            " — caller must materialize all MTP logits during forward before loss assembly");
    }

    const float per_head_loss_weight = mtp_alpha_effective / static_cast<float>(model.getMtpK());
    float mtp_loss = 0.0f;
    bool any_valid_head = false;

    for (int k = 0; k < model.getMtpK(); ++k) {
        Tensor& logits_k = mtp_logits_tensors[static_cast<size_t>(k)];
        if (payload.mtp_valid_counts[k] == 0) {
            diagnostics.head_loss.push_back(0.0f);
            diagnostics.head_acc.push_back(0.0f);
            continue;
        }
        any_valid_head = true;

        if (!logits_k.data) {
            throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: MTP head " + std::to_string(k) +
                " logits are NULL — forward must materialize valid-head logits before loss assembly");
        }

        const auto& logits_shape = require2DShape(logits_k, "MTP head logits");
        if (logits_shape.rows != payload.total_tokens || logits_shape.cols != payload.vocab_size) {
            throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: MTP head " + std::to_string(k) +
                " logits shape=[" + std::to_string(logits_shape.rows) + "," + std::to_string(logits_shape.cols) +
                "] does not match payload [total_tokens=" + std::to_string(payload.total_tokens) +
                ", vocab_size=" + std::to_string(payload.vocab_size) + "]");
        }

        const int* d_targets_k = bindings.d_mtp_shifted_targets +
            static_cast<size_t>(k) * static_cast<size_t>(payload.total_tokens);

        Tensor loss_k = autograd::unified_loss_for_mtp_head(
            logits_k,
            payload,
            bindings,
            k,
            loss_config,
            d_class_weights,
            stream
        );

        float h_loss_k = 0.0f;
        checkCuda(cudaMemcpyAsync(&h_loss_k, loss_k.data, sizeof(float), cudaMemcpyDeviceToHost, stream),
                  "cudaMemcpyAsync(MTP head loss)");

        DeviceIntScalar d_correct("mtp_d_correct", stream);
        DeviceIntScalar d_valid("mtp_d_valid", stream);
        MTP::launchMTPAccuracyKernel(
            logits_k.data,
            d_targets_k,
            payload,
            d_correct.ptr,
            d_valid.ptr,
            stream
        );

        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(MTP head telemetry)");
        if (!std::isfinite(h_loss_k)) {
            throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: MTP head k=" + std::to_string(k) +
                " loss is non-finite (" + std::to_string(h_loss_k) + ") — shift=" + std::to_string(k + 1));
        }

        int h_correct = 0;
        int h_valid = 0;
        checkCuda(cudaMemcpy(&h_correct, d_correct.ptr, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(MTP correct count)");
        checkCuda(cudaMemcpy(&h_valid, d_valid.ptr, sizeof(int), cudaMemcpyDeviceToHost),
                  "cudaMemcpy(MTP valid count)");

        if (h_valid != payload.mtp_valid_counts[k]) {
            throw std::runtime_error("computeAutogradMtpAuxiliaryLosses: MTP head k=" + std::to_string(k) +
                " valid count mismatch: GPU=" + std::to_string(h_valid) +
                " vs payload.mtp_valid_counts[k]=" + std::to_string(payload.mtp_valid_counts[k]) +
                " — buildBatchPayload and accuracy kernel disagree on masking");
        }

        const float weighted_head_loss = h_loss_k * per_head_loss_weight;
        diagnostics.head_loss.push_back(h_loss_k);
        mtp_loss += weighted_head_loss;

        const float acc_k = (h_valid > 0)
            ? (static_cast<float>(h_correct) / static_cast<float>(h_valid)) * 100.0f
            : 0.0f;
        diagnostics.head_acc.push_back(acc_k);

        Tensor scaled_k = autograd::scale_scalar(loss_k, per_head_loss_weight, stream);
        // Safe lifetime: AddGradFn captures upstream grad_fn shared_ptrs and
        // owned/non-leaf grad buffers, not raw Tensor object references. The
        // assignment may replace the local Tensor wrapper without severing the
        // graph edge to the previous scalar loss node.
        loss_tensor = autograd::add(loss_tensor, scaled_k, stream);
    }

    diagnostics.valid = any_valid_head && mtp_alpha_effective > 0.0f;
    diagnostics.L_total = text_ce_loss + mtp_loss;
    return mtp_loss;
}

}  // namespace Autograd
}  // namespace GRIM

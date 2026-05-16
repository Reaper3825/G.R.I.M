//======================================================//
//  AutogradLoss.cu
//  CUDA implementation of unified autograd-enabled loss
//  
//  Implements: Focal Loss + Label Smoothing + Cross Entropy + Entropy Regularization
//  This is the ONLY loss computation path for training.
//======================================================//

#include "AutogradLoss.hpp"
#include "CrossEntropyNLL.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
// BatchPayload + BatchDeviceBindings are part of the public unified_loss boundary.
#include "../../LogRecorder/BatchLogTape.hpp"
#include "../../VerboseLogging.hpp"  // Compile-time diagnostic guards (Issue #151)
#include "../../CudaAllocUtils.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <sstream>
#include <cmath>
#include <memory>

using GRIM::CudaAlloc::cudaMallocOrThrow;

// Access the global autograd verbose flag
extern bool g_autograd_verbose;
#define AG_TRACE(...) do { if (g_autograd_verbose) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace GRIM {
namespace autograd {

//========================================================================
// Cross-entropy / NLL kernels live in CrossEntropyNLL.cu.
// AutogradLoss.cu owns the autograd graph node and delegates CE math through
// computeCrossEntropyForwardFromLogProbs() / computeCrossEntropyBackwardToLogProbs().
//========================================================================

namespace {

__host__ const std::vector<int>& hostTargetsForSelection(
    const Batching::BatchPayload& payload,
    const CrossEntropyTargetSelection& target_selection,
    const char* caller
) {
    switch (target_selection.source) {
        case CrossEntropyTargetSource::PrimaryLm:
            if (target_selection.mtp_head_idx.has_value()) {
                throw std::runtime_error(std::string("[") + caller + "] primary LM target selection must not carry an MTP head index");
            }
            if (static_cast<int>(payload.target_ids.size()) != payload.total_tokens) {
                throw std::runtime_error(std::string("[") + caller + "] BatchPayload.target_ids.size()=" +
                    std::to_string(payload.target_ids.size()) + " != total_tokens=" +
                    std::to_string(payload.total_tokens));
            }
            return payload.target_ids;

        case CrossEntropyTargetSource::MtpShiftedHead: {
            if (!target_selection.mtp_head_idx.has_value()) {
                throw std::runtime_error(std::string("[") + caller + "] MTP shifted-target selection is missing mtp_head_idx");
            }
            const int mtp_head_idx = *target_selection.mtp_head_idx;
            if (mtp_head_idx < 0 ||
                mtp_head_idx >= static_cast<int>(payload.mtp_shifted_targets.size())) {
                throw std::runtime_error(std::string("[") + caller + "] mtp_head_idx=" +
                    std::to_string(mtp_head_idx) +
                    " out of range for BatchPayload.mtp_shifted_targets.size()=" +
                    std::to_string(payload.mtp_shifted_targets.size()));
            }
            if (static_cast<int>(payload.mtp_shifted_targets[mtp_head_idx].size()) != payload.total_tokens) {
                throw std::runtime_error(std::string("[") + caller + "] BatchPayload.mtp_shifted_targets[" +
                    std::to_string(mtp_head_idx) + "].size()=" +
                    std::to_string(payload.mtp_shifted_targets[mtp_head_idx].size()) +
                    " != total_tokens=" + std::to_string(payload.total_tokens));
            }
            return payload.mtp_shifted_targets[mtp_head_idx];
                }
    }

    throw std::runtime_error(std::string("[") + caller + "] CrossEntropyTargetSelection.source contains an unknown value");
}

} // namespace


//========================================================================
// Unified Loss GradFn - Autograd node
//========================================================================

/**
 * GradFn for unified loss (focal + smoothing + entropy reg)
 * Writes gradient directly to the logits tensor's grad field
 */
/**
 * NLLLossGradFn — Backward pass for NLL loss operating on log-probabilities.
 *
 * Architecture (replaces UnifiedLossGradFn):
 *   Forward:  logits → autograd::log_softmax() → log_probs → NLLLossForward → scalar loss
 *   Backward: scalar grad → NLLLossBackward(log_probs) → grad_log_probs → LogSoftmaxGradFn → grad_logits
 *
 * KEY IMPROVEMENT over old UnifiedLossGradFn:
 *   Old: recomputed softmax independently in backward (non-deterministic atomicAdd → different probs)
 *   New: uses SAME log_probs from forward pass (saved), exact forward/backward consistency
 *
 * Gradient chain:
 *   1. computeCrossEntropyBackwardToLogProbs() computes ∂L/∂(log_p_v) = -q_v * inv_N  (+ focal/entropy terms)
 *   2. LogSoftmaxGradFn computes ∂(log_p)/∂(logits) = δ_{ij} - p_j  (softmax Jacobian)
 *   3. Composition:  grad_logits[j] = (p_j - q_j) / N  ← standard CE gradient ✓
 *
 * Memory management:
 *   - Takes ownership of log_probs.data through a GradFn-owned RAII guard
 *   - Shares LogSoftmaxGradFn via shared_ptr
 *   - Allocates grad_log_probs_buffer for backward output and releases it via release_saved()
 */
struct NLLLossGradFn : public GradFn {
    // Saved forward data
    std::shared_ptr<float> owned_log_probs_data;        // OWNED GPU buffer: log-probabilities [num_tokens * vocab_size]
    std::shared_ptr<float> owned_grad_log_probs_buffer; // OWNED GPU buffer: backward output [num_tokens * vocab_size]
    float* log_probs_data;                              // Borrowed from owned_log_probs_data while saved
    float* grad_log_probs_buffer;                       // Borrowed from owned_grad_log_probs_buffer while saved
    
    const Batching::BatchPayload* payload;      // NOT OWNED — stable for autograd step lifetime
    Batching::BatchDeviceBindings bindings;     // View snapshot — device memory owned by TrainingState
    CrossEntropyTargetSelection target_selection;
    int valid_count;
    
    // Loss configuration (durable grouping snapshot from HyperparameterGroupings.hpp)
    HyperParameters::LossConfigHP loss_config;
    
    // Class-balanced loss: per-token weight = 1/freq(target)^β
    const float* class_weights;     // NOT OWNED — points to TrainingState::class_weights_tensor.data
    float weight_sum;               // Sum of per-token class weights for this batch
    
    // Upstream gradient chain
    std::shared_ptr<GradFn> log_probs_grad_fn;
    TensorContract::TensorShape grad_shape;
    
    __host__ NLLLossGradFn(
        float* log_probs,           // Takes ownership of this GPU buffer
        const Batching::BatchPayload& payload_,
        const Batching::BatchDeviceBindings& bindings_,
        const CrossEntropyTargetSelection& target_selection_,
        int valid_count_,
        const HyperParameters::LossConfigHP& loss_config_,
        const float* class_weights_, float weight_sum_,
        std::shared_ptr<GradFn> upstream_grad_fn,
        const TensorContract::TensorShape& shape,
        cudaStream_t stream_
    )
        : owned_log_probs_data(log_probs, [](float* p) { queueForDeferredCleanup(p); })
        , owned_grad_log_probs_buffer(nullptr)
        , log_probs_data(owned_log_probs_data.get())
        , grad_log_probs_buffer(nullptr)
        , payload(&payload_)
        , bindings(bindings_)
        , target_selection(target_selection_)
        , valid_count(valid_count_)
        , loss_config(loss_config_)
        , class_weights(class_weights_), weight_sum(weight_sum_)
        , log_probs_grad_fn(std::move(upstream_grad_fn))
        , grad_shape(shape)
    {
        if (!stream_) {
            throw std::runtime_error("[NLLLossGradFn::ctor] stream is NULL — loss GradFn requires a valid CUDA stream");
        }
        op_name = "nll_loss";
    }
    
    __host__ ~NLLLossGradFn() {
        release_saved();
    }
    
    __host__ void apply(const Tensor& grad_output, cudaStream_t stream) override {
        AG_TRACE("[NLLLossGradFn::apply] ENTER: log_probs_data=%p upstream=%p\n",
                 (void*)log_probs_data, (void*)log_probs_grad_fn.get());

        if (!payload) {
            throw std::runtime_error("[NLLLossGradFn::apply] payload pointer is NULL — BatchPayload is required for loss backward geometry");
        }
        payload->validate("NLLLossGradFn::apply");
        const int num_tokens = payload->total_tokens;
        const int vocab_size = payload->vocab_size;
        
        // ── Read upstream scalar gradient for chain rule ──
        // Loss is scalar, so grad_output is a single float. Tensor::backward()
        // seeds terminal scalar objectives with the caller-provided scale
        // (typically 1.0 for a root loss, or 1.0/accumulation_steps). When this
        // loss is composed with other losses, grad_output carries that upstream
        // derivative so magnitude accounting stays correct.
        //
        // CRITICAL: Must use the SAME stream for the D2H copy!
        // Tensor::backward() writes grad_output via cudaMemcpyAsync on the
        // training stream, which uses cudaStreamNonBlocking. Plain cudaMemcpy
        // (NULL stream) does NOT synchronize with non-blocking streams,
        // causing a data race that reads uninitialized GPU memory.
        if (!grad_output.data) {
            throw std::runtime_error("[NLLLossGradFn::apply] grad_output.data is NULL — upstream scalar gradient is required");
        }

        float upstream_scalar_grad;
        cudaError_t copy_err = cudaMemcpyAsync(&upstream_scalar_grad, grad_output.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("[NLLLossGradFn::apply] failed to copy upstream scalar gradient: ") + cudaGetErrorString(copy_err));
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error(std::string("[NLLLossGradFn::apply] failed to synchronize upstream scalar gradient copy: ") + cudaGetErrorString(sync_err));
        }
        if (!std::isfinite(upstream_scalar_grad)) {
            throw std::runtime_error("[NLLLossGradFn::apply] upstream scalar gradient is non-finite ("
                + std::to_string(upstream_scalar_grad) + ") — upstream gradient is corrupt");
        }
        AG_TRACE("[NLLLossGradFn::apply] upstream_scalar_grad=%.6f\n", upstream_scalar_grad);
        
        // OOM FIX: Lazy allocation — only allocate grad buffer when actually
        // needed for backward, not during forward pass. Saves 1.37GB peak memory.
        if (!grad_log_probs_buffer) {
            const size_t grad_bytes = static_cast<size_t>(num_tokens) * vocab_size * sizeof(float);
            float* grad_buffer = nullptr;
            cudaMallocOrThrow(reinterpret_cast<void**>(&grad_buffer), grad_bytes, "NLLLossGradFn_grad_log_probs");
            owned_grad_log_probs_buffer.reset(grad_buffer, [](float* p) { queueForDeferredCleanup(p); });
            grad_log_probs_buffer = owned_grad_log_probs_buffer.get();
        }
        
        // ── Step 1: Compute CE/NLL backward → gradient w.r.t. log_probs ──
        // upstream_scalar_grad is folded into the mean-reduction denominator in the CE module.
        computeCrossEntropyBackwardToLogProbs(
            log_probs_data,
            *payload,
            bindings,
            target_selection,
            grad_log_probs_buffer,
            valid_count,
            weight_sum,
            loss_config,
            class_weights,
            upstream_scalar_grad,
            stream
        );
        
        // Issue #152: Removed cudaStreamSynchronize error-check here.
        // Same-stream kernel ordering guarantees grad_log_probs_buffer is written
        // before the next kernel (LogSoftmaxGradFn::apply) reads it.
        // Errors will surface at the next natural sync point (loss D2H readback).
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("[NLLLossGradFn] CUDA error after NLL backward launch: ") + cudaGetErrorString(err));
        }
        
        // ── Step 2: Diagnostic — sample grad_log_probs at TARGET columns of valid tokens ──
        // FIX: Previous code sampled a contiguous block starting at token 50.
        // Problems: (a) token 50 might be masked (kernel writes 0.0f for masked rows),
        // (b) with plain CE (no smoothing), only grad[target] is non-zero — a contiguous
        // block mostly reads the 50,375 zero entries. Now we read the TARGET column of
        // each sampled valid token, which is where the actual gradient lives.
        if constexpr (GRIM::VerboseLogging::ENABLE_LOSS_BACKWARD_SAMPLING) {
            // Read BatchPayload-authored host targets to find valid positions.
            // The CE kernels resolve device targets locally from BatchDeviceBindings;
            // diagnostics do not need to cache or copy a raw target pointer.
            const int sample_max = std::min(200, num_tokens);
            const std::vector<int>& h_targets = hostTargetsForSelection(*payload, target_selection, "NLLLossGradFn::apply");
            
            float mx = 0.0f; double sq = 0.0; int valid_sampled = 0;
            for (int t = 0; t < sample_max; ++t) {
                if (h_targets[t] < 0 || h_targets[t] >= vocab_size) continue;  // skip masked
                // Read grad_log_probs[t, target[t]] — the one non-zero entry per valid row
                const size_t elem_offset = static_cast<size_t>(t) * vocab_size + h_targets[t];
                float val = 0.0f;
                cudaMemcpy(&val, grad_log_probs_buffer + elem_offset, sizeof(float), cudaMemcpyDeviceToHost);
                if (!std::isnan(val) && !std::isinf(val)) {
                    mx = std::max(mx, std::abs(val));
                    sq += static_cast<double>(val) * val;
                    valid_sampled++;
                }
            }
            float rms = (valid_sampled > 0) ? std::sqrt(static_cast<float>(sq / valid_sampled)) : 0.0f;
            
            // Expected: with plain CE, grad_log_probs[t, target[t]] = -1/N
            // With smoothing: grad_log_probs[t, target[t]] = -(1-epsilon)/N
            float expected_target_grad = 1.0f / valid_count;
            if (loss_config.smoothing_enabled) {
                expected_target_grad = (1.0f - loss_config.smoothing_epsilon) / valid_count;
            }
            
            std::ostringstream eq;
            eq << "[NLL-BWD-OUT] grad_log_probs = dL/d(log_p) [NLL backward, before LogSoftmaxGradFn]\n"
               << "  INPUTS: num_tokens=" << num_tokens << " vocab=" << vocab_size << " valid=" << valid_count << "\n"
               << "  EXPECTED |grad[t,target[t]]|=1/N=" << expected_target_grad << "\n"
               << "  ACTUAL max=" << mx << " rms=" << rms << " (sampled " << valid_sampled << " valid target entries)";
            EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::Loss, GRIM::Logging::LogPhase::LOSS_BACKWARD, -1, "NLL-BWD-OUT", eq.str().c_str());
        } // if constexpr ENABLE_LOSS_BACKWARD_SAMPLING (NLL-BWD-OUT)
        
        // ── Step 3: Chain to LogSoftmaxGradFn → computes grad w.r.t. raw logits ──
        // LogSoftmaxGradFn::apply() does:
        //   grad_logits[i] = grad_log_p[i] - exp(log_p[i]) * Σ_j grad_log_p[j]
        // Then chains to logits.grad_fn (MatMulGradFn from LM head)
        if (log_probs_grad_fn) {
            Tensor grad_log_probs_tensor;
            grad_log_probs_tensor.data = grad_log_probs_buffer;
            grad_log_probs_tensor.shape = grad_shape;
            grad_log_probs_tensor.owns_data = false;
            grad_log_probs_tensor.stream = stream;
            
            {
                std::ostringstream eq;
                eq << "[NLL-TO-LOGSOFTMAX] chaining grad_log_probs to LogSoftmaxGradFn -> grad_logits\n"
                   << "  grad_log_probs.data=" << (void*)grad_log_probs_buffer;
                EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::Loss, GRIM::Logging::LogPhase::LOSS_BACKWARD, -1, "NLL-TO-LOGSOFTMAX", eq.str().c_str());
            }
            
            AG_TRACE("[NLLLossGradFn::apply] Chaining to LogSoftmaxGradFn\n");
            log_probs_grad_fn->apply(grad_log_probs_tensor, stream);
            AG_TRACE("[NLLLossGradFn::apply] LogSoftmaxGradFn returned\n");
        } else {
            // Rule 20: upstream grad_fn is REQUIRED for backpropagation
            throw std::runtime_error("[NLLLossGradFn::apply] log_probs_grad_fn is NULL "
                "— LogSoftmaxGradFn was not attached. Cannot backpropagate.");
        }
        
        AG_TRACE("[NLLLossGradFn::apply] EXIT\n");
    }
    
    __host__ void release_saved() override {
        if (released_) return;
        GradFn::release_saved();

        // Match TensorContract GradFn lifecycle: graph-owned GPU buffers are
        // released at the tape boundary, and their deleters queue CUDA cleanup
        // rather than calling cudaFree directly from a GradFn destructor.
        log_probs_grad_fn.reset();
        owned_grad_log_probs_buffer.reset();
        owned_log_probs_data.reset();
        grad_log_probs_buffer = nullptr;
        log_probs_data = nullptr;
        payload = nullptr;
        class_weights = nullptr;
    }
};

//========================================================================
// Main API Functions (Host-only)
//========================================================================

namespace {

__host__ Tensor unifiedLossFromTargetSelection(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const CrossEntropyTargetSelection& target_selection,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream,
    const char* caller
) {
    // ══════════════════════════════════════════════════════════════════════
    // Rule 20: FAIL LOUD on invalid data
    // Public callers enter through BatchPayload + BatchDeviceBindings. Target
    // selection is a source descriptor only; CrossEntropyNLL resolves the device
    // target address from BatchDeviceBindings at the kernel launch boundary.
    // ══════════════════════════════════════════════════════════════════════
    
    payload.validate(caller);
    if (config.class_balanced_enabled && !d_class_weights) {
        throw std::runtime_error(std::string("[") + caller + "] class_balanced_enabled=true but d_class_weights is NULL");
    }
    const float* effective_class_weights = nullptr;
    if (config.class_balanced_enabled) {
        effective_class_weights = d_class_weights;
    }

    // ══════════════════════════════════════════════════════════════════════
    // Architecture: logits → log_softmax() → log_probs → NLL loss → scalar
    //
    // This is the PyTorch gold standard: F.log_softmax + F.nll_loss
    // Softmax is computed ONCE (in log_softmax), saved in LogSoftmaxGradFn.
    // NLL backward produces grad w.r.t. log_probs, which chains through
    // LogSoftmaxGradFn to produce grad w.r.t. logits: (p - q) / N
    // ══════════════════════════════════════════════════════════════════════
    
    // ── Step 1: log_softmax(logits) → log_probs ──
    // Creates LogSoftmaxGradFn if logits.requires_grad
    // OOM FIX: Pass save_output_copy=false so LogSoftmaxGradFn stores a
    // non-owning pointer to result.data instead of copying 1.37GB.
    // This is safe because NLLLossGradFn takes ownership of result.data through
    // a GradFn-owned RAII guard and keeps it alive through backward. Lifecycle:
    //   NLLLossGradFn::apply() → calls LogSoftmaxGradFn::apply() (reads shared data) → returns
    //   Tensor::backward() syncs → NLLLossGradFn::release_saved() releases LogSoftmaxGradFn and queues buffer cleanup
    Tensor log_probs = autograd::log_softmax(logits, stream, false);
    
    // ── Step 2: CE/NLL loss forward on log_probs ──
    const CrossEntropyForwardResult ce_result = computeCrossEntropyForwardFromLogProbs(
        log_probs.data,
        payload,
        bindings,
        target_selection,
        config,
        effective_class_weights,
        stream
    );
    const float mean_loss = ce_result.mean_loss;
    const int h_valid_count = ce_result.valid_count;
    const float h_weight_sum = ce_result.weight_sum;
    
    AG_TRACE("[%s] valid_count=%d mean_loss=%.6f\n",
             caller, h_valid_count, mean_loss);
    if (effective_class_weights) {
        AG_TRACE("[%s] class_balanced: weight_sum=%.2f effective_N=%.2f (vs raw N=%d)\n",
                 caller, h_weight_sum, h_weight_sum, h_valid_count);
    }
    AG_TRACE("[%s] config: focal_alpha=%.2f focal_gamma=%.2f smoothing=%.3f entropy_lambda=%.4f\n",
             caller, config.focal_alpha, config.focal_gamma, config.smoothing_epsilon, config.entropy_reg_lambda);
    
    // ── Step 4: Create scalar loss tensor ──
    float* d_loss = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_loss), sizeof(float), "unified_loss_d_loss");
    // BUG FIX Issue #61: Use SYNC copy because mean_loss is a local variable!
    cudaMemcpy(d_loss, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    
    Tensor loss;
    loss.data = d_loss;
    loss.owns_data = true;
    loss.shape = TensorContract::TensorShape::make_BSM(1, 1);
    loss.is_leaf = false;
    loss.requires_grad = logits.requires_grad;
    loss.stream = stream;
    
    // ── Step 5: Attach NLLLossGradFn ──
    if (logits.requires_grad) {
        auto grad_fn = std::make_shared<NLLLossGradFn>(
            log_probs.data,       // Takes ownership of log_probs GPU buffer
            payload,
            bindings,
            target_selection,
            h_valid_count,
            config,
            effective_class_weights, h_weight_sum,
            log_probs.grad_fn,    // Takes ownership of LogSoftmaxGradFn
            log_probs.shape,
            stream
        );
        // Transfer data ownership from Tensor to NLLLossGradFn
        log_probs.owns_data = false;     // NLLLossGradFn now owns log_probs.data
        
        loss.grad_fn = grad_fn;
    }
    
    return loss;
    // log_probs goes out of scope: data NOT freed (transferred), grad_fn NOT deleted (transferred)
}

} // namespace

__host__ Tensor unified_loss(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
) {
    payload.validate("unified_loss");
    if (!bindings.d_target_ids) {
        throw std::runtime_error("[unified_loss] BatchDeviceBindings.d_target_ids is NULL — caller MUST upload BatchPayload before loss");
    }
    if (bindings.batch_size != payload.batch_size) {
        throw std::runtime_error("[unified_loss] bindings.batch_size=" + std::to_string(bindings.batch_size) +
            " != payload.batch_size=" + std::to_string(payload.batch_size));
    }
    if (bindings.max_seq_len != payload.max_seq_len) {
        throw std::runtime_error("[unified_loss] bindings.max_seq_len=" + std::to_string(bindings.max_seq_len) +
            " != payload.max_seq_len=" + std::to_string(payload.max_seq_len));
    }

    return unifiedLossFromTargetSelection(
        logits,
        payload,
        bindings,
        CrossEntropyTargetSelection::primaryLm(),
        config,
        d_class_weights,
        stream,
        "unified_loss"
    );
}

__host__ Tensor unified_loss_for_mtp_head(
    Tensor& logits,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    int head_idx,
    const HyperParameters::LossConfigHP& config,
    const float* d_class_weights,
    cudaStream_t stream
) {
    payload.validate("unified_loss_for_mtp_head");
    if (head_idx < 0 || head_idx >= static_cast<int>(payload.mtp_shifted_targets.size())) {
        throw std::runtime_error("[unified_loss_for_mtp_head] head_idx=" + std::to_string(head_idx) +
            " out of range for payload.mtp_shifted_targets.size()=" +
            std::to_string(payload.mtp_shifted_targets.size()));
    }
    if (bindings.batch_size != payload.batch_size || bindings.max_seq_len != payload.max_seq_len) {
        throw std::runtime_error("[unified_loss_for_mtp_head] BatchDeviceBindings geometry does not match BatchPayload");
    }
    if (!bindings.d_mtp_shifted_targets) {
        throw std::runtime_error("[unified_loss_for_mtp_head] BatchDeviceBindings.d_mtp_shifted_targets is NULL");
    }
    if (payload.mtp_valid_counts[head_idx] <= 0) {
        throw std::runtime_error("[unified_loss_for_mtp_head] payload.mtp_valid_counts[" +
            std::to_string(head_idx) + "]=" + std::to_string(payload.mtp_valid_counts[head_idx]) +
            " — caller must skip zero-valid MTP heads before loss assembly");
    }

    return unifiedLossFromTargetSelection(
        logits,
        payload,
        bindings,
        CrossEntropyTargetSelection::mtpShiftedHead(head_idx),
        config,
        d_class_weights,
        stream,
        "unified_loss_for_mtp_head"
    );
}
}  // namespace autograd
}  // namespace GRIM

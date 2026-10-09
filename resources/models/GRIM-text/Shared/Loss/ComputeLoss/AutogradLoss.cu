//======================================================//
//  AutogradLoss.cu
//  CUDA implementation of unified autograd-enabled loss
//  
//  Implements: Focal Loss + Label Smoothing + Cross Entropy + Entropy Regularization
//  Primary text loss; atom insertion and retrieval have separate objectives.
//======================================================//

#include "AutogradLoss.hpp"
#include "CrossEntropyNLL.hpp"
#include "../../TensorContract/TensorContract_GPU.hpp"
#include "../../TensorContract/AutogradEngine.hpp"
#include "../../Diagnostics/MemoryAllocationTracker.hpp"
// BatchPayload + BatchDeviceBindings are part of the public unified_loss boundary.
#include "../../VerboseLogging.hpp"  // Compile-time diagnostic guards (Issue #151)
#include "../../CudaAllocUtils.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <sstream>
#include "../../LogRecorder/BatchLogTape.hpp"
#include <memory>

using GRIM::CudaAlloc::cudaMallocOrThrow;

#define AG_TRACE(...) do { if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRACE_LOGS) { fprintf(stderr, __VA_ARGS__); fflush(stderr); } } while(0)

namespace GRIM {
namespace autograd {

// One graph node owns the forward log-probabilities and delivers the fused
// NLL/log-softmax derivative directly into the logits gradient destination.
struct TextLossGradFn : public GradFn {
    std::shared_ptr<Tensor> saved_log_probs;
    std::shared_ptr<GradFn> input_producer;
    Tensor* leaf_input_gradient = nullptr;

    CrossEntropyTargetSelection target_selection;

    // Loss configuration (durable grouping snapshot from HyperparameterGroupings.hpp)
    HyperParameters::LossConfigHP loss_config;

    // Class-balanced loss: per-token weight = 1/freq(target)^β
    const float* class_weights;     // NOT OWNED — points to TrainingState::class_weights_tensor.data
    float weight_sum;               // Forward-computed sum of class weights over valid tokens (class_balanced only)
    float mean_loss;                // Forward scalar computed during capture_inputs()

    // Upstream gradient chain
    TensorContract::TensorShape grad_shape;

    __host__ TextLossGradFn()
        : target_selection(CrossEntropyTargetSelection::primaryLm())
        , loss_config{}
        , class_weights(nullptr)
        , weight_sum(0.0f)
        , mean_loss(0.0f)
        , grad_shape{}
    {
        op_name = "text_loss";
    }

    __host__ void capture_inputs(
        Tensor& logits,
        const Batching::BatchPayload& payload_,
        const Batching::BatchDeviceBindings& bindings_,
        const CrossEntropyTargetSelection& target_selection_,
        const HyperParameters::LossConfigHP& loss_config_,
        const float* class_weights_,
        cudaStream_t stream_
    ) {
        if (!stream_) {
            throw std::runtime_error("[TextLossGradFn::capture_inputs] stream is NULL — loss GradFn requires a valid CUDA stream");
        }
        logits.require("TextLossGradFn::capture_inputs");
        target_selection = target_selection_;
        loss_config = loss_config_;
        class_weights = class_weights_;
        grad_shape = logits.shape;
        if (logits.requires_grad) {
            if (logits.is_leaf) {
                if (logits.grad_fn) throw std::runtime_error("TextLossGradFn: leaf has a producer");
                logits.ensure_grad();
                leaf_input_gradient = logits.grad_.get();
                if (!leaf_input_gradient) throw std::runtime_error("TextLossGradFn: missing leaf gradient");
            } else {
                if (!logits.grad_fn) throw std::runtime_error("TextLossGradFn: non-leaf has no producer");
                input_producer = logits.grad_fn;
                register_input(input_producer);
            }
        }

        // A non-owning, non-differentiable view reuses the existing forward
        // kernel without creating a LogSoftmaxGradFn or its saved-output copy.
        // This node retains the sole owning result for the whole tape lifetime.
        Tensor logits_view;
        logits_view.data = logits.data;
        logits_view.shape = logits.shape;
        logits_view.owns_data = false;
        logits_view.requires_grad = false;
        logits_view.stream = stream_;
        saved_log_probs = std::make_shared<Tensor>(autograd::log_softmax(logits_view, stream_));
        MemoryAccounting::classify(saved_log_probs->data, MemoryAccounting::Kind::Saved);

        // CE forward scalar reduction scratch: allocated locally, used immediately, released on scope exit.
        // These device scalars are forward-only scratch — valid_count is BatchPayload-authored state
        // and must not be stored on the GradFn; weight_sum is forward-computed and stored as a host scalar.
        float* loss_sum_local = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&loss_sum_local), sizeof(float), "TextLossGradFn_fwd_loss_sum");
        std::shared_ptr<float> owned_loss_sum(loss_sum_local, [](float* p) { queueForDeferredCleanup(p); });

        int* valid_count_local = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&valid_count_local), sizeof(int), "TextLossGradFn_fwd_valid_count");
        std::shared_ptr<int> owned_valid_count(valid_count_local, [](int* p) { queueForDeferredCleanup(p); });

        float* weight_sum_local = nullptr;
        cudaMallocOrThrow(reinterpret_cast<void**>(&weight_sum_local), sizeof(float), "TextLossGradFn_fwd_weight_sum");
        std::shared_ptr<float> owned_weight_sum(weight_sum_local, [](float* p) { queueForDeferredCleanup(p); });

        const CrossEntropyForwardResult ce_result = computeCrossEntropyForwardFromLogProbs(
            saved_log_probs->data,
            payload_,
            bindings_,
            target_selection,
            CrossEntropyForwardWorkspace{
                loss_sum_local,
                valid_count_local,
                weight_sum_local,
                sizeof(float),
                sizeof(int),
                sizeof(float),
                stream_
            },
            loss_config,
            class_weights,
            stream_
        );

        mean_loss = ce_result.mean_loss;
        weight_sum = ce_result.weight_sum;
        // ce_result.valid_count is not stored — it equals the BatchPayload-authored valid token count
        // and is read directly from batch_payload in apply_impl.
    }
    
    __host__ ~TextLossGradFn() {
        release_saved();
    }
    
    __host__ void apply_impl(const Tensor& grad_output,
                             cudaStream_t stream,
                             const Batching::BatchPayload* batch_payload,
                             const Batching::BatchDeviceBindings* backward_bindings) override {
        if (applied) return;
        applied = true;
        if (!stream) throw std::runtime_error("TextLossGradFn::apply: stream is NULL");
        if (!saved_log_probs || !saved_log_probs->data) {
            throw std::runtime_error("TextLossGradFn::apply: saved log_probs are missing");
        }
        const float* log_probs_data = saved_log_probs->data;

        if (!batch_payload) {
            throw std::runtime_error("[TextLossGradFn::apply] batch_payload is NULL — orchestration MUST pass the active BatchPayload into backward()");
        }
        if (!backward_bindings) {
            throw std::runtime_error("[TextLossGradFn::apply] backward_bindings is NULL — orchestration MUST pass the active BatchDeviceBindings into backward()");
        }
        const int num_tokens = batch_payload->total_tokens;
        const int vocab_size = batch_payload->vocab_size;
        const auto saved_shape = grad_shape.as_2d();
        if (saved_shape.rows != num_tokens || saved_shape.cols != vocab_size) {
            throw std::runtime_error(
                "[TextLossGradFn::apply] backward payload geometry differs from saved log_probs");
        }

        // valid_count is BatchPayload-authored state — read directly rather than storing a copy on the GradFn.
        const int valid_count = batch_payload->lm_valid_tokens;
        if (valid_count <= 0) {
            throw std::runtime_error("[TextLossGradFn::apply] valid_count=" + std::to_string(valid_count)
                + " from batch_payload — BatchPayload must have positive valid token count");
        }
        
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
        if (!grad_output.data || grad_output.numel() != 1) {
            throw std::runtime_error("[TextLossGradFn::apply] expected one upstream scalar gradient");
        }

        float upstream_scalar_grad;
        cudaError_t copy_err = cudaMemcpyAsync(&upstream_scalar_grad, grad_output.data, sizeof(float), cudaMemcpyDeviceToHost, stream);
        if (copy_err != cudaSuccess) {
            throw std::runtime_error(std::string("[TextLossGradFn::apply] failed to copy upstream scalar gradient: ") + cudaGetErrorString(copy_err));
        }
        cudaError_t sync_err = cudaStreamSynchronize(stream);
        if (sync_err != cudaSuccess) {
            throw std::runtime_error(std::string("[TextLossGradFn::apply] failed to synchronize upstream scalar gradient copy: ") + cudaGetErrorString(sync_err));
        }
        if (!std::isfinite(upstream_scalar_grad)) {
            throw std::runtime_error("[TextLossGradFn::apply] upstream scalar gradient is non-finite ("
                + std::to_string(upstream_scalar_grad) + ") — upstream gradient is corrupt");
        }
        AG_TRACE("[TextLossGradFn::apply] upstream_scalar_grad=%.6f\n", upstream_scalar_grad);
        
        Tensor* input_gradient = input_producer
            ? &input_producer->gradient_destination(grad_shape, stream) : leaf_input_gradient;
        if (!input_gradient) {
            throw std::runtime_error("TextLossGradFn::apply: logits gradient destination is missing");
        }
        // Accumulate directly; do not pass this destination back through apply()
        // while the engine is active, which would copy/add the contribution twice.
        computeCrossEntropyBackwardToLogits(
            log_probs_data,
            *batch_payload,
            *backward_bindings,
            target_selection,
            input_gradient->data,
            valid_count,
            weight_sum,
            loss_config,
            class_weights,
            upstream_scalar_grad,
            stream
        );
        
        // This samples the accumulated logits destination, not an isolated NLL
        // contribution. Keep diagnostics on the same non-blocking CUDA stream.
        if constexpr (GRIM::VerboseLogging::ENABLE_LOSS_BACKWARD_SAMPLING) {
            float maximum = 0.0f;
            for (int t = 0; t < std::min(200, num_tokens); ++t) {
                const int target = batch_payload->target_ids[t];
                if (target < 0 || target >= vocab_size) continue;
                float value = 0.0f;
                const size_t offset = static_cast<size_t>(t) * vocab_size + target;
                const auto copy = cudaMemcpyAsync(&value, input_gradient->data + offset,
                    sizeof(float), cudaMemcpyDeviceToHost, stream);
                if (copy != cudaSuccess) throw std::runtime_error("TextLossGradFn: gradient sample copy failed");
                const auto sync = cudaStreamSynchronize(stream);
                if (sync != cudaSuccess) throw std::runtime_error("TextLossGradFn: gradient sample synchronization failed");
                if (std::isfinite(value)) maximum = std::max(maximum, std::abs(value));
            }
            std::ostringstream message;
            message << "[TEXT-LOSS-BWD] accumulated logits gradient: sampled target max_abs=" << maximum;
            EQ_LOG(GRIM::Logging::getGlobalTape(), GRIM::Logging::LogGroup::Loss,
                GRIM::Logging::LogPhase::LOSS_BACKWARD, -1, "TEXT-LOSS-BWD", message.str().c_str());
        }

        if (input_producer) {
            if (auto* engine = AutogradEngine::active()) {
                engine->contribute(input_producer.get());
            } else {
                input_producer->apply(input_producer->pending_gradient("TextLossGradFn::apply producer"),
                                      stream, batch_payload, backward_bindings);
            }
        } else {
            leaf_input_gradient->record_leaf_gradient_delivery();
        }
    }

    __host__ void release_saved() override {
        if (released_) return;
        GradFn::release_saved();

        saved_log_probs.reset();
        input_producer.reset();
        leaf_input_gradient = nullptr;
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
    
    if (!stream) {
        throw std::runtime_error(std::string("[") + caller + "] stream is NULL — caller MUST provide a valid CUDA stream");
    }
    if (config.entropy_reg_enabled && config.entropy_reg_lambda == 0.0f) {
        throw std::runtime_error(std::string("[") + caller + "] entropy_reg_enabled=true but entropy_reg_lambda is 0");
    }
    if (!config.entropy_reg_enabled && config.entropy_reg_lambda != 0.0f) {
        throw std::runtime_error(std::string("[") + caller + "] entropy_reg_enabled=false but entropy_reg_lambda=" +
            std::to_string(config.entropy_reg_lambda) + " — disable by setting lambda to 0");
    }
    // Host payload validity is established by batching before upload. The loss
    // boundary checks the tensor it consumes against that authored geometry.
    if (!logits.data || !logits.shape.is_2d_layout()) {
        throw std::runtime_error(std::string("[") + caller + "] logits must be a valid 2D tensor");
    }
    const auto logits_shape = logits.shape.as_2d();
    if (logits_shape.rows != payload.total_tokens || logits_shape.cols != payload.vocab_size) {
        throw std::runtime_error(std::string("[") + caller + "] logits shape does not match payload token rows and vocabulary size");
    }
    if (config.class_balanced_enabled && !d_class_weights) {
        throw std::runtime_error(std::string("[") + caller + "] class_balanced_enabled=true but d_class_weights is NULL");
    }
    if (!config.class_balanced_enabled && d_class_weights) {
        throw std::runtime_error(std::string("[") + caller + "] d_class_weights is non-NULL while class_balanced_enabled=false");
    }
    const float* effective_class_weights = nullptr;
    if (config.class_balanced_enabled) {
        effective_class_weights = d_class_weights;
    }

    // The loss owns forward state and the direct edge to the logits producer.
    auto grad_fn = std::make_shared<TextLossGradFn>();
    grad_fn->capture_inputs(
        logits,
        payload,
        bindings,
        target_selection,
        config,
        effective_class_weights,
        stream
    );
    const float mean_loss = grad_fn->mean_loss;
    const float h_weight_sum = grad_fn->weight_sum;
    // valid_count is BatchPayload-authored — read directly from payload, not via GradFn.
    const int h_valid_count = payload.lm_valid_tokens;

    AG_TRACE("[%s] valid_count=%d mean_loss=%.6f\n",
             caller, h_valid_count, mean_loss);
    if (effective_class_weights) {
        AG_TRACE("[%s] class_balanced: weight_sum=%.2f effective_N=%.2f (vs raw N=%d)\n",
                 caller, h_weight_sum, h_weight_sum, h_valid_count);
    }
    AG_TRACE("[%s] config: focal_alpha=%.2f focal_gamma=%.2f smoothing=%.3f entropy_lambda=%.4f\n",
             caller, config.focal_alpha, config.focal_gamma, config.smoothing_epsilon, config.entropy_reg_lambda);
    
    // Create the public scalar loss tensor.
    float* d_loss = nullptr;
    cudaMallocOrThrow(reinterpret_cast<void**>(&d_loss), sizeof(float), "unified_loss_d_loss");
    // BUG FIX Issue #61: Use SYNC copy because mean_loss is a local variable!
    cudaError_t loss_copy_err = cudaMemcpy(d_loss, &mean_loss, sizeof(float), cudaMemcpyHostToDevice);
    if (loss_copy_err != cudaSuccess) {
        throw std::runtime_error(std::string("[") + caller + "] cudaMemcpy(unified_loss_d_loss) failed: " + cudaGetErrorString(loss_copy_err));
    }
    
    Tensor loss;
    loss.data = d_loss;
    loss.owns_data = true;
    loss.shape = TensorContract::TensorShape::make_BSM(1, 1);
    loss.is_leaf = false;
    loss.requires_grad = logits.requires_grad;
    loss.stream = stream;
    
    // Only training retains the node and its saved forward state.
    if (logits.requires_grad) {
        loss.grad_fn = grad_fn;
    }
    
    // In evaluation the local node releases its forward state on return.
    return loss;
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
    if (!bindings.d_target_ids) {
        throw std::runtime_error("[unified_loss] BatchDeviceBindings.d_target_ids is NULL — caller MUST upload BatchPayload before loss");
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

}  // namespace autograd
}  // namespace GRIM

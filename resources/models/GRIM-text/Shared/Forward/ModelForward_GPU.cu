//======================================================//
//  ModelForward_GPU.cu
//  Shared full-model forward primitive
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelForward_GPU.hpp"

#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Encoding/Encoding_GPU.hpp"
#include "../../Layers/LMHead/lm_head_GPU.hpp"
#include "../../Layers/ArgSelector/ArgSelector_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"
#include "../InferenceState/KvCacheState_GPU.hpp"
#include "../CudaAllocUtils.hpp"
#include "../TensorContract/TensorContract_GPU.hpp"
#include "../TensorContract/GradFns/NumberEncoderGradFn.hpp"
#include "ModelForwardOutputs.hpp"
#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../VerboseLogging.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GRIM {
namespace Forward {

#define MFWD_INFO(msg) do { \
    if constexpr (GRIM::VerboseLogging::ENABLE_AUTOGRAD_TRAINING_LOGS) { \
        std::cerr << "[ModelForward] INFO: " << msg << std::endl; \
    } \
} while(0)

namespace {

const TensorContract::Shape2D& requireTensor2DShape(
    const Tensor& tensor,
    const char* caller,
    const char* label);

constexpr int kPresetCodebookBlockSize = 256;

void checkPresetCodebookCuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": " + cudaGetErrorString(err));
    }
}

__global__ void kernel_quantize_preset_codebook(
    const float* __restrict__ proposals,
    const float* __restrict__ codebook,
    float* __restrict__ quantized,
    int* __restrict__ selected_codes,
    int rows,
    int codebook_size,
    int preset_dim) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared_proposal[];
    const float* proposal = proposals + static_cast<size_t>(row) * preset_dim;
    for (int d = threadIdx.x; d < preset_dim; d += blockDim.x) {
        shared_proposal[d] = proposal[d];
    }
    __syncthreads();

    float best_distance = FLT_MAX;
    int best_code = 0;
    for (int code = threadIdx.x; code < codebook_size; code += blockDim.x) {
        float distance = 0.0f;
        const float* entry = codebook + static_cast<size_t>(code) * preset_dim;
        for (int d = 0; d < preset_dim; ++d) {
            const float diff = shared_proposal[d] - entry[d];
            distance = fmaf(diff, diff, distance);
        }
        if (distance < best_distance || (distance == best_distance && code < best_code)) {
            best_distance = distance;
            best_code = code;
        }
    }

    __shared__ float shared_distance[kPresetCodebookBlockSize];
    __shared__ int shared_code[kPresetCodebookBlockSize];
    shared_distance[threadIdx.x] = best_distance;
    shared_code[threadIdx.x] = best_code;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float other_distance = shared_distance[threadIdx.x + stride];
            const int other_code = shared_code[threadIdx.x + stride];
            if (other_distance < shared_distance[threadIdx.x] ||
                (other_distance == shared_distance[threadIdx.x] &&
                 other_code < shared_code[threadIdx.x])) {
                shared_distance[threadIdx.x] = other_distance;
                shared_code[threadIdx.x] = other_code;
            }
        }
        __syncthreads();
    }

    const int selected = shared_code[0];
    if (threadIdx.x == 0) {
        selected_codes[row] = selected;
    }
    for (int d = threadIdx.x; d < preset_dim; d += blockDim.x) {
        quantized[static_cast<size_t>(row) * preset_dim + d] =
            codebook[static_cast<size_t>(selected) * preset_dim + d];
    }
}

__global__ void kernel_quantize_preset_codebook_backward(
    const float* __restrict__ grad_output,
    const int* __restrict__ selected_codes,
    float* __restrict__ grad_proposals,
    float* __restrict__ grad_codebook,
    int rows,
    int preset_dim) {
    const size_t idx =
        (static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(rows) * preset_dim;
    if (idx >= total) return;
    const int row = static_cast<int>(idx / static_cast<size_t>(preset_dim));
    const int d = static_cast<int>(idx - static_cast<size_t>(row) * preset_dim);
    const float grad = grad_output[idx];
    if (grad_proposals) {
        atomicAdd(grad_proposals + idx, grad);  // straight-through estimator
    }
    if (grad_codebook) {
        const int code = selected_codes[row];
        atomicAdd(grad_codebook + static_cast<size_t>(code) * preset_dim + d, grad);
    }
}

struct PresetCodebookQuantizeGradFn final : public GradFn {
    bool proposals_require_grad = false;
    bool codebook_requires_grad = false;
    float* grad_proposals = nullptr;
    float* grad_codebook = nullptr;
    std::shared_ptr<float> owned_grad_proposals;
    std::shared_ptr<float> owned_grad_codebook;
    std::shared_ptr<int> selected_codes;
    TensorContract::TensorShape proposals_shape;
    TensorContract::TensorShape codebook_shape;
    std::shared_ptr<GradFn> proposals_grad_fn;
    std::shared_ptr<GradFn> codebook_grad_fn;
    int rows = 0;
    int preset_dim = 0;

    PresetCodebookQuantizeGradFn() { op_name = "preset_codebook_quantize"; }

    void captureTensor(Tensor& tensor,
                       bool& requires_grad,
                       float*& grad,
                       std::shared_ptr<float>& owned_grad,
                       TensorContract::TensorShape& shape,
                       std::shared_ptr<GradFn>& upstream,
                       const char* label,
                       cudaStream_t stream) {
        requires_grad = tensor.requires_grad;
        shape = tensor.shape;
        if (!requires_grad) return;
        upstream = tensor.grad_fn;
        register_input(tensor.grad_fn);
        if (tensor.is_leaf) {
            tensor.ensure_grad();
            grad = tensor.grad_data();
            return;
        }
        float* buffer = nullptr;
        ::GRIM::CudaAlloc::cudaMallocOrThrow(
            reinterpret_cast<void**>(&buffer), tensor.numel() * sizeof(float), label);
        checkPresetCodebookCuda(
            cudaMemsetAsync(buffer, 0, tensor.numel() * sizeof(float), stream),
            "PresetCodebookQuantizeGradFn: zero gradient buffer failed");
        owned_grad.reset(buffer, [](float* p) { queueForDeferredCleanup(p); });
        grad = owned_grad.get();
    }

    void apply_impl(const Tensor& grad_output,
                    cudaStream_t stream,
                    const Batching::BatchPayload* backward_payload,
                    const Batching::BatchDeviceBindings* backward_bindings) override {
        setCurrentGradFnOp("preset_codebook_quantize", this);
        if (applied) return;
        applied = true;
        if (!grad_output.data || grad_output.numel() != static_cast<size_t>(rows) * preset_dim) {
            throw std::runtime_error("PresetCodebookQuantizeGradFn: invalid output gradient");
        }

        const size_t total = static_cast<size_t>(rows) * preset_dim;
        constexpr int block = 256;
        const int blocks = static_cast<int>((total + block - 1) / block);
        const int gx = std::min(blocks, 65535);
        const int gy = (blocks + gx - 1) / gx;
        kernel_quantize_preset_codebook_backward<<<dim3(gx, gy, 1), block, 0, stream>>>(
            grad_output.data,
            selected_codes.get(),
            proposals_require_grad ? grad_proposals : nullptr,
            codebook_requires_grad ? grad_codebook : nullptr,
            rows,
            preset_dim);
        checkPresetCodebookCuda(
            cudaGetLastError(),
            "PresetCodebookQuantizeGradFn: backward kernel launch failed");

        if (proposals_require_grad && proposals_grad_fn) {
            Tensor view;
            view.data = grad_proposals;
            view.shape = proposals_shape;
            view.owns_data = false;
            view.stream = stream;
            proposals_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }
        if (codebook_requires_grad && codebook_grad_fn) {
            Tensor view;
            view.data = grad_codebook;
            view.shape = codebook_shape;
            view.owns_data = false;
            view.stream = stream;
            codebook_grad_fn->apply(view, stream, backward_payload, backward_bindings);
        }
    }

    void release_saved() override {
        GradFn::release_saved();
        proposals_grad_fn.reset();
        codebook_grad_fn.reset();
        owned_grad_proposals.reset();
        owned_grad_codebook.reset();
        selected_codes.reset();
        grad_proposals = nullptr;
        grad_codebook = nullptr;
    }
};

Tensor quantizePresetCodebook(Tensor& proposals,
                              Tensor& codebook,
                              cudaStream_t stream) {
    const auto& proposal_shape = requireTensor2DShape(
        proposals, "quantizePresetCodebook", "preset proposals");
    const auto& codebook_shape = requireTensor2DShape(
        codebook, "quantizePresetCodebook", "preset codebook");
    if (proposal_shape.cols != codebook_shape.cols || codebook_shape.rows <= 0) {
        throw std::runtime_error("quantizePresetCodebook: incompatible proposal/codebook shapes");
    }

    const bool needs_grad = proposals.requires_grad || codebook.requires_grad;
    Tensor result = Tensor::empty(
        TensorContract::TensorShape::make_BSM(proposal_shape.rows, proposal_shape.cols),
        needs_grad,
        stream,
        "latent_preset_quantized");
    int* raw_indices = nullptr;
    ::GRIM::CudaAlloc::cudaMallocOrThrow(
        reinterpret_cast<void**>(&raw_indices),
        static_cast<size_t>(proposal_shape.rows) * sizeof(int),
        "latent_preset_code_indices");
    std::shared_ptr<int> index_owner(raw_indices, [](int* p) { queueForDeferredCleanup(p); });

    const size_t proposal_shared_bytes =
        static_cast<size_t>(proposal_shape.cols) * sizeof(float);
    kernel_quantize_preset_codebook<<<
        proposal_shape.rows,
        kPresetCodebookBlockSize,
        proposal_shared_bytes,
        stream>>>(
        proposals.data,
        codebook.data,
        result.data,
        raw_indices,
        proposal_shape.rows,
        codebook_shape.rows,
        proposal_shape.cols);
    checkPresetCodebookCuda(
        cudaGetLastError(),
        "quantizePresetCodebook: forward kernel launch failed");

    if (needs_grad) {
        result.is_leaf = false;
        auto grad_fn = std::make_shared<PresetCodebookQuantizeGradFn>();
        grad_fn->captureTensor(
            proposals,
            grad_fn->proposals_require_grad,
            grad_fn->grad_proposals,
            grad_fn->owned_grad_proposals,
            grad_fn->proposals_shape,
            grad_fn->proposals_grad_fn,
            "PresetCodebookQuantizeGradFn_proposals",
            stream);
        grad_fn->captureTensor(
            codebook,
            grad_fn->codebook_requires_grad,
            grad_fn->grad_codebook,
            grad_fn->owned_grad_codebook,
            grad_fn->codebook_shape,
            grad_fn->codebook_grad_fn,
            "PresetCodebookQuantizeGradFn_codebook",
            stream);
        grad_fn->selected_codes = std::move(index_owner);
        grad_fn->rows = proposal_shape.rows;
        grad_fn->preset_dim = proposal_shape.cols;
        result.grad_fn = grad_fn;
    }
    return result;
}

const char* graphPolicyName(const ModelForwardGraphPolicy& graph) {
    if (graph.connect_parameter_graph && graph.retain_backward_graph) {
        return graph.emit_mtp_logits ? "autograd_connected+mtp" : "autograd_connected";
    }
    if (!graph.connect_parameter_graph && !graph.retain_backward_graph) {
        return graph.emit_mtp_logits ? "read_only+mtp" : "read_only";
    }
    throw std::runtime_error("ModelForward: invalid graph policy — connect_parameter_graph and retain_backward_graph must agree at this boundary");
}

const TensorContract::Shape2D& requireTensor2DShape(
    const Tensor& tensor,
    const char* caller,
    const char* label) {
    tensor.require(label);
    if (!tensor.shape.is_2d_layout()) {
        throw std::runtime_error(std::string(caller) + ": " + label + " must be a 2D tensor");
    }
    return tensor.shape.as_2d();
}

void requireCenteringSequenceLengths(const Batching::BatchPayload& payload,
                                     const char* caller) {
    if (payload.batch_size <= 0 || payload.max_seq_len <= 0) {
        throw std::runtime_error(std::string(caller) + ": invalid payload geometry batch=" +
                                 std::to_string(payload.batch_size) + " seq=" +
                                 std::to_string(payload.max_seq_len));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != batch_size (" + std::to_string(payload.batch_size) + ")");
    }
    for (int b = 0; b < payload.batch_size; ++b) {
        const int row_len = payload.seq_lengths[static_cast<size_t>(b)];
        if (row_len <= 1 || row_len > payload.max_seq_len) {
            throw std::runtime_error(std::string(caller) + ": invalid seq_lengths[" +
                                     std::to_string(b) + "]=" + std::to_string(row_len) +
                                     " for padding-aware centering over max_seq_len=" +
                                     std::to_string(payload.max_seq_len));
        }
    }
}

int requirePayloadRowLength(const Batching::BatchPayload& payload,
                            int row,
                            const char* caller) {
    if (row < 0 || row >= payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": row index " +
                                 std::to_string(row) + " out of range for batch_size=" +
                                 std::to_string(payload.batch_size));
    }
    if (static_cast<int>(payload.seq_lengths.size()) != payload.batch_size) {
        throw std::runtime_error(std::string(caller) + ": payload.seq_lengths size (" +
                                 std::to_string(payload.seq_lengths.size()) +
                                 ") != batch_size (" + std::to_string(payload.batch_size) + ")");
    }
    const int row_len = payload.seq_lengths[static_cast<size_t>(row)];
    if (row_len <= 0 || row_len > payload.max_seq_len) {
        throw std::runtime_error(std::string(caller) + ": invalid seq_lengths[" +
                                 std::to_string(row) + "]=" + std::to_string(row_len) +
                                 " for payload.max_seq_len=" + std::to_string(payload.max_seq_len));
    }
    return row_len;
}

Tensor viewCommittedTensor(const Tensor& owned,
                           cudaStream_t stream,
                           const char* debug_name,
                           const char* caller) {
    if (!owned.data) {
        throw std::runtime_error(std::string(caller) + ": committed tensor data is NULL for " + debug_name);
    }
    Tensor view = Tensor::from_ptr(
        owned.data,
        owned.shape,
        false,
        owned.requires_grad,
        debug_name);
    view.is_leaf = false;
    view.grad_fn = owned.grad_fn;
    view.stream = stream;
    return view;
}

struct MTPHeadForwardView {
    Tensor weight;
    Tensor bias;
};

Tensor buildMtpHeadParameterView(
    Tensor& parameter,
    bool connect_parameter_graph,
    cudaStream_t stream,
    const char* name)
{
    if (!parameter.data) {
        throw std::runtime_error(std::string("executeModelForward: MTP parameter data is NULL for ") + name);
    }
    if (!connect_parameter_graph) {
        return parameter.detach(stream);
    }

    Tensor view = Tensor::from_ptr(
        parameter.data,
        parameter.shape,
        false,
        parameter.requires_grad,
        name);
    view.compute_precision = parameter.compute_precision;
    view.is_leaf = parameter.is_leaf;
    view.retain_grad = parameter.retain_grad;
    view.stream = stream;
    view.grad_fn = parameter.grad_fn;
    if (parameter.requires_grad && parameter.is_leaf) {
        parameter.ensure_grad();
        view.share_grad(parameter);
    }
    return view;
}

std::vector<MTPHeadForwardView> buildMtpHeadForwardViews(
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    int mtp_k,
    bool bias_enabled,
    bool connect_parameter_graph,
    cudaStream_t stream)
{
    std::vector<MTPHeadForwardView> views;
    if (mtp_k <= 0) {
        return views;
    }

    auto& mtp_head_parameter_tensors = parameter_registry.mtpHeadParameterTensors();
    if (static_cast<int>(mtp_head_parameter_tensors.size()) != mtp_k) {
        throw std::runtime_error(
            "executeModelForward: parameter_registry MTP head count=" +
            std::to_string(mtp_head_parameter_tensors.size()) +
            " != config.mtp_k=" + std::to_string(mtp_k));
    }

    views.reserve(static_cast<size_t>(mtp_k));
    for (int k = 0; k < mtp_k; ++k) {
        auto& head = mtp_head_parameter_tensors[static_cast<std::size_t>(k)];
        if (!head.weight.data) {
            throw std::runtime_error(
                "executeModelForward: MTP head " + std::to_string(k) +
            " has NULL weight tensor");
        }
        if (bias_enabled != static_cast<bool>(head.bias.data)) {
            throw std::runtime_error(
            "executeModelForward: MTP head bias allocation does not match mtp_bias_enabled");
        }

        MTPHeadForwardView view{};
        view.weight = buildMtpHeadParameterView(
            head.weight, connect_parameter_graph, stream, "mtp_head_weight_view");
        if (bias_enabled) {
            view.bias = buildMtpHeadParameterView(
                head.bias, connect_parameter_graph, stream, "mtp_head_bias_view");
        }
        views.push_back(std::move(view));
    }

    return views;
}

void materializeForwardMtpLogits(
    const ModelForwardRequest& request,
    const Batching::BatchPayload& payload,
    const HyperParameters::LMHeadLayerConstructionHP& lm_head_hp,
    const GRIM::LMHeadParameterTensors& lm_head_parameters,
    const HyperParameters::LatentTrajectoryPresetHP& latent_preset_hp,
    ModelForwardOutputs& forward_outputs) {
    forward_outputs.mtp_logits_tensors.clear();
    if (!request.graph.emit_mtp_logits) {
        return;
    }

    const bool mtp_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*request.config, "mtp_enabled");
    const int mtp_k = HyperParameters::snapshotTrainingConfigField<int>(*request.config, "mtp_k");
    const bool mtp_bias_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*request.config, "mtp_bias_enabled");
    if (!mtp_enabled || mtp_k <= 0) {
        throw std::runtime_error("executeModelForward: graph.emit_mtp_logits=true but config MTP is disabled");
    }
    auto mtp_heads = buildMtpHeadForwardViews(
        *request.parameter_registry,
        mtp_k,
        mtp_bias_enabled,
        request.graph.connect_parameter_graph,
        request.stream);

    // ═══════════════════════════════════════════════════════════════════════
    // Discrete-preset MTP logits (latent_trajectory_preset_use_mtp_logits):
    // head k logits = MTP head k projection of positional slot k decoded from
    // the selected atomic preset.
    //
    // The codebook is the mandatory bottleneck; the MTP heads remain the
    // token-space prediction owners. Future-token CE therefore teaches the
    // selected preset what belongs at each position in its token grouping.
    // ═══════════════════════════════════════════════════════════════════════
    if (latent_preset_hp.enabled && latent_preset_hp.use_mtp_logits) {
        Tensor& decoded_slots = forward_outputs.latent_preset_mtp_hidden;
        if (!decoded_slots.data) {
            throw std::runtime_error("executeModelForward: latent_preset_mtp_hidden is NULL — "
                "decoded preset slots must be materialized before MTP logits");
        }
        const int d_model = latent_preset_hp.d_model;
        const auto& traj_shape = requireTensor2DShape(decoded_slots, "executeModelForward", "latent_preset_mtp_hidden");
        if (traj_shape.rows != payload.total_tokens ||
            traj_shape.cols != mtp_k * d_model) {
            throw std::runtime_error("executeModelForward: latent_preset_mtp_hidden shape=[" +
                std::to_string(traj_shape.rows) + "," + std::to_string(traj_shape.cols) +
                "] expected=[" + std::to_string(payload.total_tokens) + "," +
                std::to_string(mtp_k * d_model) + "]");
        }
        if (!lm_head_parameters.weights.data) {
            throw std::runtime_error("executeModelForward: LM head weights are NULL for latent MTP logits");
        }

        forward_outputs.mtp_logits_tensors.reserve(static_cast<size_t>(mtp_k));
        for (int k = 0; k < mtp_k; ++k) {
            const auto& head = mtp_heads[static_cast<size_t>(k)];
            if (!head.weight.data) {
                throw std::runtime_error("executeModelForward: latent MTP head " + std::to_string(k) +
                    " weight tensor is NULL");
            }

            Tensor slot_hidden_k = autograd::slice_columns(
                decoded_slots,
                k * d_model,
                d_model,
                request.stream);
            // Mirror the main head's pre-projection normalization so the
            // decoded positional slots are scored in the same geometry
            // as live hidden states.
            if (lm_head_parameters.final_rms_gamma.data) {
                slot_hidden_k = autograd::rms_norm(
                    slot_hidden_k,
                    lm_head_parameters.final_rms_gamma,
                    lm_head_hp.rms_epsilon,
                    request.stream);
            }
            const auto& weight_shape = requireTensor2DShape(head.weight, "executeModelForward", "latent MTP head weight");
            if (weight_shape.rows != payload.vocab_size) {
                throw std::runtime_error("executeModelForward: latent MTP head " + std::to_string(k) +
                    " weight rows=" + std::to_string(weight_shape.rows) +
                    " != payload.vocab_size=" + std::to_string(payload.vocab_size));
            }
            if (weight_shape.cols != d_model) {
                throw std::runtime_error("executeModelForward: latent MTP head " + std::to_string(k) +
                    " weight cols=" + std::to_string(weight_shape.cols) +
                    " != d_model=" + std::to_string(d_model));
            }
            if (mtp_bias_enabled) {
                const auto& bias_shape = requireTensor2DShape(head.bias, "executeModelForward", "latent MTP head bias");
                if (head.bias.numel() != static_cast<size_t>(payload.vocab_size) ||
                    (bias_shape.cols != payload.vocab_size && bias_shape.rows != payload.vocab_size)) {
                    throw std::runtime_error("executeModelForward: latent MTP head bias shape mismatch");
                }
            }
            Tensor logits_k = autograd::matmul(
                slot_hidden_k,
                head.weight,
                request.stream,
                true);
            if (mtp_bias_enabled) {
                logits_k = autograd::broadcast_add(logits_k, head.bias, request.stream);
            }
            const auto& logits_shape = requireTensor2DShape(logits_k, "executeModelForward", "latent MTP head logits");
            if (logits_shape.rows != payload.total_tokens || logits_shape.cols != payload.vocab_size) {
                throw std::runtime_error("executeModelForward: latent MTP head " + std::to_string(k) +
                    " logits shape=[" + std::to_string(logits_shape.rows) + "," + std::to_string(logits_shape.cols) +
                    "] does not match payload [total_tokens=" + std::to_string(payload.total_tokens) +
                    ", vocab_size=" + std::to_string(payload.vocab_size) + "]");
            }
            forward_outputs.mtp_logits_tensors.push_back(std::move(logits_k));
        }
        if (!latent_preset_hp.use_mtp_hidden) {
            forward_outputs.latent_preset_mtp_hidden = Tensor();
        }
        return;
    }

    Tensor* mtp_input = forward_outputs.liveLmHeadInputOrNull();
    if (!mtp_input || !mtp_input->data) {
        throw std::runtime_error("executeModelForward: live LM-head input snapshot is NULL before MTP logits materialization");
    }

    const auto& input_shape = requireTensor2DShape(*mtp_input, "executeModelForward", "mtp_input");
    if (input_shape.rows != payload.total_tokens) {
        throw std::runtime_error("executeModelForward: mtp_input rows=" +
            std::to_string(input_shape.rows) + " != payload.total_tokens=" +
            std::to_string(payload.total_tokens));
    }

    forward_outputs.mtp_logits_tensors.reserve(static_cast<size_t>(mtp_k));
    for (int k = 0; k < mtp_k; ++k) {
        const auto& head = mtp_heads[static_cast<size_t>(k)];
        if (!head.weight.data) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " weight tensor is NULL");
        }

        const auto& weight_shape = requireTensor2DShape(head.weight, "executeModelForward", "MTP head weight");
        if (weight_shape.rows != payload.vocab_size) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " weight rows=" + std::to_string(weight_shape.rows) +
                " != payload.vocab_size=" + std::to_string(payload.vocab_size));
        }
        if (weight_shape.cols != input_shape.cols) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " weight cols=" + std::to_string(weight_shape.cols) +
                " != mtp_input cols=" + std::to_string(input_shape.cols));
        }
        if (mtp_bias_enabled) {
            const auto& bias_shape = requireTensor2DShape(head.bias, "executeModelForward", "MTP head bias");
            if (head.bias.numel() != static_cast<size_t>(payload.vocab_size) ||
                (bias_shape.cols != payload.vocab_size && bias_shape.rows != payload.vocab_size)) {
                throw std::runtime_error("executeModelForward: MTP head bias shape mismatch");
            }
        }

        Tensor logits_k = autograd::matmul(
            *mtp_input,
            head.weight,
            request.stream,
            true);
        if (mtp_bias_enabled) {
            logits_k = autograd::broadcast_add(logits_k, head.bias, request.stream);
        }
        const auto& logits_shape = requireTensor2DShape(logits_k, "executeModelForward", "MTP head logits");
        if (logits_shape.rows != payload.total_tokens || logits_shape.cols != payload.vocab_size) {
            throw std::runtime_error("executeModelForward: MTP head " + std::to_string(k) +
                " logits shape=[" + std::to_string(logits_shape.rows) + "," + std::to_string(logits_shape.cols) +
                "] does not match payload [total_tokens=" + std::to_string(payload.total_tokens) +
                ", vocab_size=" + std::to_string(payload.vocab_size) + "]");
        }
        forward_outputs.mtp_logits_tensors.push_back(std::move(logits_k));
    }
}

// Arg/option selector head: encode candidate atom-entry keys (detached, from the
// NumberEncoder) and score the live LM-head hidden state against this row's
// candidate window. Emits ModelForwardOutputs::selector_logits [total_tokens,
// num_pool_atoms]. W_q carries gradient only when the parameter graph is
// connected (training); keys are always detached (NumberEncoder trains via the
// input-side fusion). No-op when the pool is empty for this batch.
void materializeForwardSelectorLogits(
    const ModelForwardRequest& request,
    const Batching::BatchPayload& payload,
    ModelForwardOutputs& forward_outputs) {
    forward_outputs.selector_logits = Tensor();
    if (!request.graph.emit_selector_logits) {
        return;
    }

    const int num_pool_atoms = request.bindings->num_pool_atoms;
    if (num_pool_atoms <= 0) {
        return;  // No candidate entries in this batch — nothing to select among.
    }

    const auto number_encoder_hp = HyperParameters::numberEncoderConstructionHP(*request.config);
    if (!number_encoder_hp.enabled) {
        throw std::runtime_error("executeModelForward: graph.emit_selector_logits=true requires the NumberEncoder to be enabled (candidate keys are NumberEncoder-derived)");
    }
    if (!request.bindings->d_pool_digit_values || !request.bindings->d_pool_digit_pow10_index ||
        !request.bindings->d_pool_digit_mask || !request.bindings->d_pool_digit_slot_features ||
        !request.bindings->d_pool_global_features || !request.bindings->d_row_atom_offset) {
        throw std::runtime_error("executeModelForward: selector requested but candidate-pool device bindings are NULL");
    }

    Tensor* sel_input = forward_outputs.liveLmHeadInputOrNull();
    if (!sel_input || !sel_input->data) {
        throw std::runtime_error("executeModelForward: live LM-head input is NULL before selector materialization");
    }

    auto& ne = request.parameter_registry->requireNumberEncoderParameters("executeModelForward(selector)");
    // Connected (training): encode keys against the registered NumberEncoder
    // leaves so the selection loss accumulates gradient into them — the selector
    // teaches the encoder which candidate entries to keep distinguishable.
    // Read-only (inference): detached copies so the keys are forward-only and no
    // graph edges are retained (mirrors the W_q detach below).
    GRIM::NumberEncoderParameterTensors ne_detached{};
    const GRIM::NumberEncoderParameterTensors* ne_src = &ne;
    if (!request.graph.connect_parameter_graph) {
        ne_detached.digit_emb = ne.digit_emb.detach(request.stream);
        ne_detached.pow10_emb = ne.pow10_emb.detach(request.stream);
        ne_detached.W_c1 = ne.W_c1.detach(request.stream);
        ne_detached.b_c1 = ne.b_c1.detach(request.stream);
        ne_detached.W_c2 = ne.W_c2.detach(request.stream);
        ne_detached.W_g1 = ne.W_g1.detach(request.stream);
        ne_detached.b_g1 = ne.b_g1.detach(request.stream);
        ne_detached.W_g2 = ne.W_g2.detach(request.stream);
        ne_src = &ne_detached;
    }
    autograd::NumberEncoderForwardParams ne_params{};
    ne_params.digit_emb = &ne_src->digit_emb;
    ne_params.pow10_emb = &ne_src->pow10_emb;
    ne_params.W_c1 = &ne_src->W_c1;
    ne_params.b_c1 = &ne_src->b_c1;
    ne_params.W_c2 = &ne_src->W_c2;
    ne_params.W_g1 = &ne_src->W_g1;
    ne_params.b_g1 = &ne_src->b_g1;
    ne_params.W_g2 = &ne_src->W_g2;

    Tensor keys = autograd::encodeAtomEntryPoolKeys(
        ne_params, number_encoder_hp,
        request.bindings->d_pool_digit_values,
        request.bindings->d_pool_digit_pow10_index,
        request.bindings->d_pool_digit_mask,
        request.bindings->d_pool_digit_slot_features,
        request.bindings->d_pool_global_features,
        num_pool_atoms, request.stream);

    auto& sel = request.parameter_registry->requireSelectorParameters("executeModelForward(selector)");
    // Connected (training): score against the registered W_q leaf so gradient
    // accumulates into the optimizer-visible buffer. Read-only (inference): a
    // detached copy so no graph edges are retained.
    Tensor W_q_detached;
    const Tensor* W_q_ptr = &sel.W_q;
    if (!request.graph.connect_parameter_graph) {
        W_q_detached = sel.W_q.detach(request.stream);
        W_q_ptr = &W_q_detached;
    }

    const int d_model = HyperParameters::snapshotTrainingConfigField<int>(*request.config, "d_model");
    const float selector_scale = 1.0f / std::sqrt(static_cast<float>(d_model));

    forward_outputs.selector_logits = ArgSelector::argSelectorForward(
        *sel_input, *W_q_ptr, keys, payload,
        request.bindings->d_row_atom_offset, num_pool_atoms, selector_scale, request.stream);
}

GRIM::FeedForwardParameterTensors detachFeedForwardParameters(
    const GRIM::FeedForwardParameterTensors& parameters,
    bool output_bias_enabled,
    cudaStream_t stream) {
    GRIM::FeedForwardParameterTensors detached{};
    detached.W_gate = parameters.W_gate.detach(stream);
    detached.W1 = parameters.W1.detach(stream);
    detached.W2 = parameters.W2.detach(stream);
    if (output_bias_enabled) {
        detached.b2 = parameters.b2.detach(stream);
    }
    return detached;
}

GRIM::EncodingLayerParameterTensors detachEncodingLayerParameters(
    const GRIM::EncodingLayerParameterTensors& parameters,
    bool qkv_bias_enabled,
    bool output_bias_enabled,
    bool use_layer_scale,
    cudaStream_t stream) {
    GRIM::EncodingLayerParameterTensors detached{};
    detached.rms1_gamma = parameters.rms1_gamma.detach(stream);
    detached.rms2_gamma = parameters.rms2_gamma.detach(stream);
    detached.W_qkv = parameters.W_qkv.detach(stream);
    detached.W_o = parameters.W_o.detach(stream);
    if (qkv_bias_enabled) {
        detached.b_qkv = parameters.b_qkv.detach(stream);
    }
    if (output_bias_enabled) {
        detached.b_o = parameters.b_o.detach(stream);
    }
    if (use_layer_scale) {
        detached.layer_scale1 = parameters.layer_scale1.detach(stream);
        detached.layer_scale2 = parameters.layer_scale2.detach(stream);
    }
    return detached;
}

GRIM::LMHeadParameterTensors detachLmHeadParameters(
    const GRIM::LMHeadParameterTensors& parameters,
    cudaStream_t stream) {
    GRIM::LMHeadParameterTensors detached{};
    detached.owns_weights = false;
    detached.weights = parameters.weights.detach(stream);
    if (parameters.bias.data) {
        detached.bias = parameters.bias.detach(stream);
    }
    if (parameters.final_rms_gamma.data) {
        detached.final_rms_gamma = parameters.final_rms_gamma.detach(stream);
    }
    if (parameters.mlp_W_gate.data) {
        detached.mlp_W_gate = parameters.mlp_W_gate.detach(stream);
    }
    if (parameters.mlp_W_up.data) {
        detached.mlp_W_up = parameters.mlp_W_up.detach(stream);
    }
    if (parameters.mlp_W_down.data) {
        detached.mlp_W_down = parameters.mlp_W_down.detach(stream);
    }
    return detached;
}

GRIM::LatentTrajectoryPresetParameterTensors detachLatentTrajectoryPresetParameters(
    const GRIM::LatentTrajectoryPresetParameterTensors& parameters,
    const HyperParameters::LatentTrajectoryPresetHP& hp,
    cudaStream_t stream) {
    GRIM::LatentTrajectoryPresetParameterTensors detached{};
    detached.W_hidden_traj = parameters.W_hidden_traj.detach(stream);
    if (hp.hidden_bias_enabled) {
        detached.b_hidden_traj = parameters.b_hidden_traj.detach(stream);
    }
    detached.W_fuse = parameters.W_fuse.detach(stream);
    if (hp.fuse_bias_enabled) {
        detached.b_fuse = parameters.b_fuse.detach(stream);
    }
    detached.W_down = parameters.W_down.detach(stream);
    if (hp.down_bias_enabled) {
        detached.b_down = parameters.b_down.detach(stream);
    }
    detached.codebook = parameters.codebook.detach(stream);
    detached.W_slots = parameters.W_slots.detach(stream);
    detached.W_up = parameters.W_up.detach(stream);
    if (hp.up_bias_enabled) {
        detached.b_up = parameters.b_up.detach(stream);
    }
    detached.W_gate = parameters.W_gate.detach(stream);
    if (hp.gate_bias_enabled) {
        detached.b_gate = parameters.b_gate.detach(stream);
    }
    detached.fuse_norm_gamma = parameters.fuse_norm_gamma.detach(stream);
    detached.preset_norm_gamma = parameters.preset_norm_gamma.detach(stream);
    return detached;
}

void clearLatentTrajectoryPresetOutputs(ModelForwardOutputs& forward_outputs) {
    forward_outputs.latent_preset_encoder_slots = Tensor();
    forward_outputs.latent_preset_mtp_hidden = Tensor();
    forward_outputs.latent_preset_future_fused = Tensor();
    forward_outputs.latent_preset_future_entropy = Tensor();
    forward_outputs.latent_preset_z = Tensor();
    forward_outputs.latent_preset_quantized = Tensor();
    forward_outputs.latent_preset_vec = Tensor();
    forward_outputs.latent_preset_gate_pre = Tensor();
    forward_outputs.latent_preset_gate = Tensor();
    forward_outputs.latent_preset_injected = Tensor();
    forward_outputs.latent_preset_h_enhanced = Tensor();
}

void materializeLatentTrajectoryPresetActivations(
    const ModelForwardRequest& request,
    const Batching::BatchPayload& payload,
    const HyperParameters::LatentTrajectoryPresetHP& latent_preset_hp,
    ModelForwardOutputs& forward_outputs) {
    clearLatentTrajectoryPresetOutputs(forward_outputs);
    if (!latent_preset_hp.enabled) {
        return;
    }
    // KV-cache decode/prefill: the latent preset is a strictly row-local
    // post-encoder transform (matmul/bias/RMSNorm/gate over each h[t] row).
    // It reads only the active window's encoder_output_tensor ([q_len, d_model])
    // and needs no cross-step latent history, so the same math runs unchanged
    // over cached windows. Read-only graph policy is already enforced by
    // ModelForwardRequest::validate(); parameters are detached below.

    Tensor& hidden = forward_outputs.encoder_output_tensor;
    const auto& hidden_shape = requireTensor2DShape(
        hidden,
        "executeModelForward(latent_preset)",
        "latent preset hidden input");
    if (hidden_shape.rows != payload.total_tokens || hidden_shape.cols != latent_preset_hp.d_model) {
        throw std::runtime_error("executeModelForward(latent_preset): hidden shape=[" +
            std::to_string(hidden_shape.rows) + "," + std::to_string(hidden_shape.cols) +
            "] expected=[" + std::to_string(payload.total_tokens) + "," +
            std::to_string(latent_preset_hp.d_model) + "]");
    }

    auto* latent_parameters =
        request.parameter_registry->getLatentTrajectoryPresetParameters();
    if (!latent_parameters) {
        throw std::runtime_error("executeModelForward: latent_trajectory_preset_enabled=true but registry owner is NULL");
    }

    GRIM::LatentTrajectoryPresetParameterTensors detached_parameters{};
    GRIM::LatentTrajectoryPresetParameterTensors* params = latent_parameters;
    if (!request.graph.connect_parameter_graph) {
        detached_parameters = detachLatentTrajectoryPresetParameters(
            *latent_parameters,
            latent_preset_hp,
            request.stream);
        params = &detached_parameters;
    }

    auto requireTensorElements = [](const Tensor& tensor,
                                    std::size_t expected,
                                    const char* label) {
        tensor.require(label);
        if (tensor.numel() != expected) {
            throw std::runtime_error(std::string("executeModelForward(latent_preset): ") +
                                     label + " numel=" + std::to_string(tensor.numel()) +
                                     " expected=" + std::to_string(expected));
        }
    };

    const std::size_t trajectory_dim =
        static_cast<std::size_t>(latent_preset_hp.mtp_k) *
        static_cast<std::size_t>(latent_preset_hp.d_model);
    requireTensorElements(params->W_hidden_traj,
                          static_cast<std::size_t>(latent_preset_hp.d_model) * trajectory_dim,
                          "latent_preset.W_hidden_traj");
    requireTensorElements(params->W_fuse,
                          trajectory_dim *
                              static_cast<std::size_t>(latent_preset_hp.fuse_dim),
                          "latent_preset.W_fuse");
    requireTensorElements(params->W_down,
                          static_cast<std::size_t>(latent_preset_hp.fuse_dim) *
                              static_cast<std::size_t>(latent_preset_hp.preset_dim),
                          "latent_preset.W_down");
    requireTensorElements(params->codebook,
                          static_cast<std::size_t>(latent_preset_hp.codebook_size) *
                              static_cast<std::size_t>(latent_preset_hp.preset_dim),
                          "latent_preset.codebook");
    requireTensorElements(params->W_slots,
                          static_cast<std::size_t>(latent_preset_hp.preset_dim) * trajectory_dim,
                          "latent_preset.W_slots");
    requireTensorElements(params->W_up,
                          static_cast<std::size_t>(latent_preset_hp.preset_dim) *
                              static_cast<std::size_t>(latent_preset_hp.d_model),
                          "latent_preset.W_up");
    requireTensorElements(params->W_gate,
                          static_cast<std::size_t>(latent_preset_hp.d_model + latent_preset_hp.fuse_dim),
                          "latent_preset.W_gate");
    const auto validateOptionalBias = [&](const Tensor& tensor,
                                          bool enabled,
                                          std::size_t expected,
                                          const char* label) {
        if (enabled) {
            requireTensorElements(tensor, expected, label);
        } else if (tensor.data) {
            throw std::runtime_error(std::string("executeModelForward(latent_preset): ") +
                                     label + " is allocated while its bias gate is false");
        }
    };
    validateOptionalBias(params->b_hidden_traj, latent_preset_hp.hidden_bias_enabled, trajectory_dim, "latent_preset.b_hidden_traj");
    validateOptionalBias(params->b_fuse, latent_preset_hp.fuse_bias_enabled, static_cast<std::size_t>(latent_preset_hp.fuse_dim), "latent_preset.b_fuse");
    validateOptionalBias(params->b_down, latent_preset_hp.down_bias_enabled, static_cast<std::size_t>(latent_preset_hp.preset_dim), "latent_preset.b_down");
    validateOptionalBias(params->b_up, latent_preset_hp.up_bias_enabled, static_cast<std::size_t>(latent_preset_hp.d_model), "latent_preset.b_up");
    validateOptionalBias(params->b_gate, latent_preset_hp.gate_bias_enabled, 1, "latent_preset.b_gate");

    // Continuous preset encoder: h[t] -> K coordinated slot features. These
    // are fused into one proposal solely to select an atomic codebook entry;
    // they cannot reach the MTP heads without passing through that entry.
    forward_outputs.latent_preset_encoder_slots = autograd::matmul(
        hidden,
        params->W_hidden_traj,
        request.stream);
    if (latent_preset_hp.hidden_bias_enabled) {
        forward_outputs.latent_preset_encoder_slots = autograd::broadcast_add(
            forward_outputs.latent_preset_encoder_slots,
            params->b_hidden_traj,
            request.stream);
    }

    forward_outputs.latent_preset_future_fused = autograd::matmul(
        forward_outputs.latent_preset_encoder_slots,
        params->W_fuse,
        request.stream);
    if (latent_preset_hp.fuse_bias_enabled) {
        forward_outputs.latent_preset_future_fused = autograd::broadcast_add(
            forward_outputs.latent_preset_future_fused,
            params->b_fuse,
            request.stream);
    }
    forward_outputs.latent_preset_future_fused = autograd::rms_norm(
        forward_outputs.latent_preset_future_fused,
        params->fuse_norm_gamma,
        1.0e-5f,
        request.stream);

    forward_outputs.latent_preset_z = autograd::matmul(
        forward_outputs.latent_preset_future_fused,
        params->W_down,
        request.stream);
    if (latent_preset_hp.down_bias_enabled) {
        forward_outputs.latent_preset_z = autograd::broadcast_add(
            forward_outputs.latent_preset_z,
            params->b_down,
            request.stream);
    }
    forward_outputs.latent_preset_z = autograd::rms_norm(
        forward_outputs.latent_preset_z,
        params->preset_norm_gamma,
        1.0e-5f,
        request.stream);

    // The discrete preset is the mandatory content bottleneck. Both positional
    // MTP slots and the main-head residual value decode from the selected entry;
    // the continuous branch only controls whether that value is injected.
    forward_outputs.latent_preset_quantized = quantizePresetCodebook(
        forward_outputs.latent_preset_z,
        params->codebook,
        request.stream);
    forward_outputs.latent_preset_mtp_hidden = autograd::matmul(
        forward_outputs.latent_preset_quantized,
        params->W_slots,
        request.stream);

    forward_outputs.latent_preset_vec = autograd::matmul(
        forward_outputs.latent_preset_quantized,
        params->W_up,
        request.stream);
    if (latent_preset_hp.up_bias_enabled) {
        forward_outputs.latent_preset_vec = autograd::broadcast_add(
            forward_outputs.latent_preset_vec,
            params->b_up,
            request.stream);
    }

    Tensor gate_input = autograd::concat(
        hidden,
        forward_outputs.latent_preset_future_fused,
        request.stream);
    forward_outputs.latent_preset_gate_pre = autograd::matmul(
        gate_input,
        params->W_gate,
        request.stream);
    if (latent_preset_hp.gate_bias_enabled) {
        forward_outputs.latent_preset_gate_pre = autograd::broadcast_add(
            forward_outputs.latent_preset_gate_pre,
            params->b_gate,
            request.stream);
    }
    forward_outputs.latent_preset_gate = autograd::sigmoid(
        forward_outputs.latent_preset_gate_pre,
        request.stream,
        forward_outputs.latent_preset_gate_pre.data);

    Tensor gated_preset = autograd::broadcast_row_mul(
        forward_outputs.latent_preset_gate,
        forward_outputs.latent_preset_vec,
        request.stream);
    forward_outputs.latent_preset_injected = autograd::mul_scalar(
        gated_preset,
        latent_preset_hp.preset_scale,
        request.stream);
    forward_outputs.latent_preset_h_enhanced = autograd::add(
        hidden,
        forward_outputs.latent_preset_injected,
        request.stream);
}

}  // namespace

void ModelForwardRequest::validate(const char* caller) const {
    if (!config) throw std::runtime_error(std::string(caller) + ": config is NULL");
    if (!gpu_encoder) throw std::runtime_error(std::string(caller) + ": gpu_encoder is NULL");
    if (!parameter_registry) throw std::runtime_error(std::string(caller) + ": parameter_registry is NULL");
    (void)parameter_registry->requireEmbeddingParameters(caller);
    (void)parameter_registry->requireLmHeadParameters(caller);
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*config, "num_layers");
    if (static_cast<int>(parameter_registry->encodingLayerParameterTensors().size()) != num_layers) {
        throw std::runtime_error(std::string(caller) + ": parameter_registry encoder tensor count mismatch. size=" +
                                 std::to_string(parameter_registry->encodingLayerParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }
    if (static_cast<int>(parameter_registry->feedForwardParameterTensors().size()) != num_layers) {
        throw std::runtime_error(std::string(caller) + ": parameter_registry FFN tensor count mismatch. size=" +
                                 std::to_string(parameter_registry->feedForwardParameterTensors().size()) +
                                 " num_layers=" + std::to_string(num_layers));
    }
    if (!this->pbm) throw std::runtime_error(std::string(caller) + ": pbm is NULL");
    if (!cublas_handle) throw std::runtime_error(std::string(caller) + ": cublas_handle is NULL");
    if (!stream) throw std::runtime_error(std::string(caller) + ": stream is NULL");
    if (!payload) throw std::runtime_error(std::string(caller) + ": payload is NULL");
    if (!bindings) throw std::runtime_error(std::string(caller) + ": bindings is NULL");
    (void)graphPolicyName(graph);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*config);
    if (execution_hp.enabled) {
        if (!execution_block_enabled) {
            throw std::runtime_error(std::string(caller) + ": execution_block_enabled=false while ExecutionBlockConstructionHP.enabled=true");
        }
        (void)parameter_registry->requireExecutionBlockParameters(caller);
    } else if (execution_block_enabled) {
        throw std::runtime_error(std::string(caller) + ": execution_block_enabled=true while ExecutionBlockConstructionHP.enabled=false");
    }
    if (payload->batch_size <= 0) throw std::runtime_error(std::string(caller) + ": BatchPayload.batch_size <= 0");
    if (payload->max_seq_len <= 0) throw std::runtime_error(std::string(caller) + ": BatchPayload.max_seq_len <= 0");
    if (static_cast<int>(payload->seq_lengths.size()) != payload->batch_size) {
        throw std::runtime_error(std::string(caller) + ": payload.seq_lengths size (" +
                                 std::to_string(payload->seq_lengths.size()) +
                                 ") != payload.batch_size (" + std::to_string(payload->batch_size) + ")");
    }
    for (int b = 0; b < payload->batch_size; ++b) {
        const int row_len = payload->seq_lengths[static_cast<size_t>(b)];
        if (row_len <= 0 || row_len > payload->max_seq_len) {
            throw std::runtime_error(std::string(caller) + ": invalid seq_lengths[" +
                                     std::to_string(b) + "]=" + std::to_string(row_len) +
                                     " for payload.max_seq_len=" + std::to_string(payload->max_seq_len));
        }
    }
    if (graph.emit_mtp_logits) {
        const bool mtp_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*config, "mtp_enabled");
        const int mtp_k = HyperParameters::snapshotTrainingConfigField<int>(*config, "mtp_k");
        if (!mtp_enabled || mtp_k <= 0) {
            throw std::runtime_error(std::string(caller) + ": graph.emit_mtp_logits=true while config MTP is disabled");
        }
        const auto& mtp_heads = parameter_registry->mtpHeadParameterTensors();
        if (static_cast<int>(mtp_heads.size()) != mtp_k) {
            throw std::runtime_error(std::string(caller) + ": parameter_registry MTP head count=" +
                                     std::to_string(mtp_heads.size()) + " != config.mtp_k=" +
                                     std::to_string(mtp_k));
        }
    }
    if (graph.emit_selector_logits) {
        const bool selector_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*config, "selector_enabled");
        if (!selector_enabled) {
            throw std::runtime_error(std::string(caller) + ": graph.emit_selector_logits=true while config.selector_enabled=false");
        }
        const bool number_encoder_enabled = HyperParameters::snapshotTrainingConfigField<bool>(*config, "number_encoder_enabled");
        if (!number_encoder_enabled) {
            throw std::runtime_error(std::string(caller) + ": graph.emit_selector_logits=true requires number_encoder_enabled=true (selector keys are NumberEncoder-derived)");
        }
        (void)parameter_registry->requireSelectorParameters(caller);
    }
    if (kv_cache) {
        if (graph.connect_parameter_graph || graph.retain_backward_graph) {
            throw std::runtime_error(std::string(caller) + ": kv_cache requires a read-only graph policy (connect_parameter_graph == retain_backward_graph == false)");
        }
        if (graph.enable_dropout) {
            throw std::runtime_error(std::string(caller) + ": kv_cache decode is read-only and cannot run with dropout");
        }
        if (payload->batch_size != 1) {
            throw std::runtime_error(std::string(caller) + ": kv_cache decode requires payload.batch_size == 1");
        }
        if (execution_block_enabled) {
            throw std::runtime_error(std::string(caller) + ": kv_cache decode does not yet support the execution block; disable it for inference");
        }
        if (!kv_cache->allocated) {
            throw std::runtime_error(std::string(caller) + ": kv_cache is not allocated (call ensureAllocated/beginSession before the forward)");
        }
        const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*config, "num_layers");
        if (kv_cache->num_layers != num_layers) {
            throw std::runtime_error(std::string(caller) + ": kv_cache.num_layers=" +
                                     std::to_string(kv_cache->num_layers) + " != config.num_layers=" +
                                     std::to_string(num_layers));
        }
        if (kv_cache->host_seqlen + payload->total_tokens > kv_cache->cache_max_seq) {
            throw std::runtime_error(std::string(caller) + ": kv_cache overflow: current fill=" +
                                     std::to_string(kv_cache->host_seqlen) + " + q_len=" +
                                     std::to_string(payload->total_tokens) + " > cache_max_seq=" +
                                     std::to_string(kv_cache->cache_max_seq));
        }
    }
}

ModelForwardOutputs executeModelForward(const ModelForwardRequest& request,
                                        ModelForwardRuntimePayload& runtime_payload) {
    request.validate("executeModelForward");
    const auto* cfg = request.config;
    const auto embedding_hp = HyperParameters::embeddingLayerConstructionHP(*cfg);
    const auto encoder_hp = HyperParameters::encoderLayerConstructionHP(*cfg);
    const auto execution_hp = HyperParameters::executionBlockConstructionHP(*cfg);
    const auto lm_head_hp = HyperParameters::lmHeadLayerConstructionHP(*cfg);
    const auto latent_preset_hp = HyperParameters::latentTrajectoryPresetHP(*cfg);
    const bool center_encoder_residuals = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "center_encoder_residuals");
    const bool lm_head_center_hidden_states = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "lm_head_center_hidden_states");
    const int d_model = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "d_model");
    const auto positional_encoding = HyperParameters::snapshotTrainingConfigField<HyperParameters::PositionalEncodingType>(*cfg, "positional_encoding");
    const float dropout_rate = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "dropout_rate");
    const int num_layers = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "num_layers");
    const bool use_layer_scale = HyperParameters::snapshotTrainingConfigField<bool>(*cfg, "use_layer_scale");
    const int execution_block_layer = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_layer");
    const int execution_block_num_steps = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_steps");
    const int execution_block_num_slots = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_slots");
    const int execution_block_num_ops = HyperParameters::snapshotTrainingConfigField<int>(*cfg, "execution_block_num_ops");
    const float execution_block_temp_start = HyperParameters::snapshotTrainingConfigField<float>(*cfg, "execution_block_temp_start");
    const bool execution_block_active = execution_hp.enabled && request.execution_block_enabled;
    auto* execution_block_parameters = execution_block_active
        ? &request.parameter_registry->requireExecutionBlockParameters("executeModelForward")
        : nullptr;

    runtime_payload.validate(
        "executeModelForward",
        execution_block_active);

    const auto& payload = *request.payload;
    auto& runtime = runtime_payload;
    ModelForwardOutputs forward_outputs;
    if (execution_block_active) {
        forward_outputs.ensureExecutionBatchGeometry(
            static_cast<size_t>(payload.batch_size),
            "executeModelForward");
        runtime.execution_runtime->ensureBatchGeometry(
            static_cast<size_t>(payload.batch_size),
            "executeModelForward");
    }
    const auto* bindings = request.bindings;
    const auto& embedding_parameters = request.parameter_registry->requireEmbeddingParameters("executeModelForward");
    const auto& lm_head_parameters = request.parameter_registry->requireLmHeadParameters("executeModelForward");
    const bool connect_parameter_graph = request.graph.connect_parameter_graph;
    const bool retain_backward_graph = request.graph.retain_backward_graph;
    const bool dropout_enabled = request.graph.enable_dropout;
    const bool emit_mtp_logits = request.graph.emit_mtp_logits;

    if (center_encoder_residuals || lm_head_center_hidden_states) {
        requireCenteringSequenceLengths(payload, "ModelForward");
    }

    const int total_tokens = payload.total_tokens;

    MFWD_INFO("forward: batch=" << payload.batch_size << " seq=" << payload.max_seq_len
              << " tokens=" << total_tokens << " vocab=" << payload.vocab_size
              << " graph=" << graphPolicyName(request.graph));

    autograd::set_autograd_cublas_handle(request.cublas_handle);

    int* token_ids = bindings->d_input_ids;
    if (!token_ids) {
        throw std::runtime_error("ModelForward: input token device pointer is NULL");
    }

    Tensor emb_weights_view;
    const Tensor* emb_weights = &embedding_parameters.token_weights;
    if (!connect_parameter_graph) {
        emb_weights_view = embedding_parameters.token_weights.detach(request.stream);
        emb_weights = &emb_weights_view;
    }
    if (!emb_weights->data) {
        throw std::runtime_error("ModelForward: embedding token_weights.data is NULL");
    }

    if (!emb_weights->shape.is_valid()) {
        throw std::runtime_error("ModelForward: embedding token_weights.shape is INVALID - EmbeddingLayer MUST initialize with correct shape [vocab_size="
                                + std::to_string(payload.vocab_size) + ", d_model=" + std::to_string(d_model) + "]");
    }

    const float embedding_scale = embedding_hp.embedding_scale;
    Tensor emb_output = autograd::embedding(
        *emb_weights,
        payload,
        *bindings,
        request.stream,
        embedding_scale);

    MFWD_INFO("Step 1b: No position embeddings (using "
              << HyperParameters::positionalEncodingTypeToString(positional_encoding)
              << " inside attention)");

    forward_outputs.embedding_tensor = std::move(emb_output);
    MFWD_INFO("Step 1: Token embedding complete, shape=[" << total_tokens << ", " << d_model
              << "] scale=" << embedding_scale);

    // ─── Step 1n: NumberEncoder numeric-meaning fusion ──────────────────────
    // x_t = token_embedding[<INT>/<FLOAT>] + number_embedding(arg_number).
    // Selection-side input path (docs/ATOM_SELECTOR_IMPLEMENTATION_PLAN.md):
    // digit-place contribution slots are pooled per numeric atom and added at
    // that atom's token position; all non-atom rows receive exact zero. The
    // channels are CURRENT-token metadata only — next-token atom metadata is
    // supervision and never enters this input boundary.
    const auto number_encoder_hp = HyperParameters::numberEncoderConstructionHP(*cfg);
    std::vector<Tensor> number_encoder_detached_params;  // keep-alive across the call window
    Tensor number_encoder_out;                           // keep-alive across the call window
    if (number_encoder_hp.enabled) {
        if (payload.number_encoder_digit_slots != number_encoder_hp.max_digit_slots) {
            throw std::runtime_error(
                "ModelForward: payload.number_encoder_digit_slots=" +
                std::to_string(payload.number_encoder_digit_slots) +
                " != config max_digit_slots=" +
                std::to_string(number_encoder_hp.max_digit_slots) +
                " — payload was built against a different NumberEncoder config");
        }
        if (payload.authoredAtomCount() > 0) {
            auto& number_encoder_parameters =
                request.parameter_registry->requireNumberEncoderParameters("executeModelForward");
            autograd::NumberEncoderForwardParams ne_params{};
            if (connect_parameter_graph) {
                ne_params.digit_emb = &number_encoder_parameters.digit_emb;
                ne_params.pow10_emb = &number_encoder_parameters.pow10_emb;
                ne_params.W_c1 = &number_encoder_parameters.W_c1;
                ne_params.b_c1 = &number_encoder_parameters.b_c1;
                ne_params.W_c2 = &number_encoder_parameters.W_c2;
                ne_params.W_g1 = &number_encoder_parameters.W_g1;
                ne_params.b_g1 = &number_encoder_parameters.b_g1;
                ne_params.W_g2 = &number_encoder_parameters.W_g2;
            } else {
                number_encoder_detached_params.reserve(8);
                number_encoder_detached_params.push_back(number_encoder_parameters.digit_emb.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.pow10_emb.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_c1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.b_c1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_c2.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_g1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.b_g1.detach(request.stream));
                number_encoder_detached_params.push_back(number_encoder_parameters.W_g2.detach(request.stream));
                ne_params.digit_emb = &number_encoder_detached_params[0];
                ne_params.pow10_emb = &number_encoder_detached_params[1];
                ne_params.W_c1 = &number_encoder_detached_params[2];
                ne_params.b_c1 = &number_encoder_detached_params[3];
                ne_params.W_c2 = &number_encoder_detached_params[4];
                ne_params.W_g1 = &number_encoder_detached_params[5];
                ne_params.b_g1 = &number_encoder_detached_params[6];
                ne_params.W_g2 = &number_encoder_detached_params[7];
            }
            number_encoder_out = autograd::number_encode(
                ne_params, number_encoder_hp, payload, *bindings, request.stream);
            forward_outputs.embedding_tensor = autograd::residual_add(
                forward_outputs.embedding_tensor, number_encoder_out, request.stream);
            MFWD_INFO("Step 1n: NumberEncoder fused into " << payload.authoredAtomCount()
                      << " numeric atom positions (digit_slots=" << number_encoder_hp.max_digit_slots
                      << ", d_hidden=" << number_encoder_hp.d_hidden << ")");
        }
    }

    if (dropout_enabled && dropout_rate > 0.0f) {
        const uint64_t emb_dropout_seed = request.batch_idx * 2654435761ULL + 500;
        constexpr uint64_t kEmbeddingDropoutMaskStream = 0x0005000000000001ULL;
        forward_outputs.embedding_tensor = autograd::dropout(
            forward_outputs.embedding_tensor,
            dropout_rate,
            emb_dropout_seed,
            request.stream,
            kEmbeddingDropoutMaskStream);
        MFWD_INFO("Step 1c: Embedding-fusion dropout applied"
                  << " (p=" << dropout_rate << ", batch_idx=" << request.batch_idx << ")");
    }

    if (!request.gpu_encoder) {
        throw std::runtime_error("ModelForward: gpu_encoder is NULL - pass encoder in request");
    }

    forward_outputs.encoder_layer_outputs.clear();
    forward_outputs.clearRetainedLayerOutputs();
    forward_outputs.embedding_tensor.is_leaf = false;

    if (!retain_backward_graph) {
        Tensor running;
        forward_outputs.reserveLayerOutputs(num_layers);
        MFWD_INFO("Step 2: Running " << num_layers << " encoder layers (no_grad)...");

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = request.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("ModelForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }
            if (layer_idx > 0) {
                cudaError_t sync_err = cudaStreamSynchronize(request.stream);
                if (sync_err != cudaSuccess) {
                    throw std::runtime_error("ModelForward(no_grad): cudaStreamSynchronize failed after layer " +
                        std::to_string(layer_idx - 1) + ": " + cudaGetErrorString(sync_err));
                }
            }

            forward_outputs.pushLayerOutputs();

            Tensor& layer_input = (layer_idx == 0) ? forward_outputs.embedding_tensor : running;

            const auto& encoding_parameters = request.parameter_registry->requireEncodingLayerParameters(
                layer_idx,
                "executeModelForward(no_grad)");
            const auto& ffn_parameters = request.parameter_registry->requireFeedForwardParameters(
                layer_idx,
                "executeModelForward(no_grad)");
            const GRIM::EncodingLayerParameterTensors* encoding_parameter_ptr = &encoding_parameters;
            const GRIM::FeedForwardParameterTensors* ffn_parameter_ptr = &ffn_parameters;
            GRIM::EncodingLayerParameterTensors detached_encoding_parameters{};
            GRIM::FeedForwardParameterTensors detached_ffn_parameters{};
            if (!connect_parameter_graph) {
                detached_encoding_parameters = detachEncodingLayerParameters(
                    encoding_parameters,
                    encoder_hp.attention_qkv_bias_enabled,
                    encoder_hp.attention_output_bias_enabled,
                    use_layer_scale,
                    request.stream);
                detached_ffn_parameters = detachFeedForwardParameters(
                    ffn_parameters,
                    encoder_hp.ffn_output_bias_enabled,
                    request.stream);
                encoding_parameter_ptr = &detached_encoding_parameters;
                ffn_parameter_ptr = &detached_ffn_parameters;
            }
            // Inference KV-cache path: every layer reads/appends its own cache at
            // the SAME cache_seqlens offset (the fill before this forward). The
            // counter is advanced once, after the layer loop.
            KvCacheLayerView cache_view{};
            const KvCacheLayerView* cache_view_ptr = nullptr;
            if (request.kv_cache) {
                cache_view = request.kv_cache->layerView(layer_idx);
                cache_view_ptr = &cache_view;
            }
            forwardEncodingLayer(
                enc_layer->hp(),
                enc_layer->requireFeedForwardCompute("executeModelForward(no_grad)"),
                layer_input,
                payload,
                *request.pbm,
                request.stream,
                request.cublas_handle,
                forward_outputs,
                request.batch_idx,
                false,
                layer_idx,
                encoding_parameter_ptr,
                ffn_parameter_ptr,
                cache_view_ptr);
            Tensor layer_output_view = viewCommittedTensor(
                forward_outputs.output_per_layer[static_cast<size_t>(layer_idx)],
                request.stream,
                "enc_layer_output",
                "executeModelForward(no_grad)");

            Tensor owned = Tensor::empty(layer_output_view.shape, false, request.stream, "no_grad_layer_output");
            const size_t bytes = static_cast<size_t>(layer_output_view.shape.total_elements()) * sizeof(float);
            cudaError_t cp_err = cudaMemcpyAsync(owned.data, layer_output_view.data, bytes, cudaMemcpyDeviceToDevice, request.stream);
            if (cp_err != cudaSuccess) {
                throw std::runtime_error("ModelForward(no_grad): copy layer output failed: " +
                    std::string(cudaGetErrorString(cp_err)));
            }
            cudaError_t sync_err = cudaStreamSynchronize(request.stream);
            if (sync_err != cudaSuccess) {
                throw std::runtime_error("ModelForward(no_grad): sync after layer output copy failed: " +
                    std::string(cudaGetErrorString(sync_err)));
            }
            running = std::move(owned);
        }

        cudaError_t enc_sync = cudaStreamSynchronize(request.stream);
        if (enc_sync != cudaSuccess) {
            throw std::runtime_error("ModelForward(no_grad): CUDA error after encoder layers: " +
                std::string(cudaGetErrorString(enc_sync)) + " (illegal access usually means a kernel wrote/read out of bounds)");
        }

        // KV-cache path: all layers have appended q_len tokens at the prior fill
        // offset. Advance the shared cache_seqlens exactly once so the next forward
        // (decode step or speculative verification) attends over the new prefix.
        if (request.kv_cache) {
            request.kv_cache->advance(total_tokens, request.stream);
        }

        forward_outputs.encoder_output_tensor = std::move(running);
        forward_outputs.encoder_output_tensor.requires_grad = false;
        forward_outputs.encoder_output_tensor.grad_fn.reset();
        forward_outputs.encoder_output_tensor.stream = request.stream;
        MFWD_INFO("Step 2: All " << num_layers << " encoder layers complete (no_grad)");
    } else {
        forward_outputs.encoder_layer_outputs.reserve(num_layers);
        forward_outputs.reserveLayerOutputs(num_layers);

        MFWD_INFO("Step 2: Running " << num_layers << " encoder layers with retained graph...");
        MFWD_INFO("  embedding_tensor.grad_fn=" << (void*)forward_outputs.embedding_tensor.grad_fn.get()
                  << " requires_grad=" << forward_outputs.embedding_tensor.requires_grad);

        int exec_layer = -1;
        int exec_K = 0;
        if (execution_block_active) {
            exec_layer = execution_block_layer;
            if (exec_layer < 0) exec_layer = num_layers - 2;
            if (exec_layer < 0) exec_layer = 0;
            if (exec_layer >= num_layers) exec_layer = num_layers - 1;
            exec_K = execution_block_num_steps;
        }

        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
            auto* enc_layer = request.gpu_encoder->getLayer(layer_idx);
            if (!enc_layer) {
                throw std::runtime_error("ModelForward: Encoder layer " + std::to_string(layer_idx) + " is NULL");
            }

            forward_outputs.pushLayerOutputs();

            const Tensor* layer_input = (layer_idx == 0)
                ? &forward_outputs.embedding_tensor
                : &forward_outputs.encoder_layer_outputs.back();
            Tensor execution_read_augmented_input;
            const auto& encoding_parameters = request.parameter_registry->requireEncodingLayerParameters(
                layer_idx,
                "executeModelForward(retained_graph)");
            const auto& ffn_parameters = request.parameter_registry->requireFeedForwardParameters(
                layer_idx,
                "executeModelForward(retained_graph)");

            if (exec_layer >= 0 && layer_idx > exec_layer
                && execution_block_active
                && !forward_outputs.exec_memories.empty()) {
                // executeStep(...) may export the immediate step result on the
                // execution layer output, but persistent ExecutionMemory is a
                // downstream side channel. Its first consumer is the next
                // layer input at the row-final token only.
                bool has_execution_readback = false;
                for (int b = 0; b < payload.batch_size; ++b) {
                    const bool row_exec_active = !payload.execution_active.empty()
                        && payload.execution_active[b];
                    if (!row_exec_active) continue;
                    const Tensor& read_source = has_execution_readback
                        ? execution_read_augmented_input
                        : *layer_input;
                    const int row_len = requirePayloadRowLength(
                        payload, b, "ModelForward ExecutionBlock next-layer input readback");
                    const int final_token_offset = b * payload.max_seq_len + row_len - 1;
                    Tensor row_delta = GRIM::executionBlockCrossAttentionRead(
                        execution_hp, read_source, forward_outputs.exec_memories[b], *execution_block_parameters,
                        total_tokens, request.stream,
                        final_token_offset, 1,
                        runtime.read_gate_accum_tensor
                            ? runtime.read_gate_accum_tensor->data
                            : nullptr);
                    Tensor padded = autograd::zero_pad(
                        row_delta, final_token_offset, total_tokens, request.stream);
                    execution_read_augmented_input = autograd::add(
                        read_source, padded, request.stream);
                    has_execution_readback = true;
                }
                if (has_execution_readback) {
                    layer_input = &execution_read_augmented_input;
                }
            }

            forwardEncodingLayer(
                enc_layer->hp(),
                enc_layer->requireFeedForwardCompute("executeModelForward(retained_graph)"),
                *layer_input,
                payload,
                *request.pbm,
                request.stream,
                request.cublas_handle,
                forward_outputs,
                request.batch_idx,
                dropout_enabled,
                layer_idx,
                &encoding_parameters,
                &ffn_parameters);
            Tensor layer_output = viewCommittedTensor(
                forward_outputs.output_per_layer[static_cast<size_t>(layer_idx)],
                request.stream,
                "enc_layer_output",
                "executeModelForward(retained_graph)");

            if (layer_idx == exec_layer && execution_block_active) {
                const int V = execution_block_num_slots;
                const int nop = execution_block_num_ops;

                float T = execution_block_temp_start;

                auto& execution_runtime = *runtime.execution_runtime;
                Forward::provisionExecutionForwardRuntime(
                    payload.execution_active,
                    payload.batch_size,
                    execution_hp.num_slots,
                    execution_hp.atom_embedding_dim,
                    execution_hp.d_model,
                    execution_hp.d_key,
                    execution_hp.d_type,
                    connect_parameter_graph,
                    request.stream,
                    forward_outputs,
                    execution_runtime);
                execution_runtime.ensureDiagnostics(request.stream);

                for (int b = 0; b < payload.batch_size; ++b) {
                    const bool row_exec_active = !payload.execution_active.empty()
                        && payload.execution_active[b];

                    if (!row_exec_active) continue;

                    auto& M_b = forward_outputs.exec_memories[b];

                    const int tok_off = b * payload.max_seq_len;
                    const int row_len = requirePayloadRowLength(payload, b, "ModelForward ExecutionBlock bootstrap");

                    if (!request.bindings || !request.bindings->d_token_to_slot_map
                        || !request.bindings->d_numeric_values) {
                        throw std::runtime_error(
                            "ModelForward: execution-active row " + std::to_string(b)
                            + " has no slot map or numeric values for bootstrap — "
                            "compiled payload marks row active but bootstrap data is missing");
                    }
                    GRIM::executionBlockBootstrapMemoryFromSlotMap(
                        execution_hp,
                        M_b,
                        *execution_block_parameters,
                        request.bindings->d_numeric_values + tok_off,
                        request.bindings->d_token_to_slot_map + tok_off,
                        row_len, request.stream);

                    for (int step = 0; step < exec_K; ++step) {
                        ExecutionBlockStepOutput step_diag;

                        GRIM::executionBlockStep(
                            execution_hp, execution_runtime.execution_diag,
                            layer_output, M_b, *execution_block_parameters,
                            payload, *request.bindings, b,
                            step, T, request.stream,
                            &step_diag,
                            runtime.execution_runtime->trace_state_by_row[b],
                            runtime.execution_runtime->execution_trace_by_row[b]);
                        runtime.execution_runtime->execution_trace_by_row[b].push_back(step_diag.record);
                        forward_outputs.exec_outputs_per_row[b].steps.push_back(std::move(step_diag));
                    }
                }
            }

            forward_outputs.encoder_layer_outputs.push_back(std::move(layer_output));
        }

        MFWD_INFO("Step 2: All " << num_layers << " encoder layers complete");
        Tensor& last = forward_outputs.encoder_layer_outputs.back();
        forward_outputs.encoder_output_tensor = Tensor::from_ptr(
            last.data,
            TensorContract::TensorShape::make_BSM(total_tokens, d_model),
            false,
            true,
            "encoder_output_for_lmhead");
        forward_outputs.encoder_output_tensor.is_leaf = false;
        forward_outputs.encoder_output_tensor.stream = request.stream;
        forward_outputs.encoder_output_tensor.grad_fn = last.grad_fn;
    }

    const GRIM::LMHeadParameterTensors* lm_head_parameter_ptr = &lm_head_parameters;
    GRIM::LMHeadParameterTensors detached_lm_head_parameters{};
    if (!connect_parameter_graph) {
        detached_lm_head_parameters = detachLmHeadParameters(lm_head_parameters, request.stream);
        lm_head_parameter_ptr = &detached_lm_head_parameters;
    }

    materializeLatentTrajectoryPresetActivations(
        request,
        payload,
        latent_preset_hp,
        forward_outputs);
    const Tensor& lm_hidden_input = forward_outputs.latent_preset_h_enhanced.data
        ? forward_outputs.latent_preset_h_enhanced
        : forward_outputs.encoder_output_tensor;

    forwardLmHead(
        lm_head_hp,
        *lm_head_parameter_ptr,
        lm_hidden_input,
        payload,
        request.stream,
        request.cublas_handle,
        forward_outputs);
    if (!forward_outputs.logits_tensor.data) {
        throw std::runtime_error("ModelForward: LMHeadLayer::forward returned logits tensor with NULL data");
    }

    const Tensor* live_lm_head_input = forward_outputs.liveLmHeadInputOrNull();
    if (!live_lm_head_input || !live_lm_head_input->data) {
        throw std::runtime_error("ModelForward: LM-head input snapshot is NULL after LMHeadLayer::forward");
    }

    materializeForwardMtpLogits(
        request,
        payload,
        lm_head_hp,
        *lm_head_parameter_ptr,
        latent_preset_hp,
        forward_outputs);
    materializeForwardSelectorLogits(request, payload, forward_outputs);

    if constexpr (GRIM::VerboseLogging::ENABLE_EXPENSIVE_DIAGNOSTICS) {
        constexpr int kSamplePositions = 1024;
        const int sample_size = std::min(kSamplePositions, total_tokens);
        std::vector<float> h_encoder(sample_size * d_model);
        std::vector<float> h_logits(sample_size * payload.vocab_size);

        cudaMemcpyAsync(h_encoder.data(), live_lm_head_input->data,
                        sample_size * d_model * sizeof(float),
                        cudaMemcpyDeviceToHost, request.stream);
        cudaMemcpyAsync(h_logits.data(), forward_outputs.logits_tensor.data,
                        sample_size * payload.vocab_size * sizeof(float),
                        cudaMemcpyDeviceToHost, request.stream);
        cudaStreamSynchronize(request.stream);

        for (int pos = 0; pos < sample_size; ++pos) {
            const float* h = h_encoder.data() + pos * d_model;
            const float* logits_row = h_logits.data() + pos * payload.vocab_size;

            int argmax_token = 0;
            float max_logit_val = logits_row[0];
            for (int v = 1; v < payload.vocab_size; ++v) {
                if (logits_row[v] > max_logit_val) {
                    max_logit_val = logits_row[v];
                    argmax_token = v;
                }
            }

            std::vector<float> h_weights_argmax(d_model);
            cudaMemcpyAsync(h_weights_argmax.data(),
                            lm_head_parameters.weights.data + static_cast<size_t>(argmax_token) * d_model,
                            d_model * sizeof(float),
                            cudaMemcpyDeviceToHost, request.stream);
            cudaStreamSynchronize(request.stream);

            float h_sum = 0.0f, h_sum_sq = 0.0f, h_min = h[0], h_max = h[0];
            for (int d = 0; d < d_model; ++d) {
                h_sum += h[d];
                h_sum_sq += h[d] * h[d];
                h_min = std::min(h_min, h[d]);
                h_max = std::max(h_max, h[d]);
            }
            float h_mean = h_sum / d_model;
            float h_rms = std::sqrt(h_sum_sq / d_model);

            float w_sum = 0.0f, w_sum_sq = 0.0f, w_min = h_weights_argmax[0], w_max = h_weights_argmax[0];
            for (int d = 0; d < d_model; ++d) {
                w_sum += h_weights_argmax[d];
                w_sum_sq += h_weights_argmax[d] * h_weights_argmax[d];
                w_min = std::min(w_min, h_weights_argmax[d]);
                w_max = std::max(w_max, h_weights_argmax[d]);
            }
            float w_mean = w_sum / d_model;
            float w_rms = std::sqrt(w_sum_sq / d_model);

            float dot_product_argmax = 0.0f;
            float positive_contrib = 0.0f, negative_contrib = 0.0f;
            for (int d = 0; d < d_model; ++d) {
                float contrib = h[d] * h_weights_argmax[d];
                dot_product_argmax += contrib;
                if (contrib > 0) positive_contrib += contrib;
                else negative_contrib += contrib;
            }

            float h_rms_val = std::sqrt(h_sum_sq / d_model);
            float w_rms_val = std::sqrt(w_sum_sq / d_model);
            float cosine_sim = (h_rms_val > 1e-8f && w_rms_val > 1e-8f)
                               ? (dot_product_argmax / (h_rms_val * w_rms_val * d_model)) : 0.0f;

            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            fprintf(stderr, "[LOGIT_ANALYSIS] Position %d: Why does logit[v] = Σ_d h[d] × W[v,d] choose token %d?\n", pos, argmax_token);
            fprintf(stderr, "═══════════════════════════════════════════════════════════════════════════\n");
            fprintf(stderr, "HIDDEN STATE h[pos=%d]:\n", pos);
            fprintf(stderr, "  Statistics: mean=%.6f (offset) rms=%.6f (magnitude) range=[%.6f, %.6f]\n",
                    h_mean, h_rms, h_min, h_max);
            fprintf(stderr, "WEIGHT ROW W[%d] (predicted token):\n", argmax_token);
            fprintf(stderr, "  Statistics: mean=%.6f rms=%.6f range=[%.6f, %.6f]\n",
                    w_mean, w_rms, w_min, w_max);
            fprintf(stderr, "DOT_PRODUCT ANALYSIS Σ_d h[d]×W[%d,d]:\n", argmax_token);
            fprintf(stderr, "  Raw computation: %.6f\n", dot_product_argmax);
            fprintf(stderr, "  ├─ Positive contributions (h×W>0): %.6f (%.1f%%)\n",
                    positive_contrib, 100.0f * positive_contrib / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Negative contributions (h×W<0): %.6f (%.1f%%)\n",
                    negative_contrib, 100.0f * std::abs(negative_contrib) / (std::abs(dot_product_argmax) + 1e-8f));
            fprintf(stderr, "  ├─ Cosine alignment: %.6f (1.0=perfect alignment, 0=orthogonal, -1=opposite)\n", cosine_sim);
            fprintf(stderr, "RESULT:\n");
            fprintf(stderr, "  logit[%d]=%.6f (PREDICTED argmax token)\n", argmax_token, max_logit_val);
            fprintf(stderr, "\n");
        }
    }

    if (emit_mtp_logits) {
        MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << payload.vocab_size
                  << "] mtp_heads=" << forward_outputs.mtp_logits_tensors.size());
    } else {
        MFWD_INFO("Forward complete: logits shape=[" << total_tokens << ", " << payload.vocab_size << "]");
    }

    return forward_outputs;
}

}  // namespace Forward
}  // namespace GRIM

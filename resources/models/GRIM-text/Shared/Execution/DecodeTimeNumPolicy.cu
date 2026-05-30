//======================================================//
//  DecodeTimeNumPolicy — decode-time selector ops
//
//  Constructs live candidate set L from ExecutionMemory,
//  assembles fixed slot features, evaluates selector
//  scores to produce Selected / Null / Ambiguous.
//======================================================//

#include "DecodeTimeResolveResult.hpp"
#include "DecodeTimeNumPolicy.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

#include "../Batching/BatchPayload.hpp"
#include "../Batching/BatchDeviceBindings.hpp"
#include "../Forward/ModelForwardExecutionRuntime.hpp"
#include "../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.hpp"
#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"

namespace GRIM {

#define POLICY_CUDA_CHECK(expr) do { \
    cudaError_t _e = (expr); \
    if (_e != cudaSuccess) { \
        throw std::runtime_error(std::string("DecodeTimeNumPolicy CUDA error at ") + \
            __FILE__ + ":" + std::to_string(__LINE__) + " — " + cudaGetErrorString(_e)); \
    } \
} while (0)

namespace {

__global__ void kernelBuildPolicyCandidateSet(
    const float* __restrict__ d_valid_mask,
    const float* __restrict__ d_values,
    const float* __restrict__ d_recent_write,
    const float* __restrict__ d_usage,
    int V,
    int scratch_slots,
    int32_t* __restrict__ d_live_slot_ids,
    float* __restrict__ d_slot_features,
    int* __restrict__ d_num_live_slots)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int count = 0;
    for (int s = scratch_slots; s < V; ++s) {
        if (d_valid_mask[s] > 0.5f) {
            d_live_slot_ids[count] = static_cast<int32_t>(s);
            float* feat = d_slot_features + count * kSlotFeatureDim;
            feat[0] = static_cast<float>(s);
            feat[1] = d_values[s];
            feat[2] = d_valid_mask[s];
            feat[3] = d_recent_write[s];
            feat[4] = d_usage[s];
            ++count;
        }
    }
    *d_num_live_slots = count;
}

__global__ void kernelEvaluatePolicyScores(
    const float* __restrict__ d_scores,
    const int32_t* __restrict__ d_live_slot_ids,
    int num_live_slots,
    float selection_margin,
    int32_t* __restrict__ d_status,
    int32_t* __restrict__ d_selected_slot,
    float* __restrict__ d_confidence)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    d_selected_slot[0] = -1;
    d_confidence[0] = 0.0f;

    if (num_live_slots == 0) {
        d_status[0] = static_cast<int32_t>(SlotSelectionStatus::Null);
        d_confidence[0] = d_scores[0];
        return;
    }

    const int total = 1 + num_live_slots;
    int top1_idx = 0;
    float top1_val = d_scores[0];
    for (int i = 1; i < total; ++i) {
        const float v = d_scores[i];
        if (v > top1_val) {
            top1_val = v;
            top1_idx = i;
        }
    }

    float top2_val = -3.4028234663852886e38f;
    for (int i = 0; i < total; ++i) {
        if (i == top1_idx) continue;
        const float v = d_scores[i];
        if (v > top2_val) {
            top2_val = v;
        }
    }

    const float margin = top1_val - top2_val;
    d_confidence[0] = margin;

    if (margin < selection_margin) {
        d_status[0] = static_cast<int32_t>(SlotSelectionStatus::Ambiguous);
        return;
    }

    if (top1_idx == 0) {
        d_status[0] = static_cast<int32_t>(SlotSelectionStatus::Null);
        return;
    }

    const int candidate_idx = top1_idx - 1;
    d_status[0] = static_cast<int32_t>(SlotSelectionStatus::Selected);
    d_selected_slot[0] = d_live_slot_ids[candidate_idx];
}

__global__ void kernelResolvePolicyTargetIndex(
    const int32_t* __restrict__ d_live_slot_ids,
    int num_live_slots,
    int32_t target_slot,
    int32_t* __restrict__ d_target_index)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    int32_t target_index = -1;
    for (int c = 0; c < num_live_slots; ++c) {
        if (d_live_slot_ids[c] == target_slot) {
            target_index = static_cast<int32_t>(1 + c);
            break;
        }
    }
    d_target_index[0] = target_index;
}

} // namespace

void validateDecodeTimeNumPolicyConfig(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp)
{
    const int max_live_slots = hp.num_slots - hp.scratch_slots;
    if (hp.selection_margin < 0.0f) {
        throw std::runtime_error("validateDecodeTimeNumPolicyConfig: selection_margin must be non-negative, got " +
                                 std::to_string(hp.selection_margin));
    }
    if (hp.num_slots <= 0) {
        throw std::runtime_error("validateDecodeTimeNumPolicyConfig: num_slots must be positive, got " +
                                 std::to_string(hp.num_slots));
    }
    if (hp.d_slot_features != kSlotFeatureDim) {
        throw std::runtime_error("validateDecodeTimeNumPolicyConfig: hp.d_slot_features=" +
                                 std::to_string(hp.d_slot_features) +
                                 " must match fixed policy feature layout width=" +
                                 std::to_string(kSlotFeatureDim));
    }
    if (hp.scratch_slots < 0 || hp.scratch_slots >= hp.num_slots) {
        throw std::runtime_error("validateDecodeTimeNumPolicyConfig: scratch_slots=" +
                                 std::to_string(hp.scratch_slots) +
                                 " out of range [0, " + std::to_string(hp.num_slots) + ")");
    }
    if (max_live_slots <= 0) {
        throw std::runtime_error("validateDecodeTimeNumPolicyConfig: value slot capacity must be positive, got " +
                                 std::to_string(max_live_slots));
    }
}

void buildDecodeTimeCandidateSet(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    const float* d_valid_mask,
    const float* d_values,
    const float* d_recent_write,
    const float* d_usage,
    Forward::DecodeTimeSelectorRuntime& runtime,
    int V,
    int scratch_slots,
    cudaStream_t stream)
{
    validateDecodeTimeNumPolicyConfig(hp);
    const int max_live_slots = hp.num_slots - hp.scratch_slots;
    if (!d_valid_mask) throw std::runtime_error("buildDecodeTimeCandidateSet: d_valid_mask is NULL");
    if (!d_values)     throw std::runtime_error("buildDecodeTimeCandidateSet: d_values is NULL");
    if (!d_recent_write) throw std::runtime_error("buildDecodeTimeCandidateSet: d_recent_write is NULL");
    if (!d_usage)      throw std::runtime_error("buildDecodeTimeCandidateSet: d_usage is NULL");
    if (!stream)       throw std::runtime_error("buildDecodeTimeCandidateSet: stream is NULL");
    if (V != hp.num_slots) {
        throw std::runtime_error("buildDecodeTimeCandidateSet: V=" + std::to_string(V) +
                                 " != hp.num_slots=" + std::to_string(hp.num_slots));
    }
    if (scratch_slots != hp.scratch_slots) {
        throw std::runtime_error("buildDecodeTimeCandidateSet: scratch_slots=" +
                                 std::to_string(scratch_slots) +
                                 " != hp.scratch_slots=" + std::to_string(hp.scratch_slots));
    }
    if (V - scratch_slots != max_live_slots) {
        throw std::runtime_error("buildDecodeTimeCandidateSet: value slot capacity=" +
                                 std::to_string(V - scratch_slots) +
                                 " != hp value-slot capacity=" + std::to_string(max_live_slots));
    }
    runtime.ensureWorkspace(max_live_slots, kSlotFeatureDim, stream);
    if (!runtime.d_slot_features()) throw std::runtime_error("buildDecodeTimeCandidateSet: runtime d_slot_features is NULL");
    if (!runtime.d_live_slot_ids()) throw std::runtime_error("buildDecodeTimeCandidateSet: runtime d_live_slot_ids is NULL");
    if (!runtime.d_num_live_slots()) throw std::runtime_error("buildDecodeTimeCandidateSet: runtime d_num_live_slots is NULL");

    kernelBuildPolicyCandidateSet<<<1, 1, 0, stream>>>(
        d_valid_mask,
        d_values,
        d_recent_write,
        d_usage,
        V,
        scratch_slots,
        runtime.d_live_slot_ids(),
        runtime.d_slot_features(),
        runtime.d_num_live_slots());
    POLICY_CUDA_CHECK(cudaGetLastError());

    POLICY_CUDA_CHECK(cudaMemcpyAsync(
        &runtime.num_live_slots,
        runtime.d_num_live_slots(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream));
    POLICY_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (runtime.num_live_slots < 0 || runtime.num_live_slots > max_live_slots) {
        throw std::runtime_error("buildDecodeTimeCandidateSet: GPU candidate count=" +
                                 std::to_string(runtime.num_live_slots) +
                                 " out of range [0, " + std::to_string(max_live_slots) + "]");
    }
}

void buildDecodeTimeCandidateSet(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    const Batching::BatchPayload& payload,
    const Batching::BatchDeviceBindings& bindings,
    const ExecutionMemory& exec_memory,
    Forward::DecodeTimeSelectorRuntime& runtime,
    int batch_row,
    cudaStream_t stream)
{
    validateDecodeTimeNumPolicyConfig(hp);
    payload.validate("buildDecodeTimeCandidateSet(payload)");
    if (batch_row < 0 || batch_row >= payload.batch_size) {
        throw std::runtime_error("buildDecodeTimeCandidateSet(payload): batch_row=" +
                                 std::to_string(batch_row) + " out of range [0, " +
                                 std::to_string(payload.batch_size) + ")");
    }
    if (payload.execution_active.empty()) {
        throw std::runtime_error("buildDecodeTimeCandidateSet(payload): payload.execution_active is empty");
    }
    if (!payload.execution_active[static_cast<size_t>(batch_row)]) {
        throw std::runtime_error("buildDecodeTimeCandidateSet(payload): row " +
                                 std::to_string(batch_row) + " is not execution-active");
    }
    if (!bindings.d_numeric_values) {
        throw std::runtime_error("buildDecodeTimeCandidateSet(payload): bindings.d_numeric_values is NULL");
    }
    if (!bindings.d_token_to_slot_map) {
        throw std::runtime_error("buildDecodeTimeCandidateSet(payload): bindings.d_token_to_slot_map is NULL");
    }

    buildDecodeTimeCandidateSet(
        hp,
        exec_memory.valid_mask.data,
        exec_memory.values.data,
        exec_memory.recent_write_mask.data,
        exec_memory.usage.data,
        runtime,
        hp.num_slots,
        hp.scratch_slots,
        stream);
}

GRIM::SlotSelectionResult evaluateDecodeTimeSelectionScores(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    const float* d_scores,
    Forward::DecodeTimeSelectorRuntime& runtime,
    int num_live_slots,
    cudaStream_t stream)
{
    validateDecodeTimeNumPolicyConfig(hp);
    if (!d_scores) throw std::runtime_error("evaluateDecodeTimeSelectionScores: d_scores is NULL");
    if (!stream)   throw std::runtime_error("evaluateDecodeTimeSelectionScores: stream is NULL");
    if (num_live_slots != runtime.num_live_slots) {
        throw std::runtime_error("evaluateDecodeTimeSelectionScores: num_live_slots=" +
                                 std::to_string(num_live_slots) +
                                 " != runtime.num_live_slots=" +
                                 std::to_string(runtime.num_live_slots));
    }
    if (!runtime.d_live_slot_ids()) throw std::runtime_error("evaluateDecodeTimeSelectionScores: runtime d_live_slot_ids is NULL");
    if (!runtime.d_selection_status()) throw std::runtime_error("evaluateDecodeTimeSelectionScores: runtime d_selection_status is NULL");
    if (!runtime.d_selection_slot()) throw std::runtime_error("evaluateDecodeTimeSelectionScores: runtime d_selection_slot is NULL");
    if (!runtime.d_selection_confidence()) throw std::runtime_error("evaluateDecodeTimeSelectionScores: runtime d_selection_confidence is NULL");

    GRIM::SlotSelectionResult result;
    result.selected_slot = -1;
    result.confidence = 0.0f;

    kernelEvaluatePolicyScores<<<1, 1, 0, stream>>>(
        d_scores,
        runtime.d_live_slot_ids(),
        num_live_slots,
        hp.selection_margin,
        runtime.d_selection_status(),
        runtime.d_selection_slot(),
        runtime.d_selection_confidence());
    POLICY_CUDA_CHECK(cudaGetLastError());

    int32_t status = -1;
    POLICY_CUDA_CHECK(cudaMemcpyAsync(&status, runtime.d_selection_status(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaMemcpyAsync(&result.selected_slot, runtime.d_selection_slot(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaMemcpyAsync(&result.confidence, runtime.d_selection_confidence(), sizeof(float), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (status == static_cast<int32_t>(SlotSelectionStatus::Selected)) {
        result.status = SlotSelectionStatus::Selected;
    } else if (status == static_cast<int32_t>(SlotSelectionStatus::Null)) {
        result.status = SlotSelectionStatus::Null;
    } else if (status == static_cast<int32_t>(SlotSelectionStatus::Ambiguous)) {
        result.status = SlotSelectionStatus::Ambiguous;
    } else {
        throw std::runtime_error("evaluateDecodeTimeSelectionScores: GPU status=" +
                                 std::to_string(status) + " is not a legal SlotSelectionStatus");
    }
    if (result.status == SlotSelectionStatus::Selected
        && (result.selected_slot < hp.scratch_slots || result.selected_slot >= hp.num_slots)) {
        throw std::runtime_error("evaluateDecodeTimeSelectionScores: selected_slot=" +
                                 std::to_string(result.selected_slot) + " out of value-slot range [" +
                                 std::to_string(hp.scratch_slots) + ", " +
                                 std::to_string(hp.num_slots) + ")");
    }

    return result;
}

int resolveDecodeTimeTargetIndexForSlot(
    const HyperParameters::DecodeTimeSelectorConstructionHP& hp,
    int32_t target_slot,
    Forward::DecodeTimeSelectorRuntime& runtime,
    cudaStream_t stream)
{
    validateDecodeTimeNumPolicyConfig(hp);
    const int max_live_slots = hp.num_slots - hp.scratch_slots;
    if (!stream) throw std::runtime_error("resolveDecodeTimeTargetIndexForSlot: stream is NULL");
    if (!runtime.d_live_slot_ids()) {
        throw std::runtime_error("resolveDecodeTimeTargetIndexForSlot: runtime d_live_slot_ids is NULL");
    }
    if (!runtime.d_target_index()) {
        throw std::runtime_error("resolveDecodeTimeTargetIndexForSlot: runtime d_target_index is NULL");
    }
    if (runtime.num_live_slots < 0 || runtime.num_live_slots > max_live_slots) {
        throw std::runtime_error("resolveDecodeTimeTargetIndexForSlot: num_live_slots=" +
                                 std::to_string(runtime.num_live_slots) +
                                 " out of range [0, " + std::to_string(max_live_slots) + "]");
    }
    if (target_slot < hp.scratch_slots || target_slot >= hp.num_slots) {
        throw std::runtime_error("resolveDecodeTimeTargetIndexForSlot: target_slot=" +
                                 std::to_string(target_slot) + " out of value-slot range [" +
                                 std::to_string(hp.scratch_slots) + ", " +
                                 std::to_string(hp.num_slots) + ")");
    }

    kernelResolvePolicyTargetIndex<<<1, 1, 0, stream>>>(
        runtime.d_live_slot_ids(),
        runtime.num_live_slots,
        target_slot,
        runtime.d_target_index());
    POLICY_CUDA_CHECK(cudaGetLastError());

    int32_t target_index = -1;
    POLICY_CUDA_CHECK(cudaMemcpyAsync(&target_index, runtime.d_target_index(), sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaStreamSynchronize(stream));
    return static_cast<int>(target_index);
}

//══════════════════════════════════════════════════════════════════════════════
//  resolveDecodeTimeNumSlotSelectionOrMask — shared selector evaluation
//══════════════════════════════════════════════════════════════════════════════

DecodeTimeResolveResult resolveDecodeTimeNumSlotSelectionOrMask(
    const DecodeTimeSlotSelector* selector,
    const HyperParameters::DecodeTimeSelectorConstructionHP& selector_hp,
    Forward::DecodeTimeSelectorRuntime& runtime,
    bool selector_enabled,
    bool exec_block_active,
    bool has_exec_memory,
    const ExecutionMemory& exec_memory,
    const float* d_hidden_state,
    cudaStream_t stream,
    cublasHandle_t cublas_handle)
{
    DecodeTimeResolveResult out;
    validateDecodeTimeNumPolicyConfig(selector_hp);

    if (!selector_enabled || !selector
        || !exec_block_active || !has_exec_memory) {
        return out;  // valid=false → caller knows selector didn't run
    }

    buildDecodeTimeCandidateSet(
        selector_hp,
        exec_memory.valid_mask.data,
        exec_memory.values.data,
        exec_memory.recent_write_mask.data,
        exec_memory.usage.data,
        runtime,
        selector_hp.num_slots,
        selector_hp.scratch_slots,
        stream);

    if (runtime.num_live_slots > 0) {
        // Create non-owning Tensor views for inference forward
        Tensor h_t_view = Tensor::from_ptr(
            const_cast<float*>(d_hidden_state),
            TensorContract::TensorShape::make_BSM(1, selector_hp.d_model),
            /*takes_ownership=*/false, /*requires_grad=*/false,
            "policy_h_t_view");
        Tensor slot_feat_view;
        if (runtime.d_slot_features() && runtime.num_live_slots > 0) {
            slot_feat_view = Tensor::from_ptr(
                const_cast<float*>(runtime.d_slot_features()),
                TensorContract::TensorShape::make_BSM(runtime.num_live_slots, selector_hp.d_slot_features),
                /*takes_ownership=*/false, /*requires_grad=*/false,
                "policy_slot_feat_view");
        }
        Forward::SelectorForwardResult fwd = forwardDecodeTimeSlotSelector(
            *selector,
            selector_hp,
            h_t_view, slot_feat_view,
            runtime.num_live_slots, stream, cublas_handle);
        GRIM::SlotSelectionResult result = evaluateDecodeTimeSelectionScores(
            selector_hp, fwd.scores.data, runtime, fwd.num_live_slots, stream);
        out.status = result.status;
        out.selected_slot = result.selected_slot;
        if (result.status == SlotSelectionStatus::Selected
            && result.selected_slot >= 0) {
            float val = 0.0f;
            cudaMemcpyAsync(&val,
                exec_memory.values.data + result.selected_slot,
                sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            out.selected_value = val;
        }
    }

    out.valid = true;
    return out;
}

} // namespace GRIM

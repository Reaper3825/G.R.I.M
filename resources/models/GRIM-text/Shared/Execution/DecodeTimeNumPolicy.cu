//======================================================//
//  DecodeTimeNumPolicy — implementation
//
//  Constructs live candidate set L from ExecutionMemory,
//  assembles fixed slot features, evaluates selector
//  scores to produce Selected / Null / Ambiguous.
//======================================================//

#include "DecodeTimeNumPolicy.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>

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

// ─── Constructor ─────────────────────────────────────

DecodeTimeNumPolicy::DecodeTimeNumPolicy(const HyperParameters::DecodeTimeSelectorConstructionHP& hp)
    : hp_(hp)
{
    if (hp_.selection_margin < 0.0f) {
        throw std::runtime_error("DecodeTimeNumPolicy: selection_margin must be non-negative, got " +
                                 std::to_string(hp_.selection_margin));
    }
    if (hp_.num_slots <= 0) {
        throw std::runtime_error("DecodeTimeNumPolicy: num_slots must be positive, got " +
                                 std::to_string(hp_.num_slots));
    }
    if (hp_.scratch_slots < 0 || hp_.scratch_slots >= hp_.num_slots) {
        throw std::runtime_error("DecodeTimeNumPolicy: scratch_slots=" +
                                 std::to_string(hp_.scratch_slots) +
                                 " out of range [0, " + std::to_string(hp_.num_slots) + ")");
    }

    const int V = hp_.num_slots;

    // Allocate host staging buffers
    h_valid_mask_   = new float[V];
    h_values_       = new float[V];
    h_recent_write_ = new float[V];
    h_usage_        = new float[V];
    h_scores_       = new float[kPolicyMaxSlots + 1]; // NULL + max candidates

    // Allocate device slot features buffer
    POLICY_CUDA_CHECK(cudaMalloc(&candidates_.d_slot_features,
                                  static_cast<size_t>(kPolicyMaxSlots) * kSlotFeatureDim * sizeof(float)));

    candidates_.num_live_slots = 0;
}

// ─── Destructor ──────────────────────────────────────

DecodeTimeNumPolicy::~DecodeTimeNumPolicy() {
    delete[] h_valid_mask_;
    delete[] h_values_;
    delete[] h_recent_write_;
    delete[] h_usage_;
    delete[] h_scores_;

    if (candidates_.d_slot_features) {
        cudaFree(candidates_.d_slot_features);
        candidates_.d_slot_features = nullptr;
    }
}

// ─── Move semantics ─────────────────────────────────

DecodeTimeNumPolicy::DecodeTimeNumPolicy(DecodeTimeNumPolicy&& other) noexcept
    : hp_(other.hp_),
      candidates_(other.candidates_),
      h_valid_mask_(other.h_valid_mask_),
      h_values_(other.h_values_),
      h_recent_write_(other.h_recent_write_),
      h_usage_(other.h_usage_),
      h_scores_(other.h_scores_)
{
    other.h_valid_mask_ = nullptr;
    other.h_values_ = nullptr;
    other.h_recent_write_ = nullptr;
    other.h_usage_ = nullptr;
    other.h_scores_ = nullptr;
    other.candidates_.d_slot_features = nullptr;
}

DecodeTimeNumPolicy& DecodeTimeNumPolicy::operator=(DecodeTimeNumPolicy&& other) noexcept {
    if (this != &other) {
        delete[] h_valid_mask_;
        delete[] h_values_;
        delete[] h_recent_write_;
        delete[] h_usage_;
        delete[] h_scores_;
        if (candidates_.d_slot_features) cudaFree(candidates_.d_slot_features);

        hp_ = other.hp_;
        candidates_ = other.candidates_;
        h_valid_mask_ = other.h_valid_mask_;
        h_values_ = other.h_values_;
        h_recent_write_ = other.h_recent_write_;
        h_usage_ = other.h_usage_;
        h_scores_ = other.h_scores_;

        other.h_valid_mask_ = nullptr;
        other.h_values_ = nullptr;
        other.h_recent_write_ = nullptr;
        other.h_usage_ = nullptr;
        other.h_scores_ = nullptr;
        other.candidates_.d_slot_features = nullptr;
    }
    return *this;
}

// ─── buildCandidateSet ──────────────────────────────

void DecodeTimeNumPolicy::buildCandidateSet(
    const float* d_valid_mask,
    const float* d_values,
    const float* d_recent_write,
    const float* d_usage,
    int V,
    int scratch_slots,
    cudaStream_t stream)
{
    if (!d_valid_mask) throw std::runtime_error("DecodeTimeNumPolicy::buildCandidateSet: d_valid_mask is NULL");
    if (!d_values)     throw std::runtime_error("DecodeTimeNumPolicy::buildCandidateSet: d_values is NULL");
    if (!d_recent_write) throw std::runtime_error("DecodeTimeNumPolicy::buildCandidateSet: d_recent_write is NULL");
    if (!d_usage)      throw std::runtime_error("DecodeTimeNumPolicy::buildCandidateSet: d_usage is NULL");
    if (!stream)       throw std::runtime_error("DecodeTimeNumPolicy::buildCandidateSet: stream is NULL");
    if (V != hp_.num_slots) {
        throw std::runtime_error("DecodeTimeNumPolicy::buildCandidateSet: V=" + std::to_string(V) +
                                 " != hp.num_slots=" + std::to_string(hp_.num_slots));
    }

    // Synchronous D2H copy of slot metadata for candidate construction
    // These are small vectors (V scalars each, V ≤ 16 typically)
    POLICY_CUDA_CHECK(cudaMemcpyAsync(h_valid_mask_, d_valid_mask,
                                       V * sizeof(float), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaMemcpyAsync(h_values_, d_values,
                                       V * sizeof(float), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaMemcpyAsync(h_recent_write_, d_recent_write,
                                       V * sizeof(float), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaMemcpyAsync(h_usage_, d_usage,
                                       V * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // Must synchronize to read host data
    POLICY_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Build L: live candidate set = valid value slots (indices [S, V))
    int count = 0;
    for (int s = scratch_slots; s < V; ++s) {
        if (h_valid_mask_[s] > 0.5f) {
            if (count >= kPolicyMaxSlots) {
                throw std::runtime_error("DecodeTimeNumPolicy::buildCandidateSet: exceeded kPolicyMaxSlots=" +
                                         std::to_string(kPolicyMaxSlots));
            }
            candidates_.live_slot_ids[count] = s;

            // Assemble fixed feature vector for this slot
            float* feat = candidates_.slot_features + count * kSlotFeatureDim;
            feat[0] = static_cast<float>(s);            // slot_id
            feat[1] = h_values_[s];                      // numeric_value
            feat[2] = h_valid_mask_[s];                  // valid_bit (always ~1.0 here)
            feat[3] = h_recent_write_[s];                // recent_write
            feat[4] = h_usage_[s];                       // usage_scalar

            ++count;
        }
    }
    candidates_.num_live_slots = count;

    // Upload features to device
    if (count > 0) {
        POLICY_CUDA_CHECK(cudaMemcpyAsync(
            candidates_.d_slot_features,
            candidates_.slot_features,
            static_cast<size_t>(count) * kSlotFeatureDim * sizeof(float),
            cudaMemcpyHostToDevice, stream));
    }
}

// ─── evaluateScores ─────────────────────────────────

SlotSelectionResult DecodeTimeNumPolicy::evaluateScores(
    const float* d_scores,
    int num_live_slots,
    cudaStream_t stream)
{
    if (!d_scores) throw std::runtime_error("DecodeTimeNumPolicy::evaluateScores: d_scores is NULL");
    if (!stream)   throw std::runtime_error("DecodeTimeNumPolicy::evaluateScores: stream is NULL");
    if (num_live_slots != candidates_.num_live_slots) {
        throw std::runtime_error("DecodeTimeNumPolicy::evaluateScores: num_live_slots=" +
                                 std::to_string(num_live_slots) +
                                 " != candidates_.num_live_slots=" +
                                 std::to_string(candidates_.num_live_slots));
    }

    const int total = 1 + num_live_slots; // NULL + candidates

    // D2H copy of score vector
    POLICY_CUDA_CHECK(cudaMemcpyAsync(h_scores_, d_scores,
                                       total * sizeof(float), cudaMemcpyDeviceToHost, stream));
    POLICY_CUDA_CHECK(cudaStreamSynchronize(stream));

    SlotSelectionResult result;
    result.selected_slot = -1;
    result.confidence = 0.0f;

    // Case: no live slots → always Null
    if (num_live_slots == 0) {
        result.status = SlotSelectionStatus::Null;
        result.confidence = h_scores_[0]; // NULL score is the only score
        return result;
    }

    // Find top-1 and top-2 across all options (NULL + candidates)
    int top1_idx = 0;
    float top1_val = h_scores_[0];
    for (int i = 1; i < total; ++i) {
        if (h_scores_[i] > top1_val) {
            top1_val = h_scores_[i];
            top1_idx = i;
        }
    }

    float top2_val = -INFINITY;
    for (int i = 0; i < total; ++i) {
        if (i != top1_idx && h_scores_[i] > top2_val) {
            top2_val = h_scores_[i];
        }
    }

    float margin = top1_val - top2_val;
    result.confidence = margin;

    // Check margin gate
    if (margin < hp_.selection_margin) {
        result.status = SlotSelectionStatus::Ambiguous;
        return result;
    }

    // Margin sufficient — check if winner is NULL or a candidate
    if (top1_idx == 0) {
        // NULL won
        result.status = SlotSelectionStatus::Null;
    } else {
        // A candidate won — map back to real slot index
        int candidate_idx = top1_idx - 1;
        if (candidate_idx < 0 || candidate_idx >= num_live_slots) {
            throw std::runtime_error("DecodeTimeNumPolicy::evaluateScores: candidate_idx=" +
                                     std::to_string(candidate_idx) + " out of range");
        }
        result.status = SlotSelectionStatus::Selected;
        result.selected_slot = candidates_.live_slot_ids[candidate_idx];
    }

    return result;
}

//══════════════════════════════════════════════════════════════════════════════
//  resolveDecodeTimeNumSlotSelectionOrMask — shared selector evaluation
//══════════════════════════════════════════════════════════════════════════════

DecodeTimeResolveResult resolveDecodeTimeNumSlotSelectionOrMask(
    DecodeTimeSlotSelectorLayer* selector,
    DecodeTimeNumPolicy* policy,
    bool selector_enabled,
    bool exec_block_active,
    bool has_exec_memory,
    const ExecutionMemory& exec_memory,
    const float* d_hidden_state,
    cudaStream_t stream,
    cublasHandle_t cublas_handle)
{
    DecodeTimeResolveResult out;

    if (!selector_enabled || !selector || !policy
        || !exec_block_active || !has_exec_memory) {
        return out;  // valid=false → caller knows selector didn't run
    }

    policy->buildCandidateSet(
        exec_memory.valid_mask.data,
        exec_memory.values.data,
        exec_memory.recent_write_mask.data,
        exec_memory.usage.data,
        policy->hp().num_slots,
        policy->hp().scratch_slots,
        stream);

    const auto& cands = policy->candidates();
    if (cands.num_live_slots > 0) {
        // Create non-owning Tensor views for inference forward
        Tensor h_t_view = Tensor::from_ptr(
            const_cast<float*>(d_hidden_state),
            TensorContract::TensorShape::make_BSM(1, selector->hp().d_model),
            /*takes_ownership=*/false, /*requires_grad=*/false,
            "policy_h_t_view");
        Tensor slot_feat_view;
        if (cands.d_slot_features && cands.num_live_slots > 0) {
            slot_feat_view = Tensor::from_ptr(
                const_cast<float*>(cands.d_slot_features),
                TensorContract::TensorShape::make_BSM(cands.num_live_slots, selector->hp().d_slot_features),
                /*takes_ownership=*/false, /*requires_grad=*/false,
                "policy_slot_feat_view");
        }
        SelectorForwardResult fwd = selector->forward(
            h_t_view, slot_feat_view,
            cands.num_live_slots, stream, cublas_handle);
        SlotSelectionResult result = policy->evaluateScores(
            fwd.scores.data, fwd.num_live_slots, stream);
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

//======================================================//
//  ModelForwardExecutionRuntime.hpp
//
//  Typed execution-trace runtime owned by one caller of shared forward.
//  Training and inference each own their own instance; shared forward only
//  receives an explicit pointer to the active owner for the current call.
//======================================================//

#pragma once

#ifdef USE_CUDA

#include <vector>
#include <stdexcept>

#include "ModelForwardOutputs.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace Forward {

struct DecodeTimeSelectorRuntime {
    int max_slots = 0;
    int slot_feature_dim = 0;
    int num_live_slots = 0;

    Tensor slot_features_tensor;          // [max_slots, slot_feature_dim]
    Tensor live_slot_ids_tensor;          // [1, max_slots] raw int32 buffer
    Tensor num_live_slots_tensor;         // [1, 1] raw int32 buffer
    Tensor selection_status_tensor;       // [1, 1] raw int32 buffer
    Tensor selection_slot_tensor;         // [1, 1] raw int32 buffer
    Tensor selection_confidence_tensor;   // [1, 1] float buffer
    Tensor target_index_tensor;           // [1, 1] raw int32 buffer

    void ensureWorkspace(int required_max_slots,
                         int required_slot_feature_dim,
                         cudaStream_t stream)
    {
        if (!stream) {
            throw std::runtime_error("DecodeTimeSelectorRuntime::ensureWorkspace: stream is NULL");
        }
        if (required_max_slots <= 0) {
            throw std::runtime_error("DecodeTimeSelectorRuntime::ensureWorkspace: required_max_slots must be > 0");
        }
        if (required_slot_feature_dim <= 0) {
            throw std::runtime_error("DecodeTimeSelectorRuntime::ensureWorkspace: required_slot_feature_dim must be > 0");
        }
        const bool already_matches =
            max_slots == required_max_slots &&
            slot_feature_dim == required_slot_feature_dim &&
            slot_features_tensor.data &&
            live_slot_ids_tensor.data &&
            num_live_slots_tensor.data &&
            selection_status_tensor.data &&
            selection_slot_tensor.data &&
            selection_confidence_tensor.data &&
            target_index_tensor.data;
        if (already_matches) {
            num_live_slots = 0;
            return;
        }

        max_slots = required_max_slots;
        slot_feature_dim = required_slot_feature_dim;
        num_live_slots = 0;

        slot_features_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(required_max_slots, required_slot_feature_dim),
            false,
            stream,
            "decode_time_selector_slot_features");
        live_slot_ids_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(1, required_max_slots),
            false,
            stream,
            "decode_time_selector_live_slot_ids");
        num_live_slots_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(1, 1),
            false,
            stream,
            "decode_time_selector_num_live_slots");
        selection_status_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(1, 1),
            false,
            stream,
            "decode_time_selector_selection_status");
        selection_slot_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(1, 1),
            false,
            stream,
            "decode_time_selector_selection_slot");
        selection_confidence_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(1, 1),
            false,
            stream,
            "decode_time_selector_selection_confidence");
        target_index_tensor = Tensor::empty(
            TensorContract::TensorShape::make_BSM(1, 1),
            false,
            stream,
            "decode_time_selector_target_index");
    }

    void reset() {
        num_live_slots = 0;
    }

    float* d_slot_features() const {
        return slot_features_tensor.data;
    }

    int32_t* d_live_slot_ids() const {
        return reinterpret_cast<int32_t*>(live_slot_ids_tensor.data);
    }

    int* d_num_live_slots() const {
        return reinterpret_cast<int*>(num_live_slots_tensor.data);
    }

    int32_t* d_selection_status() const {
        return reinterpret_cast<int32_t*>(selection_status_tensor.data);
    }

    int32_t* d_selection_slot() const {
        return reinterpret_cast<int32_t*>(selection_slot_tensor.data);
    }

    float* d_selection_confidence() const {
        return selection_confidence_tensor.data;
    }

    int32_t* d_target_index() const {
        return reinterpret_cast<int32_t*>(target_index_tensor.data);
    }
};

struct ModelForwardExecutionRuntime {
    std::vector<std::vector<ExecutionRecord>> execution_trace_by_row;
    std::vector<Tensor> trace_state_by_row;
    DecodeTimeSelectorRuntime decode_time_selector_runtime;

    void clear() {
        execution_trace_by_row.clear();
        trace_state_by_row.clear();
        decode_time_selector_runtime.reset();
    }
};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
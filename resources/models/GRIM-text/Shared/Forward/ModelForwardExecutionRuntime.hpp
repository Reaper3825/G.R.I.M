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

#include "../../Layers/ExecutionBlock/execution_block_GPU.hpp"
#include "../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
namespace Forward {

struct ModelForwardExecutionRuntime {
    std::vector<std::vector<ExecutionRecord>> execution_trace_by_row;
    std::vector<Tensor> trace_state_by_row;

    void clear() {
        execution_trace_by_row.clear();
        trace_state_by_row.clear();
    }
};

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA
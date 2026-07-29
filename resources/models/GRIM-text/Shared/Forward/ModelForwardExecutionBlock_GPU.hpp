//======================================================//
//  ModelForwardExecutionBlock_GPU.hpp
//  ExecutionBlock portions of the shared model forward
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "ModelForward_GPU.hpp"

#include <vector>

namespace GRIM {

struct ExecutionBlockParameterTensors;

namespace HyperParameters {
struct ExecutionBlockConstructionHP;
struct SlotSeedEncoderConstructionHP;
}

namespace Forward {

bool applyExecutionBlockReadback(
    const ModelForwardRequest& request,
    const HyperParameters::ExecutionBlockConstructionHP& execution_hp,
    ExecutionBlockParameterTensors* execution_block_parameters,
    int total_tokens,
    int layer_idx,
    int exec_layer,
    bool execution_block_active,
    const std::vector<bool>& execution_active_by_row,
    const Tensor& layer_input,
    Tensor& execution_read_augmented_input,
    ModelForwardRuntimePayload& runtime,
    ModelForwardOutputs& forward_outputs);

void runExecutionBlockNoGraph(
    const ModelForwardRequest& request,
    const HyperParameters::ExecutionBlockConstructionHP& execution_hp,
    const HyperParameters::SlotSeedEncoderConstructionHP& slot_seed_encoder_hp,
    ExecutionBlockParameterTensors* execution_block_parameters,
    int layer_idx,
    int exec_layer,
    int exec_step_count,
    bool execution_block_active,
    bool execution_selector_bridge_requested,
    Tensor& layer_output,
    ModelForwardRuntimePayload& runtime,
    ModelForwardOutputs& forward_outputs,
    std::vector<bool>& execution_active_by_row);

void runExecutionBlockConnectedGraph(
    const ModelForwardRequest& request,
    const HyperParameters::ExecutionBlockConstructionHP& execution_hp,
    const HyperParameters::SlotSeedEncoderConstructionHP& slot_seed_encoder_hp,
    ExecutionBlockParameterTensors* execution_block_parameters,
    int layer_idx,
    int exec_layer,
    int exec_step_count,
    bool execution_block_active,
    bool execution_selector_bridge_requested,
    Tensor& layer_output,
    ModelForwardRuntimePayload& runtime,
    ModelForwardOutputs& forward_outputs);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA

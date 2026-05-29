#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "../../../../Shared/TensorContract/TensorContract_GPU.hpp"

namespace GRIM {
class GPUGrimEncoder;
class ScratchBlockLayer;
class EmbeddingLayer;
class LMHeadLayer;
class ExecutionBlockLayer;
class DecodeTimeSlotSelectorLayer;
class DecodeTimeNumPolicy;

struct MTPHead {
    Tensor weight;
    Tensor bias;
};
}

namespace GRIMText::Training::Startup {

struct GpuModelState {
    GpuModelState();
    ~GpuModelState();

    GpuModelState(const GpuModelState&) = delete;
    GpuModelState& operator=(const GpuModelState&) = delete;
    GpuModelState(GpuModelState&&) noexcept;
    GpuModelState& operator=(GpuModelState&&) noexcept;

    std::unique_ptr<GRIM::GPUGrimEncoder> gpu_encoder;
    std::unique_ptr<GRIM::ScratchBlockLayer> scratch_block_layer;
    std::unique_ptr<GRIM::EmbeddingLayer> embedding_layer;
    std::unique_ptr<GRIM::LMHeadLayer> lm_head_layer;
    std::unique_ptr<GRIM::ExecutionBlockLayer> execution_block_layer;
    std::unique_ptr<GRIM::DecodeTimeSlotSelectorLayer> decode_time_slot_selector_layer;
    std::unique_ptr<GRIM::DecodeTimeNumPolicy> decode_time_num_policy;
    std::vector<GRIM::MTPHead> mtp_heads;
};

} // namespace GRIMText::Training::Startup

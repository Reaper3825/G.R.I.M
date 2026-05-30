#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "ParameterRegistry.hpp"

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {
class GPUGrimEncoder;
class ScratchBlockLayer;
class EmbeddingLayer;
class LMHeadLayer;
class ExecutionBlockLayer;
}

namespace GRIMText::Training::Startup {

struct ForwardTopologyView {
    GRIM::GPUGrimEncoder* gpu_encoder = nullptr;
    GRIM::ScratchBlockLayer* scratch_block = nullptr;
    GRIM::EmbeddingLayer* embedding_layer = nullptr;
    GRIM::LMHeadLayer* lm_head_layer = nullptr;
    GRIM::ExecutionBlockLayer* execution_block_layer = nullptr;
};

struct GpuModelState {
    GpuModelState();
    ~GpuModelState();

    GpuModelState(const GpuModelState&) = delete;
    GpuModelState& operator=(const GpuModelState&) = delete;
    GpuModelState(GpuModelState&&) noexcept;
    GpuModelState& operator=(GpuModelState&&) noexcept;

    GRIM::GPUGrimEncoder& requireGpuEncoder(const char* caller) {
        if (!gpu_encoder) {
            throw std::runtime_error(std::string(caller) + ": GpuModelState.gpu_encoder is NULL");
        }
        return *gpu_encoder;
    }

    GRIM::EmbeddingLayer& requireEmbeddingLayer(const char* caller) {
        if (!embedding_layer) {
            throw std::runtime_error(std::string(caller) + ": GpuModelState.embedding_layer is NULL");
        }
        return *embedding_layer;
    }

    GRIM::LMHeadLayer& requireLmHeadLayer(const char* caller) {
        if (!lm_head_layer) {
            throw std::runtime_error(std::string(caller) + ": GpuModelState.lm_head_layer is NULL");
        }
        return *lm_head_layer;
    }

    ForwardTopologyView requireForwardTopology(
        const GRIM::Config::AiConfigSnapshot& config,
        const char* caller) {
        const auto scratch_hp = GRIM::HyperParameters::scratchBlockConstructionHP(config);
        const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(config);

        ForwardTopologyView topology{};
        topology.gpu_encoder = &requireGpuEncoder(caller);
        topology.embedding_layer = &requireEmbeddingLayer(caller);
        topology.lm_head_layer = &requireLmHeadLayer(caller);
        topology.scratch_block = scratch_block_layer.get();
        topology.execution_block_layer = execution_block_layer.get();

        if (scratch_hp.enabled && !topology.scratch_block) {
            throw std::runtime_error(
                std::string(caller) + ": ScratchBlockConstructionHP.enabled=true but GpuModelState.scratch_block_layer is NULL");
        }
        if (!scratch_hp.enabled && topology.scratch_block) {
            throw std::runtime_error(
                std::string(caller) + ": ScratchBlockConstructionHP.enabled=false but GpuModelState.scratch_block_layer exists");
        }

        if (execution_hp.enabled) {
            if (!topology.execution_block_layer) {
                throw std::runtime_error(
                    std::string(caller) + ": execution_block_enabled but GpuModelState.execution_block_layer is NULL");
            }
            if (!scratch_hp.enabled || !topology.scratch_block) {
                throw std::runtime_error(
                    std::string(caller) + ": execution_block_enabled requires ScratchBlockConstructionHP.enabled=true and a constructed GpuModelState.scratch_block_layer");
            }
        } else if (topology.execution_block_layer) {
            throw std::runtime_error(
                std::string(caller) + ": execution_block_enabled=false but GpuModelState.execution_block_layer exists");
        }

        return topology;
    }

    std::unique_ptr<GRIM::GPUGrimEncoder> gpu_encoder;
    std::unique_ptr<GRIM::ScratchBlockLayer> scratch_block_layer;
    std::unique_ptr<GRIM::EmbeddingLayer> embedding_layer;
    std::unique_ptr<GRIM::LMHeadLayer> lm_head_layer;
    std::unique_ptr<GRIM::ExecutionBlockLayer> execution_block_layer;
};

} // namespace GRIMText::Training::Startup

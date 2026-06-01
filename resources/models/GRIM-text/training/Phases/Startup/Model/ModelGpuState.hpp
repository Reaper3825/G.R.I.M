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
class ExecutionBlockLayer;
}

namespace GRIMText::Training::Startup {

struct ForwardTopologyView {
    GRIM::GPUGrimEncoder* gpu_encoder = nullptr;
    GRIM::ScratchBlockLayer* scratch_block = nullptr;
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

    ForwardTopologyView requireForwardTopology(
        const GRIM::Config::AiConfigSnapshot& config,
        const char* caller) {
        const auto scratch_hp = GRIM::HyperParameters::scratchBlockConstructionHP(config);
        const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(config);

        ForwardTopologyView topology{};
        topology.gpu_encoder = &requireGpuEncoder(caller);
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
    std::unique_ptr<GRIM::ExecutionBlockLayer> execution_block_layer;
};

} // namespace GRIMText::Training::Startup

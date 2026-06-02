#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "ParameterRegistry.hpp"

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"

namespace GRIM {
class GPUGrimEncoder;
}

namespace GRIMText::Training::Startup {

struct ForwardTopologyView {
    GRIM::GPUGrimEncoder* gpu_encoder = nullptr;
    bool execution_block_enabled = false;
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
        const auto execution_hp = GRIM::HyperParameters::executionBlockConstructionHP(config);

        ForwardTopologyView topology{};
        topology.gpu_encoder = &requireGpuEncoder(caller);
        topology.execution_block_enabled = execution_hp.enabled;

        return topology;
    }

    std::unique_ptr<GRIM::GPUGrimEncoder> gpu_encoder;
};

} // namespace GRIMText::Training::Startup

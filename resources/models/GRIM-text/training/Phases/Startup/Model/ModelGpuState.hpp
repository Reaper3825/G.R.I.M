#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "ParameterRegistry.hpp"

namespace GRIM {
class GPUGrimEncoder;
}

namespace GRIMText::Training::Startup {

struct ForwardTopologyView {
    GRIM::GPUGrimEncoder* gpu_encoder = nullptr;
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

    ForwardTopologyView requireForwardTopology(const char* caller) {
        ForwardTopologyView topology{};
        topology.gpu_encoder = &requireGpuEncoder(caller);
        return topology;
    }

    std::unique_ptr<GRIM::GPUGrimEncoder> gpu_encoder;
};

} // namespace GRIMText::Training::Startup

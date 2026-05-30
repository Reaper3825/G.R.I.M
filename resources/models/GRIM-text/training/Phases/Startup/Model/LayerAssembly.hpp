#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace GRIM { namespace Config { struct AiConfigSnapshot; } }

namespace GRIMText::Training::Startup {

struct ModelAssemblyInputs {
    std::uint32_t actual_vocab_size = 0;
    std::uint64_t weight_init_seed = 0;
};

struct LayerAssembly {
    bool ready = false;
    ModelAssemblyInputs inputs;

    const LayerAssembly& requireReady(const char* caller) const {
        if (!ready) {
            throw std::runtime_error(std::string(caller) + ": LayerAssembly is not ready");
        }
        return *this;
    }
};

LayerAssembly buildLayerAssembly(
    const GRIM::Config::AiConfigSnapshot& config,
    std::uint32_t actual_vocab_size,
    std::uint64_t weight_init_seed);

} // namespace GRIMText::Training::Startup

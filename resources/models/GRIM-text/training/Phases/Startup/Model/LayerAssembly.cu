#include "LayerAssembly.hpp"

#include "../../../../Shared/HyperParameters/HyperparameterGroupings.hpp"

#include <stdexcept>

namespace GRIMText::Training::Startup {

LayerAssembly buildLayerAssembly(
    const GRIM::Config::AiConfigSnapshot& config,
    std::uint32_t actual_vocab_size,
    std::uint64_t weight_init_seed)
{
    if (actual_vocab_size == 0) {
        throw std::runtime_error("buildLayerAssembly: actual_vocab_size is 0");
    }
    if (weight_init_seed == 0) {
        throw std::runtime_error("buildLayerAssembly: weight_init_seed is 0");
    }

    const int runtime_vocab_size = GRIM::HyperParameters::snapshotTrainingConfigField<int>(config, "vocab_size");
    if (runtime_vocab_size != static_cast<int>(actual_vocab_size)) {
        throw std::runtime_error(
            "buildLayerAssembly: finalized config vocab_size does not match startup fact actual_vocab_size (config=" +
            std::to_string(runtime_vocab_size) + " actual=" + std::to_string(actual_vocab_size) + ")");
    }

    LayerAssembly assembly{};
    assembly.ready = true;
    assembly.inputs.actual_vocab_size = actual_vocab_size;
    assembly.inputs.weight_init_seed = weight_init_seed;

    return assembly;
}

} // namespace GRIMText::Training::Startup

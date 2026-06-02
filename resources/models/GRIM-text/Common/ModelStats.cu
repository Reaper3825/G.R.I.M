#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelStats.hpp"
#include "../training/Phases/Startup/Model/ParameterRegistry.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {

#ifdef USE_CUDA

::GRIM::ModelStats computeModelStats(const ::ParameterRegistry::StartupParameterRegistry& parameter_registry) {
    // Rule 20 / Rule 26: registry-owned parameter_groups is the single source of truth for
    // parameter counts. Do not estimate from config formulas; registration is
    // the only durable inventory of trainable tensors after Pattern-B layer
    // construction and feature gates.
    const auto& parameter_groups = parameter_registry.requireParameterGroups("computeModelStats");

    ::GRIM::ModelStats stats;
    for (const auto& group : parameter_groups) {
        const std::size_t group_size = group.size();

        switch (group.stats_bucket) {
            case ParamStatsBucket::EMBEDDING:
                stats.embedding_params += group_size;
                break;

            case ParamStatsBucket::LM_HEAD:
                stats.lm_head_params += group_size;
                break;

            case ParamStatsBucket::COUNT:
                throw std::runtime_error(
                    "computeModelStats: parameter group '" + group.name +
                    "' has invalid ParamStatsBucket::COUNT at " +
                    std::string(__FILE__) + ":" + std::to_string(__LINE__));
        }
    }

    stats.total_params = stats.embedding_params +
                         stats.encoder_params + stats.lm_head_params;
    stats.model_size_mb = (stats.total_params * sizeof(float)) / (1024.0f * 1024.0f);
    return stats;
}

#endif // USE_CUDA

} // namespace GRIM
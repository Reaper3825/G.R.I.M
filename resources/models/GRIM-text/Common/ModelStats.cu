#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelStats.hpp"
#include "../GRIM/grim_language_model_cuda.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {

#ifdef USE_CUDA

::GRIM::ModelStats computeModelStats(const LanguageModel& model) {
    // Rule 20 / Rule 26: parameter_groups_ is the single source of truth for
    // parameter counts. Do not estimate from config formulas; registration is
    // the only durable inventory of trainable tensors after Pattern-B layer
    // construction and feature gates.
    const auto& parameter_groups = model.parameterGroups();
    if (parameter_groups.empty()) {
        throw std::runtime_error(
            "computeModelStats called before buildParameterGroups — parameter_groups is empty at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

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

            case ParamStatsBucket::ENCODER:
                stats.encoder_params += group_size;
                if (group.type == ParamGroupType::SCRATCHBLOCK) {
                    stats.scratchblock_params += group_size;
                }
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
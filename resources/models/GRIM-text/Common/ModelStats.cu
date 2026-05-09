#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "../GRIM/grim_language_model_cuda.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace GRIM {

#ifdef USE_CUDA

LanguageModel::ModelStats LanguageModel::getModelStats() const {
    // Rule 20 / Rule 26: parameter_groups_ is the single source of truth for
    // parameter counts. Do not estimate from config formulas; registration is
    // the only durable inventory of trainable tensors after Pattern-B layer
    // construction and feature gates.
    if (parameter_groups_.empty()) {
        throw std::runtime_error(
            "getModelStats called before buildParameterGroups — parameter_groups_ is empty at " +
            std::string(__FILE__) + ":" + std::to_string(__LINE__));
    }

    ModelStats stats;
    for (const auto& group : parameter_groups_) {
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
                    "getModelStats: parameter group '" + group.name +
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
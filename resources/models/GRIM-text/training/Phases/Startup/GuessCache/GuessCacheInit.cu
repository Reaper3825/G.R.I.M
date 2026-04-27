#include "GuessCacheInit.hpp"

#include "../../Phase1_Startup.hpp"

#include <stdexcept>

namespace GRIMText::Training {

void GuessCacheReady(TrainingContext& ctx) {
    if (!ctx.model) {
        throw std::runtime_error(
            "FATAL: GuessCacheReady requires ModelAllocated to have authored ctx.model");
    }

    ctx.guess_cache_scope = GRIMTS::Training::initGuessCache(
        ctx.model->getTrainingState(),
        ctx.config.hyperparameters.guess_aux_enabled,
        ctx.config.hyperparameters.single_stream_mode,
        ctx.global_step,
        ctx.guess_cache_state);
}

} // namespace GRIMText::Training

#include "ResumeState.hpp"

#include "../../Phase1_Startup.hpp"
#include "../../../OptimizerCheckpoint.hpp"

namespace GRIMText::Training {

ResumeState captureResumeState(const TrainingContext& ctx) {
    ResumeState st;
    st.loaded_checkpoint_path = ctx.loaded_checkpoint_path;
    st.resumed = !ctx.loaded_checkpoint_path.empty();

    if (st.resumed) {
        st.optimizer_sidecar_path = optimizerSidecarPath(ctx.loaded_checkpoint_path);
    }

    st.optimizer_step = ctx.optimizer.optimizer_state.step;
    st.global_step = ctx.global_step;
    st.best_val_loss = ctx.best_val_loss;
    st.epochs_completed = ctx.epochs_completed;
    st.micro_step = ctx.optimizer.current_micro_step;
    return st;
}

} // namespace GRIMText::Training


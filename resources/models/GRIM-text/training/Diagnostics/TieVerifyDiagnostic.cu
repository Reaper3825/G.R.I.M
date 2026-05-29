//======================================================//
//  TieVerifyDiagnostic.cu
//  Lifted verbatim from Phase2_TrainingLoop.cu — the
//  "RUNTIME tie_embeddings pointer verification" scope.
//  Behavior: identical. No logic, gating, ordering, or
//  log-string changes vs. the original inline block.
//======================================================//

#include "TieVerifyDiagnostic.hpp"

#include "../Phases/Phase2_TrainingLoop.hpp"

#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdint>

namespace GRIM::Diagnostics {

void runTieVerifyDiagnostic(
    GRIMText::Training::TrainingContext& ctx,
    std::size_t batch_idx)
{
    const float* emb_w = ctx.model->getEmbeddingLayer()->tokenWeights().data;
    const float* emb_g = ctx.model->getEmbeddingLayer()->tokenWeights().grad_data();
    const float* lm_w  = ctx.model->getLmHeadLayer()->weights().data;
    const float* lm_g  = ctx.model->getLmHeadLayer()->weights().grad_data();
    const bool cfg_tied = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings");
    const bool w_same = (emb_w == lm_w);
    const bool g_same = (emb_g == lm_g);

    // Count parameter groups referencing each buffer
    int emb_w_groups = 0, lm_w_groups = 0;
    for (const auto& pg : ctx.model->parameterGroups()) {
        if (pg.tensor && pg.tensor->data == emb_w) ++emb_w_groups;
        if (pg.tensor && pg.tensor->data == lm_w)  ++lm_w_groups;
    }

    // Log every 10 batches to avoid spam, but ALWAYS log if inconsistent
    const bool inconsistent = (cfg_tied != w_same) || (cfg_tied != g_same);
    if (inconsistent || (batch_idx % 10 == 0)) {
        std::ostringstream oss;
        oss << "[TIE_VERIFY] B=" << (batch_idx + 1)
            << " step=" << ctx.optimizer.optimizer_step.step
            << " cfg_tied=" << (cfg_tied ? "yes" : "no")
            << " w_ptrs=" << (w_same ? "SAME" : "DIFF")
            << " g_ptrs=" << (g_same ? "SAME" : "DIFF")
            << " emb_w=" << (const void*)emb_w
            << " lm_w=" << (const void*)lm_w
            << " emb_g=" << (const void*)emb_g
            << " lm_g=" << (const void*)lm_g
            << " emb_w_groups=" << emb_w_groups
            << " lm_w_groups=" << lm_w_groups;
        if (inconsistent) {
            oss << " [ANOMALY] POINTER ALIASING MISMATCH — cfg says "
                << (cfg_tied ? "tied" : "untied")
                << " but weights " << (w_same ? "match" : "DIFFER")
                << " and grads " << (g_same ? "match" : "DIFFER");
        }
        ctx.logging.logger->log(oss.str());
    }

    // Rule 20: crash on mismatch — this is an architectural bug
    if (cfg_tied && !w_same) {
        throw std::runtime_error("[TIE_VERIFY] FATAL: tie_embeddings=true but weight pointers differ at batch "
            + std::to_string(batch_idx + 1) + " emb=" + std::to_string(reinterpret_cast<uintptr_t>(emb_w))
            + " lm=" + std::to_string(reinterpret_cast<uintptr_t>(lm_w)));
    }
    if (cfg_tied && !g_same) {
        throw std::runtime_error("[TIE_VERIFY] FATAL: tie_embeddings=true but grad pointers differ at batch "
            + std::to_string(batch_idx + 1) + " emb_g=" + std::to_string(reinterpret_cast<uintptr_t>(emb_g))
            + " lm_g=" + std::to_string(reinterpret_cast<uintptr_t>(lm_g)));
    }
}

} // namespace GRIM::Diagnostics

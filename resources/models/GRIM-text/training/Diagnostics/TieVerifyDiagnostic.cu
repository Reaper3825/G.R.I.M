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
    ::ParameterRegistry::StartupParameterRegistry& parameter_registry,
    std::size_t batch_idx)
{
    auto& embedding_parameters = parameter_registry.requireEmbeddingParameters("runTieVerifyDiagnostic");
    auto& lm_head_parameters = parameter_registry.requireLmHeadParameters("runTieVerifyDiagnostic");
    const float* emb_w = embedding_parameters.token_weights.data;
    const float* emb_g = embedding_parameters.token_weights.grad_data();
    const float* lm_w  = lm_head_parameters.weights.data;
    const float* lm_g  = lm_head_parameters.weights.grad_data();
    const bool cfg_tied = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "tie_embeddings");
    const bool lora_model = GRIM::HyperParameters::snapshotTrainingConfigField<bool>(ctx.config, "lora_model");
    const bool w_same = (emb_w == lm_w);
    const bool g_same = (emb_g == lm_g);
    const bool frozen_grads_absent = emb_g == nullptr && lm_g == nullptr;

    // Count parameter groups referencing each buffer
    int emb_w_groups = 0, lm_w_groups = 0;
    for (const auto& pg : ctx.parameter_registry.requireParameterGroups("runTieVerifyDiagnostic")) {
        if (pg.tensor && pg.tensor->data == emb_w) ++emb_w_groups;
        if (pg.tensor && pg.tensor->data == lm_w)  ++lm_w_groups;
    }

    // Log every 10 batches to avoid spam, but ALWAYS log if inconsistent
    const bool gradients_consistent = lora_model
        ? frozen_grads_absent
        : (cfg_tied == g_same);
    const bool inconsistent = (cfg_tied != w_same) || !gradients_consistent;
    if (inconsistent || (batch_idx % 10 == 0)) {
        std::ostringstream oss;
        oss << "[TIE_VERIFY] B=" << (batch_idx + 1)
            << " step=" << ctx.optimizer.optimizer_step.step
            << " cfg_tied=" << (cfg_tied ? "yes" : "no")
            << " lora_model=" << (lora_model ? "yes" : "no")
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
                << " and grads are "
                << (gradients_consistent ? "consistent" : "INCONSISTENT");
        }
        ctx.logging.logger->log(oss.str());
    }

    // Rule 20: crash on mismatch — this is an architectural bug
    if (cfg_tied && !w_same) {
        throw std::runtime_error("[TIE_VERIFY] FATAL: tie_embeddings=true but weight pointers differ at batch "
            + std::to_string(batch_idx + 1) + " emb=" + std::to_string(reinterpret_cast<uintptr_t>(emb_w))
            + " lm=" + std::to_string(reinterpret_cast<uintptr_t>(lm_w)));
    }
    if (lora_model && !frozen_grads_absent) {
        throw std::runtime_error("[TIE_VERIFY] FATAL: LoRA frozen embedding/LM head owns a grad buffer at batch "
            + std::to_string(batch_idx + 1) + " emb_g=" + std::to_string(reinterpret_cast<uintptr_t>(emb_g))
            + " lm_g=" + std::to_string(reinterpret_cast<uintptr_t>(lm_g)));
    }
    if (!lora_model && cfg_tied && !g_same) {
        throw std::runtime_error("[TIE_VERIFY] FATAL: tie_embeddings=true but grad pointers differ at batch "
            + std::to_string(batch_idx + 1) + " emb_g=" + std::to_string(reinterpret_cast<uintptr_t>(emb_g))
            + " lm_g=" + std::to_string(reinterpret_cast<uintptr_t>(lm_g)));
    }
}

} // namespace GRIM::Diagnostics

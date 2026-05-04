//======================================================//
//  OptimizerStepGuards.hpp
//  Three self-contained guards around the optimizer step,
//  lifted verbatim from Phase2_TrainingLoop.cu processBatch.
//
//   1. zeroNonTrainableTokenGrads (Issue #149)
//        Zero embedding + LM head grad rows for PAD / UNK
//        before clipping or stepping.
//
//   2. clipPostAccumulationGradients
//        Global gradient clipping on the accumulated + scaled gradients.
//        Mutates result.grad_rms /
//        result.grad_rms / result.gradient_clipped.
//
//   3. checkPostOptimizerWeightsFinite (Rule 20)
//        After the optimizer step, sample one weight from
//        every parameter group and throw on NaN/Inf.
//======================================================//

#pragma once

namespace GRIMText { namespace Training {
    struct TrainingContext;
    struct BatchResult;
} }

namespace GRIM::Diagnostics {

void zeroNonTrainableTokenGrads(
    GRIMText::Training::TrainingContext& ctx);

// per_token_limit comes from HyperParameters::gradientClippingHP() and is the
// effective max RMS per token. Caller checks clipping is enabled first.
void clipPostAccumulationGradients(
    GRIMText::Training::TrainingContext& ctx,
    GRIMText::Training::BatchResult& result,
    float per_token_limit,
    int batch_idx,
    float& clipping_elapsed_ms);

void checkPostOptimizerWeightsFinite(
    GRIMText::Training::TrainingContext& ctx,
    const GRIMText::Training::BatchResult& result,
    int batch_idx);

} // namespace GRIM::Diagnostics

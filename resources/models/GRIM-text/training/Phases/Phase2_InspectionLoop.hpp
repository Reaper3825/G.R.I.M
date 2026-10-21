#pragma once
//======================================================//
//  Phase2_InspectionLoop.hpp
//  Lens inspection orchestration after Phase1 startup.
//======================================================//

#include "Phase2_InferenceLoop.hpp"

namespace GRIMText::Training {

// Inspect the existing prefill through lens capture. Jacobian requests use
// activation gradients; other captures use prefill-only inference. No token
// sampling, optimizer updates, or parameter graph. The result owns the capture.
Phase2TextInferenceResult executePhase2Inspection(
    TrainingContext& ctx, GRIM::Tokenizer::UniByte& tokenizer,
    const std::string& prompt, const nlohmann::json* input_state,
    const GRIM::Lenses::LensCaptureRequest& capture);

} // namespace GRIMText::Training

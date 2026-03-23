#pragma once
#include "Serialization_requests.hpp"
#include "../../training/schemas/grim_transformer_model_generated.h"

namespace GRIM {

bool validate_checkpoint_capabilities(
    const GRIMTransformer::TransformerModel* model_fb,
    const SerializationModelConfigView& cfg,
    const CheckpointCapabilityRequirements& req,
    const SerializationLoadRequest& load_req);

} // namespace GRIM

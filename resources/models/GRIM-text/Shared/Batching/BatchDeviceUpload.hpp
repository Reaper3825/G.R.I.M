//======================================================//
//  BatchDeviceUpload.hpp
//
//  Explicit H2D sync boundary for one BatchPayload.
//  Shared/Batching owns this because it translates caller-authored host
//  payload semantics into the borrowed BatchDeviceBindings device view.
//======================================================//

#pragma once

#include "BatchPayload.hpp"
#include "BatchDeviceBindings.hpp"
#include "../HyperParameters/HyperParameters_GPU.hpp"
#include "../TrainingState/TrainingState_GPU.hpp"

namespace GRIM {
namespace Batching {

BatchDeviceBindings uploadBatchToDevice(
    const HyperParameters::LanguageModelConfig& config,
    TrainingState& training_state,
    const BatchPayload& payload);

}  // namespace Batching
}  // namespace GRIM
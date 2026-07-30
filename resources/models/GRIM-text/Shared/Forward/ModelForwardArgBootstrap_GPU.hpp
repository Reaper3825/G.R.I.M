//======================================================//
//  ModelForwardArgBootstrap_GPU.hpp
//  Argument-bootstrap seed boundary for shared forward
//======================================================//

#pragma once

#ifdef USE_CUDA

#include "ModelForward_GPU.hpp"

namespace GRIM {
namespace HyperParameters {
struct SlotSeedEncoderConstructionHP;
}

namespace Forward {

// Materializes the contextual argument-slot seeds at the configured encoder
// layer. This is deliberately the end of the bootstrap path: shared forward
// does not gate, execute, mutate registers, read results back, or retry here.
void materializeForwardArgBootstrapSeeds(
    const ModelForwardRequest& request,
    const HyperParameters::SlotSeedEncoderConstructionHP& slot_seed_encoder_hp,
    int num_slots,
    int layer_idx,
    int bootstrap_layer,
    bool arg_bootstrap_active,
    const Tensor& layer_output,
    ModelForwardOutputs& forward_outputs);

}  // namespace Forward
}  // namespace GRIM

#endif  // USE_CUDA

//======================================================//
//  ModelForwardArgBootstrap_GPU.cu
//  Argument-bootstrap seed boundary for shared forward
//======================================================//

#ifndef USE_CUDA
#define USE_CUDA
#endif

#include "ModelForwardArgBootstrap_GPU.hpp"

#include "../HyperParameters/HyperparameterGroupings.hpp"
#include "../../Layers/SlotSeedEncoder/SlotSeedEncoder_GPU.hpp"
#include "../../training/Phases/Startup/Model/ParameterRegistry.hpp"

namespace GRIM {
namespace Forward {
namespace {

GRIM::SlotSeedEncoderParameterTensors detachSlotSeedEncoderParameters(
    const GRIM::SlotSeedEncoderParameterTensors& parameters,
    const HyperParameters::SlotSeedEncoderConstructionHP& hp,
    cudaStream_t stream)
{
    GRIM::SlotSeedEncoderParameterTensors detached{};
    detached.W_seed_in = parameters.W_seed_in.detach(stream);
    detached.W_seed_out = parameters.W_seed_out.detach(stream);
    if (hp.bias_enabled) {
        detached.b_seed_in = parameters.b_seed_in.detach(stream);
        detached.b_seed_out = parameters.b_seed_out.detach(stream);
    }
    if (hp.type_embedding_enabled) {
        detached.type_embeddings = parameters.type_embeddings.detach(stream);
    }
    return detached;
}

}  // namespace

void materializeForwardArgBootstrapSeeds(
    const ModelForwardRequest& request,
    const HyperParameters::SlotSeedEncoderConstructionHP& slot_seed_encoder_hp,
    int num_slots,
    int layer_idx,
    int bootstrap_layer,
    bool arg_bootstrap_active,
    const Tensor& layer_output,
    ModelForwardOutputs& forward_outputs)
{
    if (!arg_bootstrap_active ||
        !slot_seed_encoder_hp.enabled ||
        layer_idx != bootstrap_layer) {
        return;
    }

    const auto& registered =
        request.parameter_registry->requireSlotSeedEncoderParameters(
            "executeModelForward(arg_bootstrap_seed)");
    const GRIM::SlotSeedEncoderParameterTensors* active = &registered;
    GRIM::SlotSeedEncoderParameterTensors detached{};
    if (!request.graph.connect_parameter_graph) {
        detached = detachSlotSeedEncoderParameters(
            registered,
            slot_seed_encoder_hp,
            request.stream);
        active = &detached;
    }

    SlotSeedEncoder::forward(
        slot_seed_encoder_hp,
        *active,
        layer_output,
        *request.payload,
        *request.bindings,
        num_slots,
        request.stream,
        forward_outputs);
}

}  // namespace Forward
}  // namespace GRIM

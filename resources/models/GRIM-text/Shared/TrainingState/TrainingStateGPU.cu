//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

#include "TrainingState_GPU.hpp"
#include "../../training/Autograd/AutogradTraining.hpp"  

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#ifdef USE_CUDA

namespace GRIM {

//======================================================//
//  Weight tensor accessors
//  Session 6: Embedding accessors DELETED — weights now owned by StartupParameterRegistry.
//  Access via TrainingContext::parameter_registry.getEmbeddingParameters()->token_weights.
//======================================================//

TrainingState::TrainingState() = default;
TrainingState::~TrainingState() = default;

void TrainingState::allocateReadGateWorkspace(cudaStream_t stream)
{
    StreamController::fatalIfDefaultStream(stream,
                                           "TrainingState::allocateReadGateWorkspace");

    read_gate_accum_tensor = Tensor::zeros({2}, stream, "read_gate_accum");
    std::cout << "✓ Allocated read-gate telemetry accumulator [sum,count]" << std::endl;
}

} // namespace GRIM

#endif  // USE_CUDA
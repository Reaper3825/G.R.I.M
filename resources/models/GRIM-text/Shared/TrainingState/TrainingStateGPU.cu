//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

#include "../../Layers/Embedding/Embedding_GPU.hpp"
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
//  Session 6: Embedding accessors DELETED — weights now owned by EmbeddingLayer (Pattern B).
//  Access via LanguageModel::getEmbeddingLayer()->tokenWeights().
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
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

} // namespace GRIM

#endif  // USE_CUDA
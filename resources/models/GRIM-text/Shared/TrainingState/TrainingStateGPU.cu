//======================================================//
//  TrainingStateGPU.cu
//  TrainingState implementation details
//======================================================//

// Include grim_language_model_cuda.hpp for GPUGrimEncoder, FlashAttentionBF16Scratch, etc.
#include "../../GRIM/grim_language_model_cuda.hpp"
#include "../../Layers/Embedding/Embedding_GPU.hpp"
#include "TrainingState_GPU.hpp"
#include "../../training/Autograd/AutogradTraining.hpp"  

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

} // namespace GRIM

#endif  // USE_CUDA
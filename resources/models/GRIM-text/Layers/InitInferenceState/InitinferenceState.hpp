//======================================================//
//  InitInferenceState.hpp
//  Lightweight inference-only state initialization
//  
//  Initializes only forward-pass buffers WITHOUT gradients
//  For use in grim_text_server and other inference contexts
//  
//  Key differences from initTrainingState:
//  - NO gradient buffers
//  - NO optimizer state
//  - Minimal activation caches (batch=1, short sequences)
//  - Quantization-aware if enabled in config
//  
//  Implementation: LanguageModel::initInferenceState()
//  
//  Author: GRIM Development Team
//  Date: December 9, 2025
//======================================================//

#pragma once

// This file is now just a header placeholder.
// The actual implementation is in LanguageModel::initInferenceState()
// declared in grim_language_model_cuda.hpp

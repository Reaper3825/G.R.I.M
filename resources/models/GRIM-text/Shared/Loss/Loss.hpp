#pragma once
//======================================================//
//  Loss.hpp — GUTTED (Rule 26: dead code deletion)
//
//  The entire old loss system (LossContext, LossConfig, LossBreakdown,
//  DeviceBuffers, AuxiliaryBatchViews, and all sub-module function 
//  declarations) was DEAD CODE. The production loss path is:
//
//    BatchPayload → AutogradContext → computeAutogradLoss() → unified_loss()
//
//  Functions removed:
//    - computeLossTerms()
//    - applyLabelSmoothing()
//    - accumulateDistillationKL()
//    - accumulatePreferenceKL()
//    - applyFocalLossScaling()
//    - applyTokenMasking()
//    - blendGuessFeedback()
//    - computeInfoNCELoss()
//    - computeCosineSimilarityLoss()
//    - validate()
//
//  Structs removed:
//    - Loss::Term enum
//    - Loss::LabelSmoothingConfig, DistillationConfig, PreferenceKLConfig,
//      FocalLossConfig, MaskConfig, GuessFeedbackConfig, LimitsConfig,
//      EntropyRegConfig, LossConfig (composite)
//    - Loss::LossContext (duplicated batch metadata from BatchPayload)
//    - Loss::LossBreakdown, DeviceBuffers, AuxiliaryBatchViews
//
//  Live loss code:
//    - autograd::unified_loss() → AutogradLoss.cu
//    - HyperParameters::LossConfigHP → HyperparameterGroupings.hpp
//
//  Sub-module .cu/.hpp files also deleted:
//    - CrossEntropy/CrossEntropy_GPU.cu/hpp
//    - LabelSmoothing/LabelSmoothing_GPU.cu/hpp
//    - TKML/TKML_GPU.cu/hpp
//    - Preference/Preference_KL_GPU.cu/hpp
//    - Divergence/Divergence_GPU.cu/hpp
//======================================================//

// This file is intentionally empty. It exists only so that stale #includes
// in deleted sub-module headers don't break the build during cleanup.
// Once all sub-module files are deleted, this file can also be deleted.


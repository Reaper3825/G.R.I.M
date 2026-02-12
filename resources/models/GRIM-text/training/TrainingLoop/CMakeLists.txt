# TrainingLoop CMakeLists.txt
# GPU-accelerated training executable only
cmake_minimum_required(VERSION 3.22)

# vcpkg manifest mode - uses CMAKE_PREFIX_PATH for package discovery
# Set prefix path to vcpkg_installed for package discovery
if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../vcpkg_installed/x64-windows")
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../vcpkg_installed/x64-windows" CACHE PATH "vcpkg installed packages")
endif()

# Enable CUDA (required for training loop)
enable_language(CUDA)
project(GRIM_TrainingLoop LANGUAGES CXX CUDA)
set(CUDA_ENABLED ON)
message(STATUS "CUDA enabled for training loop")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable Windows min/max macros to prevent conflicts with std::min/std::max
add_compile_definitions(NOMINMAX)

# Option: PyTorch verification for side-by-side comparison (SLOW - debug only!)
option(GRIM_PYTORCH_VERIFY "Enable PyTorch verification via subprocess" OFF)
if(GRIM_PYTORCH_VERIFY)
    add_compile_definitions(GRIM_PYTORCH_VERIFY)
    message(STATUS "PyTorch verification ENABLED (debug mode - will be slow!)")
endif()

# Enable precompiled headers for faster compilation
set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(flatbuffers CONFIG REQUIRED)

# CUDA compilation flags
# -diag-suppress=550: suppress "variable set but never used" warnings from third-party headers (FlatBuffers)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -cudart shared -diag-suppress=550")
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/arch:AVX2 /openmp\"")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG")

# Include directories
# CMAKE_CURRENT_SOURCE_DIR = training/TrainingLoop
# ../.. = training/ (wrong)
# ../../.. = GRIM-text/ (correct root)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../..)      # GRIM-text root
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)         # training/
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)            # training/
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM)    # GRIM/ folder (shared model components)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../Shared)  # Shared/ folder for TrainingState, etc.
include_directories(${CMAKE_CURRENT_SOURCE_DIR})               # TrainingLoop/

# GPU training kernels library (from parent directory)
add_library(grim_training_kernels STATIC 
    ../../Shared/Loss/CrossEntropy/CrossEntropy_GPU.cu
    ../../Shared/Gradients/GradientCC_GPU.cu
    ../../Shared/GPUBuffer/GPUBuffer.cu
    ../../Shared/GradNorm/GradNormGPU.cu
    ../../Shared/Telemetry/TelemetryLattice_GPU.cu
    ../../Shared/Telemetry/TelemetryControl_GPU.cu
    # Rule 20: Xavier.cu DELETED - Tensor::xavier_uniform_ (TensorContract) is the single source of truth
    ../../Shared/PBM/PositionalBiasMethod.cu
    # Rule 20: Old PBM modules DELETED - no backwards compatibility
)
set_target_properties(grim_training_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
)
target_include_directories(grim_training_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..      # GRIM-text root
    ${CMAKE_CURRENT_SOURCE_DIR}/../..         # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/..            # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM    # shared GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared  # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control  # control/ for ai_config_paths.hpp
)
target_link_libraries(grim_training_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)
target_compile_definitions(grim_training_kernels PRIVATE 
    USE_CUDA
)

# LogRecorder library (shared by all targets to avoid duplicate symbols)
add_library(grim_log_recorder STATIC
    ../../Shared/LogRecorder/LogRecorder.cu
)
set_target_properties(grim_log_recorder PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
)
target_include_directories(grim_log_recorder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..      # GRIM-text root
    ${CMAKE_CURRENT_SOURCE_DIR}/../..         # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared  # Shared/ folder
)
target_link_libraries(grim_log_recorder PUBLIC
    CUDA::cudart
)

# Flash Attention kernels library
add_library(grim_flash_attention STATIC 
    ../../Layers/FlashAttention/Flash_Attention_Kernal.cu
    ../../Layers/FlashAttention/AttentionDiagnostics.cu
    # Rule 20: RoPE_Verification.cu DELETED - no backwards compatibility
)
set_target_properties(grim_flash_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
)
target_compile_options(grim_flash_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/FS>
)
target_compile_definitions(grim_flash_attention PRIVATE
    GRIM_FLASHATTN_BF16_ONLY
    GRIM_FLASHATTN_HDIM64_ONLY
    GRIM_FLASHATTN_CAUSAL_ONLY
    FLASHATTENTION_DISABLE_UNEVEN_K
)
target_include_directories(grim_flash_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..      # GRIM-text root
    ${CMAKE_CURRENT_SOURCE_DIR}/../..         # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/..            # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM    # shared GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared  # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control  # control/ for ai_config_paths.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../external/flash-attention/csrc/cutlass/include
)
target_link_libraries(grim_flash_attention PUBLIC
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
    grim_log_recorder
)

# GPU transformer kernels object library
add_library(grim_cuda_kernels OBJECT
    ../../Layers/Attention/QKV_Projector.cu
    # Embedding_GPU.cu removed - compiled in grim_training_kernels to avoid duplicate symbols
    ../../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu
    ../BiasGradientKernel.cu  # Bias gradient kernel (was in deleted BackwardKernels.cu)
)
set_target_properties(grim_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "86"
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(grim_cuda_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../archive
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control  # control/ for ai_config_paths.hpp
)
target_compile_options(grim_cuda_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_link_libraries(grim_cuda_kernels PUBLIC
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)

# Language model GPU implementation object library
add_library(grim_language_model_gpu_impl OBJECT
    ../../Common/grim_language_model_gpu.cu
    ../../Common/ScaleBuffer.cu
    ../Inference_GPU.cu
    ../Forward_GPU.cu
    ../InitTrainingState.cu
    ../TrainingOps.cu
    ../../Shared/Loss/ComputeLoss/ComputeLossBatch.cu
    ../../Shared/TrainingState/TrainingStateGPU.cu
    ../../Shared/TrainingState/TrainingTensors.cu
    ../../Shared/TeacherLogits/TeacherLogits_GPU.cu
    ../../Shared/Dropout/Dropout_GPU.cu
    ../LanguageModel_Training.cu
    ../../Shared/Gradients/GradStatsCollector.cu
    ../Autograd/AutogradTraining.cu
    ../../Layers/HardcodedStates/HardcodedStates_GPU.cu
    ../../Layers/LMHead/lm_head_GPU.cu
    ../../Layers/NumericHead/numeric_head_GPU.cu
    ../../Layers/Quantization/Quantization_GPU.cu
    ../../Shared/DynaSeqs/DynaSeq_GPU.cu
    ../../Shared/Batching/Batching_GPU.cu
    ../../Shared/Batching/BatchPayload.cu
    ../../Shared/Loss/NumericLoss/NumericLoss_GPU.cu
    ../../Layers/Encoding/Encoding_GPU.cu
    ../../Layers/ScratchBlock/ScratchBlock_GPU.cu
    ../../Shared/ScratchBlock/ScratchBlock_GPU.cu
    ../../Layers/InitInferenceState/InitinferenceState.cu
    ../../Shared/TensorConversion/TensorConversion.cu
    ../../Shared/TensorContract/TensorContract_GPU.cu
    ../../Layers/FeedForward/Feed_Forward_GPU.cu
    ../../Shared/Activations/GELU/GELU.cu
    ../../Shared/StreamController/StreamController_GPU.cu
    ../../Shared/Loss/ComputeLoss/AutogradLoss.cu
    # ../Autograd/AutogradTraining.cu  # DISABLED: Dead code, replaced by HYBRID approach in LanguageModel_Training.cu
)
set_target_properties(grim_language_model_gpu_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "86"
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(grim_language_model_gpu_impl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared/TensorContract  # TensorContract module
    ${CMAKE_CURRENT_SOURCE_DIR}/../../archive
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control  # control/ for ai_config_paths.hpp
)
target_compile_options(grim_language_model_gpu_impl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_compile_definitions(grim_language_model_gpu_impl PRIVATE 
    CUDA_GPU_WRAPPER 
    USE_CUDA
)
target_link_libraries(grim_language_model_gpu_impl PUBLIC
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
)

# Model serialization library
add_library(grim_model_serialization STATIC

    ../../Layers/Serialization/Serialization_GPU.cu
    ../../Common/grim_model_serialization.cu
)
set_target_properties(grim_model_serialization PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "86"
)
target_include_directories(grim_model_serialization PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../archive
)
target_link_libraries(grim_model_serialization PRIVATE
    flatbuffers::flatbuffers
    CUDA::cudart
)
target_compile_options(grim_model_serialization PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_compile_definitions(grim_model_serialization PRIVATE USE_CUDA)

# Main GPU training executable (three-phase architecture)
add_executable(train_gpu 
    ../train_gpu.cu
    ../Phases/Phase1_Startup.cu
    ../Phases/Phase1_Startup.hpp
    ../Phases/Phase2_TrainingLoop.cu
    ../Phases/Phase2_TrainingLoop.hpp
    ../Phases/Phase3_Cleanup.cu
    ../Phases/Phase3_Cleanup.hpp
    ../training_status_writer.hpp
    ../training_logger.hpp
    ../training_data_loader.hpp
    ../metrics_collector.hpp
    ../TrainingEvents.hpp
    ../../Shared/Dynamic_LR/DynamicLR.cu
    ../../Shared/DataLoader/DataLoader.cu
    ../../Shared/TNC/Token-normalized_clipping.cu
    ../../Shared/SoftRestart/SoftRestart.cu
    ../../Layers/GRIMTS/GRIM-TS.cu
    ../../Shared/Stability/StabilityOverrides.cpp
)

set_target_properties(train_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_definitions(train_gpu PRIVATE USE_CUDA)

target_compile_options(train_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Enable precompiled headers for train_gpu
target_precompile_headers(train_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../pch.hpp
)


target_link_options(train_gpu PRIVATE 
    /NODEFAULTLIB:cudart_static.lib
)

target_link_libraries(train_gpu PRIVATE
    flatbuffers::flatbuffers
    nlohmann_json::nlohmann_json
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::nvml
    grim_flash_attention
    grim_training_kernels
    grim_cuda_kernels
    grim_language_model_gpu_impl
    grim_model_serialization
    grim_unigrambyte_tokenizer
    grim_log_recorder
    grim_equation_logging

    $<$<CXX_COMPILER_ID:GNU>:pthread gomp>
)

target_include_directories(train_gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../archive
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../core
)

# Equation logging library (centralized diagnostic logging system)
add_library(grim_equation_logging STATIC
    ../../Shared/EquationLogging/EquationLogging.cu
)
set_target_properties(grim_equation_logging PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
)
target_include_directories(grim_equation_logging PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
)
target_link_libraries(grim_equation_logging PUBLIC
    CUDA::cudart
)
target_compile_options(grim_equation_logging PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# UnigramByte tokenizer library (Unigram LM + Byte fallback + Atoms)
add_library(grim_unigrambyte_tokenizer STATIC
    ../../Shared/UnigramByte/Byte.cu
    ../../Shared/UnigramByte/Unigram.cu
    ../../Shared/UnigramByte/UniByte.cu
    ../../Shared/UnigramByte/AtomTable.cu
    ../../Shared/UnigramByte/AhoCorasick.cu
)
set_target_properties(grim_unigrambyte_tokenizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
)
target_include_directories(grim_unigrambyte_tokenizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
)
target_link_libraries(grim_unigrambyte_tokenizer PUBLIC
    CUDA::cudart
)
target_compile_options(grim_unigrambyte_tokenizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Tokenizer regression test executable
add_executable(tokenizer_self_test
    ../tokenizer_self_test.cu
)

set_target_properties(tokenizer_self_test PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(tokenizer_self_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(tokenizer_self_test PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
    grim_unigrambyte_tokenizer
)

target_include_directories(tokenizer_self_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
)

# UnigramByte tokenizer test executable
add_executable(unigrambyte_self_test
    ../../Tests/unigrambyte_self_test.cu
)

set_target_properties(unigrambyte_self_test PROPERTIES
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(unigrambyte_self_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(unigrambyte_self_test PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
    grim_unigrambyte_tokenizer
)

target_include_directories(unigrambyte_self_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
)

# Flash Attention test executable
add_executable(flash_attention_test
    ../../Tests/flash_attention_test.cu
    ../../Shared/TensorConversion/TensorConversion.cu
)

set_target_properties(flash_attention_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(flash_attention_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_compile_definitions(flash_attention_test PRIVATE
    USE_CUDA
)

target_link_libraries(flash_attention_test PRIVATE
    CUDA::cudart
    CUDA::cublas
    grim_flash_attention
)

target_include_directories(flash_attention_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
)

# Gradient Verification Test - for debugging encoder gradient collapse
add_executable(gradient_verification_test
    ../../Tests/gradient_verification_test.cu
    ../../Layers/FeedForward/Feed_Forward_GPU.cu
    ../../Layers/Attention/QKV_Projector.cu
    ../../Shared/Activations/GELU/GELU.cu
    ../../Shared/TensorConversion/TensorConversion.cu
    ../../Shared/StreamController/StreamController_GPU.cu
    ../../Shared/TensorContract/TensorContract_GPU.cu
    ../../Shared/PBM/PositionalBiasMethod.cu
)

set_target_properties(gradient_verification_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(gradient_verification_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_compile_definitions(gradient_verification_test PRIVATE
    USE_CUDA
)

target_link_libraries(gradient_verification_test PRIVATE
    CUDA::cudart
    CUDA::cublas
    grim_flash_attention
    grim_log_recorder
)

target_include_directories(gradient_verification_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Layers
)

# ============================================================================
# DIAGNOSTIC MINIMAL TRAINING LOOP
# Barebones training loop for isolating plateau bug
# No telemetry, no dynamic LR, no clipping, no restarts - just forward/backward/AdamW
# ============================================================================
add_executable(diagnostic_train_minimal
    ../diagnostic_train_minimal.cu
    ../../Shared/DataLoader/DataLoader.cu
)

set_target_properties(diagnostic_train_minimal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_definitions(diagnostic_train_minimal PRIVATE USE_CUDA)

target_compile_options(diagnostic_train_minimal PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Force exclusion of static CUDA runtime to prevent LNK2005 conflicts
target_link_options(diagnostic_train_minimal PRIVATE 
    /NODEFAULTLIB:cudart_static.lib
)

target_link_libraries(diagnostic_train_minimal PRIVATE
    flatbuffers::flatbuffers
    nlohmann_json::nlohmann_json
    CUDA::cudart
    CUDA::cublas
    grim_flash_attention
    grim_training_kernels
    grim_cuda_kernels
    grim_language_model_gpu_impl
    grim_unigrambyte_tokenizer
    grim_log_recorder
)

target_include_directories(diagnostic_train_minimal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control
)

# ============================================================================
# EMBEDDING AUTOGRAD TEST SUITE
# Autograd-enabled embedding tests using TensorContract system
message(STATUS "TrainingLoop configured:")
message(STATUS "  - train_gpu: GPU-accelerated training with comprehensive logging")
message(STATUS "  - diagnostic_train_minimal: Barebones training for plateau debugging")
message(STATUS "  - tokenizer_self_test: deterministic tokenizer regression suite (BPE)")
message(STATUS "  - unigrambyte_self_test: UnigramByte tokenizer test suite")
message(STATUS "  - flash_attention_test: Flash Attention forward/backward test suite")
message(STATUS "  - gradient_verification_test: Encoder gradient collapse debugging")
message(STATUS "  - grim_unigrambyte_tokenizer: Unigram LM + Byte fallback + AtomTable")

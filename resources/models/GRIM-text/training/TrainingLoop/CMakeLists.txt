# TrainingLoop CMakeLists.txt
# GPU-accelerated training executable only
# 3.20: Anvil (Purdue RCAC) default module; 3.18+ needed for CUDA_ARCHITECTURES / CUDAToolkit.
cmake_minimum_required(VERSION 3.20)

# vcpkg manifest mode (vcpkg.json at repo root). vcpkg install produces vcpkg_installed/<triplet>.
# We set CMAKE_PREFIX_PATH to that dir when present (relative to repo root).
# Repo root from here: ../../../../../ = resources/models/GRIM-text/training/TrainingLoop -> repo root
if(NOT DEFINED CMAKE_PREFIX_PATH)
    set(VCPKG_INSTALLED "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../vcpkg_installed")
    if(EXISTS "${VCPKG_INSTALLED}/x64-windows")
        set(CMAKE_PREFIX_PATH "${VCPKG_INSTALLED}/x64-windows" CACHE PATH "vcpkg manifest (Windows)")
    elseif(EXISTS "${VCPKG_INSTALLED}/x64-linux")
        set(CMAKE_PREFIX_PATH "${VCPKG_INSTALLED}/x64-linux" CACHE PATH "vcpkg manifest (Linux)")
    endif()
endif()

# Use CUDAToolkit_ROOT (or GRIM_CUDA_ROOT) before enabling CUDA so CMake picks that nvcc, not one from PATH.
# Needed on clusters (e.g. Anvil) where default module is CUDA 11 but project has CUDA 12 in project space.
foreach(_root CUDAToolkit_ROOT GRIM_CUDA_ROOT)
  if(DEFINED ${_root} AND NOT "${${_root}}" STREQUAL "")
    set(_cuda_root "${${_root}}")
    if(EXISTS "${_cuda_root}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc" CACHE PATH "CUDA compiler" FORCE)
      message(STATUS "Using CUDA from ${_root}=${_cuda_root}")
      break()
    endif()
  endif()
  if(DEFINED ENV{${_root}} AND NOT "$ENV{${_root}}" STREQUAL "")
    set(_cuda_root "$ENV{${_root}}")
    if(EXISTS "${_cuda_root}/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc" CACHE PATH "CUDA compiler" FORCE)
      message(STATUS "Using CUDA from ${_root}=${_cuda_root}")
      break()
    endif()
  endif()
endforeach()

# Enable CUDA (required for training loop)
enable_language(CUDA)
project(GRIM_TrainingLoop LANGUAGES CXX CUDA)
set(CUDA_ENABLED ON)
message(STATUS "CUDA enabled for training loop")

# CUDA architectures: A100 (sm_80), RTX 30xx (sm_86), H100 (sm_90) for Anvil ai partition.
# Override: GRIM_CUDA_ARCH=80 (A100-only) or 90 (H100-only) or 80;86;90 (default, portable).
if(DEFINED ENV{GRIM_CUDA_ARCH})
    set(GRIM_CUDA_ARCHITECTURES "$ENV{GRIM_CUDA_ARCH}" CACHE STRING "CUDA arch list")
else()
    set(GRIM_CUDA_ARCHITECTURES "80;86;90" CACHE STRING "CUDA arch: A100 sm_80, RTX sm_86, H100 sm_90")
endif()
# Set globally so ALL CUDA targets inherit (including object libraries)
set(CMAKE_CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}")
message(STATUS "CUDA architectures: ${GRIM_CUDA_ARCHITECTURES}")

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

# CUTLASS (flash-attention) requires CUDA 12+. Fail early with clear instructions.
if(CUDAToolkit_VERSION_MAJOR LESS 12)
  message(FATAL_ERROR
    "CUDAToolkit ${CUDAToolkit_VERSION} detected. CUTLASS (flash-attention) requires CUDA 12+.\n"
    "  Set CUDAToolkit_ROOT (or GRIM_CUDA_ROOT) to a CUDA 12+ toolkit before configuring, e.g.:\n"
    "  cmake -DCUDAToolkit_ROOT=/path/to/cuda-12.0 ..\n"
    "  Or from the run script (Anvil): ensure project CUDA 12 is installed and script will default to it.")
endif()
message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")

find_package(nlohmann_json CONFIG REQUIRED)
find_package(flatbuffers CONFIG REQUIRED)
find_package(Threads REQUIRED)

# CUDA compilation flags
# -diag-suppress=550: suppress "variable set but never used" warnings from third-party headers (FlatBuffers)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -cudart shared -diag-suppress=550")
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/arch:AVX2 /openmp\"")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()

# Compiler flags (platform-specific)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -DNDEBUG")
endif()

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
    # Rule 26: CrossEntropy_GPU.cu DELETED — dead code. Production loss uses autograd::unified_loss() in AutogradLoss.cu
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
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
)
target_include_directories(grim_log_recorder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..      # GRIM-text root
    ${CMAKE_CURRENT_SOURCE_DIR}/../..         # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared  # Shared/ folder
)
target_link_libraries(grim_log_recorder PUBLIC
    CUDA::cudart
)

# Flash Attention kernels library - requires external/flash-attention submodule (git submodule update --init)
set(FLASH_ATTN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../external/flash-attention")
set(FLASH_ATTN_SRC "${FLASH_ATTN_ROOT}/csrc/flash_attn/src")
get_filename_component(ATEN_STUB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Layers/FlashAttention/aten_stub" ABSOLUTE)
if(NOT EXISTS "${FLASH_ATTN_SRC}/flash_bwd_kernel.h")
  message(FATAL_ERROR "flash-attention submodule not found. Run: git submodule update --init --recursive")
endif()
if(NOT EXISTS "${ATEN_STUB_DIR}/ATen/cuda/detail/UnpackRaw.cuh")
  message(FATAL_ERROR "ATen stub not found at ${ATEN_STUB_DIR}. Run: git pull")
endif()
add_library(grim_flash_attention STATIC 
    ../../Layers/FlashAttention/Flash_Attention_Kernal.cu
)
set_target_properties(grim_flash_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
)
target_compile_options(grim_flash_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=177>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-Xcompiler=/FS>
)
target_compile_definitions(grim_flash_attention PRIVATE
    GRIM_FLASHATTN_BF16_ONLY
    GRIM_FLASHATTN_HDIM64_ONLY
    GRIM_FLASHATTN_CAUSAL_ONLY
    FLASHATTENTION_DISABLE_UNEVEN_K
)
target_include_directories(grim_flash_attention PRIVATE
    ${ATEN_STUB_DIR}  # ATen stub (PhiloxCudaState, unpack) - no PyTorch; must be first for <ATen/...>
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..      # GRIM-text root
    ${CMAKE_CURRENT_SOURCE_DIR}/../..         # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/..            # training/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM    # shared GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared  # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control  # control/ for ai_config_paths.hpp
    ${FLASH_ATTN_SRC}                         # flash_attn headers (namespace_config.h, etc.)
    ${FLASH_ATTN_ROOT}/csrc/cutlass/include
)
target_link_libraries(grim_flash_attention PUBLIC
    CUDA::cudart
    CUDA::cublas
    nlohmann_json::nlohmann_json
    grim_log_recorder
)

# GPU transformer kernels object library
add_library(grim_cuda_kernels OBJECT
    # QKV_Projector.cu MOVED to grim_language_model_gpu_impl — depends on TensorConversion (same library)
    # Old Embedding_GPU.cu (dead code) removed. New Embedding_GPU.cu in grim_training_kernels (Pattern B layer).
    ../../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu
    ../BiasGradientKernel.cu  # Bias gradient kernel (was in deleted BackwardKernels.cu)
)
set_target_properties(grim_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF  # OFF: QKV_Projector only CALLS TensorConversion (doesn't export device symbols)
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(grim_cuda_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared/TensorContract  # TensorContract module
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
    ../../Layers/Attention/QKV_Projector.cu  # Moved from grim_cuda_kernels — depends on TensorConversion (this library)
    ../Inference_GPU.cu
    ../Forward_GPU.cu
    ../InitTrainingState.cu
    ../TrainingOps.cu
    ../../Shared/Loss/ComputeLoss/ComputeLossBatch.cu
    ../../Shared/TrainingState/TrainingStateGPU.cu
    ../../Shared/TeacherLogits/TeacherLogits_GPU.cu
    ../LanguageModel_Training.cu
    ../../Shared/Gradients/GradStatsCollector.cu
    ../Autograd/AutogradTraining.cu
    ../../Layers/HardcodedStates/HardcodedStates_GPU.cu
    ../../Layers/Embedding/Embedding_GPU.cu
    ../../Layers/LMHead/lm_head_GPU.cu
    ../../Layers/NumericHead/numeric_head_GPU.cu
    ../../Layers/ReasoningHead/reasoning_head_GPU.cu
    ../../Layers/ExecutionBlock/execution_block_GPU.cu
    ../../Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu
    ../../Layers/ExecutionBlock/execution_block_data_stream_GPU.cu
    ../../Layers/DecodeTimeSlotSelector/decode_time_slot_selector_GPU.cu
    ../../Shared/Execution/DecodeTimeNumPolicy.cu
    ../../Layers/Quantization/Quantization_GPU.cu
    ../../Shared/DynaSeqs/DynaSeq_GPU.cu
    ../../Shared/Batching/Batching_GPU.cu
    ../../Shared/Batching/BatchPayload.cu
    ../../Shared/Execution/ExecutionPayloadValidation.cu
    ../../Layers/Encoding/Encoding_GPU.cu
    ../../Layers/ScratchBlock/ScratchBlockReasoning_GPU.cu
    ../../Shared/ScratchBlock/ScratchBlockPool_GPU.cu
    ../../Shared/Sampling/Sampling.cu
    ../../Layers/InitInferenceState/InitinferenceState.cu
    ../../Shared/TensorConversion/TensorConversion.cu
    ../../Shared/TensorContract/TensorContract_GPU.cu
    ../../Layers/FeedForward/Feed_Forward_GPU.cu
    ../../Shared/Activations/GELU/GELU.cu
    ../../Shared/StreamController/StreamController_GPU.cu
    ../../Shared/Loss/ComputeLoss/AutogradLoss.cu
)
set_target_properties(grim_language_model_gpu_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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

    ../../Layers/Serialization/Serialization_validate.cu
    ../../Layers/Serialization/Serialization_load.cu
    ../../Layers/Serialization/Serialization_save.cu
    ../../Common/grim_model_serialization.cu
)
set_target_properties(grim_model_serialization PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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
    ../Diagnostics/TrainingDiagnostics.cu
    ../Diagnostics/TrainingDiagnostics.hpp
    ../Phases/Phase3_Cleanup.cu
    ../Phases/Phase3_Cleanup.hpp
    ../OptimizerCheckpoint.cu
    ../OptimizerCheckpoint.hpp
    ../training_status_writer.hpp
    ../training_logger.hpp
    ../training_data_loader.hpp
    ../metrics_collector.hpp
    ../TrainingEvents.hpp
    ../../Shared/Dynamic_LR/DynamicLR.cu
    ../../Shared/DataLoader/DataLoader.cu
    ../../Shared/DataLoader/ConceptExecutionSequenceBuilder.cu
    ../../Shared/TNC/Token-normalized_clipping.cu
    ../../Shared/SoftRestart/SoftRestart.cu
    ../../Layers/GRIMTS/GRIM-TS.cu
)

set_target_properties(train_gpu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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


# MSVC only: avoid cudart_static conflict when using shared runtime
if(MSVC)
    target_link_options(train_gpu PRIVATE /NODEFAULTLIB:cudart_static.lib)
endif()

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
)
# Link pthread and gomp for GCC/Linux (avoid genex - nvcc device link rejects $<CXX_COMPILER_ID:GNU>)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_link_libraries(train_gpu PRIVATE pthread gomp)
endif()
# GCC: std::filesystem (C++17) is in libstdc++fs until merged into libstdc++.so (GCC 9.1+); link explicitly for compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(train_gpu PRIVATE stdc++fs)
endif()

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
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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
    ../../Shared/UnigramByte/TextUtils.cu
    ../../Shared/UnigramByte/Detectors.cu
    ../../Shared/UnigramByte/UnigramTrainer.cu
)
set_target_properties(grim_unigrambyte_tokenizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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

# Tokenizer runner — standalone validation + JSON payload output
# Launched by train_gpu before training, or from UI for vocab testing
add_executable(tokenizer_runner
    ../tokenizer_runner.cu
)

set_target_properties(tokenizer_runner PROPERTIES
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(tokenizer_runner PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(tokenizer_runner PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
    grim_unigrambyte_tokenizer
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(tokenizer_runner PRIVATE stdc++fs)
endif()

target_include_directories(tokenizer_runner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control
)

# UnigramByte tokenizer test executable
add_executable(unigrambyte_self_test
    ../../Tests/unigrambyte_self_test.cu
)

set_target_properties(unigrambyte_self_test PROPERTIES
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(unigrambyte_self_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(unigrambyte_self_test PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
    grim_unigrambyte_tokenizer
    Threads::Threads
    stdc++fs
)

target_include_directories(unigrambyte_self_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
)

# GRMT vocab metrics test executable
add_executable(grmt_vocab_metrics_test
    ../../Tests/grmt_vocab_metrics_test.cu
)

set_target_properties(grmt_vocab_metrics_test PROPERTIES
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(grmt_vocab_metrics_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(grmt_vocab_metrics_test PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
    grim_unigrambyte_tokenizer
    stdc++fs
)

target_include_directories(grmt_vocab_metrics_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
)

# ============================================================================
# STANDALONE TOKENIZER TRAINING
# Runs full tokenizer pipeline on entire corpus: train vocab + encode GRMT
# No model training, no GPU kernels beyond tokenizer
# Usage: train_tokenizer [--force]
# ============================================================================
add_executable(train_tokenizer
    ../train_tokenizer.cu
    ../../Shared/DataLoader/DataLoader.cu
    ../../Shared/DataLoader/ConceptExecutionSequenceBuilder.cu
)

set_target_properties(train_tokenizer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

target_compile_definitions(train_tokenizer PRIVATE USE_CUDA)

target_compile_options(train_tokenizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_link_libraries(train_tokenizer PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
    grim_unigrambyte_tokenizer
    Threads::Threads
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(train_tokenizer PRIVATE stdc++fs pthread gomp)
endif()

target_include_directories(train_tokenizer PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../control
)

# Flash Attention test executable
add_executable(flash_attention_test
    ../../Tests/flash_attention_test.cu
    ../../Shared/TensorConversion/TensorConversion.cu
)

set_target_properties(flash_attention_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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

# ExecutionBlock test executable
add_executable(execution_block_test
    ../../Tests/ExecutionBlockTest.cu
    ../../Layers/ExecutionBlock/execution_block_GPU.cu
    ../../Layers/ExecutionBlock/execution_block_memory_stream_GPU.cu
    ../../Layers/ExecutionBlock/execution_block_data_stream_GPU.cu
    ../../Layers/Attention/QKV_Projector.cu
    ../../Shared/Batching/BatchPayload.cu
    ../../Shared/Execution/ExecutionPayloadValidation.cu
    ../../Shared/PBM/PositionalBiasMethod.cu
    ../../Shared/TensorContract/TensorContract_GPU.cu
    ../../Shared/TensorConversion/TensorConversion.cu
)

set_target_properties(execution_block_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
     CUDA_RUNTIME_LIBRARY Shared
)

target_compile_options(execution_block_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>

)

target_compile_definitions(execution_block_test PRIVATE
    USE_CUDA
)

target_link_libraries(execution_block_test PRIVATE
    nlohmann_json::nlohmann_json
    CUDA::cudart
    CUDA::cublas
    grim_flash_attention
    grim_unigrambyte_tokenizer
)

target_include_directories(execution_block_test PRIVATE
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
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
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
message(STATUS "  - grmt_vocab_metrics_test: GRMT corpus + vocabulary metrics diagnostics")
message(STATUS "  - flash_attention_test: Flash Attention forward/backward test suite")
message(STATUS "  - gradient_verification_test: Encoder gradient collapse debugging")
message(STATUS "  - execution_block_test: ExecutionBlock batching, state transitions, and config tests")
message(STATUS "  - grim_unigrambyte_tokenizer: Unigram LM + Byte fallback + AtomTable")

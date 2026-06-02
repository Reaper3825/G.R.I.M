# GRIM Text Server CMakeLists.txt
# GPU-accelerated inference server (Ollama-compatible API)
cmake_minimum_required(VERSION 3.20)

# vcpkg manifest mode - uses CMAKE_PREFIX_PATH for package discovery
# GRIM folder is at: resources/models/GRIM-text/GRIM
# vcpkg_installed is at: vcpkg_installed/x64-windows (4 levels up from GRIM)
if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg_installed/x64-windows")
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg_installed/x64-windows" CACHE PATH "vcpkg installed packages")
endif()

# Enable CUDA
enable_language(CUDA)
project(GRIM_TextServer LANGUAGES CXX CUDA)
set(CUDA_ENABLED ON)
message(STATUS "CUDA enabled for GRIM text server")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable Windows min/max macros
add_compile_definitions(NOMINMAX)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(flatbuffers CONFIG REQUIRED)
find_package(httplib CONFIG REQUIRED)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -cudart shared")
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=\"/arch:AVX2 /openmp\"")
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG")

# Include directories
set(GRIM_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../training
)
include_directories(${GRIM_INCLUDE_DIRS})

#======================================================
# Glob all .cu files from Layers, Shared, Common, training
#======================================================

# Layers folder (recursive)
file(GLOB_RECURSE LAYERS_CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../Layers/**/*.cu")

# Shared folder (recursive)
file(GLOB_RECURSE SHARED_CU_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../Shared/**/*.cu")

# Common folder
file(GLOB COMMON_CU_FILES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../Common/*.cu")

# Training folder (only specific files needed for inference)
set(TRAINING_CU_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Forward_GPU.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Phase1_Startup.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Phase2_InferenceLoop.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Rng.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Logging.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Capacity/CapacityStem.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Capacity/MemorySnapshot.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Data/TrainingData.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Model/ModelAllocationState.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/CheckpointLoad.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Resume/ResumeState.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Epoch/EpochPlan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Payload/PayloadBuildInputs.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Batching/PlannedBatches.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Telemetry/TelemetryInitInputs.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Validation/StartupValidation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Validation/Phase2Handoff.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/SlidingWindow.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/ClassBalancedWeights.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/InitFacts.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Subprocess/tokenizer_subprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Model/ModelGpuAssembly.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Autograd/AutogradExecutionLoss.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Autograd/AutogradMtpAuxiliaryLoss.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Autograd/AutogradSelectorSupervisionLoss.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Autograd/AutogradTraining.cu
)

# Combine all .cu files
set(ALL_GPU_SOURCES
    ${LAYERS_CU_FILES}
    ${SHARED_CU_FILES}
    ${COMMON_CU_FILES}
    ${TRAINING_CU_FILES}
)

# Remove Flash Attention from main list (needs special handling for sm_80+)
list(FILTER ALL_GPU_SOURCES EXCLUDE REGEX "Flash_Attention")

# Remove problematic files that have broken includes or aren't needed for inference
# Feed_Forward_GPU exclusion removed - implementation now complete with CRTP design
list(FILTER ALL_GPU_SOURCES EXCLUDE REGEX "GRIM-TS")
list(FILTER ALL_GPU_SOURCES EXCLUDE REGEX "LayerNorm_GPU\\.cu")  # Layer class - using RMSNorm instead
# Keep LayerNorm_Kernal_GPU.cu - launchLayerNorm is still referenced by live shared-forward/model sources

# Exclude SoftRestart - references training-only LanguageModel::resetOptimizerMoments
list(FILTER ALL_GPU_SOURCES EXCLUDE REGEX "SoftRestart")

# Exclude deprecated grim_embedding_gpu.cu (GPUEmbeddingStack implementation)
# The newer Embedding_GPU.hpp/cu is used instead
list(FILTER ALL_GPU_SOURCES EXCLUDE REGEX "grim_embedding_gpu\\.cu")

# Debug: Print found files
list(LENGTH ALL_GPU_SOURCES NUM_CU_FILES)
message(STATUS "Found ${NUM_CU_FILES} CUDA source files")

#======================================================
# Flash Attention library (requires sm_80+ only)
#======================================================
add_library(grim_server_flash_attention EXCLUDE_FROM_ALL STATIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../Layers/FlashAttention/Flash_Attention_Kernal.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../Layers/FlashAttention/AttentionDiagnostics.cu
)
set_target_properties(grim_server_flash_attention PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES "86"  # target local GPU only for faster builds
)
target_compile_options(grim_server_flash_attention PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
target_compile_definitions(grim_server_flash_attention PRIVATE
    GRIM_FLASHATTN_BF16_ONLY
    GRIM_FLASHATTN_HDIM64_ONLY
    GRIM_FLASHATTN_CAUSAL_ONLY
    FLASHATTENTION_DISABLE_UNEVEN_K
)
target_include_directories(grim_server_flash_attention PRIVATE ${GRIM_INCLUDE_DIRS})
target_include_directories(grim_server_flash_attention PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../external/flash-attention/csrc/cutlass/include
)
target_link_libraries(grim_server_flash_attention PUBLIC CUDA::cudart CUDA::cublas)

#======================================================
# Main GPU implementation library (all other .cu files)
#======================================================
add_library(grim_server_gpu_impl EXCLUDE_FROM_ALL OBJECT ${ALL_GPU_SOURCES})
set_target_properties(grim_server_gpu_impl PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "86"  # target local GPU only for faster builds
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(grim_server_gpu_impl PRIVATE ${GRIM_INCLUDE_DIRS})
target_compile_options(grim_server_gpu_impl PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

target_compile_definitions(grim_server_gpu_impl PRIVATE 
    USE_CUDA 
    GRIM_GPU_ONLY
    CUDA_GPU_WRAPPER
)

target_link_libraries(grim_server_gpu_impl PUBLIC
    CUDA::cudart
    CUDA::cublas
    flatbuffers::flatbuffers
)

#======================================================
# Main GRIM text server executable
#======================================================
add_executable(grim_text_server 
    ${CMAKE_CURRENT_SOURCE_DIR}/grim_text_server.cpp
)

set_target_properties(grim_text_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

target_link_libraries(grim_text_server PRIVATE
    nlohmann_json::nlohmann_json
    httplib::httplib
    $<$<CXX_COMPILER_ID:GNU>:pthread gomp>
    $<$<PLATFORM_ID:Windows>:ws2_32>
)

target_include_directories(grim_text_server PRIVATE ${GRIM_INCLUDE_DIRS})

# Set output directory to match expected training location
set_target_properties(grim_text_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}/../training/build/Release"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}/../training/build/Debug"
)

message(STATUS "GRIM Text Server configured:")
message(STATUS "  - grim_text_server: Ollama-compatible HTTP bridge to train_gpu inference worker")
message(STATUS "  - Output: training/build/Release/grim_text_server.exe")


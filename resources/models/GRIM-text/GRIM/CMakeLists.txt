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

# Ablation: disable the causal mask (full bidirectional self-attention). Mirrors
# the TrainingLoop option. DEFAULT IS ON (causal mask disabled). ON drops
# GRIM_FLASHATTN_CAUSAL_ONLY (compiling the non-causal kernels) and defines
# GRIM_ABLATE_DISABLE_CAUSAL_MASK so Ablation::kDisableCausalMask is true in
# lockstep. To restore the causal mask, configure with -DGRIM_ABLATE_DISABLE_CAUSAL_MASK=OFF.
option(GRIM_ABLATE_DISABLE_CAUSAL_MASK "Ablation: build non-causal FlashAttention and disable the causal mask (AblationFlags::kDisableCausalMask)" OFF)
if(GRIM_ABLATE_DISABLE_CAUSAL_MASK)
    add_compile_definitions(GRIM_ABLATE_DISABLE_CAUSAL_MASK)
    message(WARNING "[GRIM] Causal-mask ablation ENABLED: building NON-causal FlashAttention kernels (GRIM_FLASHATTN_CAUSAL_ONLY dropped). Do NOT ship this binary for production.")
    message(STATUS  "[GRIM] Compiled causal mask = DISABLED (bidirectional self-attention; kDisableCausalMask=true). effective_causal = causal_mask && !kDisableCausalMask resolves to FALSE regardless of the runtime `causal_mask` HP.")
else()
    message(STATUS  "[GRIM] Compiled causal mask = ENABLED (autoregressive; kDisableCausalMask=false, GRIM_FLASHATTN_CAUSAL_ONLY defined).")
endif()

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
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Model/ModelAllocationState.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/CheckpointLoad.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Resume/ResumeState.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Epoch/EpochPlan.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Batching/PlannedBatches.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Telemetry/TelemetryInitInputs.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Validation/StartupValidation.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Validation/Phase2Handoff.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/SlidingWindow.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/ClassBalancedWeights.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/InitFacts.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Subprocess/tokenizer_subprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/Phases/Startup/Model/ModelGpuAssembly.cu
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
    ${CMAKE_CURRENT_SOURCE_DIR}/../Layers/FlashAttention/AttentionBreadthDiagnostic.cu
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
    # Causal-only unless the causal-mask ablation is enabled (then the non-causal
    # kernel templates must be compiled in so a non-causal call is valid).
    $<$<NOT:$<BOOL:${GRIM_ABLATE_DISABLE_CAUSAL_MASK}>>:GRIM_FLASHATTN_CAUSAL_ONLY>
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

if(WIN32)
    set(GRIM_SERVER_BROTLI_DLL_SOURCE_DIR "")
    foreach(_prefix IN LISTS CMAKE_PREFIX_PATH)
        if(EXISTS "${_prefix}/bin/brotlienc.dll")
            set(GRIM_SERVER_BROTLI_DLL_SOURCE_DIR "${_prefix}/bin")
            break()
        endif()
    endforeach()

    if(NOT GRIM_SERVER_BROTLI_DLL_SOURCE_DIR)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg_installed/x64-windows/bin/brotlienc.dll")
            set(GRIM_SERVER_BROTLI_DLL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg_installed/x64-windows/bin")
        elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../external/vcpkg/installed/x64-windows/bin/brotlienc.dll")
            set(GRIM_SERVER_BROTLI_DLL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../external/vcpkg/installed/x64-windows/bin")
        endif()
    endif()

    if(GRIM_SERVER_BROTLI_DLL_SOURCE_DIR)
        set(GRIM_SERVER_BROTLI_DLLS)
        foreach(_dll IN ITEMS brotlicommon.dll brotlidec.dll brotlienc.dll)
            if(EXISTS "${GRIM_SERVER_BROTLI_DLL_SOURCE_DIR}/${_dll}")
                list(APPEND GRIM_SERVER_BROTLI_DLLS "${GRIM_SERVER_BROTLI_DLL_SOURCE_DIR}/${_dll}")
            else()
                message(WARNING "Expected Brotli runtime DLL missing at configure time: ${GRIM_SERVER_BROTLI_DLL_SOURCE_DIR}/${_dll}")
            endif()
        endforeach()

        if(GRIM_SERVER_BROTLI_DLLS)
            add_custom_command(TARGET grim_text_server POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different ${GRIM_SERVER_BROTLI_DLLS} $<TARGET_FILE_DIR:grim_text_server>
                COMMAND_EXPAND_LISTS
                COMMENT "Deploying Brotli runtime DLLs for grim_text_server"
            )
            message(STATUS "GRIM text server runtime DLL source: ${GRIM_SERVER_BROTLI_DLL_SOURCE_DIR}")
        else()
            message(WARNING "No Brotli runtime DLLs were found to deploy for grim_text_server")
        endif()
    else()
        message(WARNING "Could not locate brotlienc.dll in CMAKE_PREFIX_PATH or known vcpkg bin directories; grim_text_server may fail with STATUS_DLL_NOT_FOUND")
    endif()
endif()

message(STATUS "GRIM Text Server configured:")
message(STATUS "  - grim_text_server: Ollama-compatible HTTP bridge to train_gpu inference worker")
message(STATUS "  - Output: training/build/Release/grim_text_server.exe")


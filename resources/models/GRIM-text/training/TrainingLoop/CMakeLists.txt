# TrainingLoop CMakeLists.txt
# GPU-accelerated training executable only
# 3.20: Anvil (Purdue RCAC) default module; 3.18+ needed for CUDA_ARCHITECTURES / CUDAToolkit.
cmake_minimum_required(VERSION 3.20)

# TrainingLoop can consume manually staged header-only deps from training/third_party.
# If those headers are absent, fall back to the minimal vendored vcpkg manifest.
get_filename_component(GRIM_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../" ABSOLUTE)
get_filename_component(GRIM_TRAINING_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GRIM_TRAINING_MANUAL_DEPS_ROOT "${GRIM_TRAINING_ROOT}/third_party" CACHE PATH "Root directory for manually staged TrainingLoop dependencies")
set(GRIM_TRAINING_NLOHMANN_JSON_INCLUDE_DIR "${GRIM_TRAINING_MANUAL_DEPS_ROOT}/nlohmann_json/include" CACHE PATH "Directory containing nlohmann/json.hpp for TrainingLoop")
set(GRIM_TRAINING_FLATBUFFERS_INCLUDE_DIR "${GRIM_TRAINING_MANUAL_DEPS_ROOT}/flatbuffers/include" CACHE PATH "Directory containing flatbuffers/flatbuffers.h for TrainingLoop")
set(GRIM_TRAINING_HTTPLIB_INCLUDE_DIR "${GRIM_TRAINING_MANUAL_DEPS_ROOT}/cpp-httplib/include" CACHE PATH "Directory containing httplib.h for TrainingLoop")
option(GRIM_TRAINING_USE_MANUAL_DEPS "Prefer manually staged header-only deps in resources/models/GRIM-text/training/third_party" ON)
option(GRIM_TRAINING_ENABLE_VCPKG_FALLBACK "Allow fallback to vendored vcpkg when manual deps are not staged" ON)
option(GRIM_TRAINING_BUILD_TEST_EXECUTABLES "Build GRIM-text TrainingLoop test executables" OFF)

# Ablation: disable the causal mask (full bidirectional self-attention).
# DEFAULT IS ON (causal mask disabled) for the current collapse investigation.
# This is NOT a header-only toggle: the causal-mask behavior is baked into which
# FlashAttention kernel templates compile. When ON, this drops GRIM_FLASHATTN_CAUSAL_ONLY
# (so the non-causal kernels are compiled) AND defines GRIM_ABLATE_DISABLE_CAUSAL_MASK
# directory-wide so Ablation::kDisableCausalMask is true in lockstep.
# To restore the causal (autoregressive) mask, configure with
# -DGRIM_ABLATE_DISABLE_CAUSAL_MASK=OFF (re-adds CAUSAL_ONLY, flag back to false).
option(GRIM_ABLATE_DISABLE_CAUSAL_MASK "Ablation: build non-causal FlashAttention and disable the causal mask (AblationFlags::kDisableCausalMask)" OFF)
if(GRIM_ABLATE_DISABLE_CAUSAL_MASK)
    add_compile_definitions(GRIM_ABLATE_DISABLE_CAUSAL_MASK)
    message(WARNING "[GRIM] Causal-mask ablation ENABLED: building NON-causal FlashAttention kernels (GRIM_FLASHATTN_CAUSAL_ONLY dropped). Do NOT ship this binary for production training.")
    message(STATUS  "[GRIM] Compiled causal mask = DISABLED (bidirectional self-attention; kDisableCausalMask=true). effective_causal = causal_mask && !kDisableCausalMask resolves to FALSE regardless of the runtime `causal_mask` HP.")
else()
    message(STATUS  "[GRIM] Compiled causal mask = ENABLED (autoregressive; kDisableCausalMask=false, GRIM_FLASHATTN_CAUSAL_ONLY defined).")
endif()

set(GRIM_TRAINING_MANUAL_DEPS_READY OFF)
if(GRIM_TRAINING_USE_MANUAL_DEPS)
    if(EXISTS "${GRIM_TRAINING_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp"
       AND EXISTS "${GRIM_TRAINING_FLATBUFFERS_INCLUDE_DIR}/flatbuffers/flatbuffers.h"
       AND EXISTS "${GRIM_TRAINING_HTTPLIB_INCLUDE_DIR}/httplib.h")
        set(GRIM_TRAINING_MANUAL_DEPS_READY ON)
        message(STATUS "TrainingLoop manual deps: using headers staged under ${GRIM_TRAINING_MANUAL_DEPS_ROOT}")
    elseif(NOT GRIM_TRAINING_ENABLE_VCPKG_FALLBACK)
        message(FATAL_ERROR
            "TrainingLoop manual deps requested, but the staged headers are incomplete.\n"
            "  Expected:\n"
            "    ${GRIM_TRAINING_NLOHMANN_JSON_INCLUDE_DIR}/nlohmann/json.hpp\n"
            "    ${GRIM_TRAINING_FLATBUFFERS_INCLUDE_DIR}/flatbuffers/flatbuffers.h\n"
            "    ${GRIM_TRAINING_HTTPLIB_INCLUDE_DIR}/httplib.h\n"
            "  Run resources/models/GRIM-text/training/scripts/prepare_manual_deps.py or provide the include dirs explicitly.")
    else()
        message(STATUS "TrainingLoop manual deps not fully staged; falling back to vendored vcpkg")
    endif()
endif()

set(GRIM_VCPKG_ROOT "${GRIM_REPO_ROOT}/external/vcpkg")
set(GRIM_VCPKG_TOOLCHAIN "${GRIM_VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
set(GRIM_TRAINING_VCPKG_INSTALLED_DIR "${GRIM_TRAINING_ROOT}/vcpkg_installed")
option(GRIM_TRAINING_ALLOW_REPO_VCPKG_INSTALLED_DIR
    "Allow TrainingLoop to share the repo-root vcpkg_installed directory with the main GRIM manifest"
    OFF)

if(NOT DEFINED VCPKG_TARGET_TRIPLET)
    if(WIN32)
        set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "vcpkg target triplet for GRIM-text training")
    elseif(APPLE)
        set(VCPKG_TARGET_TRIPLET "arm64-osx" CACHE STRING "vcpkg target triplet for GRIM-text training")
    else()
        set(VCPKG_TARGET_TRIPLET "x64-linux" CACHE STRING "vcpkg target triplet for GRIM-text training")
    endif()
endif()

if(NOT GRIM_TRAINING_MANUAL_DEPS_READY AND NOT DEFINED CMAKE_TOOLCHAIN_FILE AND EXISTS "${GRIM_VCPKG_TOOLCHAIN}")
    set(CMAKE_TOOLCHAIN_FILE "${GRIM_VCPKG_TOOLCHAIN}" CACHE FILEPATH "vendored vcpkg toolchain")
    message(STATUS "Using vendored vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
endif()

if(NOT GRIM_TRAINING_MANUAL_DEPS_READY AND NOT DEFINED VCPKG_MANIFEST_DIR AND EXISTS "${GRIM_TRAINING_ROOT}/vcpkg.json")
    set(VCPKG_MANIFEST_DIR "${GRIM_TRAINING_ROOT}" CACHE PATH "GRIM-text training vcpkg manifest")
endif()

if(NOT GRIM_TRAINING_MANUAL_DEPS_READY AND NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${GRIM_TRAINING_VCPKG_INSTALLED_DIR}" CACHE PATH "TrainingLoop vcpkg installed directory")
elseif(NOT GRIM_TRAINING_MANUAL_DEPS_READY AND NOT GRIM_TRAINING_ALLOW_REPO_VCPKG_INSTALLED_DIR AND "${VCPKG_INSTALLED_DIR}" STREQUAL "${GRIM_REPO_ROOT}/vcpkg_installed")
    message(STATUS "TrainingLoop manifest install root moved from repo-root vcpkg_installed to ${GRIM_TRAINING_VCPKG_INSTALLED_DIR}")
    message(STATUS "  Pass -DGRIM_TRAINING_ALLOW_REPO_VCPKG_INSTALLED_DIR=ON only if you intentionally want the TrainingLoop manifest to prune the main GRIM manifest install tree.")
    set(VCPKG_INSTALLED_DIR "${GRIM_TRAINING_VCPKG_INSTALLED_DIR}" CACHE PATH "TrainingLoop vcpkg installed directory" FORCE)
endif()

set(GRIM_VCPKG_TRIPLET_DIR "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}")
if(NOT GRIM_TRAINING_MANUAL_DEPS_READY AND EXISTS "${GRIM_VCPKG_TRIPLET_DIR}")
    list(PREPEND CMAKE_PREFIX_PATH
        "${GRIM_VCPKG_TRIPLET_DIR}"
        "${GRIM_VCPKG_TRIPLET_DIR}/share"
    )
    list(REMOVE_DUPLICATES CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE STRING "Package search path" FORCE)
endif()

if(GRIM_TRAINING_MANUAL_DEPS_READY)
    message(STATUS "TrainingLoop dependency mode: manual headers")
else()
    message(STATUS "TrainingLoop dependency mode: vendored vcpkg")
    message(STATUS "TrainingLoop vcpkg triplet: ${VCPKG_TARGET_TRIPLET}")
    message(STATUS "TrainingLoop manifest: ${VCPKG_MANIFEST_DIR}")
    message(STATUS "TrainingLoop installed dir: ${GRIM_VCPKG_TRIPLET_DIR}")
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

# Enable precompiled headers for faster compilation
set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

function(grim_training_repair_vcpkg_package_dir package_dir cache_var)
    set(_pkg_dir "${GRIM_VCPKG_TRIPLET_DIR}/share/${package_dir}")
    if(EXISTS "${_pkg_dir}")
        set(${cache_var} "${_pkg_dir}" CACHE PATH "${package_dir} package config directory from the TrainingLoop manifest install root" FORCE)
    elseif(DEFINED ${cache_var} AND "${${cache_var}}" MATCHES "-NOTFOUND$")
        unset(${cache_var} CACHE)
    endif()
endfunction()

if(GRIM_TRAINING_MANUAL_DEPS_READY)
    message(STATUS "Using nlohmann-json headers from ${GRIM_TRAINING_NLOHMANN_JSON_INCLUDE_DIR}")
    message(STATUS "Using FlatBuffers headers from ${GRIM_TRAINING_FLATBUFFERS_INCLUDE_DIR}")
    message(STATUS "Using cpp-httplib headers from ${GRIM_TRAINING_HTTPLIB_INCLUDE_DIR}")

    if(NOT TARGET grim_training_nlohmann_json)
        add_library(grim_training_nlohmann_json INTERFACE)
        target_include_directories(grim_training_nlohmann_json INTERFACE "${GRIM_TRAINING_NLOHMANN_JSON_INCLUDE_DIR}")
        target_compile_definitions(grim_training_nlohmann_json INTERFACE HAVE_NLOHMANN_JSON=1)
        add_library(nlohmann_json::nlohmann_json ALIAS grim_training_nlohmann_json)
    endif()

    if(NOT TARGET grim_training_flatbuffers)
        add_library(grim_training_flatbuffers INTERFACE)
        target_include_directories(grim_training_flatbuffers INTERFACE "${GRIM_TRAINING_FLATBUFFERS_INCLUDE_DIR}")
        add_library(flatbuffers::flatbuffers ALIAS grim_training_flatbuffers)
    endif()

    # Header staging cannot provide the schema compiler. Locate a host flatc
    # explicitly so generated checkpoint/config headers remain available in
    # manual-dependency builds as well.
    if(NOT TARGET flatbuffers::flatc)
        find_program(GRIM_TRAINING_FLATC_EXECUTABLE
            NAMES flatc
            HINTS
                "${GRIM_TRAINING_MANUAL_DEPS_ROOT}/flatbuffers/bin"
                "${GRIM_VCPKG_TRIPLET_DIR}/tools/flatbuffers"
            REQUIRED
        )
        add_executable(grim_training_flatc IMPORTED GLOBAL)
        set_target_properties(grim_training_flatc PROPERTIES
            IMPORTED_LOCATION "${GRIM_TRAINING_FLATC_EXECUTABLE}"
        )
        add_executable(flatbuffers::flatc ALIAS grim_training_flatc)
    endif()

    if(NOT TARGET grim_training_httplib)
        add_library(grim_training_httplib INTERFACE)
        target_include_directories(grim_training_httplib INTERFACE "${GRIM_TRAINING_HTTPLIB_INCLUDE_DIR}")
        target_link_libraries(grim_training_httplib INTERFACE
            Threads::Threads
            $<$<PLATFORM_ID:Windows>:ws2_32>
            $<$<PLATFORM_ID:Windows>:crypt32>
        )
        add_library(httplib::httplib ALIAS grim_training_httplib)
    endif()
else()
    find_path(GRIM_TRAINING_HTTPLIB_INCLUDE_DIR
        NAMES httplib.h
        HINTS "${GRIM_VCPKG_TRIPLET_DIR}/include"
        REQUIRED
    )
    message(STATUS "Using cpp-httplib headers from ${GRIM_TRAINING_HTTPLIB_INCLUDE_DIR}")
    if(NOT TARGET grim_training_httplib)
        add_library(grim_training_httplib INTERFACE)
        target_include_directories(grim_training_httplib INTERFACE "${GRIM_TRAINING_HTTPLIB_INCLUDE_DIR}")
        target_link_libraries(grim_training_httplib INTERFACE
            Threads::Threads
            $<$<PLATFORM_ID:Windows>:ws2_32>
            $<$<PLATFORM_ID:Windows>:crypt32>
        )
        add_library(httplib::httplib ALIAS grim_training_httplib)
    endif()

    grim_training_repair_vcpkg_package_dir("nlohmann_json" nlohmann_json_DIR)
    grim_training_repair_vcpkg_package_dir("flatbuffers" flatbuffers_DIR)

    find_package(nlohmann_json CONFIG REQUIRED)
    find_package(flatbuffers CONFIG REQUIRED)
endif()

# CUTLASS (flash-attention) requires CUDA 12+. Fail early with clear instructions.
if(CUDAToolkit_VERSION_MAJOR LESS 12)
  message(FATAL_ERROR
    "CUDAToolkit ${CUDAToolkit_VERSION} detected. CUTLASS (flash-attention) requires CUDA 12+.\n"
    "  Set CUDAToolkit_ROOT (or GRIM_CUDA_ROOT) to a CUDA 12+ toolkit before configuring, e.g.:\n"
    "  cmake -DCUDAToolkit_ROOT=/path/to/cuda-12.0 ..\n"
    "  Or from the run script (Anvil): ensure project CUDA 12 is installed and script will default to it.")
endif()
message(STATUS "CUDAToolkit version: ${CUDAToolkit_VERSION}")

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

# TensorContract sources: glob recursively so GradFn splits in subfolders
# automatically link everywhere (CONFIGURE_DEPENDS triggers reconfigure).
set(GRIM_TENSORCONTRACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Shared/TensorContract")
file(GLOB_RECURSE GRIM_TENSORCONTRACT_CU CONFIGURE_DEPENDS
    "${GRIM_TENSORCONTRACT_DIR}/*.cu"
)

# GPU training kernels library (from parent directory)
add_library(grim_training_kernels STATIC 
    # Rule 26: CrossEntropy_GPU.cu DELETED — dead code. Production loss uses autograd::unified_loss() in AutogradLoss.cu
    ../../Shared/Gradients/GradientCC_GPU.cu
    ../../Shared/GPUBuffer/GPUBuffer.cu
    ../../Shared/GradNorm/GradNormGPU.cu
    ../../Shared/Telemetry/TelemetryLattice_GPU.cu
    ../../Shared/Telemetry/TelemetryControl_GPU.cu
    ../../Shared/Telemetry/TelemetryCsvLogger.cu
    ../../Shared/Telemetry/TelemetryUpdate.cu
    # Rule 20: Xavier.cu DELETED - Tensor::xavier_uniform_ (TensorContract) is the single source of truth
    ../../Shared/PBM/PositionalBiasMethod.cu
    ../../Shared/PBM/PBMStateOwner.cu
    # Rule 20: Old PBM modules DELETED - current PBM path only
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
    ../../Shared/LogRecorder/BatchLogTape.cu
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
    ../../Layers/FlashAttention/AttentionDiagnostics.cu
    ../../Layers/FlashAttention/AttentionBreadthDiagnostic.cu
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
    # Causal-only unless the causal-mask ablation is enabled (then the non-causal
    # kernel templates must be compiled in so a non-causal call is valid).
    $<$<NOT:$<BOOL:${GRIM_ABLATE_DISABLE_CAUSAL_MASK}>>:GRIM_FLASHATTN_CAUSAL_ONLY>
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
    # QKV projector wrappers deleted; autograd attention calls TensorConversion directly.
    # Old Embedding_GPU.cu (dead code) removed. New Embedding_GPU.cu in grim_training_kernels (Pattern B layer).
    ../../Shared/Optimizers/OptimizerUpdate_GPU.cu
    ../../Shared/Optimizers/OptimizerUpdateTrace.cu
    ../../Shared/Optimizers/AdamW/AdamW_Kernal_GPU.cu
    ../../Shared/Optimizers/RAdamW/RAdamW_Kernal_GPU.cu
)
set_target_properties(grim_cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
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
    ../../Common/ModelStats.cu
    ../../Common/ScaleBuffer.cu
    ../Forward_GPU.cu
    ../Phases/Startup/Model/ModelGpuAssembly.cu
    ../../Shared/TrainingState/CublasHandleOwner_GPU.cu
    ../../Shared/TrainingState/DeviceAllocation_GPU.cu
    ../../Shared/TrainingState/TrainingStateGPU.cu
    ../../Shared/Optimizers/OptimizerState_GPU.cu
    ../../Shared/TeacherLogits/TeacherLogits_GPU.cu
    ../Phases/Startup/Model/ParameterGroupRegistration.cu
    ../Phases/Startup/Model/ParameterGroupRegistration.hpp
    ../../Shared/Gradients/GradStatsCollector.cu
    ../../Shared/Forward/ModelForward_GPU.cu
    ../../Shared/Forward/ModelForwardArgBootstrap_GPU.cu
    ../../Shared/Goal/MeanPool_GPU.cu
    ../../Shared/Goal/MeanPool_GPU.hpp
    ../../Shared/InferenceState/KvCacheState_GPU.cu
    ../Autograd/AutogradContext.cu
    ../Autograd/AutogradTraining.cu
    ../../Layers/HardcodedStates/HardcodedStates_GPU.cu
    ../../Layers/Embedding/Embedding_GPU.cu
    ../../Layers/LMHead/lm_head_GPU.cu
    ../../Layers/ArgSelector/ArgSelector_GPU.cu
    ../../Layers/SlotSeedEncoder/SlotSeedEncoder_GPU.cu
    ../../Layers/Quantization/Quantization_GPU.cu
    ../../Shared/Batching/Batching_GPU.cu
    ../../Shared/Batching/BatchDeviceUpload.cu
    ../../Shared/Batching/BatchPayload.cu
    ../../Shared/Batching/EpochBatching.cu
    ../../Layers/Encoding/Encoding_GPU.cu
    ../../Layers/Encoding/EncoderDiagnostics.cu
    ../../Layers/FlashAttention/EncoderSelfAttention_GPU.cu
    ../../Shared/Sampling/Sampling.cu
    ../../Shared/TensorConversion/TensorConversion.cu
    ${GRIM_TENSORCONTRACT_CU}
    ../../Layers/FeedForward/Feed_Forward_GPU.cu
    ../../Shared/Activations/GELU/GELU.cu
    ../../Shared/StreamController/StreamController_GPU.cu
    ../../Shared/Loss/ComputeLoss/AutogradLoss.cu
    ../../Shared/Loss/ComputeLoss/ArgSelectorLoss.cu
    ../../Shared/Loss/ComputeLoss/CrossEntropyNLL.cu
    ../../Shared/Loss/LossSignals/LossSignals.cu
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

# Also available as its own CXX-only CMake project under ConfigCompiler/.
add_subdirectory(
    ../ConfigCompiler
    ${CMAKE_CURRENT_BINARY_DIR}/ConfigCompiler
)

# Registry parameter checkpoint schema and I/O library
set(GRIM_PARAMETER_CHECKPOINT_SCHEMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../schemas/grim_parameter_checkpoint.fbs
)
set(GRIM_PARAMETER_CHECKPOINT_GENERATED_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/grim_parameter_checkpoint_schema
)
set(GRIM_PARAMETER_CHECKPOINT_GENERATED_HEADER
    ${GRIM_PARAMETER_CHECKPOINT_GENERATED_DIR}/grim_parameter_checkpoint_generated.h
)

# BuildFlatBuffers.cmake's flatbuffers_generate_headers() calls source_group()
# with CMAKE_CURRENT_SOURCE_DIR as its tree root, so it cannot accept the
# canonical sibling training/schemas directory. Keep the schema there and own
# the one generated-header command explicitly.
add_custom_command(
    OUTPUT ${GRIM_PARAMETER_CHECKPOINT_GENERATED_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${GRIM_PARAMETER_CHECKPOINT_GENERATED_DIR}
    COMMAND $<TARGET_FILE:flatbuffers::flatc>
            -o ${GRIM_PARAMETER_CHECKPOINT_GENERATED_DIR}
            -c ${GRIM_PARAMETER_CHECKPOINT_SCHEMA}
    DEPENDS
        flatbuffers::flatc
        ${GRIM_PARAMETER_CHECKPOINT_SCHEMA}
    COMMENT "Generating GRIM registry parameter checkpoint FlatBuffer header"
    VERBATIM
)
add_custom_target(generate_grim_parameter_checkpoint_schema
    DEPENDS ${GRIM_PARAMETER_CHECKPOINT_GENERATED_HEADER}
)

add_library(grim_parameter_checkpoint STATIC
    ../../Common/ParameterCheckpoint.cu
    ../../Common/ParameterCheckpoint.hpp
    ${GRIM_PARAMETER_CHECKPOINT_GENERATED_HEADER}
)
add_dependencies(grim_parameter_checkpoint
    generate_grim_parameter_checkpoint_schema
)
set_target_properties(grim_parameter_checkpoint PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "${GRIM_CUDA_ARCHITECTURES}"
)
target_include_directories(grim_parameter_checkpoint PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared             # Shared/ folder for TrainingState, etc.
    ${CMAKE_CURRENT_SOURCE_DIR}/../../archive
    ${GRIM_PARAMETER_CHECKPOINT_GENERATED_DIR}
)
target_link_libraries(grim_parameter_checkpoint PRIVATE
    flatbuffers::flatbuffers
    CUDA::cudart
)
target_compile_options(grim_parameter_checkpoint PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)
target_compile_definitions(grim_parameter_checkpoint PRIVATE USE_CUDA)

# Main GPU training executable (three-phase architecture)
add_executable(train_gpu 
    ../train_gpu.cu
    ../Phases/Phase1_Startup.cu
    ../Phases/Phase1_Startup.hpp
    ../Phases/Startup/Rng.cu
    ../Phases/Startup/Rng.hpp
    ../Phases/Startup/Logging.cu
    ../Phases/Startup/Logging.hpp
    ../Phases/Startup/Capacity/CapacityStem.cu
    ../Phases/Startup/Capacity/CapacityStem.hpp
    ../Phases/Startup/Data/DataInfo.hpp
    ../Phases/Startup/Model/ModelAllocationState.cu
    ../Phases/Startup/Model/ModelAllocationState.hpp
    ../Phases/Startup/CheckpointLoad.cu
    ../Phases/Startup/CheckpointLoad.hpp
    ../Phases/Startup/OptimizerReady.cu
    ../Phases/Startup/OptimizerReady.hpp
    ../Phases/Startup/Epoch/EpochPlan.cu
    ../Phases/Startup/Epoch/EpochPlan.hpp
    ../Phases/Startup/Batching/PlannedBatches.cu
    ../Phases/Startup/Batching/PlannedBatches.hpp
    ../Phases/Startup/Telemetry/TelemetryInitInputs.cu
    ../Phases/Startup/Telemetry/TelemetryInitInputs.hpp
    ../Phases/Startup/Validation/StartupValidation.cu
    ../Phases/Startup/Validation/StartupValidation.hpp
    ../Phases/Startup/Validation/Phase2Handoff.cu
    ../Phases/Startup/Validation/Phase2Handoff.hpp
    ../Phases/Startup/SlidingWindow.cu
    ../Phases/Startup/SlidingWindow.hpp
    ../Phases/Startup/ClassBalancedWeights.cu
    ../Phases/Startup/ClassBalancedWeights.hpp
    ../Phases/Startup/InitFacts.cu
    ../Phases/Startup/InitFacts.hpp
    ../Phases/ConfigDump.cu
    ../Phases/ConfigDump.hpp
    ../Phases/DataStatsDump.cu
    ../Phases/DataStatsDump.hpp
    ../Phases/Phase2_TrainingLoop.cu
    ../Phases/Phase2_TrainingLoop.hpp
    ../Phases/Phase2_InferenceLoop.cu
    ../Phases/Phase2_InferenceLoop.hpp
    ../Diagnostics/DiagnosticInference.cu
    ../Diagnostics/DiagnosticInference.hpp
    ../Diagnostics/RhoDiagnostic.cu
    ../Diagnostics/RhoDiagnostic.hpp
    ../Diagnostics/LMHeadWeightStats.cu
    ../Diagnostics/LMHeadWeightStats.hpp
    ../Diagnostics/LogitScaleDiagnostic.cu
    ../Diagnostics/LogitScaleDiagnostic.hpp
    ../Diagnostics/BoundaryDiagnostic.cu
    ../Diagnostics/BoundaryDiagnostic.hpp
    ../Diagnostics/SpecialTokenDiagnostic.cu
    ../Diagnostics/SpecialTokenDiagnostic.hpp
    ../Diagnostics/AtomStatsDiagnostic.cu
    ../Diagnostics/AtomStatsDiagnostic.hpp
    ../Diagnostics/LossBaselineDiagnostic.cu
    ../Diagnostics/LossBaselineDiagnostic.hpp
    ../Diagnostics/LossStatsDiagnostic.cu
    ../Diagnostics/LossStatsDiagnostic.hpp
    ../Diagnostics/GradientNormDiagnostic.cu
    ../Diagnostics/GradientNormDiagnostic.hpp
    ../Diagnostics/PostClipParamGradEmbLmEquation.cu
    ../Diagnostics/PostClipParamGradEmbLmEquation.hpp
    ../Diagnostics/OptimizerStepGuards.cu
    ../Diagnostics/OptimizerStepGuards.hpp
    ../Diagnostics/TieVerifyDiagnostic.cu
    ../Diagnostics/TieVerifyDiagnostic.hpp
    ../Diagnostics/OptimizerMomentDiagnostic.cu
    ../Diagnostics/OptimizerMomentDiagnostic.hpp
    ../Diagnostics/PostOptimizerWeightTrace.cu
    ../Diagnostics/PostOptimizerWeightTrace.hpp
    ../Diagnostics/PredictionDistributionDiagnostic.cu
    ../Diagnostics/PredictionDistributionDiagnostic.hpp
    ../Diagnostics/DiagnosticGates.cu
    ../Diagnostics/DiagnosticGates.hpp
    ../Diagnostics/HistogramDiagnostic.cu
    ../Diagnostics/HistogramDiagnostic.hpp
    ../Phases/Phase3_Cleanup.cu
    ../Phases/Phase3_Cleanup.hpp
    ../Subprocess/subprocess_status.hpp
    ../Subprocess/subprocess_status_io.hpp
    ../Subprocess/subprocess_status_io.cpp
    ../Subprocess/subprocess_manager.hpp
    ../Subprocess/subprocess_manager.cpp
    ../Subprocess/tokenizer_subprocess.hpp
    ../Subprocess/tokenizer_subprocess.cpp
    ../OptimizerCheckpoint.cu
    ../OptimizerCheckpoint.hpp
    ../training_status_writer.hpp
    ../training_logger.hpp
    ../metrics_collector.hpp
    ../TrainingEvents.hpp
    ../../Shared/DataLoader/DataLoader.cu
    ../../Shared/SoftRestart/SoftRestart.cu
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
    httplib::httplib
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
    CUDA::nvml
    grim_flash_attention
    grim_training_kernels
    grim_cuda_kernels
    grim_language_model_gpu_impl
    grim_parameter_checkpoint
    grim_unigrambyte_tokenizer
    grim_log_recorder
    $<$<PLATFORM_ID:Windows>:ws2_32>
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

# UnigramByte tokenizer library (Unigram LM + TokenLayout byte fallback + Atoms)
add_library(grim_unigrambyte_tokenizer STATIC
    ../../Shared/UnigramByte/Unigram.cu
    ../../Shared/UnigramByte/UnigramViterbi.cu
    ../../Shared/UnigramByte/UnigramGpuMemory.cu
    ../../Shared/UnigramByte/UniByte.cu
    ../../Shared/UnigramByte/AtomTable.cu
    ../../Shared/UnigramByte/AhoCorasick.cu
    ../../Shared/UnigramByte/TextUtils.cu
    ../../Shared/UnigramByte/Detectors/DetectorRegistry.cu
    ../../Shared/UnigramByte/Detectors/NumericDetectors.cu
    ../../Shared/UnigramByte/Detectors/TextFeatureDetectors.cu
    ../../Shared/UnigramByte/Training/SubwordMining.cu
    ../../Shared/UnigramByte/Training/UnigramForwardBackward.cu
    ../../Shared/UnigramByte/UnigramTrainer.cu
    ../../Shared/TokenizerArtifacts/GrmtCorpusIO.cu
    ../../Shared/TokenizerArtifacts/VocabArtifactIO.cu
    ../../Shared/TokenizerArtifacts/TokenizerArtifactBundle.cu
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
    nlohmann_json::nlohmann_json
)
target_compile_options(grim_unigrambyte_tokenizer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
)

# Tokenizer regression test executable
if(GRIM_TRAINING_BUILD_TEST_EXECUTABLES)
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
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(unigrambyte_self_test PRIVATE stdc++fs)
endif()

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
)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_link_libraries(grmt_vocab_metrics_test PRIVATE stdc++fs)
endif()

target_include_directories(grmt_vocab_metrics_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../GRIM
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared
)
endif()

# ============================================================================
# STANDALONE TOKENIZER TRAINING SUBPROCESS
# Spawned by train_gpu via Subprocess/tokenizer_subprocess.cpp.
# Runs full tokenizer pipeline on entire corpus: train vocab + encode GRMT.
# Writes a JSON status file the parent reads.
# Usage: train_tokenizer --status-file <path> [--config <ai_config.json>]
# ============================================================================
add_executable(train_tokenizer
    ../train_tokenizer.cu
    ../Subprocess/subprocess_status_io.cpp
    ../Phases/Startup/SlidingWindow.cu
    ../Phases/ConfigDump.cu
    ../Phases/DataStatsDump.cu
    ../../Shared/DataLoader/DataLoader.cu
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
    grim_log_recorder
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
if(GRIM_TRAINING_BUILD_TEST_EXECUTABLES)
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
    ../../Shared/Activations/GELU/GELU.cu
    ../../Shared/TensorConversion/TensorConversion.cu
    ../../Shared/StreamController/StreamController_GPU.cu
    ${GRIM_TENSORCONTRACT_CU}
    ../../Shared/PBM/PositionalBiasMethod.cu
    ../../Shared/PBM/PBMStateOwner.cu
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

endif()

# ============================================================================
# EMBEDDING AUTOGRAD TEST SUITE
# Autograd-enabled embedding tests using TensorContract system
message(STATUS "TrainingLoop configured:")
message(STATUS "  - train_gpu: GPU-accelerated training with comprehensive logging")
message(STATUS "  - grim_unigrambyte_tokenizer: Unigram LM + byte-layout fallback + AtomTable")
message(STATUS "  - train_tokenizer: standalone tokenizer training subprocess")
if(GRIM_TRAINING_BUILD_TEST_EXECUTABLES)
    message(STATUS "  - tokenizer_self_test: deterministic tokenizer regression suite (BPE)")
    message(STATUS "  - unigrambyte_self_test: UnigramByte tokenizer test suite")
    message(STATUS "  - grmt_vocab_metrics_test: GRMT corpus + vocabulary metrics diagnostics")
    message(STATUS "  - flash_attention_test: Flash Attention forward/backward test suite")
    message(STATUS "  - gradient_verification_test: Encoder gradient collapse debugging")
else()
    message(STATUS "  - TrainingLoop test executables: disabled (set -DGRIM_TRAINING_BUILD_TEST_EXECUTABLES=ON to enable)")
endif()

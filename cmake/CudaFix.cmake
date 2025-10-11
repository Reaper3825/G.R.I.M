# ====================================================================
# GRIM CUDA toolchain compatibility layer
# Fixes MSVC/NVCC flag mismatches for CUDA 12.5+ on VS2022
# ====================================================================

if (CMAKE_CUDA_COMPILER)
    message(STATUS "[GRIM] Applying CUDA 12.x + MSVC compatibility fixes")

    # Use modern C++17
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)


    # Allow new MSVC toolset, silence deprecated warnings
    string(APPEND CMAKE_CUDA_FLAGS " --allow-unsupported-compiler -Wno-deprecated-gpu-targets")


    # Remove problematic /FS flag injected by MSVC generator
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=-FS")


    # Keep CUDA linking static (ggml-cuda requires it)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)


    # Ensure device linking is enabled
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()
# Suppress MSVC D9025 "overriding '/w' with '/W1'" warning
if (MSVC)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/wd9025")
endif()


# Try to detect GPU compute capability if not set manually
if (NOT CMAKE_CUDA_ARCHITECTURES)
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
        OUTPUT_VARIABLE GPU_ARCH_LIST
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (GPU_ARCH_LIST MATCHES "compute_([0-9]+)")
        string(REGEX REPLACE ".*compute_([0-9]+).*" "\\1" DETECTED_ARCH "${GPU_ARCH_LIST}")
        message(STATUS "[GRIM] Detected CUDA architecture: ${DETECTED_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCH} CACHE STRING "Auto-detected CUDA arch" FORCE)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "Default CUDA arch (RTX 3080 Ti)" FORCE)
    endif()
endif()

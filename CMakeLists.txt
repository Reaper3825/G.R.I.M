# =========================================================
# G.R.I.M - Core Build Configuration (Safe Persistent Version)
# =========================================================
cmake_minimum_required(VERSION 3.22)
project(GRIM LANGUAGES CXX CUDA)

# =========================================================
# MSVC Compiler Conformance Fixes
# =========================================================
if (MSVC)
    # Ensure accurate __cplusplus macro and modern preprocessor
    add_compile_options(/Zc:__cplusplus /Zc:preprocessor)
endif()

# =========================================================
# BX Configuration Macro
# =========================================================
# BX expects one of these defines to be present; since we use prebuilt BX/BGFX
# headers, we set it manually based on current configuration.
add_compile_definitions(
    $<$<CONFIG:Debug>:BX_CONFIG_DEBUG=1>
    $<$<NOT:$<CONFIG:Debug>>:BX_CONFIG_DEBUG=0>
)


# =========================================================
# Canonical GRIM Root Path (shared with runtime)
# =========================================================
set(GRIM_ROOT_DIR ${CMAKE_SOURCE_DIR} CACHE PATH "Root of GRIM project")
add_compile_definitions(GRIM_ROOT_DIR="${GRIM_ROOT_DIR}")

# =========================================================
# MSVC Debug Fix → prevent PDB access conflicts in subprojects
# =========================================================
if (MSVC)
    # Apply /FS only for normal C++ files, not CUDA ones
    add_compile_options($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/FS>)
endif()

# =========================================================
# CUDA Environment Sanity
# =========================================================
enable_language(CUDA OPTIONAL)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "[GRIM] Sanitizing CUDA environment...")

    # Clean inherited flags that cause NVCC parse errors
    unset(CMAKE_CUDA_FLAGS CACHE)
    unset(CMAKE_CUDA_FLAGS_DEBUG CACHE)
    unset(CMAKE_CUDA_FLAGS_RELEASE CACHE)

    # Apply clean CUDA defaults
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    # Disable CMAKE_INTDIR duplication
    add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:_DISABLE_CMAKE_INTDIR_DEFS>)
endif()

# =========================================================
# GRIM CUDA Compatibility Layer
# =========================================================
include(${CMAKE_SOURCE_DIR}/cmake/CudaFix.cmake)

# =========================================================
# CUDA Flag Sanitization
# =========================================================
if (CMAKE_CUDA_COMPILER)
    string(REPLACE "-DCMAKE_INTDIR=Release" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    string(REPLACE "-DCMAKE_INTDIR=\"Debug\"" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    string(REPLACE "-D\"CMAKE_INTDIR=\\\"Debug\\\"\"" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
    string(REPLACE "-D\"CMAKE_INTDIR=\\\"Release\\\"\"" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

    foreach(var CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE)
        if(DEFINED ${var})
            string(REPLACE "-DCMAKE_INTDIR=Release" "" ${var} "${${var}}")
            string(REPLACE "-DCMAKE_INTDIR=\"Debug\"" "" ${var} "${${var}}")
            string(REPLACE "-D\"CMAKE_INTDIR=\\\"Debug\\\"\"" "" ${var} "${${var}}")
            string(REPLACE "-D\"CMAKE_INTDIR=\\\"Release\\\"\"" "" ${var} "${${var}}")
        endif()
    endforeach()
endif()

# =========================================================
# Core Configuration & Sources
# =========================================================
include(cmake/Options.cmake)
include(cmake/Config.cmake)
include(cmake/Sources.cmake)

# =========================================================
# Core Configuration & Sources
# =========================================================
include(cmake/Options.cmake)
include(cmake/Config.cmake)
include(cmake/Sources.cmake)

# =========================================================
# Main GRIM Executable
# =========================================================
add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})

# Precompiled headers and includes
target_precompile_headers(GRIM PRIVATE pch.hpp)
target_include_directories(GRIM PRIVATE ${CMAKE_SOURCE_DIR})

# Apply linker flags for MSVC
if (MSVC)
    target_link_options(GRIM PRIVATE /SUBSYSTEM:CONSOLE)
endif()

# =========================================================
# Dependencies & Resources (must come AFTER add_executable)
# =========================================================
include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Resources.cmake)

# Link all dependencies safely


# =========================================================
# Final Target Properties
# =========================================================
set_target_properties(GRIM PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

message(STATUS "[GRIM] Build configuration complete.")

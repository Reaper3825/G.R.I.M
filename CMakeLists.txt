# =========================================================
# G.R.I.M - Core Build Configuration (Unified Release)
# =========================================================
cmake_minimum_required(VERSION 3.22)

set(VCPKG_FEATURE_FLAGS "manifests,versions" CACHE STRING "")
set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed")
include_directories("${VCPKG_INSTALLED_DIR}/x64-windows/include")
link_directories("${VCPKG_INSTALLED_DIR}/x64-windows/lib")

project(GRIM LANGUAGES CXX CUDA)

if (MSVC)
    add_compile_options(/Zc:__cplusplus /Zc:preprocessor)
    add_compile_options($<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/FS>)
endif()

add_compile_definitions(
    $<$<CONFIG:Debug>:BX_CONFIG_DEBUG=1>
    $<$<NOT:$<CONFIG:Debug>>:BX_CONFIG_DEBUG=0>
)

set(GRIM_ROOT_DIR ${CMAKE_SOURCE_DIR} CACHE PATH "Root of GRIM project")
add_compile_definitions(GRIM_ROOT_DIR="${GRIM_ROOT_DIR}")

enable_language(CUDA OPTIONAL)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "[GRIM] Sanitizing CUDA environment...")

    unset(CMAKE_CUDA_FLAGS CACHE)
    unset(CMAKE_CUDA_FLAGS_DEBUG CACHE)
    unset(CMAKE_CUDA_FLAGS_RELEASE CACHE)

    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    add_compile_definitions($<$<COMPILE_LANGUAGE:CUDA>:_DISABLE_CMAKE_INTDIR_DEFS>)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/CudaFix.cmake)

if (CMAKE_CUDA_COMPILER)
    foreach(var CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_DEBUG CMAKE_CUDA_FLAGS_RELEASE)
        if(DEFINED ${var})
            string(REPLACE "-DCMAKE_INTDIR=Release" "" ${var} "${${var}}")
            string(REPLACE "-DCMAKE_INTDIR=\"Debug\"" "" ${var} "${${var}}")
            string(REPLACE "-D\"CMAKE_INTDIR=\\\"Debug\\\"\"" "" ${var} "${${var}}")
            string(REPLACE "-D\"CMAKE_INTDIR=\\\"Release\\\"\"" "" ${var} "${${var}}")
        endif()
    endforeach()
endif()

include(cmake/Options.cmake)

include(cmake/Sources.cmake)

add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})
# Allow GRIM.exe to export symbols for possible plugin use
set_target_properties(GRIM PROPERTIES ENABLE_EXPORTS ON)

# Compile the core plugin directly into the host
target_sources(GRIM PRIVATE ${CMAKE_SOURCE_DIR}/plugins/core_plugin.cpp)
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/plugins/core_plugin.cpp
    PROPERTIES COMPILE_DEFINITIONS "GRIM_BUILD_HOST"
)

target_compile_definitions(GRIM PRIVATE GRIM_BUILD_HOST)
target_precompile_headers(GRIM PRIVATE pch.hpp)

include(cmake/Config.cmake)

target_include_directories(GRIM PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/ai
    ${CMAKE_SOURCE_DIR}/commands
    ${CMAKE_SOURCE_DIR}/voice
    ${CMAKE_SOURCE_DIR}/wake
    ${CMAKE_SOURCE_DIR}/memory
    ${CMAKE_SOURCE_DIR}/nlp
    ${CMAKE_SOURCE_DIR}/ui
    ${CMAKE_SOURCE_DIR}/helpers
    ${CMAKE_SOURCE_DIR}/bootstrap
    ${CMAKE_SOURCE_DIR}/device_setups
    ${CMAKE_SOURCE_DIR}/popup_ui
    ${CMAKE_SOURCE_DIR}/deps/include
)

if (MSVC)
    target_link_options(GRIM PRIVATE /SUBSYSTEM:CONSOLE)
endif()

include(${CMAKE_SOURCE_DIR}/cmake/Dependencies.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/Resources.cmake)

set_target_properties(GRIM PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
)


# =========================================================
# Expose GRIM target to plugins
# =========================================================
add_library(GRIM_EXPORT INTERFACE)
target_link_libraries(GRIM_EXPORT INTERFACE GRIM)

target_include_directories(GRIM PRIVATE
    ${CMAKE_SOURCE_DIR}/external/porcupine/include
)
target_link_directories(GRIM PRIVATE
    ${CMAKE_SOURCE_DIR}/external/porcupine/lib/windows/amd64
)
target_link_libraries(GRIM PRIVATE pv_porcupine)

# =========================================================
# DLL Copy (Per-Configuration Safe)
# =========================================================
set(_runtime_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")

if (NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/external/vcpkg/installed")
endif()

set(_dll_dir_whisper "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/bin")
set(_dll_dir_vcpkg_release "${VCPKG_INSTALLED_DIR}/x64-windows/bin")
set(_dll_dir_vcpkg_debug   "${VCPKG_INSTALLED_DIR}/x64-windows/debug/bin")

# =========================================================
# DLL Copy (Per-Configuration Safe)
# =========================================================
set(_runtime_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")

if (NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/external/vcpkg/installed")
endif()

set(_dll_dir_whisper "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/bin")
set(_dll_dir_vcpkg_release "${VCPKG_INSTALLED_DIR}/x64-windows/bin")
set(_dll_dir_vcpkg_debug   "${VCPKG_INSTALLED_DIR}/x64-windows/debug/bin")

# =========================================================
# DLL Copy (Per-Configuration Safe)
# =========================================================
set(_runtime_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")

if (NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/external/vcpkg/installed")
endif()

set(_dll_dir_whisper "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/bin")
set(_dll_dir_vcpkg_release "${VCPKG_INSTALLED_DIR}/x64-windows/bin")
set(_dll_dir_vcpkg_debug   "${VCPKG_INSTALLED_DIR}/x64-windows/debug/bin")

add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "[GRIM] Copying runtime DLLs..."

    # Whisper / GGML (Debug + Release)
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${_dll_dir_whisper}/Debug"   "${_runtime_dir}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${_dll_dir_whisper}/Release" "${_runtime_dir}"

    # Vcpkg DLLs (Config-specific)
    COMMAND ${CMAKE_COMMAND} -E echo "[GRIM] Copying $<CONFIG> vcpkg DLLs..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        "$<IF:$<CONFIG:Debug>,${_dll_dir_vcpkg_debug},${_dll_dir_vcpkg_release}>"
        "${_runtime_dir}"

    # Porcupine
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/external/porcupine/lib/windows/amd64/libpv_porcupine.dll"
        "${_runtime_dir}/libpv_porcupine.dll"

    COMMAND ${CMAKE_COMMAND} -E echo "[GRIM] DLL copy complete."
)

# =========================================================
#                   Whisper Linking
# =========================================================
set(WHISPER_LIB_DEBUG   "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src/Debug")
set(WHISPER_LIB_RELEASE "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src/Release")

target_link_directories(GRIM PRIVATE
    $<$<CONFIG:Debug>:${WHISPER_LIB_DEBUG}>
    $<$<CONFIG:Release>:${WHISPER_LIB_RELEASE}>
)


# =========================================================
# BX / BIMG / BGFX Linking (Proper order + per-config paths)
# =========================================================
set(DEPS_LIB_DEBUG   "${CMAKE_SOURCE_DIR}/deps/lib/Debug")
set(DEPS_LIB_RELEASE "${CMAKE_SOURCE_DIR}/deps/lib/Release")

target_link_directories(GRIM PRIVATE
    $<$<CONFIG:Debug>:${DEPS_LIB_DEBUG}>
    $<$<CONFIG:Release>:${DEPS_LIB_RELEASE}>
)
# order: bx → bimg → bgfx
target_link_libraries(GRIM PRIVATE bx bimg bgfx)
target_link_libraries(GRIM PRIVATE whisper)


# =========================================================
# Force using release OpenAL for all configurations
# =========================================================
set(_openal_release_lib "${VCPKG_INSTALLED_DIR}/x64-windows/lib/OpenAL32.lib")
set(_openal_release_dll "${VCPKG_INSTALLED_DIR}/x64-windows/bin/OpenAL32.dll")

target_link_libraries(GRIM PRIVATE "${_openal_release_lib}")

add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${_openal_release_dll}" "${CMAKE_BINARY_DIR}/$<CONFIG>/OpenAL32.dll"
)


message(STATUS "[GRIM] Build configuration complete.")

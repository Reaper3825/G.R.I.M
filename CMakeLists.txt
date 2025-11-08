# =========================================================
# G.R.I.M - Core Build Configuration (Unified Release)
# =========================================================
cmake_minimum_required(VERSION 3.22)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(VCPKG_FEATURE_FLAGS "manifests,versions" CACHE STRING "")
set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed")
include_directories("${VCPKG_INSTALLED_DIR}/x64-windows/include")
include_directories("${VCPKG_INSTALLED_DIR}/x64-windows/include/opencv4")

# Add library search paths for both Debug and Release
link_directories("${VCPKG_INSTALLED_DIR}/x64-windows/lib")
link_directories("${VCPKG_INSTALLED_DIR}/x64-windows/debug/lib")

project(GRIM LANGUAGES CXX C)

# Enable CUDA support for libraries that use GPU (ONNX Runtime, Whisper, etc.)
# Note: We don't compile .cu files in main GRIM build anymore (GRIM-text is separate)
enable_language(CUDA OPTIONAL)

if (CMAKE_CUDA_COMPILER)
    message(STATUS "[GRIM] CUDA detected - GPU acceleration available for ONNX Runtime, Whisper")
    set(CMAKE_CUDA_ARCHITECTURES 86)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

if (MSVC)
    add_compile_options(/Zc:__cplusplus)
    add_compile_options(/Zc:preprocessor)
    add_compile_options($<$<CONFIG:Debug>:/FS>)
endif()

add_compile_definitions(
    NOMINMAX
    WIN32_LEAN_AND_MEAN
    $<$<CONFIG:Debug>:BX_CONFIG_DEBUG=1>
    $<$<NOT:$<CONFIG:Debug>>:BX_CONFIG_DEBUG=0>
)

set(GRIM_ROOT_DIR ${CMAKE_SOURCE_DIR} CACHE PATH "Root of GRIM project")
add_compile_definitions(GRIM_ROOT_DIR="${GRIM_ROOT_DIR}")

# =========================================================
# WORKAROUND: vcpkg's onnxruntime-gpu has a corrupted dependency that adds '\openal32.lib'
# This filters it out from link commands
# =========================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC" OR CMAKE_CXX_SIMULATE_ID MATCHES "MSVC")
    # Remove the malformed \openal32.lib from link line (vcpkg onnxruntime bug)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /IGNORE:4078")
    
    # Hook into link command to filter out the broken library
    set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")
    string(REPLACE "\\openal32.lib" "" CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")
    string(REPLACE "/openal32.lib" "" CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_LINK_EXECUTABLE}")
endif()

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

# =========================================================
# Copy STB headers from bimg 3rdparty to deps/include
# =========================================================
file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/deps/include/stb")
file(GLOB STB_HEADERS "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/bimg/3rdparty/stb/*.h")
foreach(STB_HEADER ${STB_HEADERS})
    get_filename_component(HEADER_NAME ${STB_HEADER} NAME)
    configure_file(
        ${STB_HEADER}
        "${CMAKE_SOURCE_DIR}/deps/include/stb/${HEADER_NAME}"
        COPYONLY
    )
endforeach()

add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})
# Allow GRIM.exe to export symbols for possible plugin use
set_target_properties(GRIM PROPERTIES ENABLE_EXPORTS ON)

# ✅ Compile the core plugin directly into the host
target_sources(GRIM PRIVATE ${CMAKE_SOURCE_DIR}/plugins/core_plugin.cpp)
set_source_files_properties(
    ${CMAKE_SOURCE_DIR}/plugins/core_plugin.cpp
    PROPERTIES COMPILE_DEFINITIONS "GRIM_BUILD_HOST"
)

target_compile_definitions(GRIM PRIVATE GRIM_BUILD_HOST)

include(cmake/Config.cmake)

# Ensure ONNX Runtime includes are available if perception is enabled
if(GRIM_USE_PERCEPTION)
    target_include_directories(GRIM PRIVATE "${VCPKG_INSTALLED_DIR}/x64-windows/include")
endif()

# Perception libraries are now linked in Config.cmake
# (removed hardcoded linking here to avoid conflicts)

target_include_directories(GRIM PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/ai
    ${CMAKE_SOURCE_DIR}/commands
    ${CMAKE_SOURCE_DIR}/voice
    ${CMAKE_SOURCE_DIR}/wake
    ${CMAKE_SOURCE_DIR}/memory
    ${CMAKE_SOURCE_DIR}/nlp
    ${CMAKE_SOURCE_DIR}/control
    ${CMAKE_SOURCE_DIR}/ui
    ${CMAKE_SOURCE_DIR}/helpers
    ${CMAKE_SOURCE_DIR}/bootstrap
    ${CMAKE_SOURCE_DIR}/device_setups
    ${CMAKE_SOURCE_DIR}/popup_ui
    ${CMAKE_SOURCE_DIR}/external_collector
    ${CMAKE_SOURCE_DIR}/perception
    ${CMAKE_SOURCE_DIR}/vision
    ${CMAKE_SOURCE_DIR}/deps/include
)

# Configure PCH after include directories are set
target_precompile_headers(GRIM PRIVATE ${CMAKE_SOURCE_DIR}/pch.hpp)

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
# Whisper Linking
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


message(STATUS "[GRIM] Build configuration complete.")

# Hot-Reloadable Plugins
# Note: core_plugin is built directly into GRIM.exe, not as a hot-reloadable DLL
# Only independent plugins with minimal dependencies should be hot-reloadable
add_subdirectory(plugins/osint_plugin)
add_subdirectory(plugins/network_test_plugin)



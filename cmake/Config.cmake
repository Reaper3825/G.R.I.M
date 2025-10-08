# =========================================================
# GRIM Compiler & Build Configuration
# =========================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Debug or Release" FORCE)
endif()

# =========================================================
# Compiler warnings
# =========================================================
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>
                      $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()

# =========================================================
# Speed up rebuilds with ccache
# =========================================================
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# =========================================================
# GPU acceleration setup
# =========================================================
if(GRIM_USE_CUDA)
    add_definitions(-DWHISPER_USE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "Enable CUDA in ggml")

    # Force architecture to sm_86 only
    set(CMAKE_CUDA_ARCHITECTURES 86)
    add_definitions(-DGGML_CUDA_ARCH=86)

    if(MSVC)
        add_compile_definitions(CMAKE_INTDIR=Release)
    endif()
endif()

if(GRIM_USE_METAL)
    add_definitions(-DWHISPER_USE_METAL)
    set(GGML_METAL ON CACHE BOOL "Enable Metal in ggml")
endif()

if(GRIM_USE_VULKAN)
    add_definitions(-DWHISPER_USE_VULKAN)
    set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan in ggml")
endif()

if(GRIM_USE_OPENCL)
    add_definitions(-DWHISPER_USE_OPENCL)
    set(GGML_OPENCL ON CACHE BOOL "Enable OpenCL in ggml")
endif()

# =========================================================
# UI Font Path Option
# =========================================================
set(GRIM_FONT_PATH "" CACHE STRING "Path to TTF font file for UI")
if(NOT GRIM_FONT_PATH STREQUAL "")
    add_definitions(-DGRIM_FONT_PATH=\"${GRIM_FONT_PATH}\")
endif()

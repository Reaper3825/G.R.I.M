# =========================================================
# GRIM CUDA pre-configuration fix
# =========================================================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Force Ampere / RTX 3000 GPU architecture to avoid CMake's invalid sm_52 default
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures" FORCE)
endif()

# Prevent nvcc debug build issues during compiler ID detection
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler -Xcompiler=/w")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS_INIT}" CACHE STRING "Initial CUDA flags" FORCE)
set(GRIM_ROOT_DIR ${CMAKE_SOURCE_DIR} CACHE PATH "Root of GRIM project")
add_compile_definitions(GRIM_ROOT_DIR="${GRIM_ROOT_DIR}")

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
# =========================================================
# Perception/Vision AI Models (Optional)
# =========================================================
if(GRIM_USE_PERCEPTION)
    message(STATUS "[GRIM] Perception models enabled - finding OpenCV and Tesseract")
    
    find_package(OpenCV QUIET COMPONENTS core imgproc dnn)
    find_package(Tesseract CONFIG QUIET)
    
    if(OpenCV_FOUND AND Tesseract_FOUND)
        message(STATUS "[GRIM] ✓ OpenCV ${OpenCV_VERSION} found")
        message(STATUS "[GRIM] ✓ Tesseract found")
        add_compile_definitions(GRIM_HAS_PERCEPTION)
        set(GRIM_PERCEPTION_AVAILABLE TRUE)
    else()
        if(NOT OpenCV_FOUND)
            message(WARNING "[GRIM] ✗ OpenCV not found - perception models disabled")
        endif()
        if(NOT Tesseract_FOUND)
            message(WARNING "[GRIM] ✗ Tesseract not found - OCR model disabled")
        endif()
        message(STATUS "[GRIM] Install with: vcpkg install opencv[dnn,contrib] tesseract leptonica")
        set(GRIM_PERCEPTION_LIBS "")
    endif()
else()
    message(STATUS "[GRIM] Perception models disabled (set GRIM_USE_PERCEPTION=ON to enable)")
    set(GRIM_PERCEPTION_LIBS "")
endif()

# =========================================================
# Link Dependencies (static/import libraries)
# =========================================================
link_directories(
    "${DEPS_LIB_DIR}"
    "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src/${CMAKE_CFG_INTDIR}"
    "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bgfx/${CMAKE_CFG_INTDIR}"
    "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bimg/${CMAKE_CFG_INTDIR}"
    "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bx/${CMAKE_CFG_INTDIR}"
)

target_link_libraries(GRIM PRIVATE


    # Audio / Speech
    portaudio
    pv_porcupine
    whisper

    # HTTP / Networking
    cpr

    # Graphics backend
    bgfx bimg bx

    ${GRIM_PERCEPTION_LIBS}
)



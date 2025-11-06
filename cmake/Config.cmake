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
# GRIM_ROOT_DIR is already defined in root CMakeLists.txt - don't redefine it here

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
        # Do NOT manually add CMAKE_INTDIR here.
        # The MSVC generator already injects a CMAKE_INTDIR define for per-configuration
        # compilation. Manually adding it here caused duplicate -DCMAKE_INTDIR=...
        # definitions when other parts of the pipeline (or the Visual Studio
        # generator) also add the define. If a specific target truly requires a
        # hard-coded value, use target_compile_definitions(...) on that target
        # only, guarded by an appropriate condition.
        # add_compile_definitions(CMAKE_INTDIR=Release)  # intentionally disabled
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
    
    find_package(OpenCV QUIET COMPONENTS core imgproc dnn imgcodecs photo)
    find_package(Tesseract CONFIG QUIET)
    find_package(Leptonica CONFIG QUIET)
    
    set(GRIM_PERCEPTION_LIBS "")
    
    if(OpenCV_FOUND)
        message(STATUS "[GRIM] ✓ OpenCV ${OpenCV_VERSION} found")
        list(APPEND GRIM_PERCEPTION_LIBS ${OpenCV_LIBS})
        target_include_directories(GRIM PRIVATE ${OpenCV_INCLUDE_DIRS})
        add_compile_definitions(GRIM_HAS_OPENCV)
    else()
        message(WARNING "[GRIM] ✗ OpenCV not found - vision models disabled")
    endif()
    
    if(Tesseract_FOUND)
        message(STATUS "[GRIM] ✓ Tesseract found")
        list(APPEND GRIM_PERCEPTION_LIBS Tesseract::libtesseract)
        add_compile_definitions(GRIM_HAS_TESSERACT)
    
        if(Leptonica_FOUND)
            message(STATUS "[GRIM] ✓ Leptonica found")
            # Leptonica is already linked via Tesseract::libtesseract dependency
        endif()
    else()
        message(WARNING "[GRIM] ✗ Tesseract not found - OCR model disabled")
    endif()
    
    # ONNX Runtime - use IMPORTED target to avoid transitive dependency issues
    set(ONNX_LIB_PATH "${VCPKG_INSTALLED_DIR}/x64-windows/lib/onnxruntime.lib")
    if(EXISTS "${ONNX_LIB_PATH}")
        message(STATUS "[GRIM] ✓ ONNX Runtime found")
        
        # Create IMPORTED target to prevent CMake from resolving broken transitive dependencies
        if(NOT TARGET onnxruntime_imported)
            add_library(onnxruntime_imported STATIC IMPORTED GLOBAL)
            set_target_properties(onnxruntime_imported PROPERTIES
                IMPORTED_LOCATION "${ONNX_LIB_PATH}"
                INTERFACE_INCLUDE_DIRECTORIES "${VCPKG_INSTALLED_DIR}/x64-windows/include"
                IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            )
        endif()
        
        list(APPEND GRIM_PERCEPTION_LIBS onnxruntime_imported)
        target_include_directories(GRIM PRIVATE "${VCPKG_INSTALLED_DIR}/x64-windows/include")
        add_compile_definitions(GRIM_HAS_ONNXRUNTIME)
    else()
        message(WARNING "[GRIM] ✗ ONNX Runtime not found at ${ONNX_LIB_PATH}")
    endif()
  
    if(OpenCV_FOUND AND Tesseract_FOUND)
        add_compile_definitions(GRIM_HAS_PERCEPTION)
        set(GRIM_PERCEPTION_AVAILABLE TRUE)
        message(STATUS "[GRIM] ✓ Perception models fully available")
    else()
        message(STATUS "[GRIM] Install missing packages with: vcpkg install opencv[dnn,contrib] tesseract leptonica onnxruntime-gpu")
    endif()
else()
    message(STATUS "[GRIM] Perception models disabled (set GRIM_USE_PERCEPTION=ON to enable)")
    set(GRIM_PERCEPTION_LIBS "")
endif()

# =========================================================
# Link Dependencies (static/import libraries)
# =========================================================
# Handle both multi-config (VS) and single-config (Ninja) generators
if(CMAKE_CONFIGURATION_TYPES)
    # Multi-config generator (Visual Studio)
    set(_build_config_dir "${CMAKE_CFG_INTDIR}")
else()
    # Single-config generator (Ninja, Make)
    set(_build_config_dir "${CMAKE_BUILD_TYPE}")
endif()

link_directories(
    "${DEPS_LIB_DIR}"
    "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src/${_build_config_dir}"
    "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bgfx/${_build_config_dir}"
    "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bimg/${_build_config_dir}"
    "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bx/${_build_config_dir}"
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

# =========================================================
# OpenAL - Force release version for all configurations
# =========================================================
# Create IMPORTED target to avoid CUDA device link issues with raw library paths
set(_openal_release_lib "${VCPKG_INSTALLED_DIR}/x64-windows/lib/OpenAL32.lib")
if(NOT TARGET OpenAL32_IMPORTED)
    add_library(OpenAL32_IMPORTED STATIC IMPORTED GLOBAL)
    set_target_properties(OpenAL32_IMPORTED PROPERTIES
        IMPORTED_LOCATION "${_openal_release_lib}"
    )
endif()
target_link_libraries(GRIM PRIVATE OpenAL32_IMPORTED)



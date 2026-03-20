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
# GPU acceleration setup (CUDA not available on macOS)
# =========================================================
if(GRIM_USE_CUDA AND NOT APPLE)
    add_definitions(-DWHISPER_USE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "Enable CUDA in ggml")

    # Force architecture to sm_86 only
    set(CMAKE_CUDA_ARCHITECTURES 86)
    add_definitions(-DGGML_CUDA_ARCH=86)
    
    # Find and link CUDA runtime
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "[GRIM] CUDA Toolkit found - linking cudart")
        target_link_libraries(GRIM PRIVATE CUDA::cudart)
    else()
        message(WARNING "[GRIM] CUDA Toolkit not found - CUDA detection may fail at runtime")
    endif()

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
    if(WIN32)
        set(ONNX_LIB_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/onnxruntime.lib")
    else()
        set(ONNX_LIB_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libonnxruntime.a")
    endif()
    # On Mac/Linux, vcpkg splits onnxruntime into component libraries
    set(_ort_session "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libonnxruntime_session.a")
    if(EXISTS "${ONNX_LIB_PATH}")
        message(STATUS "[GRIM] ✓ ONNX Runtime found (single lib)")
        
        if(NOT TARGET onnxruntime_imported)
            add_library(onnxruntime_imported STATIC IMPORTED GLOBAL)
            set_target_properties(onnxruntime_imported PROPERTIES
                IMPORTED_LOCATION "${ONNX_LIB_PATH}"
                INTERFACE_INCLUDE_DIRECTORIES "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include"
                IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
            )
        endif()
        
        list(APPEND GRIM_PERCEPTION_LIBS onnxruntime_imported)
        target_include_directories(GRIM PRIVATE "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include")
        add_compile_definitions(GRIM_HAS_ONNXRUNTIME)
    elseif(EXISTS "${_ort_session}")
        message(STATUS "[GRIM] ✓ ONNX Runtime found (component libs)")
        set(_ort_libs
            onnxruntime_session
            onnxruntime_providers
            onnxruntime_optimizer
            onnxruntime_framework
            onnxruntime_graph
            onnxruntime_mlas
            onnxruntime_common
            onnxruntime_util
            onnxruntime_lora
            onnxruntime_flatbuffers
            onnx
            onnx_proto
            re2
            cpuinfo
            protobuf
        )
        foreach(_ort_lib ${_ort_libs})
            set(_ort_lib_path "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/lib${_ort_lib}.a")
            if(EXISTS "${_ort_lib_path}")
                target_link_libraries(GRIM PRIVATE ${_ort_lib})
            endif()
        endforeach()
        target_include_directories(GRIM PRIVATE
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include"
            "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include/onnxruntime"
        )
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
if(CMAKE_CONFIGURATION_TYPES)
    # Multi-config generator (Visual Studio) — $<CONFIG> resolves at build time
    target_link_directories(GRIM PRIVATE
        "${CMAKE_SOURCE_DIR}/deps/lib/$<CONFIG>"
        "${CMAKE_SOURCE_DIR}/deps/lib"
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src/$<CONFIG>"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bgfx/$<CONFIG>"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bimg/$<CONFIG>"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bx/$<CONFIG>"
    )
    if(DEFINED DEPS_LIB_DIR AND DEPS_LIB_DIR)
        target_link_directories(GRIM PRIVATE "${DEPS_LIB_DIR}")
    endif()
    set(_flat_link_dirs
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src"
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/ggml/src"
    )
else()
    # Single-config generator (Ninja, Make) — resolve paths at configure time
    set(_flat_link_dirs
        "${CMAKE_SOURCE_DIR}/deps/lib/${CMAKE_BUILD_TYPE}"
        "${CMAKE_SOURCE_DIR}/deps/lib"
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src/${CMAKE_BUILD_TYPE}"
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/src"
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/ggml/src"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bgfx/${CMAKE_BUILD_TYPE}"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bimg/${CMAKE_BUILD_TYPE}"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/bx/${CMAKE_BUILD_TYPE}"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/cmake/bgfx"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/cmake/bimg"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/cmake/bx"
    )
    if(DEFINED DEPS_LIB_DIR AND DEPS_LIB_DIR)
        list(INSERT _flat_link_dirs 0 "${DEPS_LIB_DIR}")
    endif()
endif()
if(APPLE)
    list(APPEND _flat_link_dirs
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/ggml/src/ggml-metal"
        "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/ggml/src/ggml-blas"
    )
endif()
foreach(_dir ${_flat_link_dirs})
    if(EXISTS "${_dir}")
        target_link_directories(GRIM PRIVATE "${_dir}")
    endif()
endforeach()

target_link_libraries(GRIM PRIVATE


    # Audio / Speech
    portaudio
    whisper
    ggml ggml-base ggml-cpu

    # HTTP / Networking
    cpr

    # Graphics backend
    bgfx bimg bx

    ${GRIM_PERCEPTION_LIBS}
)

if(APPLE)
    target_link_libraries(GRIM PRIVATE ggml-metal ggml-blas)
    target_link_libraries(GRIM PRIVATE "-framework Accelerate" "-framework Foundation")
endif()

# =========================================================
# OpenAL - Force release version for all configurations
# =========================================================
if(WIN32)
    set(_openal_release_lib "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/OpenAL32.lib")
else()
    set(_openal_release_lib "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/libopenal.a")
endif()
if(EXISTS "${_openal_release_lib}")
    if(NOT TARGET OpenAL32_IMPORTED)
        add_library(OpenAL32_IMPORTED STATIC IMPORTED GLOBAL)
        set_target_properties(OpenAL32_IMPORTED PROPERTIES
            IMPORTED_LOCATION "${_openal_release_lib}"
        )
    endif()
    target_link_libraries(GRIM PRIVATE OpenAL32_IMPORTED)
elseif(APPLE)
    target_link_libraries(GRIM PRIVATE OpenAL)
endif()



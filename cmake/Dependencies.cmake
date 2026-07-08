# =========================================================
# Dependencies.cmake — GRIM Runtime Library Management
# =========================================================
# This file ensures all required runtime DLLs are copied into
# the active build configuration directory after compilation.
# Compatible with MSBuild, Ninja, and CLI cmake --build.

# =========================================================
# Base runtime and dependency paths
# =========================================================
# Handle both multi-config (VS) and single-config (Ninja) generators
if(CMAKE_CONFIGURATION_TYPES)
    # Multi-config generator (Visual Studio)
    set(_runtime_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")
else()
    # Single-config generator (Ninja, Make)
    set(_runtime_dir "${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}")
endif()

if (NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/vcpkg_installed")
endif()

set(_dll_dir_vcpkg "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
set(_dll_dir_whisper "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/bin")
set(_dll_dir_porcupine "${CMAKE_SOURCE_DIR}/external/porcupine/lib/windows/amd64")

# =========================================================
# Helper macro for DLL copies (Windows only)
# =========================================================
if(WIN32)
function(grim_copy_dlls SRC_DIR)
    foreach(_dll IN LISTS ARGN)
        if (EXISTS "${SRC_DIR}/${_dll}")
            add_custom_command(TARGET GRIM POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${SRC_DIR}/${_dll}"
                    "${_runtime_dir}/${_dll}"
                COMMENT "[GRIM] Copied ${_dll} → ${_runtime_dir}"
            )
        endif()
    endforeach()
endfunction()
else()
function(grim_copy_dlls SRC_DIR)
    # No-op on non-Windows (dylib/so are linked, not copied)
endfunction()
endif()

# =========================================================
# Whisper / GGML DLLs
# =========================================================
foreach(_cfg IN ITEMS Debug Release)
    grim_copy_dlls("${_dll_dir_whisper}/${_cfg}"
        ggml.dll
        ggml-base.dll
        ggml-cpu.dll
        ggml-cuda.dll
        whisper.dll
    )
endforeach()

# =========================================================
# Porcupine DLL
# =========================================================
grim_copy_dlls("${_dll_dir_porcupine}"
    libpv_porcupine.dll
)


# =========================================================
# PortAudio and CPR
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    portaudio.dll
    cpr.dll
)

# =========================================================
# Audio codec libraries
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    vorbis.dll
    vorbisfile.dll
    vorbisenc.dll
    FLAC.dll
    ogg.dll
)

# =========================================================
# FFmpeg media stack
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    avcodec-60.dll
    avdevice-60.dll
    avfilter-9.dll
    avformat-60.dll
    avutil-58.dll
    swresample-4.dll
    swscale-7.dll
)

# =========================================================
# SentencePiece runtime (tokenizer)
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    sentencepiece.dll
    sentencepiece_train.dll
    abseil_dll.dll
)

# =========================================================
# Graphics dependencies
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    freetype.dll
    libpng16.dll
    zlib1.dll
    libbz2.dll
    jpeg62.dll
    zip.dll
)

# =========================================================
# OpenAL (Audio backend)
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    OpenAL32.dll
)
# =========================================================
# Perception/Vision DLLs (OpenCV, Tesseract, ONNX) - Optional
# =========================================================
if(GRIM_USE_PERCEPTION)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio video dnn photo calib3d objdetect)
    find_package(Tesseract CONFIG QUIET)
    find_package(onnxruntime CONFIG QUIET)
    
    if(OpenCV_FOUND)
        message(STATUS "[GRIM] Copying OpenCV DLLs for perception models")
        grim_copy_dlls("${_dll_dir_vcpkg}"
            opencv_core4.dll
            opencv_imgproc4.dll
            opencv_dnn4.dll
            opencv_photo4.dll
            opencv_core480.dll
            opencv_imgproc480.dll
            opencv_dnn480.dll
            opencv_photo480.dll
        )
    endif()
    
    if(Tesseract_FOUND)
        message(STATUS "[GRIM] Copying Tesseract DLLs for OCR model")
        grim_copy_dlls("${_dll_dir_vcpkg}"
            tesseract50.dll
            tesseract51.dll
            leptonica-1.82.0.dll
            leptonica-1.83.0.dll
        )
    endif()
    
    # ONNX Runtime DLLs (library already linked in Config.cmake)
    if(EXISTS "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/onnxruntime.dll")
        grim_copy_dlls("${_dll_dir_vcpkg}"
            onnxruntime.dll
            onnxruntime_providers_shared.dll
        )
    endif()
endif()

# =========================================================
# uWebSockets (WebSocket server)
# =========================================================
find_package(unofficial-uwebsockets CONFIG REQUIRED)
target_link_libraries(GRIM PRIVATE unofficial::uwebsockets::uwebsockets)

# libuv DLL (required by uWebSockets)
grim_copy_dlls("${_dll_dir_vcpkg}"
    uv.dll
)

# =========================================================
# FlatBuffers (Binary serialization)
# =========================================================
find_package(flatbuffers CONFIG REQUIRED)
target_link_libraries(GRIM PRIVATE flatbuffers::flatbuffers)

# =========================================================
# nlohmann_json (JSON parsing for verification system)
# =========================================================
find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(GRIM PRIVATE nlohmann_json::nlohmann_json)

# =========================================================
# libzip (Archive extraction for DataCollection)
# =========================================================
find_package(libzip CONFIG REQUIRED)
target_link_libraries(GRIM PRIVATE libzip::zip)

# =========================================================
# Poppler (PDF text extraction for HuggingFace datasets)
# =========================================================
find_package(unofficial-poppler CONFIG REQUIRED)
target_link_libraries(GRIM PRIVATE unofficial::poppler::poppler-cpp)

grim_copy_dlls("${_dll_dir_vcpkg}"
    poppler.dll
    poppler-cpp.dll
)

# =========================================================
# Cesium Native (geospatial runtime; rendered through bgfx adapters)
# =========================================================
find_package(cesium-native CONFIG REQUIRED)
target_link_libraries(GRIM PRIVATE
    Cesium3DTilesSelection
    CesiumRasterOverlays
    CesiumIonClient
    CesiumClientCommon
    CesiumCurl
    CesiumGeospatial
    CesiumGltf
)
target_compile_definitions(GRIM PRIVATE GRIM_HAS_CESIUM_NATIVE=1)

# --- Ensure runtime finds our DLL first ---
set_target_properties(GRIM PROPERTIES
    VS_DEBUGGER_ENVIRONMENT "PATH=${DEPS_BIN_DIR};%PATH%"
)

# =========================================================
# FFmpeg (media/transcoding helpers) - linked directly from vcpkg libs
# =========================================================
set(_grim_ffmpeg_libs
    avcodec
    avdevice
    avfilter
    avformat
    avutil
    swresample
    swscale
)
target_link_libraries(GRIM PRIVATE ${_grim_ffmpeg_libs})

# =========================================================
# SentencePiece (tokenizer runtime for verifier)
# =========================================================
set(_sentencepiece_core_libs
    sentencepiece
    sentencepiece_train
)
if(WIN32)
    list(APPEND _sentencepiece_core_libs libprotobuf-lite abseil_dll)
else()
    list(APPEND _sentencepiece_core_libs protobuf-lite)
endif()

set(_sentencepiece_absl_libs
    absl_decode_rust_punycode
    absl_demangle_rust
    absl_flags_commandlineflag_internal
    absl_flags_commandlineflag
    absl_flags_config
    absl_flags_internal
    absl_flags_marshalling
    absl_flags_parse
    absl_flags_private_handle_accessor
    absl_flags_program_name
    absl_flags_reflection
    absl_flags_usage_internal
    absl_flags_usage
    absl_log_flags
    absl_log_internal_structured_proto
    absl_poison
    absl_tracing_internal
    absl_utf8_for_code_point
)

target_link_libraries(GRIM PRIVATE
    ${_sentencepiece_core_libs}
    ${_sentencepiece_absl_libs}
)

# =========================================================
# Final message
# =========================================================
message(STATUS "[GRIM] Dependencies.cmake runtime copy setup complete.")



# =========================================================
# Dependencies.cmake — GRIM Runtime Library Management
# =========================================================
# This file ensures all required runtime DLLs are copied into
# the active build configuration directory after compilation.
# Compatible with MSBuild, Ninja, and CLI cmake --build.

# =========================================================
# Base runtime and dependency paths
# =========================================================
set(_runtime_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}")

if (NOT DEFINED VCPKG_INSTALLED_DIR)
    set(VCPKG_INSTALLED_DIR "${CMAKE_SOURCE_DIR}/external/vcpkg/installed")
endif()

set(_dll_dir_vcpkg "${VCPKG_INSTALLED_DIR}/x64-windows/bin")
set(_dll_dir_whisper "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build/bin")
set(_dll_dir_porcupine "${CMAKE_SOURCE_DIR}/external/porcupine/lib/windows/amd64")

# =========================================================
# Helper macro for DLL copies
# =========================================================
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
# Graphics dependencies
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    freetype.dll
    libpng16.dll
    zlib1.dll
    libbz2.dll
    jpeg62.dll
)

# =========================================================
# OpenAL (Audio backend)
# =========================================================
grim_copy_dlls("${_dll_dir_vcpkg}"
    OpenAL32.dll
)

# --- Ensure correct OpenAL linkage ---
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_link_libraries(GRIM PRIVATE "${DEPS_LIB_DIR}/openal32.lib")
else()
    target_link_libraries(GRIM PRIVATE "${DEPS_LIB_DIR}/openal32.lib")
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

# --- Ensure runtime finds our DLL first ---
set_target_properties(GRIM PROPERTIES
    VS_DEBUGGER_ENVIRONMENT "PATH=${DEPS_BIN_DIR};%PATH%"
)

# =========================================================
# Final message
# =========================================================
message(STATUS "[GRIM] Dependencies.cmake runtime copy setup complete.")

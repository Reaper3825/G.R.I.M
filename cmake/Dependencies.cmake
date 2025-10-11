# =========================================================
# G.R.I.M - Dependency Configuration
# =========================================================

message(STATUS "[GRIM] Configuring dependencies...")

# =========================================================
#  Dependency Directories
# =========================================================
set(GRIM_DEPS_DIR "${CMAKE_SOURCE_DIR}/deps")
set(GRIM_DEPS_INCLUDE "${GRIM_DEPS_DIR}/include")
set(GRIM_DEPS_LIB "${GRIM_DEPS_DIR}/lib")

if (EXISTS "${GRIM_DEPS_DIR}")
    message(STATUS "[GRIM] Using consolidated dependency directory: ${GRIM_DEPS_DIR}")
else()
    message(WARNING "[GRIM] Dependency directory not found: ${GRIM_DEPS_DIR}")
endif()

# =========================================================
#  SFML (via vcpkg)
# =========================================================
find_package(SFML CONFIG REQUIRED COMPONENTS System Window Graphics Audio Network)

# =========================================================
# Link All Dependencies
# =========================================================
target_link_libraries(GRIM PRIVATE
    # Core Libraries
    SFML::System
    SFML::Window
    SFML::Graphics
    SFML::Audio
    SFML::Network
    glfw
    bgfx
    bx
    bimg
    cpr::cpr
    nlohmann_json::nlohmann_json

    # Audio + Speech
    portaudio
    whisper
)

# ---- GLFW ----
find_package(glfw3 CONFIG REQUIRED)
if (TARGET glfw3::glfw)
    target_link_libraries(GRIM PRIVATE glfw3::glfw)
elseif (TARGET glfw)
    target_link_libraries(GRIM PRIVATE glfw)
else()
    message(FATAL_ERROR "[GRIM] GLFW target not found in vcpkg.")
endif()



# =========================================================
#  OpenAL
# =========================================================
find_path(OPENAL_INCLUDE_DIR AL/al.h)
find_library(OPENAL_LIBRARY OpenAL32)
if (OPENAL_INCLUDE_DIR AND OPENAL_LIBRARY)
    target_include_directories(GRIM PRIVATE ${OPENAL_INCLUDE_DIR})
    target_link_libraries(GRIM PRIVATE ${OPENAL_LIBRARY})
else()
    message(WARNING "[GRIM] OpenAL not found — using fallback if available.")
endif()

# =========================================================
#  CPR (via vcpkg, manifest-safe)
# =========================================================
find_package(cpr CONFIG REQUIRED)

if (TARGET cpr::cpr)
    message(STATUS "[GRIM] Found CPR target: cpr::cpr")
    # Remove any accidental extra libraries from old config
    target_link_libraries(GRIM PRIVATE cpr::cpr)
else()
    message(FATAL_ERROR "[GRIM] CPR target not found — check vcpkg manifest and toolchain settings")
endif()

# Force correct library search path (MSVC sometimes misses it in manifest mode)
target_link_directories(GRIM PRIVATE "C:/vcpkg/installed/x64-windows/debug/lib")
target_link_directories(GRIM PRIVATE "C:/vcpkg/installed/x64-windows/lib")


# =========================================================
#  nlohmann-json
# =========================================================
find_package(nlohmann_json CONFIG QUIET)
if (nlohmann_json_FOUND)
    target_link_libraries(GRIM PRIVATE nlohmann_json::nlohmann_json)
else()
    message(WARNING "[GRIM] nlohmann-json not found — using fallback if available.")
endif()



target_link_directories(GRIM PRIVATE ${CMAKE_SOURCE_DIR}/deps/lib)
# =========================================================
# BGFX / BX / BIMG
# =========================================================
set(GRIM_DEPS_LIB "${CMAKE_SOURCE_DIR}/deps/lib")

if (EXISTS "${GRIM_DEPS_LIB}/bgfx.lib")
    message(STATUS "[GRIM] Linking prebuilt BGFX libraries from ${GRIM_DEPS_LIB}")

    # Add the deps/lib directory to the linker search path
    target_link_directories(GRIM PRIVATE "${GRIM_DEPS_LIB}")

    # Link all required static libs explicitly
    target_link_libraries(GRIM PRIVATE
        bgfx
        bx
        bimg
    )
else()
    message(WARNING "[GRIM] BGFX libraries not found in ${GRIM_DEPS_LIB}")
endif()

# =========================================================
# Whisper (Prebuilt)
# =========================================================
set(WHISPER_DIR "${CMAKE_SOURCE_DIR}/external/whisper.cpp/build")
set(WHISPER_LIB "${WHISPER_DIR}/src/Debug/whisper.lib")
set(WHISPER_DLL "${WHISPER_DIR}/bin/Debug/whisper.dll")

if (EXISTS "${WHISPER_LIB}")
    message(STATUS "[GRIM] Linking prebuilt Whisper from ${WHISPER_DIR}")
    target_link_libraries(GRIM PRIVATE "${WHISPER_LIB}")

    # Copy whisper.dll next to GRIM.exe after build
    add_custom_command(TARGET GRIM POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WHISPER_DLL}"
            "$<TARGET_FILE_DIR:GRIM>/whisper.dll"
        COMMENT "[GRIM] Copied whisper.dll to output directory"
    )
else()
    message(FATAL_ERROR "[GRIM] Prebuilt whisper.lib not found at ${WHISPER_LIB}")
endif()


# Copy all ggml runtime DLLs
foreach(_dll IN ITEMS ggml.dll ggml-base.dll ggml-cpu.dll)
    add_custom_command(TARGET GRIM POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${WHISPER_DIR}/bin/Debug/${_dll}"
            "$<TARGET_FILE_DIR:GRIM>/${_dll}"
        COMMENT "[GRIM] Copied ${_dll} to output directory"
    )
endforeach()

# =========================================================
#  Centralized Include Directory
# =========================================================
target_include_directories(GRIM PRIVATE
    ${GRIM_DEPS_INCLUDE}
)

# =========================================================
#  Library Directory
# =========================================================
if (EXISTS "${GRIM_DEPS_LIB}")
    message(STATUS "[GRIM] Adding library search path: ${GRIM_DEPS_LIB}")
    link_directories(${GRIM_DEPS_LIB})
else()
    message(WARNING "[GRIM] Library directory not found: ${GRIM_DEPS_LIB}")
endif()

# =========================================================
#  Finalization
# =========================================================
message(STATUS "[GRIM] Dependency configuration complete.")

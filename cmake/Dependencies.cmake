# =========================================================
# GRIM - External Dependencies
# =========================================================

# ---- GLFW ----
find_package(glfw3 CONFIG REQUIRED)
if(TARGET glfw)
    target_link_libraries(GRIM PRIVATE glfw)
elseif(TARGET glfw3::glfw)
    target_link_libraries(GRIM PRIVATE glfw3::glfw)
else()
    message(FATAL_ERROR "[GRIM] Could not find GLFW target in glfw3Config.cmake")
endif()

# ---- SFML ----
find_package(SFML CONFIG REQUIRED COMPONENTS System Window Graphics Audio Network)
if(TARGET SFML::System)
    target_link_libraries(GRIM PRIVATE
        SFML::System
        SFML::Window
        SFML::Graphics
        SFML::Audio
        SFML::Network
    )
else()
    message(FATAL_ERROR "[GRIM] SFML components not found or improperly configured.")
endif()

# ---- bgfx / bx / bimg ----
add_subdirectory(${CMAKE_SOURCE_DIR}/external/bgfx.cmake)
if(TARGET bgfx)
    target_link_libraries(GRIM PRIVATE bgfx bx bimg)
else()
    message(FATAL_ERROR "[GRIM] bgfx subproject not found.")
endif()

# ---- OpenAL Soft ----
find_package(OpenAL CONFIG QUIET)
if(TARGET OpenAL::OpenAL)
    target_link_libraries(GRIM PRIVATE OpenAL::OpenAL)
elseif(EXISTS "${VCPKG_INSTALLED_DIR}/x64-windows/lib/OpenAL32.lib")
    target_link_libraries(GRIM PRIVATE OpenAL32)
else()
    message(WARNING "[GRIM] OpenAL not found through CMake; linking fallback OpenAL32.")
    target_link_libraries(GRIM PRIVATE OpenAL32)
endif()

# ---- cpr (HTTP) ----
find_package(cpr CONFIG QUIET)
if(cpr_FOUND)
    target_link_libraries(GRIM PRIVATE cpr::cpr)
else()
    message(WARNING "[GRIM] cpr not found; skipping HTTP support.")
endif()

# ---- nlohmann_json ----
find_package(nlohmann_json CONFIG REQUIRED)
if(TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(GRIM PRIVATE nlohmann_json::nlohmann_json)
else()
    message(FATAL_ERROR "[GRIM] nlohmann_json target not found — ensure vcpkg installed nlohmann-json.")
endif()

# ---- PortAudio ----
find_package(portaudio CONFIG QUIET)
if(portaudio_FOUND)
    if(TARGET PortAudio::PortAudio)
        target_link_libraries(GRIM PRIVATE PortAudio::PortAudio)
    else()
        target_link_libraries(GRIM PRIVATE portaudio)
    endif()
else()
    message(WARNING "[GRIM] PortAudio not found; microphone input will be disabled.")
endif()

# =========================================================
# Optional: Whisper / GGML (if present in external folder)
# =========================================================
if(EXISTS "${CMAKE_SOURCE_DIR}/external/whisper.cpp")
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/whisper.cpp)
    if(TARGET whisper)
        target_link_libraries(GRIM PRIVATE whisper)
    else()
        message(WARNING "[GRIM] Whisper build skipped (no target generated).")
    endif()
endif()

# =========================================================
# Dependency Summary
# =========================================================
message(STATUS "[GRIM] Dependency configuration complete.")

# =========================================================
# GRIM External Dependencies
# =========================================================

# ---- GLFW ----
find_package(glfw3 CONFIG REQUIRED)
if(TARGET glfw)
    target_link_libraries(GRIM PRIVATE glfw)
elseif(TARGET glfw3::glfw)
    target_link_libraries(GRIM PRIVATE glfw3::glfw)
else()
    message(FATAL_ERROR "Could not find GLFW target in glfw3Config.cmake")
endif()

# ---- SFML ----
find_package(SFML CONFIG REQUIRED COMPONENTS System Window Graphics Audio Network)
target_link_libraries(GRIM PRIVATE SFML::System SFML::Window SFML::Graphics SFML::Audio SFML::Network)

# ---- bgfx ----
add_subdirectory(external/bgfx.cmake)
target_link_libraries(GRIM PRIVATE bgfx bx bimg)

# ---- cpr ----
find_package(cpr QUIET)
if(cpr_FOUND)
  target_link_libraries(GRIM PRIVATE cpr::cpr)
else()
  message(WARNING "cpr not found; using /usr/local fallback")
  target_include_directories(GRIM PRIVATE /usr/local/include)
  target_link_directories(GRIM PRIVATE /usr/local/lib)
  target_link_libraries(GRIM PRIVATE cpr ssl crypto)
endif()

# ---- nlohmann_json ----
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(GRIM PRIVATE nlohmann_json::nlohmann_json)
else()
  message(WARNING "nlohmann_json not found; using header-only include")
  target_include_directories(GRIM PRIVATE ${CMAKE_SOURCE_DIR}/external/nlohmann_json)
endif()

# ---- Whisper.cpp ----
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
add_subdirectory(external/whisper.cpp EXCLUDE_FROM_ALL)
target_link_libraries(GRIM PRIVATE whisper)

# ---- PortAudio ----
find_path(PORTAUDIO_INCLUDE_DIR portaudio.h)
find_library(PORTAUDIO_LIBRARY portaudio)
if(PORTAUDIO_INCLUDE_DIR AND PORTAUDIO_LIBRARY)
  target_include_directories(GRIM PRIVATE ${PORTAUDIO_INCLUDE_DIR})
  target_link_libraries(GRIM PRIVATE ${PORTAUDIO_LIBRARY})
else()
  message(FATAL_ERROR "PortAudio not found. Please install it.")
endif()

# ---- Windows specific ----
if(WIN32)
    target_link_libraries(GRIM PRIVATE sapi ole32 oleaut32 shlwapi)
endif()

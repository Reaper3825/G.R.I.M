cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

# ---- Language / build settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Debug or Release" FORCE)
endif()

# ---- Warnings ----
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Speed up rebuilds with ccache ----
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

# ---- Portable mode ----
option(GRIM_PORTABLE_ONLY "Force resources to live next to executable" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ---- Sources ----
file(GLOB COMMAND_SOURCES "commands/*.cpp")
file(GLOB COMMAND_HEADERS "commands/*.hpp")

set(GRIM_SOURCES
    main.cpp
    ai.cpp
    bootstrap.cpp
    resources.cpp
    nlp.cpp
    nlp_rules.cpp
    synonyms.cpp
    aliases.cpp
    console_history.cpp
    ui_helpers.cpp
    ui_draw.cpp
    ui_events.cpp
    voice.cpp
    voice_stream.cpp 
    system_detect.cpp
    voice_speak.cpp     
    response_manager.cpp
    error_manager.cpp
    ${COMMAND_SOURCES}
)

set(GRIM_HEADERS
    ai.hpp
    bootstrap.hpp
    console_history.hpp
    nlp.hpp
    intent.hpp
    aliases.hpp
    synonyms.hpp
    resources.hpp
    ui_config.hpp
    ui_helpers.hpp
    ui_draw.hpp
    ui_events.hpp
    timer.hpp
    voice.hpp
    voice_stream.hpp
    voice_speak.hpp 
    response_manager.hpp
    error_manager.hpp
    ${COMMAND_HEADERS}
)


# ---- Define target early ----
add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})

# ---- Dependencies ----

# SFML
find_package(PkgConfig REQUIRED)
pkg_check_modules(SFML REQUIRED sfml-graphics sfml-window sfml-system sfml-audio)
target_include_directories(GRIM PRIVATE ${SFML_INCLUDE_DIRS})
target_link_directories(GRIM PRIVATE ${SFML_LIBRARY_DIRS})
target_link_libraries(GRIM PRIVATE ${SFML_LIBRARIES})

# CPR
find_package(cpr QUIET)
if(cpr_FOUND)
  target_link_libraries(GRIM PRIVATE cpr::cpr)
else()
  message(WARNING "cpr not found via find_package, falling back to /usr/local")
  target_include_directories(GRIM PRIVATE /usr/local/include)
  target_link_directories(GRIM PRIVATE /usr/local/lib)
  target_link_libraries(GRIM PRIVATE cpr ssl crypto)
endif()

# nlohmann_json with fallback
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(GRIM PRIVATE nlohmann_json::nlohmann_json)
else()
  message(WARNING "nlohmann_json not found, falling back to header-only include")
  target_include_directories(GRIM PRIVATE ${CMAKE_SOURCE_DIR}/external/nlohmann_json)
endif()

# ---- Whisper GPU backend auto-detection ----
if (WIN32)
    # Prefer CUDA on Windows
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA detected: enabling GPU support for Whisper")
        set(GGML_CUDA ON CACHE BOOL "Enable CUDA support" FORCE)
    else()
        message(WARNING "CUDA not found, Whisper will run on CPU")
    endif()
elseif(APPLE)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        message(STATUS "Apple Silicon detected: enabling Metal backend")
        set(GGML_METAL ON CACHE BOOL "Enable Metal support" FORCE)
    endif()
else() # Linux / other
    find_package(CUDAToolkit QUIET)
    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA detected: enabling GPU support for Whisper")
        set(GGML_CUDA ON CACHE BOOL "Enable CUDA support" FORCE)
    elseif (EXISTS "/usr/include/vulkan/vulkan.h")
        message(STATUS "Vulkan headers found: enabling Vulkan support for Whisper")
        set(GGML_VULKAN ON CACHE BOOL "Enable Vulkan support" FORCE)
    elseif (EXISTS "/usr/include/CL/cl.h")
        message(STATUS "OpenCL headers found: enabling OpenCL support for Whisper")
        set(GGML_OPENCL ON CACHE BOOL "Enable OpenCL support" FORCE)
    else()
        message(WARNING "No GPU backend found, Whisper will run on CPU")
    endif()
endif()

# Whisper.cpp
add_subdirectory(external/whisper.cpp EXCLUDE_FROM_ALL)
target_link_libraries(GRIM PRIVATE whisper)

# PortAudio
find_path(PORTAUDIO_INCLUDE_DIR portaudio.h)
find_library(PORTAUDIO_LIBRARY portaudio)
if(PORTAUDIO_INCLUDE_DIR AND PORTAUDIO_LIBRARY)
  target_include_directories(GRIM PRIVATE ${PORTAUDIO_INCLUDE_DIR})
  target_link_libraries(GRIM PRIVATE ${PORTAUDIO_LIBRARY})
else()
  message(FATAL_ERROR "PortAudio not found. Please install it.")
endif()

# ---- Voice integration ----
target_include_directories(GRIM PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external/whisper.cpp
    ${PORTAUDIO_INCLUDE_DIR}
)

# ---- Compile-time resource paths ----
include(GNUInstallDirs)

if(GRIM_PORTABLE_ONLY)
  target_compile_definitions(GRIM PRIVATE GRIM_PORTABLE_ONLY=1)
else()
  set(GRIM_DATADIR "${CMAKE_INSTALL_FULL_DATADIR}/grim")
  target_compile_definitions(GRIM PRIVATE GRIM_DATA_DIR="${GRIM_DATADIR}")
endif()

# ---- JSON Resources ----
set(GRIM_RESOURCES
    nlp_rules.json
    synonyms.json
    app_aliases.json
    ai_config.json
    memory.json
)

foreach(FILE ${GRIM_RESOURCES})
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/resources/${FILE}" AND NOT EXISTS "${CMAKE_BINARY_DIR}/${FILE}")
    file(WRITE "${CMAKE_BINARY_DIR}/${FILE}" "{}\n")
  endif()
endforeach()

foreach(FILE ${GRIM_RESOURCES})
  if(EXISTS "${CMAKE_SOURCE_DIR}/resources/${FILE}")
    set(SRC_FILE "${CMAKE_SOURCE_DIR}/resources/${FILE}")
  else()
    set(SRC_FILE "${CMAKE_BINARY_DIR}/${FILE}")
  endif()
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SRC_FILE}"
            "$<TARGET_FILE_DIR:GRIM>/${FILE}"
  )
endforeach()

# ---- Fonts (auto-detect) ----
set(GRIM_FONT_SRC "")

if(GRIM_FONT_PATH)
  set(GRIM_FONT_SRC "${GRIM_FONT_PATH}")
  message(STATUS "GRIM: Using user-specified font path: ${GRIM_FONT_SRC}")

elseif(EXISTS "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
  set(GRIM_FONT_SRC "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
  message(STATUS "GRIM: Found bundled font at ${GRIM_FONT_SRC}")

elseif(EXISTS "C:/Windows/Fonts/arial.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/arial.ttf")
elseif(EXISTS "C:/Windows/Fonts/ARIAL.TTF")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/ARIAL.TTF")
elseif(EXISTS "C:/Windows/Fonts/segoeui.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/segoeui.ttf")

elseif(EXISTS "/c/Windows/Fonts/arial.ttf")
  set(GRIM_FONT_SRC "/c/Windows/Fonts/arial.ttf")
elseif(EXISTS "/c/Windows/Fonts/ARIAL.TTF")
  set(GRIM_FONT_SRC "/c/Windows/Fonts/ARIAL.TTF")
elseif(EXISTS "/c/Windows/Fonts/segoeui.ttf")
  set(GRIM_FONT_SRC "/c/Windows/Fonts/segoeui.ttf")

else()
  message(WARNING "GRIM: No font found during CMake configure. UI may fail unless you provide -DGRIM_FONT_PATH=...")
endif()

# ---- Whisper model (auto-download) ----
set(WHISPER_MODEL_DIR "${CMAKE_SOURCE_DIR}/external/whisper.cpp/models")
set(WHISPER_MODEL_FILE "${WHISPER_MODEL_DIR}/ggml-small.bin")

if(NOT EXISTS "${WHISPER_MODEL_FILE}")
  file(MAKE_DIRECTORY "${WHISPER_MODEL_DIR}")
  add_custom_command(
    OUTPUT "${WHISPER_MODEL_FILE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Downloading Whisper model (ggml-small.bin)..."
    COMMAND curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
            -o "${WHISPER_MODEL_FILE}"
    COMMENT "Fetching Whisper model"
    VERBATIM
  )
  add_custom_target(fetch_whisper_model ALL DEPENDS "${WHISPER_MODEL_FILE}")
  add_dependencies(GRIM fetch_whisper_model)
endif()

# ---- Resources folder ----
if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
  add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/resources"
            "$<TARGET_FILE_DIR:GRIM>/resources"
  )
  add_dependencies(GRIM copy_resources)
endif()

# ---- Install rules ----
install(TARGETS GRIM RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

foreach(FILE ${GRIM_RESOURCES})
  if(EXISTS "${CMAKE_SOURCE_DIR}/resources/${FILE}")
    install(FILES "${CMAKE_SOURCE_DIR}/resources/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
  else()
    install(FILES "${CMAKE_BINARY_DIR}/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
  endif()
endforeach()

if(GRIM_FONT_SRC)
  install(FILES "${GRIM_FONT_SRC}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/resources" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
endif()

if(EXISTS "${WHISPER_MODEL_FILE}")
  install(FILES "${WHISPER_MODEL_FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim/models)
endif()

if (WIN32)
    target_link_libraries(GRIM PRIVATE sapi ole32)
endif()

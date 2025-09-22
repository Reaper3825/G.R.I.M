cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

# =========================================================
# Language / build settings
# =========================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Release build
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Debug or Release" FORCE)
endif()

# =========================================================
# GPU acceleration options
# =========================================================
option(GRIM_USE_CUDA "Enable CUDA acceleration for Whisper/AI" ON)
option(GRIM_USE_METAL "Enable Metal acceleration for Whisper/AI" OFF)
option(GRIM_USE_VULKAN "Enable Vulkan acceleration for Whisper/AI" OFF)
option(GRIM_USE_OPENCL "Enable OpenCL acceleration for Whisper/AI" OFF)

# ---- CUDA arch for RTX 3080 Ti (Ampere, sm_86) ----
if(GRIM_USE_CUDA)
    add_definitions(-DWHISPER_USE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "Enable CUDA in ggml")

    # Force architecture to sm_86 only
    set(CMAKE_CUDA_ARCHITECTURES 86)
    add_definitions(-DGGML_CUDA_ARCH=86)

    # Fix bad quoting of CMAKE_INTDIR passed to nvcc
    if(MSVC)
        add_compile_definitions(CMAKE_INTDIR=Release)
    endif()
endif()

# ---------------------------------------------------------
# Workaround: remove broken quoted CMAKE_INTDIR define for NVCC on MSVC
# ---------------------------------------------------------
if(MSVC AND GRIM_USE_CUDA)
    foreach(flag_var
        CMAKE_CUDA_FLAGS
        CMAKE_CUDA_FLAGS_RELEASE
        CMAKE_CUDA_FLAGS_DEBUG
        CMAKE_CUDA_FLAGS_RELWITHDEBINFO
        CMAKE_CUDA_FLAGS_MINSIZEREL)
        string(REPLACE "-D\"CMAKE_INTDIR=\\\"Release\\\"\"" "" ${flag_var} "${${flag_var}}")
        set(${flag_var} "${${flag_var}}" CACHE STRING "Overridden" FORCE)
    endforeach()
endif()

# =========================================================
# UI font path option
# =========================================================
set(GRIM_FONT_PATH "" CACHE STRING "Path to TTF font file to use in the UI")
if(NOT GRIM_FONT_PATH STREQUAL "")
    add_definitions(-DGRIM_FONT_PATH=\"${GRIM_FONT_PATH}\")
endif()

# =========================================================
# GPU backend auto-detection
# =========================================================
if(GRIM_USE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        include(CheckLanguage)
        check_language(CUDA)

        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            message(STATUS "✅ CUDA detected: ${CUDAToolkit_VERSION} at ${CUDAToolkit_LIBRARY_ROOT}")
            set(GGML_CUDA ON CACHE BOOL "Enable CUDA in ggml" FORCE)
        else()
            message(WARNING "⚠️ CUDA Toolkit installed, but no working GPU compiler found. Falling back to CPU.")
            set(GRIM_USE_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
            set(GGML_CUDA OFF CACHE BOOL "" FORCE)
        endif()
    else()
        message(WARNING "⚠️ CUDA requested but not found. Falling back to CPU.")
        set(GRIM_USE_CUDA OFF CACHE BOOL "Disable CUDA" FORCE)
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
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
# Compiler warnings
# =========================================================
if(MSVC)
  # Only apply MSVC flags to C++ sources, not CUDA
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>
                      $<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
  # Only apply GCC/Clang warnings to C++ sources
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
# Portable mode
# =========================================================
option(GRIM_PORTABLE_ONLY "Force resources to live next to executable" OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =========================================================
# Sources
# =========================================================
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
    bootstrap_config.cpp
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
    bootstrap_config.hpp
    ${COMMAND_HEADERS}
)

# =========================================================
# Target
# =========================================================
add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})
target_precompile_headers(GRIM PRIVATE pch.hpp)

# =========================================================
# Dependencies
# =========================================================

# ---- SFML ----
find_package(SFML CONFIG REQUIRED COMPONENTS System Window Graphics Audio Network)
target_link_libraries(GRIM PRIVATE SFML::System SFML::Window SFML::Graphics SFML::Audio SFML::Network)

# ---- CPR ----
find_package(cpr QUIET)
if(cpr_FOUND)
  target_link_libraries(GRIM PRIVATE cpr::cpr)
else()
  message(WARNING "cpr not found, falling back to /usr/local")
  target_include_directories(GRIM PRIVATE /usr/local/include)
  target_link_directories(GRIM PRIVATE /usr/local/lib)
  target_link_libraries(GRIM PRIVATE cpr ssl crypto)
endif()

# ---- nlohmann_json ----
find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
  target_link_libraries(GRIM PRIVATE nlohmann_json::nlohmann_json)
else()
  message(WARNING "nlohmann_json not found, falling back to header-only include")
  target_include_directories(GRIM PRIVATE ${CMAKE_SOURCE_DIR}/external/nlohmann_json)
endif()

# ---- Whisper.cpp ----
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

# =========================================================
# Voice / Resources
# =========================================================
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

# =========================================================
# JSON resources auto-copy
# =========================================================
set(GRIM_RESOURCES
    nlp_rules.json
    synonyms.json
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

# =========================================================
# Font auto-detection
# =========================================================
set(GRIM_FONT_SRC "")
if(GedaadsasaaaaaaaaaaaaaaaRIM_FONT_PATH)
  set(GRIM_FONT_SRC "${GRIM_FONT_PATH}")
  message(STATUS "Using user-specified font: ${GRIM_FONT_SRC}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
  set(GRIM_FONT_SRC "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
elseif(EXISTS "C:/Windows/Fonts/arial.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/arial.ttf")
elseif(EXISTS "C:/Windows/Fonts/segoeui.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/segoeui.ttf")
else()
  message(WARNING "No font found. Provide -DGRIM_FONT_PATH=...")
endif()

# =========================================================
# Whisper model auto-download
# =========================================================
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

# =========================================================
# Resources folder copy
# =========================================================
if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
  add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/resources"
            "$<TARGET_FILE_DIR:GRIM>/resources"
  )
  add_dependencies(GRIM copy_resources)
endif()

# =========================================================
# Install rules
# =========================================================
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

# =========================================================
# Windows-specific links
# =========================================================
if(WIN32)
    target_link_libraries(GRIM PRIVATE sapi ole32 oleaut32 shlwapi)
endif()

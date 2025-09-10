cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

# ---- Language / build settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Debug or Release" FORCE)
endif()

# Output all executables to build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---- Warnings ----
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Portable mode option ----
option(GRIM_PORTABLE_ONLY "Force resources to live next to executable" OFF)

# ---- Sources ----
set(GRIM_SOURCES
    main.cpp
    ai.cpp
    bootstrap.cpp
    resources.cpp
    commands.cpp
    nlp.cpp
    nlp_rules.cpp
    synonyms.cpp
    aliases.cpp
    console_history.cpp
    ui_helpers.cpp
    ui_draw.cpp
    ui_events.cpp
    voice.cpp
)

set(GRIM_HEADERS
    ai.hpp
    bootstrap.hpp
    commands.hpp
    console_history.hpp
    nlp.hpp
    aliases.hpp
    synonyms.hpp
    resources.hpp
    ui_config.hpp
    ui_helpers.hpp
    ui_draw.hpp
    ui_events.hpp
    timer.hpp
)

add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})

# ---- Dependencies ----

# SFML via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(SFML REQUIRED sfml-graphics sfml-window sfml-system)
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

# ---- Voice demo ----
add_executable(voice voice.cpp)
target_include_directories(voice PRIVATE
    ${CMAKE_SOURCE_DIR}/external/whisper.cpp
    ${PORTAUDIO_INCLUDE_DIR}
)
target_link_libraries(voice PRIVATE whisper ${PORTAUDIO_LIBRARY})


# ---- Compile-time resource paths ----
include(GNUInstallDirs)

if(GRIM_PORTABLE_ONLY)
  target_compile_definitions(GRIM PRIVATE GRIM_PORTABLE_ONLY=1)
else()
  set(GRIM_DATADIR "${CMAKE_INSTALL_FULL_DATADIR}/grim")
  target_compile_definitions(GRIM PRIVATE GRIM_DATA_DIR="${GRIM_DATADIR}")
endif()

# ---- Default JSONs ----
set(GRIM_RESOURCES
    nlp_rules.json
    synonyms.json
    app_aliases.json
    ai_config.json
    memory.json
)

foreach(FILE ${GRIM_RESOURCES})
  if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${FILE}")
    file(WRITE "${CMAKE_BINARY_DIR}/${FILE}" "{}\n")
  endif()
endforeach()

# Copy resources into build dir
foreach(FILE ${GRIM_RESOURCES})
  if(EXISTS "${CMAKE_SOURCE_DIR}/${FILE}")
    set(SRC_FILE "${CMAKE_SOURCE_DIR}/${FILE}")
  else()
    set(SRC_FILE "${CMAKE_BINARY_DIR}/${FILE}")
  endif()
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SRC_FILE}"
            "$<TARGET_FILE_DIR:GRIM>/${FILE}"
  )
endforeach()

# ---- Fonts ----
set(GRIM_FONT_SRC "")
if(EXISTS "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
  set(GRIM_FONT_SRC "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
elseif(EXISTS "C:/Windows/Fonts/arial.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/arial.ttf")
elseif(EXISTS "C:/Windows/Fonts/segoeui.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/segoeui.ttf")
endif()

if(GRIM_FONT_SRC)
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GRIM_FONT_SRC}"
            "$<TARGET_FILE_DIR:GRIM>/DejaVuSans.ttf"
  )
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
install(TARGETS voice RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

foreach(FILE ${GRIM_RESOURCES})
  if(EXISTS "${CMAKE_SOURCE_DIR}/${FILE}")
    install(FILES "${CMAKE_SOURCE_DIR}/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
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

cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

# ---- Language / build settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose Debug or Release" FORCE)
endif()

# Enable warnings
if(MSVC)
  add_compile_options(/W4 /permissive-)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- Portable mode option ----
option(GRIM_PORTABLE "Enable portable mode (resources next to exe)" ON)

# ---- Sources ----
set(GRIM_SOURCES
    main.cpp
    commands.cpp
    nlp.cpp
    nlp_rules.cpp
    ai.cpp
    synonyms.cpp
    aliases.cpp
    console_history.cpp
    resources.cpp
    ui_helpers.cpp
    ui_draw.cpp
    ui_events.cpp
    resources.cpp
)

set(GRIM_HEADERS
    commands.hpp
    console_history.hpp
    nlp.hpp
    ai.hpp
    aliases.hpp
    synonyms.hpp
    resources.hpp
    ui_config.hpp
    ui_helpers.hpp
    ui_draw.hpp
    ui_events.hpp
    timer.hpp
    resources.hpp
)

add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})

# ---- Dependencies ----

# SFML via pkg-config (works reliably on Pi)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SFML REQUIRED sfml-graphics sfml-window sfml-system)

include_directories(${SFML_INCLUDE_DIRS})
link_directories(${SFML_LIBRARY_DIRS})
target_link_libraries(GRIM PRIVATE ${SFML_LIBRARIES})

# CPR (manual linking: installed under /usr/local)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
target_link_libraries(GRIM PRIVATE cpr ssl crypto)

target_include_directories(GRIM PRIVATE ${CMAKE_SOURCE_DIR})
target_sources(GRIM PRIVATE ${GRIM_HEADERS})

# ---- Compile-time resource paths ----
include(GNUInstallDirs)

if(GRIM_PORTABLE)
  target_compile_definitions(GRIM PRIVATE GRIM_PORTABLE_ONLY=1)
else()
  set(GRIM_DATADIR "${CMAKE_INSTALL_FULL_DATADIR}/grim")
  target_compile_definitions(GRIM PRIVATE GRIM_DATA_DIR="${GRIM_DATADIR}")
endif()

# ---- Asset handling ----

set(GRIM_RULES_SRC "${CMAKE_SOURCE_DIR}/nlp_rules.json")
if(EXISTS "${GRIM_RULES_SRC}")
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GRIM_RULES_SRC}"
            "$<TARGET_FILE_DIR:GRIM>/nlp_rules.json"
  )
else()
  message(WARNING "nlp_rules.json not found in source; generating default.")
  set(GRIM_RULES_DEFAULT
    "[\n"
    "  {\"intent\":\"open_app\",\"pattern\":\"^\\\\s*(open|launch|start)\\\\s+([\\\\w\\\\.\\\\-]+)\\\\s*$\",\"slot_names\":[\"verb\",\"app\"],\"score_boost\":0.3,\"case_insensitive\":true},\n"
    "  {\"intent\":\"search_web\",\"pattern\":\"^(google|search|look up)\\\\s+(.+)$\",\"slot_names\":[\"verb\",\"query\"],\"score_boost\":0.2,\"case_insensitive\":true},\n"
    "  {\"intent\":\"set_timer\",\"pattern\":\"^(set\\\\s+)?timer\\\\s+for\\\\s+(\\\\d+)\\\\s*(seconds|s|minutes|min|m|hours|h)\\\\b\",\"slot_names\":[\"_opt\",\"value\",\"unit\"],\"score_boost\":0.25,\"case_insensitive\":true}\n"
    "]\n"
  )
  file(WRITE "${CMAKE_BINARY_DIR}/nlp_rules.json" "${GRIM_RULES_DEFAULT}")
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/nlp_rules.json"
            "$<TARGET_FILE_DIR:GRIM>/nlp_rules.json"
  )
endif()

set(GRIM_SYNONYMS_SRC "${CMAKE_SOURCE_DIR}/synonyms.json")
if(EXISTS "${GRIM_SYNONYMS_SRC}")
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GRIM_SYNONYMS_SRC}"
            "$<TARGET_FILE_DIR:GRIM>/synonyms.json"
  )
endif()

set(GRIM_ALIASES_SRC "${CMAKE_SOURCE_DIR}/app_aliases.json")
if(EXISTS "${GRIM_ALIASES_SRC}")
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GRIM_ALIASES_SRC}"
            "$<TARGET_FILE_DIR:GRIM>/app_aliases.json"
  )
endif()

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

if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
  add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/resources"
            "$<TARGET_FILE_DIR:GRIM>/resources"
  )
  add_dependencies(GRIM copy_resources)
endif()

# ---- Install rules ----

install(TARGETS GRIM
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

foreach(FILE nlp_rules.json synonyms.json app_aliases.json)
  if(EXISTS "${CMAKE_SOURCE_DIR}/${FILE}")
    install(FILES "${CMAKE_SOURCE_DIR}/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
  endif()
endforeach()

if(GRIM_FONT_SRC)
  install(FILES "${GRIM_FONT_SRC}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
  install(DIRECTORY "${CMAKE_SOURCE_DIR}/resources" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
endif()

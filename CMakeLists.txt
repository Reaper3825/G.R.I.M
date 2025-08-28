cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

# ---- Language / build settings ----
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Sources ----
add_executable(GRIM
    main.cpp
    commands.cpp
    nlp.cpp
    nlp_rules.cpp
    ai.cpp
    NLP.cpp
)

# ---- Dependencies ----
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
find_package(cpr CONFIG REQUIRED)

target_link_libraries(GRIM PRIVATE sfml-graphics sfml-window sfml-system cpr::cpr)

# ---- Asset handling ----

# 1) NLP rules
set(GRIM_RULES_SRC "${CMAKE_SOURCE_DIR}/nlp_rules.json")
if (EXISTS "${GRIM_RULES_SRC}")
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
    "  {\"intent\":\"set_timer\",\"pattern\":\"^(set\\\\s+)?timer\\\\s+for\\\\s+(\\\\d+)\\\\s*(minutes|min|m)\\\\b\",\"slot_names\":[\"_opt\",\"minutes\"],\"score_boost\":0.25,\"case_insensitive\":true}\n"
    "]\n"
  )
  file(WRITE "${CMAKE_BINARY_DIR}/nlp_rules.json" "${GRIM_RULES_DEFAULT}")
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/nlp_rules.json"
            "$<TARGET_FILE_DIR:GRIM>/nlp_rules.json"
  )
endif()

# 2) Fonts: copy the first .ttf we find in resources/
file(GLOB FONT_FILES "${CMAKE_SOURCE_DIR}/resources/*.ttf")
list(GET FONT_FILES 0 GRIM_FONT_SRC)

if (GRIM_FONT_SRC)
  message(STATUS "Using default font: ${GRIM_FONT_SRC}")
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GRIM_FONT_SRC}"
            "$<TARGET_FILE_DIR:GRIM>/DefaultFont.ttf"
  )
else()
  message(WARNING "No .ttf font found in resources/. Text may not render.")
endif()


# 3) Always copy the entire resources/ folder (future-proof asset handling)
if (EXISTS "${CMAKE_SOURCE_DIR}/resources")
  add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/resources"
            "$<TARGET_FILE_DIR:GRIM>/resources"
  )
  add_dependencies(GRIM copy_resources)
else()
  message(WARNING "No resources/ folder found. Skipping copy.")
endif()

# ---- Optional nicer output dirs ----
# set_target_properties(GRIM PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )

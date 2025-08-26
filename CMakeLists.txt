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
# SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# CPR (installed via MSYS2: mingw-w64-x86_64-cpr)
find_package(cpr CONFIG REQUIRED)

# Link
target_link_libraries(GRIM PRIVATE sfml-graphics sfml-window sfml-system cpr::cpr)

# ---- Asset handling (robust) ----
# 1) NLP rules: copy if present; else generate a minimal default
set(GRIM_RULES_SRC "${CMAKE_SOURCE_DIR}/nlp_rules.json")
if (EXISTS "${GRIM_RULES_SRC}")
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GRIM_RULES_SRC}"
            "$<TARGET_FILE_DIR:GRIM>/nlp_rules.json"
    VERBATIM
  )
else()
  message(WARNING "nlp_rules.json not found in source; generating a minimal default.")
  # Important: JSON string with proper escaping for backslashes
  set(GRIM_RULES_DEFAULT
    "[\n"
    "  {\"intent\":\"open_app\",\"pattern\":\"^\\\\s*(open|launch|start)\\\\s+([\\\\w\\\\.\\\\-]+)\\\\s*$\",\"slot_names\":[\"verb\",\"app\"],\"score_boost\":0.3,\"case_insensitive\":true},\n"
    "  {\"intent\":\"search_web\",\"pattern\":\"^(google|search|look up)\\\\s+(.+)$\",\"slot_names\":[\"verb\",\"query\"],\"score_boost\":0.2,\"case_insensitive\":true},\n"
    "  {\"intent\":\"set_timer\",\"pattern\":\"^(set\\\\s+)?timer\\\\s+for\\\\s+(\\\\d+)\\\\s*(minutes|min|m)\\\\b\",\"slot_names\":[\"_opt\",\"minutes\"],\"score_boost\":0.25,\"case_insensitive\":true}\n"
    "]\n"
  )
  # Write default rules into build dir at configure time
  file(WRITE "${CMAKE_BINARY_DIR}/nlp_rules.json" "${GRIM_RULES_DEFAULT}")
  # Ensure the exe dir gets the file after build
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_BINARY_DIR}/nlp_rules.json"
            "$<TARGET_FILE_DIR:GRIM>/nlp_rules.json"
    VERBATIM
  )
endif()

# 2) Font: prefer project DejaVuSans.ttf; else fall back to Windows fonts if available
set(GRIM_FONT_SRC "")
if (EXISTS "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
  set(GRIM_FONT_SRC "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
elseif (EXISTS "C:/Windows/Fonts/arial.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/arial.ttf")
elseif (EXISTS "C:/Windows/Fonts/segoeui.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/segoeui.ttf")
endif()

if (GRIM_FONT_SRC)
  # Copy to exe dir as DejaVuSans.ttf (even if source is Arial/Segoe)
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${GRIM_FONT_SRC}"
            "$<TARGET_FILE_DIR:GRIM>/DejaVuSans.ttf"
    VERBATIM
  )
else()
  message(WARNING "No font found (DejaVuSans.ttf or Windows fonts). Text may not render.")
endif()

# ---- (Optional) nicer output dirs ----
# set_target_properties(GRIM PROPERTIES
#   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )

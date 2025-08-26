cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(GRIM
    main.cpp
    commands.cpp
    nlp.cpp
    nlp_rules.cpp
    ai.cpp
    NLP.cpp
)

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Find CPR
find_package(cpr CONFIG REQUIRED)

# Link against SFML + CPR
target_link_libraries(GRIM PRIVATE sfml-graphics sfml-window sfml-system cpr::cpr)

# --- Asset copy step (new) ---
# List of assets that should always be copied to the build folder
set(GRIM_ASSETS
    nlp_rules.json
    DejaVuSans.ttf
)

# Copy each asset next to the built exe after every build
foreach(f ${GRIM_ASSETS})
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_SOURCE_DIR}/${f}
            $<TARGET_FILE_DIR:GRIM>/${f}
  )
endforeach()

cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add your sources
add_executable(GRIM
    main.cpp
    commands.cpp
    nlp.cpp
    nlp_rules.cpp
    ai.cpp
)

# Find SFML
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Find CPR
find_package(cpr CONFIG REQUIRED)

# Link against SFML + CPR
target_link_libraries(GRIM PRIVATE sfml-graphics sfml-window sfml-system cpr::cpr)

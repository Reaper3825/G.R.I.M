cmake_minimum_required(VERSION 3.22)
project(osint_plugin LANGUAGES CXX)

# OSINT plugin DLL - defensive self-audit tools
add_library(osint_plugin SHARED
    ${CMAKE_SOURCE_DIR}/commands/commands_osint.cpp
    ${CMAKE_SOURCE_DIR}/plugins/osint_plugin.cpp
    ${CMAKE_SOURCE_DIR}/external_collector/osit.cpp
    ${CMAKE_SOURCE_DIR}/external_collector/osit_secrets.cpp
)

target_include_directories(osint_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/commands
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/helpers
    ${CMAKE_SOURCE_DIR}/logger
    ${CMAKE_SOURCE_DIR}/external_collector
)

# Build as plugin
target_compile_definitions(osint_plugin PRIVATE GRIM_BUILD_PLUGIN)

# Link against GRIM export lib for command registration
target_link_libraries(osint_plugin PRIVATE GRIM_EXPORT)
add_dependencies(osint_plugin GRIM)

set_target_properties(osint_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/plugins"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/plugins"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/plugins"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/plugins"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

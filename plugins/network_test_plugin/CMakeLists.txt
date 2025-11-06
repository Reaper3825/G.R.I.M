cmake_minimum_required(VERSION 3.22)
project(network_test_plugin LANGUAGES CXX)

# Network Test plugin DLL - legitimate network stress testing tool
add_library(network_test_plugin SHARED
    network_test_plugin.cpp
)

target_include_directories(network_test_plugin PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/commands
    ${CMAKE_SOURCE_DIR}/core
    ${CMAKE_SOURCE_DIR}/helpers
)

# Build as plugin
target_compile_definitions(network_test_plugin PRIVATE GRIM_BUILD_PLUGIN)

# Link against GRIM export lib for command registration
target_link_libraries(network_test_plugin PRIVATE GRIM_EXPORT)
add_dependencies(network_test_plugin GRIM)

# Link with ws2_32 on Windows for socket support
if(WIN32)
    target_link_libraries(network_test_plugin PRIVATE ws2_32)
endif()

set_target_properties(network_test_plugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/plugins"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/plugins"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/plugins"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/plugins"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/plugins"
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# GRIM-text Training Control Server Build
cmake_minimum_required(VERSION 3.22)
project(TrainingControlServer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set prefix path to vcpkg_installed for package discovery
if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-windows")
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../vcpkg_installed/x64-windows" CACHE PATH "vcpkg installed packages")
endif()

# Find dependencies
find_package(flatbuffers CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_path(HTTPLIB_INCLUDE_DIR NAMES httplib.h)

# Control server executable
add_executable(training_control_server
    training_control_server.cpp
)

target_include_directories(training_control_server PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HTTPLIB_INCLUDE_DIR}
)

target_link_libraries(training_control_server PRIVATE
    flatbuffers::flatbuffers
    nlohmann_json::nlohmann_json
)

if(WIN32)
    target_link_libraries(training_control_server PRIVATE ws2_32)
endif()

# Copy FlatBuffer schema to build output for reference
add_custom_command(TARGET training_control_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/training_control.fbs
        $<TARGET_FILE_DIR:training_control_server>/training_control.fbs
    COMMENT "Copying FlatBuffer schema to output directory"
)

cmake_minimum_required(VERSION 3.22)
project(tool_inspection_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED
    HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
add_library(tool_inspection_plugin SHARED tool_inspection_plugin.cpp inspection_commands.cpp)
grim_configure_plugin(tool_inspection_plugin)
target_link_libraries(tool_inspection_plugin PRIVATE nlohmann_json::nlohmann_json)

cmake_minimum_required(VERSION 3.22)
project(system_info_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED
    HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
add_library(system_info_plugin SHARED system_info_plugin.cpp system_info_command.cpp)
grim_configure_plugin(system_info_plugin)
target_link_libraries(system_info_plugin PRIVATE nlohmann_json::nlohmann_json)

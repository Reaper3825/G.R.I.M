cmake_minimum_required(VERSION 3.22)
project(aliases_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED
    HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
add_library(aliases_plugin SHARED aliases_plugin.cpp alias_commands.cpp)
grim_configure_plugin(aliases_plugin)
target_link_libraries(aliases_plugin PRIVATE nlohmann_json::nlohmann_json)

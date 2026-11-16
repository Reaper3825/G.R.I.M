cmake_minimum_required(VERSION 3.22)
project(utility_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED
    HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
add_library(utility_plugin SHARED utility_plugin.cpp)
target_link_libraries(utility_plugin PRIVATE nlohmann_json::nlohmann_json)
grim_configure_plugin(utility_plugin)

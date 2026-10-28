cmake_minimum_required(VERSION 3.22)
project(osint_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED
    HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
find_package(Threads REQUIRED)
add_library(osint_plugin SHARED
    "${GRIM_PLUGIN_ROOT}/commands/commands_osint.cpp"
    "${GRIM_PLUGIN_ROOT}/plugins/osint_plugin.cpp"
    "${GRIM_PLUGIN_ROOT}/external_collector/osit.cpp"
    "${GRIM_PLUGIN_ROOT}/external_collector/osit_secrets.cpp")
grim_configure_plugin(osint_plugin)
target_link_libraries(osint_plugin PRIVATE nlohmann_json::nlohmann_json Threads::Threads)

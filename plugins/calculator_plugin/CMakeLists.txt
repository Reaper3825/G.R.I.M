cmake_minimum_required(VERSION 3.22)
project(calculator_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED
    HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
add_library(calculator_plugin SHARED calculator_plugin.cpp)
target_link_libraries(calculator_plugin PRIVATE nlohmann_json::nlohmann_json)
grim_configure_plugin(calculator_plugin
    TOOL_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/calculator_plugin.tools.json")

option(GRIM_CALCULATOR_TESTS "Build focused calculator plugin/registry tests" OFF)
if(GRIM_CALCULATOR_TESTS)
    enable_testing()
    add_executable(calculator_plugin_tests
        "${GRIM_PLUGIN_ROOT}/tests/calculator_plugin_tests.cpp"
        "${GRIM_PLUGIN_ROOT}/MMO/Core/Registries/ProviderRegistry.cpp"
        "${GRIM_PLUGIN_ROOT}/MMO/Core/ToolResultJson.cpp"
        "${GRIM_PLUGIN_ROOT}/MMO/Core/Registries/ToolRegistry.cpp")
    target_compile_features(calculator_plugin_tests PRIVATE cxx_std_20)
    target_include_directories(calculator_plugin_tests PRIVATE "${GRIM_PLUGIN_ROOT}"
        "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/include")
    target_compile_definitions(calculator_plugin_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN GRIM_PROVIDER_REGISTRY_TESTS)
    target_link_libraries(calculator_plugin_tests PRIVATE nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
    add_dependencies(calculator_plugin_tests calculator_plugin)
    add_test(NAME calculator_plugin_tests COMMAND calculator_plugin_tests $<TARGET_FILE:calculator_plugin>)
endif()

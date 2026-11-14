cmake_minimum_required(VERSION 3.22)
project(osint_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED
    HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
find_package(Threads REQUIRED)
add_library(osint_plugin SHARED process.cpp
    "${CMAKE_CURRENT_LIST_DIR}/commands_osint.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/osint_plugin.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/osit.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/osit_secrets.cpp")
grim_configure_plugin(osint_plugin)
target_link_libraries(osint_plugin PRIVATE nlohmann_json::nlohmann_json Threads::Threads)

option(GRIM_OSINT_PROVIDER_TESTS "Build isolated OSINT provider tests" OFF)
if(GRIM_OSINT_PROVIDER_TESTS)
 enable_testing()
 add_executable(osint_provider_tests "${GRIM_PLUGIN_ROOT}/tests/osint_provider_tests.cpp" osint_plugin.cpp commands_osint.cpp osit.cpp osit_secrets.cpp "${GRIM_PLUGIN_ROOT}/plugins/plugin_logging.cpp")
 target_compile_features(osint_provider_tests PRIVATE cxx_std_17)
 target_compile_definitions(osint_provider_tests PRIVATE GRIM_BUILD_PLUGIN NOMINMAX WIN32_LEAN_AND_MEAN)
 target_include_directories(osint_provider_tests PRIVATE "${GRIM_PLUGIN_ROOT}" "${GRIM_PLUGIN_ROOT}/core" "${GRIM_PLUGIN_ROOT}/commands")
 target_link_libraries(osint_provider_tests PRIVATE nlohmann_json::nlohmann_json Threads::Threads)
 if(MSVC)
  target_compile_options(osint_provider_tests PRIVATE /utf-8)
 endif()
 add_test(NAME osint_provider COMMAND osint_provider_tests)
 add_executable(osint_process_tests "${GRIM_PLUGIN_ROOT}/tests/osint_process_tests.cpp" process.cpp)
 target_compile_features(osint_process_tests PRIVATE cxx_std_17)
 target_include_directories(osint_process_tests PRIVATE "${GRIM_PLUGIN_ROOT}")
 target_compile_definitions(osint_process_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
 target_link_libraries(osint_process_tests PRIVATE Threads::Threads)
 add_test(NAME osint_process COMMAND osint_process_tests)
endif()

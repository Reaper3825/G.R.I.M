cmake_minimum_required(VERSION 3.22)
project(memory_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
find_package(flatbuffers CONFIG REQUIRED HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/flatbuffers")

if(NOT TARGET generate_grim_unified_memory_schema)
    set(GRIM_UNIFIED_MEMORY_SCHEMA "${GRIM_PLUGIN_ROOT}/memory/unified_memory.fbs")
    set(GRIM_UNIFIED_MEMORY_GENERATED_ROOT "${CMAKE_CURRENT_BINARY_DIR}/generated")
    set(GRIM_UNIFIED_MEMORY_GENERATED_DIR "${GRIM_UNIFIED_MEMORY_GENERATED_ROOT}/grim_memory")
    set(GRIM_UNIFIED_MEMORY_GENERATED_HEADER "${GRIM_UNIFIED_MEMORY_GENERATED_DIR}/unified_memory_generated.h")
    add_custom_command(OUTPUT "${GRIM_UNIFIED_MEMORY_GENERATED_HEADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${GRIM_UNIFIED_MEMORY_GENERATED_DIR}"
        COMMAND $<TARGET_FILE:flatbuffers::flatc> --cpp --gen-object-api
            -o "${GRIM_UNIFIED_MEMORY_GENERATED_DIR}" "${GRIM_UNIFIED_MEMORY_SCHEMA}"
        DEPENDS flatbuffers::flatc "${GRIM_UNIFIED_MEMORY_SCHEMA}"
        VERBATIM)
    add_custom_target(generate_grim_unified_memory_schema DEPENDS "${GRIM_UNIFIED_MEMORY_GENERATED_HEADER}")
endif()

add_library(memory_plugin SHARED memory_plugin.cpp memory_commands.cpp
    "${GRIM_PLUGIN_ROOT}/memory/unified_memory_storage.cpp")
add_dependencies(memory_plugin generate_grim_unified_memory_schema)
target_include_directories(memory_plugin PRIVATE "${GRIM_UNIFIED_MEMORY_GENERATED_ROOT}")
grim_configure_plugin(memory_plugin
    TOOL_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/memory_plugin.tools.json")
target_link_libraries(memory_plugin PRIVATE nlohmann_json::nlohmann_json flatbuffers::flatbuffers)

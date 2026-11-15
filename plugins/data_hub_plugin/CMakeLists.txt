cmake_minimum_required(VERSION 3.22)
project(data_hub_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
add_library(data_hub_plugin SHARED data_hub_plugin.cpp
    data_hub/ui_data_hub_core.cpp
    data_hub/ui_data_hub_home_tab.cpp
    data_hub/ui_data_hub_sources_tab.cpp
    data_hub/ui_data_hub_huggingface_tab.cpp
    data_hub/ui_data_hub_structurer_tab.cpp
    data_hub/ui_data_hub_generator_tab.cpp
    data_hub/ui_data_hub_curriculum_tab.cpp
    data_hub/ui_data_hub_jobs.cpp)
grim_configure_plugin(data_hub_plugin)
target_compile_features(data_hub_plugin PRIVATE cxx_std_20)
target_include_directories(data_hub_plugin PRIVATE "${GRIM_PLUGIN_ROOT}/ui")
target_link_libraries(data_hub_plugin PRIVATE nlohmann_json::nlohmann_json)
if(TARGET GRIM)
    target_link_libraries(data_hub_plugin PRIVATE GRIM_EXPORT)
elseif(WIN32)
    set(GRIM_HOST_IMPORT_LIBRARY "" CACHE FILEPATH "Matching GRIM host import library for native UI plugins")
    if(NOT EXISTS "${GRIM_HOST_IMPORT_LIBRARY}")
        message(FATAL_ERROR "data_hub_plugin requires GRIM_HOST_IMPORT_LIBRARY from a matching host build")
    endif()
    target_link_libraries(data_hub_plugin PRIVATE "${GRIM_HOST_IMPORT_LIBRARY}")
elseif(APPLE)
    target_link_options(data_hub_plugin PRIVATE "-undefined" "dynamic_lookup")
endif()

cmake_minimum_required(VERSION 3.22)
project(model_workspace_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
find_package(nlohmann_json CONFIG REQUIRED HINTS "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share/nlohmann_json")
add_library(model_workspace_plugin SHARED model_workspace_plugin.cpp ui_training_panel.cpp
    training_panel/ui_training_panel_home_tab.cpp
    training_panel/ui_training_panel_knowledge_gaps_tab.cpp
    training_panel/ui_training_panel_tool_gaps_tab.cpp
    training_panel/ui_training_panel_training_tab.cpp
    training_panel/ui_training_panel_tokenizer_tab.cpp)
grim_configure_plugin(model_workspace_plugin
    TOOL_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/model_workspace_plugin.tools.json")
target_compile_features(model_workspace_plugin PRIVATE cxx_std_20)
target_include_directories(model_workspace_plugin PRIVATE "${GRIM_PLUGIN_ROOT}/ui")
target_link_libraries(model_workspace_plugin PRIVATE nlohmann_json::nlohmann_json)
# Import the one host UI/configuration implementation; never compile a second copy.
if(TARGET GRIM)
    target_link_libraries(model_workspace_plugin PRIVATE GRIM_EXPORT)
elseif(WIN32)
    set(GRIM_HOST_IMPORT_LIBRARY "" CACHE FILEPATH "Matching GRIM host import library for native UI plugins")
    if(NOT EXISTS "${GRIM_HOST_IMPORT_LIBRARY}")
        message(FATAL_ERROR "model_workspace_plugin requires GRIM_HOST_IMPORT_LIBRARY from a matching host build")
    endif()
    target_link_libraries(model_workspace_plugin PRIVATE "${GRIM_HOST_IMPORT_LIBRARY}")
elseif(APPLE)
    target_link_options(model_workspace_plugin PRIVATE "-undefined" "dynamic_lookup")
endif()

if(WIN32)
    target_link_libraries(model_workspace_plugin PRIVATE ws2_32 crypt32)
endif()

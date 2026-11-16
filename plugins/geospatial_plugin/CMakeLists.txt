cmake_minimum_required(VERSION 3.22)
project(geospatial_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")

if(WIN32 AND NOT TARGET GRIM)
    list(APPEND CMAKE_PREFIX_PATH "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows")
endif()
set(_vcpkg_share "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows/share")
find_package(cesium-native CONFIG REQUIRED HINTS "${_vcpkg_share}/cesium-native")
find_package(nlohmann_json CONFIG REQUIRED HINTS "${_vcpkg_share}/nlohmann_json")
find_package(spdlog CONFIG REQUIRED HINTS "${_vcpkg_share}/spdlog")
find_package(Threads REQUIRED)

add_library(geospatial_plugin SHARED
    geospatial_plugin.cpp ui_geospatial_panel.cpp ui_camera.cpp
    geospatial/geospatial_runtime.cpp geospatial/geospatial_render_aggregation.cpp
    geospatial/geospatial_geometry.cpp geospatial/geospatial_catalog.cpp
    geospatial/geospatial_group.cpp geospatial/geospatial_point.cpp geospatial/geospatial_area.cpp
    geospatial/cesium_bgfx_render_adapter.cpp geospatial/cesium_bgfx_shaders.cpp
    stb_image.cpp)
grim_configure_plugin(geospatial_plugin DEPLOY_RUNTIME_DLLS
    TOOL_MANIFEST "${CMAKE_CURRENT_SOURCE_DIR}/geospatial_plugin.tools.json")
target_compile_features(geospatial_plugin PRIVATE cxx_std_20)
if(NOT TARGET GRIM)
    set(GRIM_ROOT_DIR "${GRIM_PLUGIN_ROOT}" CACHE PATH "Root of GRIM project")
    target_compile_definitions(geospatial_plugin PRIVATE
        GRIM_ROOT_DIR="${GRIM_ROOT_DIR}"
        $<$<CONFIG:Debug>:BX_CONFIG_DEBUG=1>
        $<$<NOT:$<CONFIG:Debug>>:BX_CONFIG_DEBUG=0>)
endif()
if(MSVC)
    # The installed Async++ headers still use std::result_of.
    target_compile_options(geospatial_plugin PRIVATE /Zc:__cplusplus /Zc:preprocessor)
    target_compile_definitions(geospatial_plugin PRIVATE _HAS_DEPRECATED_RESULT_OF=1)
endif()
target_compile_definitions(geospatial_plugin PRIVATE BGFX_PLATFORM_SUPPORTS_WGSL=0)
target_include_directories(geospatial_plugin PRIVATE
    "${GRIM_PLUGIN_ROOT}/ui"
    "${GRIM_PLUGIN_ROOT}/external/bgfx.cmake/bgfx/include"
    "${GRIM_PLUGIN_ROOT}/external/bgfx.cmake/bx/include"
    "${GRIM_PLUGIN_ROOT}/external/bgfx.cmake/bimg/3rdparty")
target_link_libraries(geospatial_plugin PRIVATE
    Cesium3DTilesSelection CesiumRasterOverlays CesiumIonClient CesiumClientCommon
    CesiumCurl CesiumGeospatial CesiumGltf nlohmann_json::nlohmann_json spdlog::spdlog Threads::Threads)

# Native UI and renderer calls import the same implementations used by the host.
if(TARGET GRIM)
    target_link_libraries(geospatial_plugin PRIVATE GRIM_EXPORT)
elseif(WIN32)
    set(GRIM_HOST_IMPORT_LIBRARY "" CACHE FILEPATH "Matching GRIM host import library for native UI plugins")
    if(NOT EXISTS "${GRIM_HOST_IMPORT_LIBRARY}")
        message(FATAL_ERROR "geospatial_plugin requires GRIM_HOST_IMPORT_LIBRARY from a matching host build")
    endif()
    target_link_libraries(geospatial_plugin PRIVATE "${GRIM_HOST_IMPORT_LIBRARY}")
elseif(APPLE)
    target_link_options(geospatial_plugin PRIVATE "-undefined" "dynamic_lookup")
endif()

set(GRIM_SHADER_ROOT "${GRIM_PLUGIN_ROOT}")
set(GRIM_SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/shaders")
include("${GRIM_PLUGIN_ROOT}/cmake/CompileShaders.cmake")
if(BGFX_SHADERC)
    grim_compile_shader(NAME vs_cesium_terrain TYPE VERTEX
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vs_cesium_terrain.sc"
        VARYING_DEF "${CMAKE_CURRENT_SOURCE_DIR}/shaders/varying_cesium_terrain.def.sc")
    grim_compile_shader(NAME fs_cesium_terrain TYPE FRAGMENT
        SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/fs_cesium_terrain.sc"
        VARYING_DEF "${CMAKE_CURRENT_SOURCE_DIR}/shaders/varying_cesium_terrain.def.sc")
    add_custom_target(geospatial_shaders DEPENDS
        ${GRIM_SHADER_OUTPUTS_vs_cesium_terrain} ${GRIM_SHADER_OUTPUTS_fs_cesium_terrain})
    add_dependencies(geospatial_plugin geospatial_shaders)
    target_include_directories(geospatial_plugin PRIVATE "${GRIM_SHADER_OUTPUT_DIR}")
endif()

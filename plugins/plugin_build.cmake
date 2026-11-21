# Shared by standalone builds and optional host subdirectories.
get_filename_component(GRIM_PLUGIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GRIM_PLUGIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/plugins" CACHE PATH "Plugin library output folder")
set(_grim_deploy_plugins_default OFF)
if(TARGET GRIM)
    set(_grim_deploy_plugins_default ON)
endif()
option(GRIM_DEPLOY_PLUGINS "Publish built plugins into GRIM's watched folder" ${_grim_deploy_plugins_default})
# Upgrade build trees that still have the old in-source default cached.
if(GRIM_PLUGIN_DEPLOY_DIR STREQUAL "${GRIM_PLUGIN_ROOT}/plugins")
    unset(GRIM_PLUGIN_DEPLOY_DIR CACHE)
endif()
set(GRIM_PLUGIN_DEPLOY_DIR "${GRIM_PLUGIN_ROOT}/deployed_plugins" CACHE PATH "Watched plugin deployment folder")
function(grim_configure_plugin target)
    cmake_parse_arguments(PLUGIN "DEPLOY_RUNTIME_DLLS" "TOOL_MANIFEST" "" ${ARGN})
    if(PLUGIN_TOOL_MANIFEST)
        add_custom_target(manifest_${target}
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${PLUGIN_TOOL_MANIFEST}"
                "$<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_PREFIX:${target}>$<TARGET_FILE_BASE_NAME:${target}>.tools.json"
            DEPENDS "${PLUGIN_TOOL_MANIFEST}"
            VERBATIM)
        add_dependencies(${target} manifest_${target})
    endif()
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE GRIM_BUILD_PLUGIN NOMINMAX WIN32_LEAN_AND_MEAN)
    target_include_directories(${target} PRIVATE
        "${GRIM_PLUGIN_ROOT}" "${GRIM_PLUGIN_ROOT}/commands"
        "${GRIM_PLUGIN_ROOT}/core" "${GRIM_PLUGIN_ROOT}/helpers")
    target_sources(${target} PRIVATE "${GRIM_PLUGIN_ROOT}/plugins/plugin_logging.cpp")
    set_target_properties(${target} PROPERTIES
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES
        RUNTIME_OUTPUT_DIRECTORY "${GRIM_PLUGIN_OUTPUT_DIR}"
        LIBRARY_OUTPUT_DIRECTORY "${GRIM_PLUGIN_OUTPUT_DIR}")
    foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        set_target_properties(${target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_${config} "${GRIM_PLUGIN_OUTPUT_DIR}"
            LIBRARY_OUTPUT_DIRECTORY_${config} "${GRIM_PLUGIN_OUTPUT_DIR}")
    endforeach()
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(${target} PRIVATE -fno-gnu-unique)
    endif()
    if(GRIM_DEPLOY_PLUGINS)
        set(runtime_argument)
        if(WIN32 AND PLUGIN_DEPLOY_RUNTIME_DLLS)
            set(runtime_argument "-DPLUGIN_RUNTIME_DLLS=$<TARGET_RUNTIME_DLLS:${target}>")
        endif()
        # Run even when linking is up to date: repair stale/missing deployments.
        # Depend on the library so publication only sees a completed output.
        add_custom_target(deploy_${target} ALL
            COMMAND "${CMAKE_COMMAND}"
                "-DPLUGIN_FILE=$<TARGET_FILE:${target}>"
                "-DDEPLOY_DIR=${GRIM_PLUGIN_DEPLOY_DIR}"
                "-DPLUGIN_MANIFEST=${PLUGIN_TOOL_MANIFEST}"
                ${runtime_argument}
                -P "${GRIM_PLUGIN_ROOT}/plugins/deploy_plugin.cmake"
            DEPENDS ${target}
            COMMENT "Checking deployment of ${target}"
            VERBATIM)
        if(TARGET GRIM AND NOT target STREQUAL "schedule_plugin" AND NOT target STREQUAL "console_plugin" AND NOT target STREQUAL "settings_plugin" AND NOT target STREQUAL "model_workspace_plugin" AND NOT target STREQUAL "storage_plugin" AND NOT target STREQUAL "data_hub_plugin" AND NOT target STREQUAL "geospatial_plugin")
            add_dependencies(GRIM deploy_${target})
        endif()
    endif()
endfunction()

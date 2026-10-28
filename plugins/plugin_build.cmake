# Shared by standalone builds and optional host subdirectories.
get_filename_component(GRIM_PLUGIN_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(GRIM_PLUGIN_OUTPUT_DIR "${CMAKE_BINARY_DIR}/plugins" CACHE PATH "Plugin library output folder")
function(grim_configure_plugin target)
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
endfunction()

# =========================================================
# GRIM Shader Compilation (bgfx shaderc)
# =========================================================
# Compiles .sc shaders to embedded C headers (.bin.h) using
# bgfx's shaderc tool, then assembles them into a custom
# target that GRIM depends on.
#
# Requires: shaderc built from external/bgfx.cmake.
# Build it once with:
#   cd external/bgfx.cmake/build/cmake
#   cmake ../.. -DBGFX_BUILD_TOOLS=ON -DBGFX_BUILD_TOOLS_SHADER=ON 
#               -DBGFX_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
#   cmake --build . --target shaderc --config Release
# =========================================================

set(BGFX_SHADER_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/bgfx/src")
set(GRIM_SHADER_DIR "${CMAKE_SOURCE_DIR}/resources/shaders")
set(GRIM_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated/shaders")

# ---- Locate shaderc ----
find_program(BGFX_SHADERC
    NAMES shaderc
    HINTS
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/cmake/bgfx"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/cmake/bgfx/Release"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake/cmake/bgfx/Debug"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/Release"
        "${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/Debug"
    DOC "Path to bgfx shaderc compiler"
)

if(NOT BGFX_SHADERC)
    message(WARNING
        "[GRIM] shaderc not found — popup model shaders will NOT be compiled.\n"
        "  Build shaderc with:\n"
        "    cd ${CMAKE_SOURCE_DIR}/external/bgfx.cmake/build/cmake\n"
        "    cmake ../.. -DBGFX_BUILD_TOOLS=ON -DBGFX_BUILD_TOOLS_SHADER=ON "
        "-DBGFX_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release\n"
        "    cmake --build . --target shaderc --config Release\n"
        "  Then re-run cmake for GRIM."
    )
    return()
endif()

message(STATUS "[GRIM] Found shaderc: ${BGFX_SHADERC}")

# ---- Platform / profiles ----
if(APPLE AND NOT IOS)
    set(_SHADERC_PLATFORM "osx")
    set(_SHADERC_VS_PROFILES  "120" "300_es" "spirv" "metal" "wgsl")
    set(_SHADERC_FS_PROFILES  "120" "300_es" "spirv" "metal" "wgsl")
elseif(IOS)
    set(_SHADERC_PLATFORM "ios")
    set(_SHADERC_VS_PROFILES  "300_es" "spirv" "metal")
    set(_SHADERC_FS_PROFILES  "300_es" "spirv" "metal")
elseif(WIN32)
    set(_SHADERC_PLATFORM "windows")
    set(_SHADERC_VS_PROFILES  "120" "300_es" "spirv" "s_4_0" "s_5_0" "wgsl")
    set(_SHADERC_FS_PROFILES  "120" "300_es" "spirv" "s_4_0" "s_5_0" "wgsl")
elseif(UNIX)
    set(_SHADERC_PLATFORM "linux")
    set(_SHADERC_VS_PROFILES  "120" "300_es" "spirv" "wgsl")
    set(_SHADERC_FS_PROFILES  "120" "300_es" "spirv" "wgsl")
endif()

# Map profile names to the extension suffixes that embedded_shader.h expects
function(_profile_to_ext PROFILE OUT_VAR)
    if(PROFILE STREQUAL "120")
        set(${OUT_VAR} "glsl" PARENT_SCOPE)
    elseif(PROFILE STREQUAL "300_es")
        set(${OUT_VAR} "essl" PARENT_SCOPE)
    elseif(PROFILE STREQUAL "spirv")
        set(${OUT_VAR} "spv" PARENT_SCOPE)
    elseif(PROFILE STREQUAL "metal")
        set(${OUT_VAR} "mtl" PARENT_SCOPE)
    elseif(PROFILE STREQUAL "s_4_0")
        set(${OUT_VAR} "dx10" PARENT_SCOPE)
    elseif(PROFILE STREQUAL "s_5_0")
        set(${OUT_VAR} "dx11" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "${PROFILE}" PARENT_SCOPE)
    endif()
endfunction()

# Map profile to the platform shaderc expects (spirv always uses "linux")
function(_profile_to_platform PROFILE BASE_PLATFORM OUT_VAR)
    if(PROFILE STREQUAL "spirv" OR PROFILE STREQUAL "wgsl")
        set(${OUT_VAR} "linux" PARENT_SCOPE)
    else()
        set(${OUT_VAR} "${BASE_PLATFORM}" PARENT_SCOPE)
    endif()
endfunction()

# ---- Compile one shader for all profiles → one .bin.h per profile ----
# Each .bin.h contains a C array named <shader_name>_<profile_ext>
# All per-profile .bin.h files are then combined into one umbrella header.
function(grim_compile_shader)
    cmake_parse_arguments(ARGS "" "NAME;TYPE;SOURCE;VARYING_DEF" "" ${ARGN})

    if(ARGS_TYPE STREQUAL "VERTEX")
        set(_type "vertex")
        set(_profiles ${_SHADERC_VS_PROFILES})
    elseif(ARGS_TYPE STREQUAL "FRAGMENT")
        set(_type "fragment")
        set(_profiles ${_SHADERC_FS_PROFILES})
    else()
        message(FATAL_ERROR "grim_compile_shader: TYPE must be VERTEX or FRAGMENT")
    endif()

    set(_all_outputs "")
    set(_include_lines "")

    foreach(_profile ${_profiles})
        _profile_to_ext(${_profile} _ext)
        _profile_to_platform(${_profile} ${_SHADERC_PLATFORM} _plat)

        set(_bin2c_name "${ARGS_NAME}_${_ext}")
        set(_output "${GRIM_SHADER_OUTPUT_DIR}/${ARGS_NAME}_${_ext}.bin.h")

        add_custom_command(
            OUTPUT "${_output}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${GRIM_SHADER_OUTPUT_DIR}"
            COMMAND "${BGFX_SHADERC}"
                -f "${ARGS_SOURCE}"
                -o "${_output}"
                --type "${_type}"
                --platform "${_plat}"
                -p "${_profile}"
                --varyingdef "${ARGS_VARYING_DEF}"
                -i "${BGFX_SHADER_INCLUDE_PATH}"
                --bin2c "${_bin2c_name}"
            DEPENDS "${ARGS_SOURCE}" "${ARGS_VARYING_DEF}"
            COMMENT "Compiling shader ${ARGS_NAME} [${_ext}]"
            VERBATIM
        )

        list(APPEND _all_outputs "${_output}")
        list(APPEND _include_lines "#include \"${ARGS_NAME}_${_ext}.bin.h\"")
    endforeach()

    # Write an umbrella header that includes all profile variants
    set(_umbrella "${GRIM_SHADER_OUTPUT_DIR}/${ARGS_NAME}.bin.h")
    string(REPLACE ";" "\n" _includes_content "${_include_lines}")
    file(WRITE "${_umbrella}.in"
        "// Auto-generated — do not edit\n"
        "// Compiled from ${ARGS_SOURCE}\n"
        "#pragma once\n"
        "${_includes_content}\n"
    )
    add_custom_command(
        OUTPUT "${_umbrella}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_umbrella}.in" "${_umbrella}"
        DEPENDS ${_all_outputs}
        COMMENT "Generating umbrella header ${ARGS_NAME}.bin.h"
        VERBATIM
    )

    # Export outputs to parent scope
    set(GRIM_SHADER_OUTPUTS_${ARGS_NAME} ${_all_outputs} "${_umbrella}" PARENT_SCOPE)
endfunction()

# ---- Compile popup model shaders ----
grim_compile_shader(
    NAME vs_popup_model
    TYPE VERTEX
    SOURCE "${GRIM_SHADER_DIR}/vs_popup_model.sc"
    VARYING_DEF "${GRIM_SHADER_DIR}/varying_popup_model.def.sc"
)

grim_compile_shader(
    NAME fs_popup_model
    TYPE FRAGMENT
    SOURCE "${GRIM_SHADER_DIR}/fs_popup_model.sc"
    VARYING_DEF "${GRIM_SHADER_DIR}/varying_popup_model.def.sc"
)

# ---- Custom target so GRIM depends on shader compilation ----
add_custom_target(grim_shaders ALL
    DEPENDS
        ${GRIM_SHADER_OUTPUTS_vs_popup_model}
        ${GRIM_SHADER_OUTPUTS_fs_popup_model}
)

# Make the generated headers includable
target_include_directories(GRIM PRIVATE "${GRIM_SHADER_OUTPUT_DIR}")
add_dependencies(GRIM grim_shaders)

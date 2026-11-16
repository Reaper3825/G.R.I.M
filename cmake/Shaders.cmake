set(GRIM_SHADER_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(GRIM_SHADER_DIR "${GRIM_SHADER_ROOT}/resources/shaders")
set(GRIM_SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/generated/shaders")
include("${CMAKE_CURRENT_LIST_DIR}/CompileShaders.cmake")
if(NOT BGFX_SHADERC)
    return()
endif()
target_compile_definitions(GRIM PRIVATE BGFX_PLATFORM_SUPPORTS_WGSL=0)

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
grim_compile_shader(
    NAME vs_observatory TYPE VERTEX
    SOURCE "${GRIM_SHADER_DIR}/vs_observatory.sc"
    VARYING_DEF "${GRIM_SHADER_DIR}/varying_observatory.def.sc"
)
grim_compile_shader(
    NAME fs_observatory TYPE FRAGMENT
    SOURCE "${GRIM_SHADER_DIR}/fs_observatory.sc"
    VARYING_DEF "${GRIM_SHADER_DIR}/varying_observatory.def.sc"
)
add_custom_target(grim_shaders ALL
    DEPENDS
        ${GRIM_SHADER_OUTPUTS_vs_popup_model}
        ${GRIM_SHADER_OUTPUTS_fs_popup_model}
        ${GRIM_SHADER_OUTPUTS_vs_observatory}
        ${GRIM_SHADER_OUTPUTS_fs_observatory}
)

# Make the generated headers includable
target_include_directories(GRIM PRIVATE "${GRIM_SHADER_OUTPUT_DIR}")
add_dependencies(GRIM grim_shaders)

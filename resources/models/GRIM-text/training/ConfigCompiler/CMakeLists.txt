cmake_minimum_required(VERSION 3.20)

# This directory is both a true host-only project and a TrainingLoop
# subdirectory. Standalone configuration therefore never enables CUDA.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(GRIMModelConfigCompiler LANGUAGES CXX)
    find_package(flatbuffers CONFIG REQUIRED)
    find_package(nlohmann_json CONFIG REQUIRED)
endif()

if(NOT TARGET flatbuffers::flatc)
    message(FATAL_ERROR
        "compile_model_config requires the flatc executable target. "
        "Configure with the FlatBuffers package/toolchain, not headers alone.")
endif()

set(GRIM_COMPILED_HYPERPARAMETERS_SCHEMA
    ${CMAKE_CURRENT_SOURCE_DIR}/../schemas/grim_compiled_hyperparameters.fbs
)
set(GRIM_COMPILED_HYPERPARAMETERS_GENERATED_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
set(GRIM_COMPILED_HYPERPARAMETERS_GENERATED_HEADER
    ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_DIR}/grim_compiled_hyperparameters_generated.h
)

add_custom_command(
    OUTPUT ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory
            ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_DIR}
    COMMAND $<TARGET_FILE:flatbuffers::flatc>
            -o ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_DIR}
            -c --gen-mutable ${GRIM_COMPILED_HYPERPARAMETERS_SCHEMA}
    DEPENDS
        flatbuffers::flatc
        ${GRIM_COMPILED_HYPERPARAMETERS_SCHEMA}
    COMMENT "Generating GRIM compiled model hyperparameter FlatBuffer header"
    VERBATIM
)
add_custom_target(generate_grim_compiled_hyperparameters_schema
    DEPENDS ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_HEADER}
)

# Low-level runtime reader for the compiler's immutable model artifact. Keeping
# this in the CXX-only compiler project lets TrainingLoop consume the same wire
# contract without adding a CUDA or trainer dependency to configuration loading.
add_library(grim_compiled_model_config STATIC
    ../../Shared/ModelConfig/CompiledModelConfig.cpp
    ../../Shared/ModelConfig/CompiledModelConfig.hpp
    ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_HEADER}
)
add_dependencies(grim_compiled_model_config
    generate_grim_compiled_hyperparameters_schema
)
target_include_directories(grim_compiled_model_config
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../Shared/ModelConfig
    PRIVATE
        ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_DIR}
)
target_link_libraries(grim_compiled_model_config PUBLIC
    flatbuffers::flatbuffers
    nlohmann_json::nlohmann_json
)
target_compile_features(grim_compiled_model_config PUBLIC cxx_std_17)

add_executable(compile_model_config
    compile_model_config.cpp
    ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_HEADER}
)
add_dependencies(compile_model_config
    generate_grim_compiled_hyperparameters_schema
)
target_include_directories(compile_model_config PRIVATE
    ${GRIM_COMPILED_HYPERPARAMETERS_GENERATED_DIR}
)
target_link_libraries(compile_model_config PRIVATE
    flatbuffers::flatbuffers
    nlohmann_json::nlohmann_json
)
target_compile_features(compile_model_config PRIVATE cxx_std_17)

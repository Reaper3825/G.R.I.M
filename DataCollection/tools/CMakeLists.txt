cmake_minimum_required(VERSION 3.20)

project(atom_curriculum_builder LANGUAGES CXX)

find_package(flatbuffers CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

if(NOT TARGET flatbuffers::flatc)
    message(FATAL_ERROR
        "atom_curriculum_builder requires the flatc executable target.")
endif()

set(CONCEPT_BLOCK_SCHEMA ${CMAKE_CURRENT_SOURCE_DIR}/../concept_block.fbs)
set(CONCEPT_BLOCK_GENERATED_ROOT ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(CONCEPT_BLOCK_GENERATED_DIR ${CONCEPT_BLOCK_GENERATED_ROOT}/grim_concept)
set(CONCEPT_BLOCK_GENERATED_HEADER
    ${CONCEPT_BLOCK_GENERATED_DIR}/concept_block_generated.h)

add_custom_command(
    OUTPUT ${CONCEPT_BLOCK_GENERATED_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CONCEPT_BLOCK_GENERATED_DIR}
    COMMAND $<TARGET_FILE:flatbuffers::flatc>
            --cpp --scoped-enums
            -o ${CONCEPT_BLOCK_GENERATED_DIR}
            ${CONCEPT_BLOCK_SCHEMA}
    DEPENDS flatbuffers::flatc ${CONCEPT_BLOCK_SCHEMA}
    COMMENT "Generating ConceptBlock FlatBuffer schema"
    VERBATIM
)

add_executable(atom_curriculum_builder
    atom_curriculum_builder.cpp
    ../io/concept_block_io_flatbuffer.cpp
    ${CONCEPT_BLOCK_GENERATED_HEADER}
)

target_include_directories(atom_curriculum_builder PRIVATE
    ..
    ${CONCEPT_BLOCK_GENERATED_ROOT}
)
target_link_libraries(atom_curriculum_builder PRIVATE
    flatbuffers::flatbuffers
    nlohmann_json::nlohmann_json
)
target_compile_features(atom_curriculum_builder PRIVATE cxx_std_20)

if(WIN32)
    target_compile_definitions(atom_curriculum_builder PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
endif()

if(MSVC)
    target_compile_options(atom_curriculum_builder PRIVATE /utf-8 /W4)
endif()

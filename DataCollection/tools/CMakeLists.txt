cmake_minimum_required(VERSION 3.20)

project(atom_curriculum_builder LANGUAGES CXX)

find_package(flatbuffers CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(atom_curriculum_builder
    atom_curriculum_builder.cpp
    ../io/concept_block_io_flatbuffer.cpp
)

target_include_directories(atom_curriculum_builder PRIVATE ..)
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

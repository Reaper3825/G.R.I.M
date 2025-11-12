# DataCollection CMakeLists.txt
# Web collector, verifier, preprocessor, and data splitter pipeline
cmake_minimum_required(VERSION 3.22)

# vcpkg manifest mode - uses CMAKE_PREFIX_PATH or environment VCPKG_ROOT
# Set prefix path to vcpkg_installed for package discovery
if(NOT DEFINED CMAKE_PREFIX_PATH AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg_installed/x64-windows")
    set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../../vcpkg_installed/x64-windows" CACHE PATH "vcpkg installed packages")
endif()

project(GRIM_DataCollection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Disable Windows min/max macros to prevent conflicts with std::min/std::max
add_compile_definitions(NOMINMAX)

# Find required packages
find_package(CURL REQUIRED)  # Uses FindCURL.cmake (not CONFIG)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(flatbuffers CONFIG REQUIRED)
find_package(unofficial-gumbo CONFIG REQUIRED)

# Compiler flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /O2 /DNDEBUG")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Header-only high-performance components
set(DATA_COLLECTION_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/streaming_downloader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/async_crawler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/web_collector.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/verifier.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_preprocessor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/data_splitter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/html_extractor.hpp
)

# UNIFIED DATA PIPELINE - Single executable for collect/verify/merge
add_executable(grim_data_pipeline 
    grim_data_pipeline.cpp
    verifier.cpp
    ${DATA_COLLECTION_HEADERS}
)

target_compile_definitions(grim_data_pipeline PRIVATE HAVE_NLOHMANN_JSON)

target_link_libraries(grim_data_pipeline PRIVATE 
    CURL::libcurl
    nlohmann_json::nlohmann_json
    flatbuffers::flatbuffers
    unofficial::gumbo::gumbo
    $<$<CXX_COMPILER_ID:GNU>:pthread>
)

target_include_directories(grim_data_pipeline PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/control
)

# Copy required DLLs to output directory on Windows
if(WIN32)
    add_custom_command(TARGET grim_data_pipeline POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Ensuring runtime DLLs are present for grim_data_pipeline..."
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libcurl.dll"
            "$<TARGET_FILE_DIR:grim_data_pipeline>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/zlib1.dll"
            "$<TARGET_FILE_DIR:grim_data_pipeline>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libssl-3-x64.dll"
            "$<TARGET_FILE_DIR:grim_data_pipeline>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libcrypto-3-x64.dll"
            "$<TARGET_FILE_DIR:grim_data_pipeline>" || (exit 0)
        COMMENT "Runtime DLLs for grim_data_pipeline"
        VERBATIM
    )
endif()

message(STATUS "  - grim_data_pipeline: Unified tool (collect → verify → merge)")

# Legacy executables (DEPRECATED - use grim_data_pipeline instead)
add_executable(collect_data 
    main_data_collection.cpp
    verifier.cpp
    ${DATA_COLLECTION_HEADERS}
)

target_compile_definitions(collect_data PRIVATE HAVE_NLOHMANN_JSON)

target_link_libraries(collect_data PRIVATE 
    CURL::libcurl
    nlohmann_json::nlohmann_json
    flatbuffers::flatbuffers
    unofficial::gumbo::gumbo
    $<$<CXX_COMPILER_ID:GNU>:pthread>
)

target_include_directories(collect_data PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/control
)

if(WIN32)
    add_custom_command(TARGET collect_data POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Runtime DLLs for collect_data (legacy)..."
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libcurl.dll"
            "$<TARGET_FILE_DIR:collect_data>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/zlib1.dll"
            "$<TARGET_FILE_DIR:collect_data>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libssl-3-x64.dll"
            "$<TARGET_FILE_DIR:collect_data>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libcrypto-3-x64.dll"
            "$<TARGET_FILE_DIR:collect_data>" || (exit 0)
        VERBATIM
    )
endif()

message(STATUS "  - collect_data: DEPRECATED - use grim_data_pipeline instead")

# Data Verifier executable (standalone)
add_executable(verifier 
    main_verifier.cpp
    verifier.cpp
)

target_compile_options(verifier PRIVATE 
    $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    $<$<CXX_COMPILER_ID:GNU>:-mavx2>
)

target_link_libraries(verifier PRIVATE 
    nlohmann_json::nlohmann_json
    flatbuffers::flatbuffers
    unofficial::gumbo::gumbo
    $<$<CXX_COMPILER_ID:GNU>:pthread>
)

target_include_directories(verifier PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/control
)

message(STATUS "  - verifier: Standalone data verification with FlatBuffer output")

# Checkpoint merger executable
add_executable(merge_checkpoints
    merge_checkpoints.cpp
    verifier.cpp
    ${DATA_COLLECTION_HEADERS}
)

target_compile_definitions(merge_checkpoints PRIVATE HAVE_NLOHMANN_JSON)

target_link_libraries(merge_checkpoints PRIVATE
    CURL::libcurl
    nlohmann_json::nlohmann_json
    flatbuffers::flatbuffers
    unofficial::gumbo::gumbo
    $<$<CXX_COMPILER_ID:GNU>:pthread>
)

target_include_directories(merge_checkpoints PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../training/control
)

# Copy required DLLs to output directory on Windows
if(WIN32)
    add_custom_command(TARGET merge_checkpoints POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo "Ensuring runtime DLLs are present for merge_checkpoints..."
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libcurl.dll"
            "$<TARGET_FILE_DIR:merge_checkpoints>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/zlib1.dll"
            "$<TARGET_FILE_DIR:merge_checkpoints>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libssl-3-x64.dll"
            "$<TARGET_FILE_DIR:merge_checkpoints>" || (exit 0)
        COMMAND ${CMAKE_COMMAND} -E copy_if_different 
            "D:/G.R.I.M/vcpkg_installed/x64-windows/bin/libcrypto-3-x64.dll"
            "$<TARGET_FILE_DIR:merge_checkpoints>" || (exit 0)
        COMMENT "Runtime DLLs for merge_checkpoints"
        VERBATIM
    )
endif()

message(STATUS "  - merge_checkpoints: Merges verified checkpoint files")

message(STATUS "DataCollection pipeline configured")
message(STATUS "  Included components:")
message(STATUS "    - StreamingDownloader: Memory-efficient downloads (40-50GB budget)")
message(STATUS "    - AsyncCrawler: Recursive crawling (50 workers, respects crawl_depth)")
message(STATUS "    - WebCollector: Fetches data from URLs with streaming")
message(STATUS "    - Verifier: Validates and filters data quality")
message(STATUS "    - DataPreprocessor: Text normalization and cleaning")
message(STATUS "    - DataSplitter: Train/validation/test splits")
message(STATUS "    - Tokenizer: BPE tokenization (via grim_tokenizer.hpp)")

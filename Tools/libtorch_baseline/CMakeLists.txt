cmake_minimum_required(VERSION 3.18)
project(grim_libtorch_baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CUDA_PATH})
    set(CUDAToolkit_ROOT "$ENV{CUDA_PATH}")
endif ()

# Prefer NVTX3 headers from the system CUDA toolkit to avoid nvToolsExt link errors.
set(USE_SYSTEM_NVTX ON CACHE BOOL "" FORCE)

find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

add_executable(grim_libtorch_baseline
    main.cpp
)

target_include_directories(grim_libtorch_baseline
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/../../resources/models/llama.cpp/vendor
)

target_link_libraries(grim_libtorch_baseline "${TORCH_LIBRARIES}")

if (MSVC)
    target_compile_options(grim_libtorch_baseline PRIVATE /bigobj)
    
    # Copy torch DLLs to output directory for runtime loading
    if (CMAKE_BUILD_TYPE STREQUAL "Release" OR NOT CMAKE_BUILD_TYPE)
        file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
        foreach(DLL_FILE ${TORCH_DLLS})
            add_custom_command(TARGET grim_libtorch_baseline POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${DLL_FILE}
                $<TARGET_FILE_DIR:grim_libtorch_baseline>)
        endforeach()
    endif()
endif ()

set_property(TARGET grim_libtorch_baseline PROPERTY CXX_STANDARD 17)

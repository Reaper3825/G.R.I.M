cmake_minimum_required(VERSION 3.20)
project(GRIM CXX)

include(cmake/Options.cmake)
include(cmake/Config.cmake)
include(cmake/Sources.cmake)

add_executable(GRIM ${GRIM_SOURCES} ${GRIM_HEADERS})
target_precompile_headers(GRIM PRIVATE pch.hpp)
target_include_directories(GRIM PRIVATE ${CMAKE_SOURCE_DIR})

if(MSVC)
    target_link_options(GRIM PRIVATE /SUBSYSTEM:CONSOLE)
endif()

include(cmake/Dependencies.cmake)
include(cmake/Resources.cmake)

cmake_minimum_required(VERSION 3.14)
project(SimpleQtWindow)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Qt6 COMPONENTS Widgets REQUIRED)

add_executable(SimpleQtWindow main.cpp)
target_link_libraries(SimpleQtWindow Qt6::Widgets)

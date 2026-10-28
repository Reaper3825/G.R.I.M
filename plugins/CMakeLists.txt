cmake_minimum_required(VERSION 3.22)
project(grim_plugins LANGUAGES CXX)
# This entry point deliberately does not configure the GRIM application.
add_subdirectory(osint_plugin)
add_subdirectory(network_test_plugin)

# Host-only episode plumbing reused by the Phase1-containing targets.
if(NOT TARGET grim_environment_runtime)
    get_filename_component(GRIM_ENVIRONMENT_REPO "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
    add_library(grim_environment_runtime STATIC
        EnvironmentSession.cpp
        EnvironmentInput.cpp
        EnvironmentBoundary.cpp
        "${GRIM_ENVIRONMENT_REPO}/ui/world_lab/world_document.cpp"
        "${GRIM_ENVIRONMENT_REPO}/ui/world_lab/textworld_client.cpp"
        "${GRIM_ENVIRONMENT_REPO}/MMO/Core/ToolRegistry.cpp")
    target_compile_features(grim_environment_runtime PRIVATE cxx_std_20)
    target_compile_definitions(grim_environment_runtime PRIVATE NOMINMAX)
    target_include_directories(grim_environment_runtime PRIVATE "${GRIM_ENVIRONMENT_REPO}")
    target_link_libraries(grim_environment_runtime PUBLIC nlohmann_json::nlohmann_json)
    if(UNIX)
        find_package(Threads REQUIRED)
        target_link_libraries(grim_environment_runtime PRIVATE Threads::Threads)
    endif()
endif()

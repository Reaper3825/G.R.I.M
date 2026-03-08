# =========================================================
# GRIM Source File Collection
# =========================================================

file(GLOB COMMAND_SOURCES "commands/*.cpp")
file(GLOB COMMAND_HEADERS "commands/*.hpp")
file(GLOB POPUP_UI_SOURCES "popup_ui/*.cpp")
file(GLOB POPUP_UI_HEADERS "popup_ui/*.hpp")
file(GLOB DEVICE_SETUPS_SOURCES "device_setups/*.cpp")
file(GLOB DEVICE_SETUPS_HEADERS "device_setups/*.hpp")
file(GLOB WAKE_SOURCES "wake/*.cpp")
file(GLOB WAKE_HEADERS "wake/*.hpp")
file(GLOB VOICE_SOURCES "voice/*.cpp")
file(GLOB VOICE_HEADERS "voice/*.hpp")
file(GLOB ai_SOURCES "ai/*.cpp")
file(GLOB ai_HEADERS "ai/*.hpp")
file(GLOB nlp_SOURCES "nlp/*.cpp")
file(GLOB nlp_HEADERS "nlp/*.hpp")
file(GLOB_RECURSE ui_SOURCES "${CMAKE_SOURCE_DIR}/ui/*.cpp")
file(GLOB_RECURSE ui_HEADERS "${CMAKE_SOURCE_DIR}/ui/*.hpp")
file(GLOB bootstrap_SOURCES "bootstrap/*.cpp")
file(GLOB bootstrap_HEADERS "bootstrap/*.hpp")
file(GLOB helpers_SOURCES "helpers/*.cpp")
file(GLOB helpers_HEADERS "helpers/*.hpp")
file(GLOB memory_SOURCES "memory/*.cpp")
file(GLOB memory_HEADERS "memory/*.hpp")
file(GLOB core_SOURCES "core/*.cpp")
file(GLOB core_HEADERS "core/*.hpp")
file(GLOB input_SOURCES "core/input/*.cpp")
file(GLOB input_HEADERS "core/input/*.hpp")
file(GLOB NET_SOURCES "net/*.cpp")
file(GLOB NET_HEADERS "net/*.hpp")
file(GLOB EXTERNAL_COLLECTOR_SOURCES "external_collector/*.cpp")
file(GLOB EXTERNAL_COLLECTOR_HEADERS "external_collector/*.hpp")
file(GLOB PERCEPTION_SOURCES "perception/*.cpp")
file(GLOB PERCEPTION_HEADERS "perception/*.hpp")
file(GLOB VISION_SOURCES "vision/*.cpp")
file(GLOB VISION_HEADERS "vision/*.hpp")
file(GLOB REWARD_LEARNING_SOURCES "Reward_Learning/*.cpp")
file(GLOB REWARD_LEARNING_HEADERS "Reward_Learning/*.hpp")
file(GLOB DataCollection_SOURCES "DataCollection/*.cpp")
file(GLOB DataCollection_HEADERS "DataCollection/*.hpp")
file(GLOB hardware_SOURCES "hardware/*.cpp")
file(GLOB hardware_HEADERS "hardware/*.hpp")
file(GLOB_RECURSE MMO_SOURCES "MMO/*.cpp")
file(GLOB_RECURSE MMO_HEADERS "MMO/*.hpp")
# Exclude standalone entry point files (these have main() functions)
list(FILTER DataCollection_SOURCES EXCLUDE REGEX "main_data_collection\\.cpp$")
list(FILTER DataCollection_SOURCES EXCLUDE REGEX "merge_checkpoints\\.cpp$")
list(FILTER DataCollection_SOURCES EXCLUDE REGEX "main_verifier\\.cpp$")
# =========================================================
# GRIM-text Model: Standalone Build (2025-11-05)
# =========================================================
# GRIM-text is built separately like Ollama, not compiled into GRIM.exe
# Use resources/models/GRIM-text/training/CMakeLists.txt to build the model
# GRIM loads the model at runtime as an external dependency
# =========================================================
set(GRIM_GPU_SOURCES "")

set(GRIM_SOURCES
    main.cpp
    resources.cpp
    synonyms.cpp
    aliases.cpp
    console_history.cpp
    response_manager.cpp
    error_manager.cpp
    logger.cpp
    location.cpp
    ${COMMAND_SOURCES}
    ${POPUP_UI_SOURCES}
    ${DEVICE_SETUPS_SOURCES}
    ${VOICE_SOURCES}
    ${ai_SOURCES}
    ${WAKE_SOURCES}
    ${nlp_SOURCES}
    ${ui_SOURCES}
    ${bootstrap_SOURCES}
    ${helpers_SOURCES}
    ${memory_SOURCES}
    ${core_SOURCES}
    ${input_SOURCES}
    ${NET_SOURCES}
    ${EXTERNAL_COLLECTOR_SOURCES}
    ${PERCEPTION_SOURCES}
    ${VISION_SOURCES}
    ${REWARD_LEARNING_SOURCES}
    ${GRIM_GPU_SOURCES}
    ${DataCollection_SOURCES}
    ${hardware_SOURCES}
    ${MMO_SOURCES}
)

set(GRIM_HEADERS
    console_history.hpp
    intent.hpp
    aliases.hpp
    synonyms.hpp
    resources.hpp
    timer.hpp
    response_manager.hpp
    error_manager.hpp
    logger.hpp
    location.hpp
    ${COMMAND_HEADERS}
    ${POPUP_UI_HEADERS}
    ${DEVICE_SETUPS_HEADERS}
    ${VOICE_HEADERS}
    ${ai_HEADERS}
    ${WAKE_HEADERS}
    ${nlp_HEADERS}
    ${ui_HEADERS}
    ${bootstrap_HEADERS}
    ${helpers_HEADERS}
    ${memory_HEADERS}
    ${core_HEADERS}
    ${input_HEADERS}
    ${NET_HEADERS}
    ${EXTERNAL_COLLECTOR_HEADERS}
    ${PERCEPTION_HEADERS}
    ${VISION_HEADERS}
    ${REWARD_LEARNING_HEADERS}
    ${DataCollection_HEADERS}
    ${hardware_HEADERS}
    ${MMO_HEADERS}
)

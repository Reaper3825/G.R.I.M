# =========================================================
# GRIM Source File Collection
# =========================================================

file(GLOB COMMAND_SOURCES "commands/*.cpp")
file(GLOB COMMAND_HEADERS "commands/*.hpp")
file(GLOB POPUP_UI_SOURCES CONFIGURE_DEPENDS "popup_ui/*.cpp")
file(GLOB POPUP_UI_HEADERS CONFIGURE_DEPENDS "popup_ui/*.hpp")
file(GLOB POPUP_UI_OBJECTS_SOURCES CONFIGURE_DEPENDS "popup_ui/objects/*.cpp")
file(GLOB POPUP_UI_OBJECTS_HEADERS CONFIGURE_DEPENDS "popup_ui/objects/*.hpp")
if(APPLE)
  file(GLOB POPUP_UI_SOURCES_MM CONFIGURE_DEPENDS "popup_ui/*.mm")
  list(APPEND POPUP_UI_SOURCES ${POPUP_UI_SOURCES_MM})
endif()
file(GLOB DEVICE_SETUPS_SOURCES "device_setups/*.cpp")
file(GLOB DEVICE_SETUPS_HEADERS "device_setups/*.hpp")
file(GLOB WAKE_SOURCES "wake/*.cpp")
file(GLOB WAKE_HEADERS "wake/*.hpp")
file(GLOB VOICE_SOURCES "voice/*.cpp")
file(GLOB VOICE_HEADERS "voice/*.hpp")
if(APPLE)
  file(GLOB VOICE_SOURCES_MM "voice/*.mm")
  list(APPEND VOICE_SOURCES ${VOICE_SOURCES_MM})
endif()
file(GLOB ai_SOURCES "ai/*.cpp")
file(GLOB ai_HEADERS "ai/*.hpp")
file(GLOB nlp_SOURCES "nlp/*.cpp")
file(GLOB nlp_HEADERS "nlp/*.hpp")
file(GLOB_RECURSE ui_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/ui/*.cpp")
list(FILTER ui_SOURCES EXCLUDE REGEX "ui_DataCollection_OLD\\.cpp$")
list(FILTER ui_SOURCES EXCLUDE REGEX "ui_model_panel\\.cpp$")
list(FILTER ui_SOURCES EXCLUDE REGEX "\\.bak$")
file(GLOB_RECURSE ui_HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/ui/*.hpp")
file(GLOB bootstrap_SOURCES "bootstrap/*.cpp")
file(GLOB bootstrap_HEADERS "bootstrap/*.hpp")
file(GLOB helpers_SOURCES "helpers/*.cpp")
file(GLOB helpers_HEADERS "helpers/*.hpp")
file(GLOB memory_SOURCES "memory/*.cpp")
file(GLOB memory_HEADERS "memory/*.hpp")
file(GLOB core_SOURCES "core/*.cpp")
if(APPLE)
  file(GLOB core_SOURCES_MM "core/*.mm")
  list(APPEND core_SOURCES ${core_SOURCES_MM})
endif()
file(GLOB core_HEADERS "core/*.hpp")
file(GLOB input_SOURCES "core/input/*.cpp")
file(GLOB input_HEADERS "core/input/*.hpp")
file(GLOB NET_SOURCES "net/*.cpp")
file(GLOB NET_HEADERS "net/*.hpp")
file(GLOB EXTERNAL_COLLECTOR_SOURCES "external_collector/*.cpp")
file(GLOB EXTERNAL_COLLECTOR_HEADERS "external_collector/*.hpp")
file(GLOB_RECURSE GEOSPATIAL_SOURCES "geospatial/*.cpp")
file(GLOB_RECURSE GEOSPATIAL_HEADERS "geospatial/*.hpp")
file(GLOB_RECURSE PERCEPTION_SOURCES "perception/*.cpp")
file(GLOB_RECURSE PERCEPTION_HEADERS "perception/*.hpp")
# Platform split for perception/physical/PhysicalNicScan_*: keep only the host OS impl
list(FILTER PERCEPTION_SOURCES EXCLUDE REGEX "perception/physical/PhysicalNicScan_(macos|win32|linux)\\.(cpp|mm)$")
if(WIN32)
  list(APPEND PERCEPTION_SOURCES ${CMAKE_SOURCE_DIR}/perception/physical/PhysicalNicScan_win32.cpp)
elseif(APPLE)
  list(APPEND PERCEPTION_SOURCES ${CMAKE_SOURCE_DIR}/perception/physical/PhysicalNicScan_macos.mm)
else()
  list(APPEND PERCEPTION_SOURCES ${CMAKE_SOURCE_DIR}/perception/physical/PhysicalNicScan_linux.cpp)
endif()
file(GLOB VISION_SOURCES "vision/*.cpp")
file(GLOB VISION_HEADERS "vision/*.hpp")
# Vision/DataCollection files with direct ONNX dependencies are excluded when ONNX is unavailable
# (GRIM_HAS_ONNXRUNTIME is set in Config.cmake when the library is found)
file(GLOB REWARD_LEARNING_SOURCES "Reward_Learning/*.cpp")
file(GLOB REWARD_LEARNING_HEADERS "Reward_Learning/*.hpp")
file(GLOB DataCollection_SOURCES "DataCollection/*.cpp")
file(GLOB DataCollection_HEADERS "DataCollection/*.hpp")
file(GLOB DataCollection_Pipeline_SOURCES "DataCollection/pipeline/*.cpp")
file(GLOB DataCollection_Pipeline_HEADERS "DataCollection/pipeline/*.hpp")
file(GLOB DataCollection_PipelineStages_SOURCES "DataCollection/pipeline/stages/*.cpp")
file(GLOB DataCollection_PipelineStages_HEADERS "DataCollection/pipeline/stages/*.hpp")
file(GLOB DataCollection_IO_SOURCES CONFIGURE_DEPENDS "DataCollection/io/*.cpp")
file(GLOB DataCollection_IO_HEADERS CONFIGURE_DEPENDS "DataCollection/io/*.hpp")
file(GLOB hardware_SOURCES "hardware/*.cpp")
file(GLOB hardware_HEADERS "hardware/*.hpp")
file(GLOB_RECURSE MMO_SOURCES "MMO/*.cpp")
file(GLOB_RECURSE MMO_HEADERS "MMO/*.hpp")
file(GLOB settings_SOURCES "settings/*.cpp")
file(GLOB settings_HEADERS "settings/*.hpp")
file(GLOB_RECURSE DEVICE_COMM_SOURCES "control/devices/*.cpp")
file(GLOB_RECURSE DEVICE_COMM_HEADERS "control/devices/*.hpp")
# Exclude standalone entry point files (these have main() functions)
list(FILTER DataCollection_SOURCES EXCLUDE REGEX "main_data_collection\\.cpp$")
list(FILTER DataCollection_SOURCES EXCLUDE REGEX "main_verifier\\.cpp$")
# Exclude replaced monolithic files
list(FILTER DataCollection_SOURCES EXCLUDE REGEX "grim_data_pipeline\\.cpp$")
list(FILTER DataCollection_SOURCES EXCLUDE REGEX "data_collection_manager\\.cpp$")
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
    error_manager.cpp
    logger.cpp
    location.cpp
    ${COMMAND_SOURCES}
    ${POPUP_UI_SOURCES}
    ${POPUP_UI_OBJECTS_SOURCES}
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
    ${GEOSPATIAL_SOURCES}
    ${PERCEPTION_SOURCES}
    ${VISION_SOURCES}
    ${REWARD_LEARNING_SOURCES}
    ${GRIM_GPU_SOURCES}
    ${DataCollection_SOURCES}
    ${DataCollection_Pipeline_SOURCES}
    ${DataCollection_PipelineStages_SOURCES}
    ${DataCollection_IO_SOURCES}
    ${hardware_SOURCES}
    ${MMO_SOURCES}
    ${settings_SOURCES}
    ${DEVICE_COMM_SOURCES}
)

set(GRIM_HEADERS
    console_history.hpp
    intent.hpp
    aliases.hpp
    synonyms.hpp
    resources.hpp
    timer.hpp
    error_manager.hpp
    logger.hpp
    location.hpp
    ${COMMAND_HEADERS}
    ${POPUP_UI_HEADERS}
    ${POPUP_UI_OBJECTS_HEADERS}
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
    ${GEOSPATIAL_HEADERS}
    ${PERCEPTION_HEADERS}
    ${VISION_HEADERS}
    ${REWARD_LEARNING_HEADERS}
    ${DataCollection_HEADERS}
    ${DataCollection_Pipeline_HEADERS}
    ${DataCollection_PipelineStages_HEADERS}
    ${DataCollection_IO_HEADERS}
    ${hardware_HEADERS}
    ${MMO_HEADERS}
    ${settings_HEADERS}
    ${DEVICE_COMM_HEADERS}
)

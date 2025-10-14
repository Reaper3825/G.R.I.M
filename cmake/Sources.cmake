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
file(GLOB ui_SOURCES "ui/*.cpp")
file(GLOB ui_HEADERS "ui/*.hpp")
file(GLOB bootstrap_SOURCES "bootstrap/*.cpp")
file(GLOB bootstrap_HEADERS "bootstrap/*.hpp")
file(GLOB helpers_SOURCES "helpers/*.cpp")
file(GLOB helpers_HEADERS "helpers/*.hpp")
file(GLOB memory_SOURCES "memory/*.cpp")
file(GLOB memory_HEADERS "memory/*.hpp")
file(GLOB core_SOURCES "core/*.cpp")
file(GLOB core_HEADERS "core/*.hpp")
set(GRIM_SOURCES
    main.cpp
    resources.cpp
    synonyms.cpp
    aliases.cpp
    console_history.cpp
    response_manager.cpp
    error_manager.cpp
    logger.cpp
    system_detect.cpp
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
    system_detect.hpp
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
)

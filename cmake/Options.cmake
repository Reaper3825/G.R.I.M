# =========================================================
# GRIM Build Options
# =========================================================
option(GRIM_USE_CUDA "Enable CUDA acceleration for Whisper/AI" ON)
option(GRIM_USE_METAL "Enable Metal acceleration for Whisper/AI" OFF)
option(GRIM_USE_VULKAN "Enable Vulkan acceleration for Whisper/AI" OFF)
option(GRIM_USE_OPENCL "Enable OpenCL acceleration for Whisper/AI" OFF)
option(GRIM_USE_PERCEPTION "Enable perception/vision AI models (OpenCV, Tesseract)" ON)
option(GRIM_PORTABLE_ONLY "Force resources to live next to executable" OFF)

# Optional local-only hand interaction backend. The checked-in setup script is
# the only download path; GRIM itself never fetches a runtime or model.
set(GRIM_MEDIAPIPE_ROOT "${CMAKE_SOURCE_DIR}/external/mediapipe" CACHE PATH
    "MediaPipe source/install root containing mediapipe/tasks/c headers")
if(WIN32)
    set(_GRIM_MEDIAPIPE_LIBRARY_DEFAULT
        "${GRIM_MEDIAPIPE_ROOT}/bazel-bin/mediapipe/tasks/c/libmediapipe.dll")
elseif(APPLE)
    set(_GRIM_MEDIAPIPE_LIBRARY_DEFAULT
        "${GRIM_MEDIAPIPE_ROOT}/bazel-bin/mediapipe/tasks/c/libmediapipe.dylib")
else()
    set(_GRIM_MEDIAPIPE_LIBRARY_DEFAULT
        "${GRIM_MEDIAPIPE_ROOT}/bazel-bin/mediapipe/tasks/c/libmediapipe.so")
endif()
set(GRIM_MEDIAPIPE_TASKS_C_LIBRARY "${_GRIM_MEDIAPIPE_LIBRARY_DEFAULT}"
    CACHE FILEPATH
    "Path to the locally built MediaPipe Tasks C library")
set(_GRIM_MEDIAPIPE_DEFAULT OFF)
if(EXISTS
   "${GRIM_MEDIAPIPE_ROOT}/mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h"
   AND EXISTS "${GRIM_MEDIAPIPE_TASKS_C_LIBRARY}")
    set(_GRIM_MEDIAPIPE_DEFAULT ON)
endif()
option(GRIM_USE_MEDIAPIPE_HAND_GESTURES
       "Enable the optional MediaPipe Tasks C hand-gesture backend"
       ${_GRIM_MEDIAPIPE_DEFAULT})
set(GRIM_MEDIAPIPE_VERSION "0.10.35" CACHE STRING
    "MediaPipe Tasks C API version used by the supplied local library")
set(GRIM_MEDIAPIPE_OFFLINE_AUDIT_STAMP
    "${GRIM_MEDIAPIPE_ROOT}/.grim-offline-audit.json" CACHE FILEPATH
    "Audit stamp emitted by scripts/setup_mediapipe_backend.ps1 -Build")
option(GRIM_MEDIAPIPE_OFFLINE_LOGGER_VERIFIED
       "Manually confirm a custom MediaPipe build has usage metrics disabled" OFF)
unset(_GRIM_MEDIAPIPE_DEFAULT)
unset(_GRIM_MEDIAPIPE_LIBRARY_DEFAULT)

# =========================================================
# Export compile commands for tooling
# =========================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

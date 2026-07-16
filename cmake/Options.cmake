# =========================================================
# GRIM Build Options
# =========================================================
option(GRIM_USE_CUDA "Enable CUDA acceleration for Whisper/AI" ON)
option(GRIM_USE_METAL "Enable Metal acceleration for Whisper/AI" OFF)
option(GRIM_USE_VULKAN "Enable Vulkan acceleration for Whisper/AI" OFF)
option(GRIM_USE_OPENCL "Enable OpenCL acceleration for Whisper/AI" OFF)
option(GRIM_USE_PERCEPTION "Enable perception/vision AI models (OpenCV, Tesseract)" ON)
option(GRIM_PORTABLE_ONLY "Force resources to live next to executable" OFF)

# Optional local-only hand interaction backend. OFF preserves the existing
# dependency set and base offline build. GRIM never downloads MediaPipe or a
# gesture model; callers provide both explicitly when opting in.
option(GRIM_USE_MEDIAPIPE_HAND_GESTURES
       "Enable the optional MediaPipe Tasks C hand-gesture backend" OFF)
set(GRIM_MEDIAPIPE_ROOT "" CACHE PATH
    "MediaPipe source/install root containing mediapipe/tasks/c headers")
set(GRIM_MEDIAPIPE_TASKS_C_LIBRARY "" CACHE FILEPATH
    "Path to the locally built MediaPipe Tasks C library")
set(GRIM_MEDIAPIPE_VERSION "0.10.35" CACHE STRING
    "MediaPipe Tasks C API version used by the supplied local library")
option(GRIM_MEDIAPIPE_OFFLINE_LOGGER_VERIFIED
       "Confirm supplied MediaPipe was built with usage metrics transport disabled" OFF)

# =========================================================
# Export compile commands for tooling
# =========================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

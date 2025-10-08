# =========================================================
# GRIM Build Options
# =========================================================
option(GRIM_USE_CUDA "Enable CUDA acceleration for Whisper/AI" ON)
option(GRIM_USE_METAL "Enable Metal acceleration for Whisper/AI" OFF)
option(GRIM_USE_VULKAN "Enable Vulkan acceleration for Whisper/AI" OFF)
option(GRIM_USE_OPENCL "Enable OpenCL acceleration for Whisper/AI" OFF)
option(GRIM_PORTABLE_ONLY "Force resources to live next to executable" OFF)

# =========================================================
# Export compile commands for tooling
# =========================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

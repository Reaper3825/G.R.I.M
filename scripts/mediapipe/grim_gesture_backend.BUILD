load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

licenses(["notice"])

# Keep GRIM's native runtime limited to the exports it consumes. Building
# //mediapipe/tasks/c:mediapipe_windows links every Tasks C API and pulls in
# unrelated audio, text, GenAI, and vision dependencies.
cc_binary(
    name = "libmediapipe.dll",
    linkshared = True,
    linkstatic = True,
    tags = [
        "manual",
        "nobuilder",
        "notap",
    ],
    deps = [
        "//mediapipe/tasks/c/core:common",
        "//mediapipe/tasks/c/vision/core:image_c_lib",
        "//mediapipe/tasks/c/vision/gesture_recognizer:gesture_recognizer_c_lib",
    ],
)

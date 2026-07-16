# OpenCV adapter for GRIM's vcpkg x64-windows installation.

licenses(["notice"])

cc_library(
    name = "opencv",
    srcs = glob(
        [
            "bin/opencv_*.dll",
            "lib/opencv_*.lib",
        ],
        allow_empty = False,
    ),
    hdrs = glob(
        ["include/opencv4/opencv2/**/*.h*"],
        allow_empty = False,
    ),
    includes = ["include/opencv4"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)

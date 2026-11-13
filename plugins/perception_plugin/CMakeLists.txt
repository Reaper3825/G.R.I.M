cmake_minimum_required(VERSION 3.22)
project(perception_plugin LANGUAGES CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../plugin_build.cmake")
if(WIN32 AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    list(APPEND CMAKE_PREFIX_PATH "${GRIM_PLUGIN_ROOT}/vcpkg_installed/x64-windows")
endif()
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc dnn imgcodecs videoio video photo calib3d objdetect)
find_package(Tesseract CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_path(PERCEPTION_ONNX_INCLUDE onnxruntime_cxx_api.h
    PATH_SUFFIXES onnxruntime onnxruntime/core/session REQUIRED)
find_library(PERCEPTION_ONNX_LIBRARY NAMES onnxruntime REQUIRED)
file(GLOB provider_sources CONFIGURE_DEPENDS "digital/*.cpp" "physical/*.cpp")
list(FILTER provider_sources EXCLUDE REGEX "PhysicalNicScan_(macos|win32|linux)\\.cpp$")
if(WIN32)
    list(APPEND provider_sources physical/PhysicalNicScan_win32.cpp)
elseif(APPLE)
    enable_language(OBJCXX)
    list(APPEND provider_sources physical/PhysicalNicScan_macos.mm)
else()
    list(APPEND provider_sources physical/PhysicalNicScan_linux.cpp)
endif()
add_library(perception_plugin SHARED perception_plugin.cpp perception_runtime.cpp perception_config.cpp ${provider_sources})
if(APPLE)
    set_target_properties(perception_plugin PROPERTIES OBJCXX_STANDARD 20 OBJCXX_STANDARD_REQUIRED YES)
endif()
grim_configure_plugin(perception_plugin)
target_compile_features(perception_plugin PRIVATE cxx_std_20)
target_compile_definitions(perception_plugin PRIVATE GRIM_HAS_OPENCV GRIM_HAS_TESSERACT GRIM_ROOT_DIR="${GRIM_PLUGIN_ROOT}")
target_include_directories(perception_plugin PRIVATE ${OpenCV_INCLUDE_DIRS} ${PERCEPTION_ONNX_INCLUDE})
target_link_libraries(perception_plugin PRIVATE nlohmann_json::nlohmann_json ${OpenCV_LIBS} Tesseract::libtesseract Threads::Threads ${PERCEPTION_ONNX_LIBRARY} ${CMAKE_DL_LIBS})
if(WIN32)
    target_link_libraries(perception_plugin PRIVATE user32 gdi32 ole32 oleaut32 uuid uiautomationcore iphlpapi ws2_32 strmiids)
elseif(APPLE)
    target_link_libraries(perception_plugin PRIVATE "-framework Foundation" "-framework AVFoundation")
endif()
option(GRIM_PERCEPTION_PROVIDER_TESTS "Build isolated provider tests without devices or inference" OFF)
if(GRIM_PERCEPTION_PROVIDER_TESTS)
    enable_testing()
    add_executable(perception_provider_tests "${GRIM_PLUGIN_ROOT}/tests/perception_provider_tests.cpp")
    target_compile_features(perception_provider_tests PRIVATE cxx_std_20)
    target_include_directories(perception_provider_tests PRIVATE "${GRIM_PLUGIN_ROOT}")
    target_link_libraries(perception_provider_tests PRIVATE nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS})
    add_test(NAME perception_provider COMMAND perception_provider_tests $<TARGET_FILE:perception_plugin>)
endif()

# MediaPipe hand gestures are an explicitly opt-in local dependency. The
# default GRIM target compiles the visible "backend unavailable" adapter and
# remains fully functional without MediaPipe. No FetchContent or download path
# is allowed here because base operation must remain offline-capable.
if(GRIM_USE_MEDIAPIPE_HAND_GESTURES)
    if(NOT EXISTS "${GRIM_MEDIAPIPE_OFFLINE_AUDIT_STAMP}" AND
       NOT GRIM_MEDIAPIPE_OFFLINE_LOGGER_VERIFIED)
        message(FATAL_ERROR
            "MediaPipe hand gestures require "
            "scripts/setup_mediapipe_backend.ps1 -Build, or explicit "
            "-DGRIM_MEDIAPIPE_OFFLINE_LOGGER_VERIFIED=ON for an independently "
            "audited custom build.")
    endif()
    if(NOT EXISTS
       "${GRIM_MEDIAPIPE_ROOT}/mediapipe/tasks/c/vision/gesture_recognizer/gesture_recognizer.h")
        message(FATAL_ERROR
            "GRIM_MEDIAPIPE_ROOT does not contain the MediaPipe Tasks C headers")
    endif()
    if(NOT EXISTS "${GRIM_MEDIAPIPE_TASKS_C_LIBRARY}")
        message(FATAL_ERROR
            "GRIM_MEDIAPIPE_TASKS_C_LIBRARY must name an existing local library")
    endif()
    if(EXISTS "${GRIM_MEDIAPIPE_OFFLINE_AUDIT_STAMP}")
        file(READ "${GRIM_MEDIAPIPE_OFFLINE_AUDIT_STAMP}"
             _GRIM_MEDIAPIPE_AUDIT_JSON)
        string(JSON _GRIM_MEDIAPIPE_AUDITED_HASH ERROR_VARIABLE
               _GRIM_MEDIAPIPE_AUDIT_ERROR GET
               "${_GRIM_MEDIAPIPE_AUDIT_JSON}" runtime_sha256)
        if(NOT "${_GRIM_MEDIAPIPE_AUDIT_ERROR}" STREQUAL "NOTFOUND")
            message(FATAL_ERROR
                "MediaPipe offline audit stamp is invalid: "
                "${_GRIM_MEDIAPIPE_AUDIT_ERROR}")
        endif()
        file(SHA256 "${GRIM_MEDIAPIPE_TASKS_C_LIBRARY}"
             _GRIM_MEDIAPIPE_RUNTIME_HASH)
        if(NOT "${_GRIM_MEDIAPIPE_RUNTIME_HASH}" STREQUAL
           "${_GRIM_MEDIAPIPE_AUDITED_HASH}")
            message(FATAL_ERROR
                "MediaPipe runtime changed after its offline audit; rerun "
                "scripts/setup_mediapipe_backend.ps1 -Build")
        endif()
        unset(_GRIM_MEDIAPIPE_AUDIT_JSON)
        unset(_GRIM_MEDIAPIPE_AUDITED_HASH)
        unset(_GRIM_MEDIAPIPE_AUDIT_ERROR)
        unset(_GRIM_MEDIAPIPE_RUNTIME_HASH)
    endif()

    target_include_directories(perception_plugin PRIVATE "${GRIM_MEDIAPIPE_ROOT}")
    # Resolve the C exports dynamically. Windows DLL packages do not need an
    # import library, and a missing/wrong runtime becomes a visible backend
    # state rather than a process-start failure.
    if(CMAKE_DL_LIBS)
        target_link_libraries(perception_plugin PRIVATE ${CMAKE_DL_LIBS})
    endif()
    file(TO_CMAKE_PATH "${GRIM_MEDIAPIPE_TASKS_C_LIBRARY}"
         _GRIM_MEDIAPIPE_RUNTIME_PATH)
    target_compile_definitions(perception_plugin PRIVATE
        GRIM_HAS_MEDIAPIPE_TASKS_C=1
        GRIM_MEDIAPIPE_TELEMETRY_DISABLED=1
        GRIM_MEDIAPIPE_RUNTIME_LIBRARY_PATH="${_GRIM_MEDIAPIPE_RUNTIME_PATH}"
        GRIM_MEDIAPIPE_VERSION="${GRIM_MEDIAPIPE_VERSION}")
    if(WIN32)
        set(_GRIM_MEDIAPIPE_RUNTIME_NAME "libmediapipe.dll")
    elseif(APPLE)
        set(_GRIM_MEDIAPIPE_RUNTIME_NAME "libmediapipe.dylib")
    else()
        set(_GRIM_MEDIAPIPE_RUNTIME_NAME "libmediapipe.so")
    endif()
    unset(_GRIM_MEDIAPIPE_RUNTIME_PATH)
    unset(_GRIM_MEDIAPIPE_RUNTIME_NAME)
    message(STATUS
        "GRIM MediaPipe hand gestures: local backend enabled "
        "(${GRIM_MEDIAPIPE_VERSION}); network fallback disabled")
else()
    message(STATUS
        "GRIM MediaPipe hand gestures: disabled (base offline build unchanged)")
endif()

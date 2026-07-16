# MediaPipe hand-gesture backend

GRIM uses the MediaPipe Tasks C `GestureRecognizer` through a native runtime
that is built locally from the pinned v0.10.35 public source. Camera frames,
landmarks, and gesture results stay in-process. The runtime never downloads a
model and GRIM has no network fallback.

The official Python wheel is intentionally not used. Its Windows DLL contains
Google usage-logging code and the Clearcut endpoint. The v0.10.35 public source
build removed those analytics references, and GRIM's setup script rejects a
compiled DLL if the corresponding logger symbols or endpoint are present.

## Prepare without building

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_mediapipe_backend.ps1
```

This explicit setup step downloads checksum-pinned source, Bazelisk, and the
official gesture model. It does not compile anything. These local assets are
ignored by Git because they are generated dependencies, not GRIM source.

## Build the backend

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_mediapipe_backend.ps1 -Build
```

The first Bazelisk run downloads Bazel 7.4.1 and MediaPipe's build
dependencies, so this one-time development step requires internet access.
The resulting `libmediapipe.dll`, model, and GRIM runtime need no connection.
The build pins TensorFlow's dependency environment to hermetic Python 3.12.
LLVM's repository setup also needs a runnable host Python of any supported
version; the script resolves the real interpreter behind WindowsApps aliases
and passes its directory explicitly to Bazel.
The Windows build also supplies MSVC's native `/std:c11` option for C-only
dependencies such as pthreadpool, whose upstream BUILD file uses GCC syntax.

The script builds MediaPipe CPU-only, scans the DLL for forbidden logging
markers, and writes `external/mediapipe/.grim-offline-audit.json`. CMake will
not enable the backend without that stamp unless the developer explicitly
asserts that a custom runtime was independently audited.

On Windows, the build reuses GRIM's `vcpkg_installed/x64-windows` OpenCV 4
headers and import libraries. A GRIM-specific Bazel target links only the
gesture recognizer and image C APIs instead of the complete MediaPipe Tasks C
bundle. The generated runtime therefore uses the same OpenCV DLL family already
deployed for GRIM's physical perception layer.

## Build GRIM

On an existing CMake cache, enable the feature once:

```powershell
cmake -S . -B out/build -DGRIM_USE_MEDIAPIPE_HAND_GESTURES=ON
cmake --build out/build --config Release
```

On a fresh configure after the audited DLL exists, the option defaults to ON.
CMake copies `libmediapipe.dll` beside `GRIM.exe`. The adapter loads it by
absolute local path, validates every required C export, and reports load or ABI
errors through the Physical Environment interaction status instead of failing
process startup.

Deleting the local MediaPipe files only disables this optional perception
backend; GRIM's base offline operation and other physical perception paths do
not depend on it.

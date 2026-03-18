# Building G.R.I.M on macOS

This document describes how to build G.R.I.M on macOS. The codebase has been updated for cross-platform compilation; several features (ProcessManager, Voice TTS, Popup UI, Perception screen capture) remain Windows-implementations and will need separate ports for full macOS functionality.

## Prerequisites

1. **Xcode Command Line Tools** (or full Xcode)
   ```bash
   xcode-select --install
   ```

2. **vcpkg** – Install and bootstrap:
   ```bash
   git clone https://github.com/Microsoft/vcpkg.git
   cd vcpkg && ./bootstrap-vcpkg.sh
   ```

3. **vcpkg dependencies** – The root `vcpkg.json` uses `onnxruntime-gpu` (CUDA). On macOS, use `onnxruntime` (CPU) instead. Either:
   - Create a `vcpkg-configuration.json` override, or
   - Manually install packages:
   ```bash
   export VCPKG_ROOT=/path/to/vcpkg
   cd /path/to/G.R.I.M
   vcpkg install --triplet arm64-osx \
     curl cpr nlohmann-json opencv4[dnn,contrib] tesseract leptonica \
     onnxruntime flatbuffers gumbo libzip poppler ffmpeg sentencepiece \
     portaudio glfw3 openal-soft uwebsockets simdjson
   ```

4. **External libraries** – Build for macOS:
   - **whisper.cpp**: `cmake -B build -DWHISPER_METAL=ON && cmake --build build`
   - **bgfx** (bx, bimg, bgfx): Build with Metal backend for macOS
   - **Porcupine**: Obtain macOS libraries from Picovoice and place under `external/porcupine/lib/macos/arm64` (or `macos/x86_64` for Intel)

## Build

```bash
# From repo root
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
  -DVCPKG_TARGET_TRIPLET=arm64-osx
cmake --build .
```

Or use the helper script:
```bash
export VCPKG_ROOT=/path/to/vcpkg
./scripts/build_macos.sh Release
```

## Post-build

- Executable: `build/GRIM` (or `build/Release/GRIM` depending on generator)
- Run from repo root so resources and plugins are found:
  ```bash
  cd /path/to/G.R.I.M && ./build/GRIM
  ```

## Features on macOS

| Feature              | Status              |
|----------------------|---------------------|
| Core runtime         | Should run          |
| Plugin loading       | `.dylib` supported  |
| Input / Clipboard    | Implemented         |
| ProcessManager       | Not implemented     |
| Voice TTS            | Not implemented     |
| Popup UI / Overlay   | Not implemented     |
| Perception (screen)  | Not implemented     |

These will need Unix/macOS implementations (e.g. `posix_spawn` for ProcessManager, NSSpeechSynthesizer for TTS) to match Windows behavior.

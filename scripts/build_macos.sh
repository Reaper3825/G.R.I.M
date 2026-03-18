#!/usr/bin/env bash
# =========================================================
# G.R.I.M macOS Build Script
# =========================================================
# Prerequisites:
#   1. Install vcpkg:  git clone https://github.com/Microsoft/vcpkg.git && ./vcpkg/bootstrap-vcpkg.sh
#   2. Install dependencies:  cd G.R.I.M && vcpkg install --triplet arm64-osx  (or x64-osx for Intel)
#      Note: vcpkg.json has onnxruntime-gpu (CUDA) - for macOS use: vcpkg install --triplet arm64-osx \
#        curl cpr nlohmann-json opencv tesseract ... (see vcpkg.json, use onnxruntime not onnxruntime-gpu)
#   3. Build whisper.cpp, bgfx, porcupine for macOS (see external/ layout)
#
# Usage:
#   ./scripts/build_macos.sh [Debug|Release]
# =========================================================

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BUILD_TYPE="${1:-Release}"

# Default to arm64 on Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
  TRIPLET="arm64-osx"
else
  TRIPLET="x64-osx"
fi

# vcpkg toolchain (if available)
if [[ -n "$VCPKG_ROOT" ]] && [[ -f "$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake" ]]; then
  CMAKE_OPTS="-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=$TRIPLET"
elif [[ -f "$ROOT/external/vcpkg/scripts/buildsystems/vcpkg.cmake" ]]; then
  CMAKE_OPTS="-DCMAKE_TOOLCHAIN_FILE=$ROOT/external/vcpkg/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=$TRIPLET"
else
  CMAKE_OPTS="-DVCPKG_TARGET_TRIPLET=$TRIPLET"
fi

mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE" $CMAKE_OPTS
cmake --build . --config "$BUILD_TYPE"

echo ""
echo "[GRIM] Build complete. Run: ./build/GRIM (or ./build/$BUILD_TYPE/GRIM)"

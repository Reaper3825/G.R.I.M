#!/bin/bash
#=============================================#
# GRIM Training Tools - Build Script (MSYS2)
#=============================================#

set -e  # Exit on error

echo "=== GRIM Training Tools Build ==="
echo ""

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found!"
    echo "Please run this script from MSYS2 MinGW64 terminal"
    echo "Or install gcc with: pacman -S mingw-w64-x86_64-gcc"
    exit 1
fi

echo "✓ Found g++ compiler: $(g++ --version | head -n1)"

# Check for libcurl
if ! pkg-config --exists libcurl; then
    echo "WARNING: libcurl not found!"
    echo "Install with: pacman -S mingw-w64-x86_64-curl"
    echo "Attempting to continue anyway..."
fi

# Create directories
echo ""
echo "Creating directories..."
mkdir -p bin
mkdir -p data/raw data/processed data/tokenized
mkdir -p checkpoints
mkdir -p models
mkdir -p logs

# Build data collection tool
echo ""
echo "Building data collection tool..."
g++ -std=c++17 -O3 -march=native \
    main_data_collection.cpp \
    -o bin/collect_data.exe \
    -lcurl -lpthread \
    2>&1 | tee logs/build_collect.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Built: bin/collect_data.exe"
else
    echo "✗ Failed to build collect_data.exe (see logs/build_collect.log)"
    exit 1
fi

# Build training tool
echo ""
echo "Building training tool..."
g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp \
    train_model.cpp \
    -o bin/train_model.exe \
    -lpthread \
    2>&1 | tee logs/build_train.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✓ Built: bin/train_model.exe"
else
    echo "✗ Failed to build train_model.exe (see logs/build_train.log)"
    exit 1
fi

echo ""
echo "=== Build Complete ==="
echo ""
echo "Next steps:"
echo "  1. Collect training data:  ./bin/collect_data.exe"
echo "  2. Train the model:        ./bin/train_model.exe data/tokenized/train.bin"
echo ""
echo "Or use dummy data for testing:"
echo "  ./bin/train_model.exe"
echo ""

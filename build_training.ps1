#=============================================#
# GRIM Training Tools - CMake Build Script
#=============================================#

Write-Host "=== GRIM Training Tools Build (CMake) ===" -ForegroundColor Cyan
Write-Host ""

# Check for cmake
$cmakePath = Get-Command cmake -ErrorAction SilentlyContinue
if (-not $cmakePath) {
    Write-Host "ERROR: cmake not found!" -ForegroundColor Red
    Write-Host "Please install CMake from https://cmake.org/download/"
    exit 1
}

Write-Host "✓ Found CMake: $($cmakePath.Source)" -ForegroundColor Green

# Create build directory for training tools
$trainBuildDir = "build_training"
Write-Host "Creating build directory: $trainBuildDir" -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path $trainBuildDir | Out-Null

# Create output directories
New-Item -ItemType Directory -Force -Path bin | Out-Null
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
New-Item -ItemType Directory -Force -Path data\processed | Out-Null
New-Item -ItemType Directory -Force -Path data\tokenized | Out-Null
New-Item -ItemType Directory -Force -Path checkpoints | Out-Null
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path logs | Out-Null

Write-Host "✓ Directories created" -ForegroundColor Green

Write-Host ""
Write-Host "NOTE: The training code has complex dependencies." -ForegroundColor Yellow
Write-Host "It requires the full GRIM project to be built first." -ForegroundColor Yellow
Write-Host ""
Write-Host "To build the full GRIM project:" -ForegroundColor Cyan
Write-Host "  1. Open this folder in Visual Studio" -ForegroundColor White
Write-Host "  2. Or use CMake from command line:" -ForegroundColor White
Write-Host "     cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=vcpkg/scripts/buildsystems/vcpkg.cmake" -ForegroundColor DarkGray
Write-Host "     cmake --build build --config Release" -ForegroundColor DarkGray
Write-Host ""
Write-Host "The training code at line 1047 in lm_trainer.hpp has a TODO:" -ForegroundColor Yellow
Write-Host "  - Implement encoder backprop (access encoder layer parameters)" -ForegroundColor White
Write-Host "  - Implement optimizer parameter updates for encoder layers" -ForegroundColor White
Write-Host ""
Write-Host "You mentioned 'another chat implemented the stubs'." -ForegroundColor Cyan
Write-Host "Please check if those implementations exist elsewhere in the codebase." -ForegroundColor Cyan
Write-Host ""

#=============================================#
# GRIM Training Tools - Build Script
#=============================================#

Write-Host "=== GRIM Training Tools Build ===" -ForegroundColor Cyan
Write-Host ""

# Try common MSYS2 locations
$msys2Paths = @(
    "C:\msys64\mingw64\bin",
    "C:\msys32\mingw64\bin",
    "$env:USERPROFILE\msys64\mingw64\bin",
    "$env:USERPROFILE\msys32\mingw64\bin"
)

$mingwBin = $null
foreach ($path in $msys2Paths) {
    if (Test-Path "$path\g++.exe") {
        $mingwBin = $path
        Write-Host "✓ Found MinGW64 at: $mingwBin" -ForegroundColor Green
        break
    }
}

if (-not $mingwBin) {
    # Check if g++ is already in PATH
    $gppPath = Get-Command g++ -ErrorAction SilentlyContinue
    if (-not $gppPath) {
        Write-Host "ERROR: g++ not found!" -ForegroundColor Red
        Write-Host "Please install MinGW-w64 or MSYS2 with gcc"
        Write-Host ""
        Write-Host "For MSYS2, install gcc with:"
        Write-Host "  pacman -S mingw-w64-x86_64-gcc"
        exit 1
    }
    $mingwBin = Split-Path -Parent $gppPath.Source
}

# Add to PATH for this session
$env:PATH = "$mingwBin;$env:PATH"
Write-Host "✓ Added to PATH: $mingwBin" -ForegroundColor Green

# Verify g++ works
try {
    $version = & g++ --version 2>&1 | Select-Object -First 1
    Write-Host "✓ Compiler: $version" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Failed to run g++" -ForegroundColor Red
    exit 1
}

# Create directories
Write-Host ""
Write-Host "Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path bin | Out-Null
New-Item -ItemType Directory -Force -Path data\raw | Out-Null
New-Item -ItemType Directory -Force -Path data\processed | Out-Null
New-Item -ItemType Directory -Force -Path data\tokenized | Out-Null
New-Item -ItemType Directory -Force -Path checkpoints | Out-Null
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path logs | Out-Null
Write-Host "✓ Directories created" -ForegroundColor Green

# Set source directory
$srcDir = "resources\models\GRIM-text\training"
$vcpkgInclude = "vcpkg_installed\x64-windows\include"
$vcpkgLib = "vcpkg_installed\x64-windows\lib"

# Build data collection tool
Write-Host ""
Write-Host "Building data collection tool..." -ForegroundColor Yellow
$collectCmd = "g++ -std=c++17 -O3 -march=native -I. -I$srcDir -I$vcpkgInclude $srcDir\main_data_collection.cpp -o bin/collect_data.exe -L$vcpkgLib -lcurl -lpthread"
Write-Host "  Command: $collectCmd" -ForegroundColor DarkGray

& cmd /c "$collectCmd 2>&1" | Tee-Object -FilePath logs\build_collect.log | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Built: bin\collect_data.exe" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to build collect_data.exe" -ForegroundColor Red
    Write-Host "  See logs\build_collect.log for details" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  - Missing libcurl: pacman -S mingw-w64-x86_64-curl" -ForegroundColor DarkGray
    exit 1
}

# Build training tool
Write-Host ""
Write-Host "Building training tool..." -ForegroundColor Yellow
$trainCmd = "g++ -std=c++17 -O3 -march=native -mavx2 -mfma -fopenmp -I. -I$srcDir -I$vcpkgInclude $srcDir\train_model.cpp -o bin/train_model.exe -L$vcpkgLib -lpthread"
Write-Host "  Command: $trainCmd" -ForegroundColor DarkGray

& cmd /c "$trainCmd 2>&1" | Tee-Object -FilePath logs\build_train.log | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Built: bin\train_model.exe" -ForegroundColor Green
} else {
    Write-Host "✗ Failed to build train_model.exe" -ForegroundColor Red
    Write-Host "  See logs\build_train.log for details" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== Build Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Collect training data:  .\bin\collect_data.exe" -ForegroundColor White
Write-Host "  2. Train the model:        .\bin\train_model.exe data\tokenized\train.bin" -ForegroundColor White
Write-Host ""
Write-Host "Or use dummy data for testing:" -ForegroundColor Cyan
Write-Host "  .\bin\train_model.exe" -ForegroundColor White
Write-Host ""

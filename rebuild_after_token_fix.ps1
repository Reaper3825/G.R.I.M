# Rebuild GRIM Text Server After Token Fix
# Run this script to rebuild and test the fixed model

Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  GRIM Text Server - Rebuild After Token Fix" -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

# Paths
$GRIM_ROOT = "D:\G.R.I.M"
$BUILD_DIR = "$GRIM_ROOT\resources\models\GRIM-text\GRIM\build"
$SERVER_EXE = "$GRIM_ROOT\resources\models\GRIM-text\training\build\Release\grim_text_server.exe"

# Step 1: Navigate to build directory
Write-Host "[1/4] Navigating to build directory..." -ForegroundColor Yellow
if (-not (Test-Path $BUILD_DIR)) {
    Write-Host "  ERROR: Build directory not found: $BUILD_DIR" -ForegroundColor Red
    exit 1
}
Push-Location $BUILD_DIR

# Step 2: Clean previous build (optional)
Write-Host "[2/4] Cleaning previous build..." -ForegroundColor Yellow
if (Test-Path ".\Release") {
    Write-Host "  Removing old Release folder..." -ForegroundColor Gray
    Remove-Item -Recurse -Force ".\Release" -ErrorAction SilentlyContinue
}

# Step 3: Build the project
Write-Host "[3/4] Building project (Release, CUDA)..." -ForegroundColor Yellow
Write-Host "  This may take 2-5 minutes..." -ForegroundColor Gray
cmake --build . --config Release --target grim_text_server

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Build failed with exit code $LASTEXITCODE" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host "  Build successful!" -ForegroundColor Green

# Step 4: Verify executable exists
Write-Host "[4/4] Verifying executable..." -ForegroundColor Yellow
if (-not (Test-Path $SERVER_EXE)) {
    Write-Host "  ERROR: Server executable not found: $SERVER_EXE" -ForegroundColor Red
    Pop-Location
    exit 1
}

$fileSize = (Get-Item $SERVER_EXE).Length / 1MB
Write-Host "  Server executable: $SERVER_EXE" -ForegroundColor Green
Write-Host "  Size: $([math]::Round($fileSize, 2)) MB" -ForegroundColor Green

Pop-Location

Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Build Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Run the test: python test_grim_model.py" -ForegroundColor White
Write-Host "  2. Check for '[ERROR] Invalid token' messages in output" -ForegroundColor White
Write-Host "  3. Verify responses are now valid words" -ForegroundColor White
Write-Host ""
Write-Host "If you still see errors, check TOKEN_GENERATION_FIX.md" -ForegroundColor Cyan

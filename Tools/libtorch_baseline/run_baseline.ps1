#!/usr/bin/env pwsh
# Launch script for libtorch baseline with proper DLL paths

# Detect PyTorch installation
$PYTHON_TORCH_PATH = python -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>$null

if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrEmpty($PYTHON_TORCH_PATH)) {
    Write-Host "ERROR: Could not find PyTorch installation" -ForegroundColor Red
    Write-Host "Make sure PyTorch is installed: pip install torch" -ForegroundColor Yellow
    exit 1
}

$TORCH_LIB = Join-Path $PYTHON_TORCH_PATH "lib"

# Check if torch DLLs exist
if (-not (Test-Path (Join-Path $TORCH_LIB "torch_cpu.dll"))) {
    Write-Host "ERROR: PyTorch DLLs not found at: $TORCH_LIB" -ForegroundColor Red
    exit 1
}

# Add PyTorch lib directory to PATH
$env:PATH = "$TORCH_LIB;$env:PATH"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "LibTorch Baseline Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PyTorch Path: $PYTHON_TORCH_PATH" -ForegroundColor Green
Write-Host "DLL Path:     $TORCH_LIB" -ForegroundColor Green
Write-Host ""

# Get executable path
$EXE_PATH = Join-Path $PSScriptRoot "build\Release\grim_libtorch_baseline.exe"

if (-not (Test-Path $EXE_PATH)) {
    Write-Host "ERROR: Executable not found: $EXE_PATH" -ForegroundColor Red
    Write-Host "Run: cmake --build build --config Release" -ForegroundColor Yellow
    exit 1
}

# Change to repo root for data access
Push-Location (Join-Path $PSScriptRoot "..\..") -ErrorAction Stop

Write-Host "Working Directory: $(Get-Location)" -ForegroundColor Green
Write-Host "Launching: $EXE_PATH" -ForegroundColor Green
Write-Host ""

# Launch with arguments
& $EXE_PATH $args

$EXIT_CODE = $LASTEXITCODE
Pop-Location

if ($EXIT_CODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Process exited with code $EXIT_CODE" -ForegroundColor Red
}

exit $EXIT_CODE

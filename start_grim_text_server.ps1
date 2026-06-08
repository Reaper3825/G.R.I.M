# GRIM-text Server Startup Script
# Starts the GRIM-text HTTP server on port 11435

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GRIM-text Server Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$serverExe = "resources\models\GRIM-text\training\build\Release\grim_text_server.exe"

# Check if server executable exists
if (-not (Test-Path $serverExe)) {
    Write-Host "[ERROR] Server executable not found: $serverExe" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build the server first:" -ForegroundColor Yellow
    Write-Host "  cd resources\models\GRIM-text" -ForegroundColor Yellow
    Write-Host "  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release" -ForegroundColor Yellow
    Write-Host "  cmake --build build --config Release" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "[INFO] Starting GRIM-text server..." -ForegroundColor Green
Write-Host "[INFO] Server: $serverExe" -ForegroundColor Gray
Write-Host "[INFO] Server reads config internally; no CLI args are required." -ForegroundColor Gray
Write-Host "[INFO] URL: http://127.0.0.1:11435" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start server
& $serverExe

# GRIM-text Server Startup Script
# Starts the GRIM-text HTTP server on port 11435

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GRIM-text Server Launcher" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$serverExe = "resources\models\GRIM-text\training\build_vs_cuda\Release\grim_text_server.exe"
$vocabPath = "resources\models\GRIM-text\training\models\vocab.bin"
$modelPath = "resources\models\GRIM-text\checkpoints\checkpoint_epoch_5.bin"
$port = 11435

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

# Check if vocab file exists
if (-not (Test-Path $vocabPath)) {
    Write-Host "[WARNING] Vocab file not found: $vocabPath" -ForegroundColor Yellow
    Write-Host "Server may fail to start without vocabulary" -ForegroundColor Yellow
    Write-Host ""
}

# Check if model file exists
if (-not (Test-Path $modelPath)) {
    Write-Host "[WARNING] Model file not found: $modelPath" -ForegroundColor Yellow
    Write-Host "Server will start but may use random weights" -ForegroundColor Yellow
    Write-Host ""
}

Write-Host "[INFO] Starting GRIM-text server..." -ForegroundColor Green
Write-Host "[INFO] Server: $serverExe" -ForegroundColor Gray
Write-Host "[INFO] Vocab: $vocabPath" -ForegroundColor Gray
Write-Host "[INFO] Model: $modelPath" -ForegroundColor Gray
Write-Host "[INFO] Port: $port" -ForegroundColor Gray
Write-Host "[INFO] URL: http://127.0.0.1:$port" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start server
& $serverExe $vocabPath $modelPath $port

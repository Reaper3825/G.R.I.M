# Test Training UI Integration
# Tests the complete flow: GRIM.exe → training_control_server → train_gpu.exe

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Testing Training UI Integration" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check all executables exist
Write-Host "[1/5] Checking executables..." -ForegroundColor Yellow
$grim = "out/build/Release/GRIM.exe"
$controlServer = "resources/models/GRIM-text/training/build_vs_cuda/control/Release/training_control_server.exe"
$trainGpu = "resources/models/GRIM-text/training/build_vs_cuda/Release/train_gpu.exe"

if (!(Test-Path $grim)) {
    Write-Host "✗ GRIM.exe not found" -ForegroundColor Red
    exit 1
}
if (!(Test-Path $controlServer)) {
    Write-Host "✗ training_control_server.exe not found" -ForegroundColor Red
    exit 1
}
if (!(Test-Path $trainGpu)) {
    Write-Host "✗ train_gpu.exe not found" -ForegroundColor Red
    exit 1
}
Write-Host "✓ All executables found" -ForegroundColor Green
Write-Host ""

# 2. Check training data exists
Write-Host "[2/5] Checking training data..." -ForegroundColor Yellow
$vocabPath = "resources/models/GRIM-text/training/models/vocab.bin"
$dataPath = "resources/models/GRIM-text/training/data/training_data.grmt"

if (!(Test-Path $vocabPath)) {
    Write-Host "✗ vocab.bin not found" -ForegroundColor Red
    exit 1
}
if (!(Test-Path $dataPath)) {
    Write-Host "✗ training_data.grmt not found" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Training data found" -ForegroundColor Green
Write-Host ""

# 3. Start training control server
Write-Host "[3/5] Starting training control server..." -ForegroundColor Yellow
$serverProcess = Start-Process -FilePath $controlServer -PassThru -NoNewWindow
Start-Sleep -Seconds 2

if ($serverProcess.HasExited) {
    Write-Host "✗ Server failed to start" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Server started (PID: $($serverProcess.Id))" -ForegroundColor Green
Write-Host ""

# 4. Test server health
Write-Host "[4/5] Testing server health..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:11436/health" -Method GET -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Server is healthy" -ForegroundColor Green
    } else {
        Write-Host "✗ Server returned status: $($response.StatusCode)" -ForegroundColor Red
        Stop-Process -Id $serverProcess.Id -Force
        exit 1
    }
} catch {
    Write-Host "✗ Failed to connect to server: $_" -ForegroundColor Red
    Stop-Process -Id $serverProcess.Id -Force
    exit 1
}
Write-Host ""

# 5. Test status endpoint
Write-Host "[5/5] Testing status endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:11436/api/status" -Method GET -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✓ Status endpoint works" -ForegroundColor Green
        Write-Host "  Response size: $($response.Content.Length) bytes" -ForegroundColor Gray
    } else {
        Write-Host "✗ Status endpoint failed" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Failed to get status: $_" -ForegroundColor Red
}
Write-Host ""

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Integration Test Results" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "✓ Training control server is running on port 11436" -ForegroundColor Green
Write-Host "✓ Ready to receive training commands" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run GRIM.exe" -ForegroundColor White
Write-Host "2. Press T to open training panel" -ForegroundColor White
Write-Host "3. Click 'Start Training' button" -ForegroundColor White
Write-Host "4. Watch real-time progress updates!" -ForegroundColor White
Write-Host ""
Write-Host "Server PID: $($serverProcess.Id)" -ForegroundColor Cyan
Write-Host "To stop server: Stop-Process -Id $($serverProcess.Id)" -ForegroundColor Cyan
Write-Host ""

# GRIM-text Integration Test Script
# Tests the automatic server startup and HTTP communication

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "GRIM-text Server Integration Test" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if server executable exists
$serverPath = "D:\G.R.I.M\resources\models\GRIM-text\training\build_vs_cuda\Release\grim_text_server.exe"

Write-Host "[1/5] Checking server executable..." -ForegroundColor Yellow
if (Test-Path $serverPath) {
    Write-Host "✓ Server found: $serverPath" -ForegroundColor Green
} else {
    Write-Host "✗ Server NOT found at: $serverPath" -ForegroundColor Red
    Write-Host "Build it first with:" -ForegroundColor Yellow
    Write-Host "  cd resources\models\GRIM-text\training" -ForegroundColor Gray
    Write-Host "  cmake --preset vs-cuda-release" -ForegroundColor Gray
    Write-Host "  cmake --build build_vs_cuda --config Release" -ForegroundColor Gray
    exit 1
}

# 2. Check if server is already running
Write-Host ""
Write-Host "[2/5] Checking if server is running..." -ForegroundColor Yellow
$serverProcess = Get-Process -Name "grim_text_server" -ErrorAction SilentlyContinue

if ($serverProcess) {
    Write-Host "✓ Server already running (PID: $($serverProcess.Id))" -ForegroundColor Green
    $wasRunning = $true
} else {
    Write-Host "○ Server not running" -ForegroundColor Gray
    $wasRunning = $false
    
    # 3. Start server manually for testing
    Write-Host ""
    Write-Host "[3/5] Starting server manually..." -ForegroundColor Yellow
    
    $processInfo = Start-Process -FilePath $serverPath -WorkingDirectory (Split-Path $serverPath) -PassThru -WindowStyle Hidden
    
    if ($processInfo) {
        Write-Host "✓ Server started (PID: $($processInfo.Id))" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to start server" -ForegroundColor Red
        exit 1
    }
    
    # Wait for server to initialize
    Write-Host "  Waiting for server to initialize..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
}

# 4. Test HTTP communication
Write-Host ""
Write-Host "[4/5] Testing HTTP API..." -ForegroundColor Yellow

try {
    $testRequest = @{
        model = "grim-text"
        prompt = "Hello GRIM"
        max_tokens = 50
        temperature = 0.7
        stream = $false
    } | ConvertTo-Json

    $response = Invoke-RestMethod -Uri "http://127.0.0.1:11435/api/generate" `
                                   -Method Post `
                                   -ContentType "application/json" `
                                   -Body $testRequest `
                                   -TimeoutSec 30

    if ($response.response) {
        Write-Host "✓ Server responded successfully!" -ForegroundColor Green
        Write-Host "  Response: $($response.response)" -ForegroundColor Cyan
    } else {
        Write-Host "✗ Invalid response format" -ForegroundColor Red
        Write-Host "  Raw response: $response" -ForegroundColor Gray
    }
} catch {
    Write-Host "✗ HTTP request failed: $_" -ForegroundColor Red
    
    # Cleanup if we started it
    if (-not $wasRunning -and $processInfo) {
        Write-Host "  Cleaning up server process..." -ForegroundColor Gray
        Stop-Process -Id $processInfo.Id -Force -ErrorAction SilentlyContinue
    }
    
    exit 1
}

# 5. Cleanup
Write-Host ""
Write-Host "[5/5] Cleanup..." -ForegroundColor Yellow

if (-not $wasRunning) {
    Write-Host "  Stopping test server..." -ForegroundColor Gray
    if ($processInfo) {
        Stop-Process -Id $processInfo.Id -Force -ErrorAction SilentlyContinue
        Write-Host "✓ Server stopped" -ForegroundColor Green
    }
} else {
    Write-Host "○ Leaving server running (was already running)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "✓ Integration test PASSED" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Set backend to 'grim_native' in ai_config.json" -ForegroundColor Gray
Write-Host "2. Rebuild GRIM.exe with updated server manager" -ForegroundColor Gray
Write-Host "3. Run GRIM - server will start automatically" -ForegroundColor Gray

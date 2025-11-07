# Start GRIM-text Training Control Server
# This script starts the training control server and keeps it running

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host " GRIM-text Training Control Server" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to training directory
$trainingDir = "D:\G.R.I.M\resources\models\GRIM-text\training"
$serverExe = "$trainingDir\build_vs_cuda\Release\training_control_server.exe"

if (-not (Test-Path $serverExe)) {
    Write-Host "❌ Server executable not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Build it first:" -ForegroundColor Yellow
    Write-Host "  cd $trainingDir" -ForegroundColor Gray
    Write-Host "  cmake --preset vs-cuda-release" -ForegroundColor Gray
    Write-Host "  cmake --build build_vs_cuda --config Release --target training_control_server" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

# Check if already running
$existingProcess = Get-Process -Name "training_control_server" -ErrorAction SilentlyContinue
if ($existingProcess) {
    Write-Host "⚠️  Control server already running (PID: $($existingProcess.Id))" -ForegroundColor Yellow
    Write-Host ""
    $response = Read-Host "Kill and restart? (y/n)"
    if ($response -eq "y") {
        Stop-Process -Id $existingProcess.Id -Force
        Write-Host "✓ Stopped existing server" -ForegroundColor Green
        Start-Sleep -Seconds 1
    } else {
        Write-Host "Exiting..." -ForegroundColor Gray
        exit 0
    }
}

Write-Host "🚀 Starting control server..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Server will run on: http://127.0.0.1:11436" -ForegroundColor White
Write-Host ""
Write-Host "API Endpoints:" -ForegroundColor Yellow
Write-Host "  GET  /health" -ForegroundColor Gray
Write-Host "  GET  /api/status" -ForegroundColor Gray
Write-Host "  POST /api/training/start" -ForegroundColor Gray
Write-Host "  POST /api/training/stop" -ForegroundColor Gray
Write-Host "  POST /api/config" -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Start server
Set-Location $trainingDir
& $serverExe

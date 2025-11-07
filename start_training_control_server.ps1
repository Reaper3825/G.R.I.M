# Start GRIM-text Training Control Server
# This script starts the training control server in the background

$serverPath = ".\resources\models\GRIM-text\training\build_vs_cuda\control\Release\training_control_server.exe"
$workingDir = Get-Location

Write-Host "Starting Training Control Server..." -ForegroundColor Cyan
Write-Host "Server: $serverPath" -ForegroundColor Gray
Write-Host "Working Directory: $workingDir" -ForegroundColor Gray
Write-Host ""

# Kill any existing instances
Get-Process | Where-Object {$_.ProcessName -eq "training_control_server"} | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Start the server
$process = Start-Process -FilePath $serverPath `
    -WorkingDirectory $workingDir `
    -PassThru `
    -WindowStyle Hidden

if ($process) {
    Write-Host "✓ Server started (PID: $($process.Id))" -ForegroundColor Green
    Write-Host "  Waiting for server to bind..." -ForegroundColor Gray
    Start-Sleep -Seconds 3
    
    # Test connection
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:11436/health" -Method Get -TimeoutSec 5 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Host "✓ Server is responding on port 11436" -ForegroundColor Green
        }
    } catch {
        Write-Host "✗ Server not responding: $_" -ForegroundColor Red
        Write-Host "  Check if port 11436 is available" -ForegroundColor Yellow
    }
} else {
    Write-Host "✗ Failed to start server" -ForegroundColor Red
}

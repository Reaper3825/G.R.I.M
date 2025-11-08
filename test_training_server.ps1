# GRIM Training Server Debug & Test Script
# Tests server startup, connectivity, and API endpoints

Write-Host "======================================" -ForegroundColor Cyan
Write-Host " GRIM Training Server Debug Tool" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host ""

# Server configuration
$serverExe = ".\resources\models\GRIM-text\training\build_vs_cuda\control\Release\training_control_server.exe"
$serverPort = 11436
$serverUrl = "http://127.0.0.1:$serverPort"
$workingDir = Get-Location

# Function to test HTTP endpoint
function Test-Endpoint {
    param(
        [string]$Url,
        [string]$Method = "GET",
        [string]$Name = "Unknown"
    )
    
    try {
        Write-Host "  Testing $Name..." -NoNewline -ForegroundColor Gray
        $response = Invoke-WebRequest -Uri $Url -Method $Method -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
        Write-Host " ✓ OK (Status: $($response.StatusCode))" -ForegroundColor Green
        
        # Show response content if available
        if ($response.Content) {
            $contentPreview = $response.Content.Substring(0, [Math]::Min(100, $response.Content.Length))
            Write-Host "    Response preview: $contentPreview..." -ForegroundColor DarkGray
        }
        
        return $true
    } catch {
        Write-Host " ✗ FAILED" -ForegroundColor Red
        Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor DarkRed
        return $false
    }
}

# Function to check if port is in use
function Test-Port {
    param([int]$Port)
    
    try {
        $connection = New-Object System.Net.Sockets.TcpClient("127.0.0.1", $Port)
        $connection.Close()
        return $true
    } catch {
        return $false
    }
}

# Step 1: Check if server executable exists
Write-Host "[1/7] Checking server executable..." -ForegroundColor Yellow
if (Test-Path $serverExe) {
    Write-Host "  ✓ Found: $serverExe" -ForegroundColor Green
    $fileInfo = Get-Item $serverExe
    Write-Host "    Size: $($fileInfo.Length) bytes" -ForegroundColor Gray
    Write-Host "    Modified: $($fileInfo.LastWriteTime)" -ForegroundColor Gray
} else {
    Write-Host "  ✗ NOT FOUND: $serverExe" -ForegroundColor Red
    Write-Host "    This is a critical error - server cannot start without the executable" -ForegroundColor Yellow
    Write-Host "    Please build the training control server first" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 2: Check if port is already in use
Write-Host "[2/7] Checking port availability..." -ForegroundColor Yellow
$portInUse = Test-Port -Port $serverPort

if ($portInUse) {
    Write-Host "  ⚠ Port $serverPort is already in use" -ForegroundColor Yellow
    Write-Host "    Attempting to connect to existing server..." -ForegroundColor Gray
    
    if (Test-Endpoint -Url "$serverUrl/health" -Name "Health Check") {
        Write-Host "    ✓ Server already running and responding!" -ForegroundColor Green
        $useExisting = $true
    } else {
        Write-Host "    ✗ Port is occupied but server not responding" -ForegroundColor Red
        Write-Host "    Kill the process using port $serverPort and try again" -ForegroundColor Yellow
        
        # Try to find process using the port
        $netstat = netstat -ano | Select-String ":$serverPort "
        if ($netstat) {
            Write-Host "    Processes on port ${serverPort}:" -ForegroundColor Yellow
            $netstat | ForEach-Object {
                Write-Host "      $_" -ForegroundColor Gray
            }
        }
        exit 1
    }
} else {
    Write-Host "  ✓ Port $serverPort is available" -ForegroundColor Green
    $useExisting = $false
}

Write-Host ""

# Step 3: Start server if needed
if (-not $useExisting) {
    Write-Host "[3/7] Starting training control server..." -ForegroundColor Yellow
    Write-Host "  Command: $serverExe --port $serverPort" -ForegroundColor Gray
    Write-Host "  Working Dir: $workingDir" -ForegroundColor Gray
    
    # Kill any existing instances
    Get-Process | Where-Object {$_.ProcessName -eq "training_control_server"} | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    
    try {
        $process = Start-Process -FilePath $serverExe `
            -ArgumentList "--port $serverPort" `
            -WorkingDirectory $workingDir `
            -PassThru `
            -WindowStyle Hidden `
            -ErrorAction Stop
        
        Write-Host "  ✓ Process started (PID: $($process.Id))" -ForegroundColor Green
        Write-Host "    Waiting for server to bind to port..." -ForegroundColor Gray
        
        # Wait for server to bind to port (max 10 seconds)
        $maxWait = 10
        $waited = 0
        $serverReady = $false
        
        while ($waited -lt $maxWait) {
            Start-Sleep -Milliseconds 500
            $waited += 0.5
            
            if (Test-Port -Port $serverPort) {
                Write-Host "  ✓ Server bound to port after $waited seconds" -ForegroundColor Green
                $serverReady = $true
                break
            }
            
            # Show progress every 2 seconds
            if ($waited % 2 -eq 0) {
                Write-Host "    Still waiting... ($waited/$maxWait seconds)" -ForegroundColor Gray
            }
        }
        
        if (-not $serverReady) {
            Write-Host "  ✗ Server failed to bind to port after $maxWait seconds" -ForegroundColor Red
            Write-Host "    Check server logs for errors" -ForegroundColor Yellow
            exit 1
        }
        
    } catch {
        Write-Host "  ✗ Failed to start server: $($_.Exception.Message)" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[3/7] Using existing server instance" -ForegroundColor Yellow
}

Write-Host ""

# Step 4: Test health endpoint
Write-Host "[4/7] Testing server health..." -ForegroundColor Yellow
$healthOk = Test-Endpoint -Url "$serverUrl/health" -Name "GET /health"

if (-not $healthOk) {
    Write-Host "  ✗ Health check failed - server may not be fully initialized" -ForegroundColor Red
    Write-Host "    Wait a few more seconds and try again" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Step 5: Test status endpoint
Write-Host "[5/7] Testing status API..." -ForegroundColor Yellow
$statusOk = Test-Endpoint -Url "$serverUrl/api/status" -Name "GET /api/status"

if (-not $statusOk) {
    Write-Host "  ⚠ Status endpoint not responding correctly" -ForegroundColor Yellow
    Write-Host "    Server may be running but API not fully initialized" -ForegroundColor Gray
}

Write-Host ""

# Step 6: Test training start endpoint (dry run - just check if endpoint exists)
Write-Host "[6/7] Testing training API availability..." -ForegroundColor Yellow
Write-Host "  Note: Not actually starting training, just checking if endpoint exists" -ForegroundColor Gray

try {
    # Send empty POST to see if endpoint responds (it should error but respond)
    $response = Invoke-WebRequest -Uri "$serverUrl/api/training/start" `
        -Method POST `
        -ContentType "application/octet-stream" `
        -Body @() `
        -TimeoutSec 5 `
        -UseBasicParsing `
        -ErrorAction SilentlyContinue
    
    Write-Host "  ✓ Training endpoint accessible" -ForegroundColor Green
} catch {
    # 400/500 errors mean endpoint exists but rejected our empty request (expected)
    if ($_.Exception.Response.StatusCode -eq 400 -or $_.Exception.Response.StatusCode -eq 500) {
        Write-Host "  ✓ Training endpoint accessible (rejected empty request as expected)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Training endpoint may not be available" -ForegroundColor Yellow
        Write-Host "    Error: $($_.Exception.Message)" -ForegroundColor Gray
    }
}

Write-Host ""

# Step 7: Summary
Write-Host "[7/7] Test Summary" -ForegroundColor Yellow
Write-Host "  Server Executable: $(if (Test-Path $serverExe) {'✓'} else {'✗'}) Found" -ForegroundColor $(if (Test-Path $serverExe) {'Green'} else {'Red'})
Write-Host "  Port $serverPort`: $(if (Test-Port -Port $serverPort) {'✓'} else {'✗'}) In Use" -ForegroundColor $(if (Test-Port -Port $serverPort) {'Green'} else {'Red'})
Write-Host "  Health Check: $(if ($healthOk) {'✓'} else {'✗'}) $(if ($healthOk) {'Passed'} else {'Failed'})" -ForegroundColor $(if ($healthOk) {'Green'} else {'Red'})
Write-Host "  Status API: $(if ($statusOk) {'✓'} else {'✗'}) $(if ($statusOk) {'Passed'} else {'Failed'})" -ForegroundColor $(if ($statusOk) {'Green'} else {'Red'})

Write-Host ""

if ($healthOk -and $statusOk) {
    Write-Host "✓ All tests passed! Server is ready for training" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now start training from the GRIM UI" -ForegroundColor Cyan
    Write-Host "Or test manually with:" -ForegroundColor Cyan
    Write-Host "  curl http://127.0.0.1:11436/health" -ForegroundColor Gray
    Write-Host "  curl http://127.0.0.1:11436/api/status" -ForegroundColor Gray
} else {
    Write-Host "⚠ Some tests failed - check errors above" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "  1. Server exe not built - run build.ps1 first" -ForegroundColor Gray
    Write-Host "  2. Port already in use - kill existing process" -ForegroundColor Gray
    Write-Host "  3. Firewall blocking localhost - check Windows Firewall" -ForegroundColor Gray
    Write-Host "  4. Server crashing on startup - check server logs" -ForegroundColor Gray
}

Write-Host ""
Write-Host "======================================" -ForegroundColor Cyan

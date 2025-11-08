# Test the connection pool implementation
Write-Host "`n=== Connection Pool Test ===" -ForegroundColor Cyan
Write-Host "Testing thread-safe connection pool with persistent connections`n"

# Start monitoring connections
$initialConnections = (netstat -ano | Select-String "11436" | Measure-Object).Count
Write-Host "Initial connections on port 11436: $initialConnections" -ForegroundColor Yellow

# Test 1: Sequential requests (should reuse connections from pool)
Write-Host "`n[Test 1] Sequential requests (50 requests)..." -ForegroundColor Cyan
for ($i = 1; $i -le 50; $i++) {
    try {
        $null = Invoke-WebRequest -Uri "http://127.0.0.1:11436/api/status" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($i % 10 -eq 0) {
            $current = (netstat -ano | Select-String "11436" | Measure-Object).Count
            $timeWait = (netstat -ano | Select-String "11436" | Select-String "TIME_WAIT" | Measure-Object).Count
            Write-Host "  After $i requests: $current total connections ($timeWait TIME_WAIT)" -ForegroundColor Gray
        }
    } catch {
        Write-Host "  Request $i failed: $_" -ForegroundColor Red
    }
    Start-Sleep -Milliseconds 50
}

Start-Sleep -Seconds 2

# Test 2: Check final connection state
Write-Host "`n[Test 2] Final connection state..." -ForegroundColor Cyan
$finalConnections = (netstat -ano | Select-String "11436").Count
$timeWaitConnections = (netstat -ano | Select-String "11436" | Select-String "TIME_WAIT").Count
$establishedConnections = (netstat -ano | Select-String "11436" | Select-String "ESTABLISHED").Count

Write-Host "  Total connections: $finalConnections" -ForegroundColor Yellow
Write-Host "  TIME_WAIT: $timeWaitConnections" -ForegroundColor $(if ($timeWaitConnections -lt 20) { "Green" } else { "Red" })
Write-Host "  ESTABLISHED: $establishedConnections" -ForegroundColor Yellow
Write-Host "  Growth: +$($finalConnections - $initialConnections)" -ForegroundColor $(if (($finalConnections - $initialConnections) -lt 30) { "Green" } else { "Red" })

# Test 3: Check for checkpoint files
Write-Host "`n[Test 3] Checking for checkpoint files..." -ForegroundColor Cyan
$checkpointDirs = @("data", "data\checkpoints", "resources\models\GRIM-text\training\data", "resources\models\GRIM-text\training\data\checkpoints")
foreach ($dir in $checkpointDirs) {
    if (Test-Path $dir) {
        $checkpoints = Get-ChildItem $dir -Filter "checkpoint_*.json" -ErrorAction SilentlyContinue
        if ($checkpoints) {
            Write-Host "  Found $($checkpoints.Count) checkpoints in: $dir" -ForegroundColor Green
            $checkpoints | Select-Object -First 3 | ForEach-Object { Write-Host "    - $($_.Name)" -ForegroundColor Gray }
        }
    }
}

# Verdict
Write-Host "`n=== Results ===" -ForegroundColor Cyan
if ($timeWaitConnections -lt 20 -and ($finalConnections - $initialConnections) -lt 30) {
    Write-Host "✅ PASS: Connection pool working correctly!" -ForegroundColor Green
    Write-Host "   Connections properly reused, minimal leak" -ForegroundColor Green
    exit 0
} else {
    Write-Host "❌ FAIL: Connection leak still present" -ForegroundColor Red
    Write-Host "   Too many connections accumulated" -ForegroundColor Red
    exit 1
}

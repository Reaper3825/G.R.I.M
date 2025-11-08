# Test connection leak by simulating UI polling behavior
# This mimics the 200ms polling interval that happens in the UI

Write-Host "`n=== Connection Leak Test ===" -ForegroundColor Cyan
Write-Host "Simulating UI polling (200ms interval, 100 requests)" -ForegroundColor Cyan
Write-Host ""

$initialConnections = (netstat -ano | Select-String "11436").Count
Write-Host "Initial connections on port 11436: $initialConnections"

# Simulate 100 status polls (20 seconds of UI activity)
for ($i = 1; $i -le 100; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:11436/api/status" -Method GET -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($i % 10 -eq 0) {
            $currentConnections = (netstat -ano | Select-String "11436").Count
            $timeWait = (netstat -ano | Select-String "11436" | Select-String "TIME_WAIT").Count
            Write-Host "After $i requests: $currentConnections total, $timeWait TIME_WAIT" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Request $i failed: $_" -ForegroundColor Red
    }
    Start-Sleep -Milliseconds 200
}

Write-Host ""
Write-Host "=== Final Results ===" -ForegroundColor Cyan
$finalConnections = (netstat -ano | Select-String "11436").Count
$timeWaitConnections = (netstat -ano | Select-String "11436" | Select-String "TIME_WAIT").Count
$establishedConnections = (netstat -ano | Select-String "11436" | Select-String "ESTABLISHED").Count

Write-Host "Initial connections: $initialConnections"
Write-Host "Final connections: $finalConnections"
Write-Host "  - TIME_WAIT: $timeWaitConnections"
Write-Host "  - ESTABLISHED: $establishedConnections"
Write-Host "  - LISTENING: 1"

if ($timeWaitConnections -gt 50) {
    Write-Host "`nRESULT: CONNECTION LEAK DETECTED!" -ForegroundColor Red
    Write-Host "Too many TIME_WAIT connections accumulated." -ForegroundColor Red
    exit 1
} else {
    Write-Host "`nRESULT: No connection leak - connections properly reused!" -ForegroundColor Green
    exit 0
}

# show_current_log.ps1
# Helper script to identify and display info about the current training log

$logDir = "d:\G.R.I.M\resources\models\GRIM-text\training\logs"

Write-Host "`n=== GRIM-text Training Log Status ===`n" -ForegroundColor Cyan

# Find most recent log file
$latest = Get-ChildItem "$logDir\*.log" | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $latest) {
    Write-Host "No log files found in $logDir" -ForegroundColor Red
    exit 1
}

Write-Host "📁 Log Directory: $logDir" -ForegroundColor Gray
Write-Host "📝 Current Log: $($latest.Name)" -ForegroundColor Green
Write-Host "📅 Started: $($latest.LastWriteTime)" -ForegroundColor Gray
Write-Host "📊 Size: $([math]::Round($latest.Length / 1MB, 2)) MB`n" -ForegroundColor Gray

# Show basic stats
$content = Get-Content $latest.FullName
$lineCount = $content.Count
$lossLines = $content | Select-String "POST-FORWARD loss=" | Select-Object -Last 1

Write-Host "Statistics:" -ForegroundColor Yellow
Write-Host "  Total lines: $lineCount"
Write-Host "  Latest loss: $($lossLines.Line -replace '.*loss=', '' -replace ' .*', '')"

# Check for common issues
$zeroLoss = ($content | Select-String "loss=0\.0000").Count
$nanLoss = ($content | Select-String "loss=nan").Count
$infLoss = ($content | Select-String "loss=inf").Count

Write-Host "`nHealth Check:" -ForegroundColor Yellow
if ($zeroLoss -gt 0) {
    Write-Host "  ⚠️  Found $zeroLoss loss=0.0000 events" -ForegroundColor Red
} else {
    Write-Host "  ✅ No loss=0.0000 events" -ForegroundColor Green
}

if ($nanLoss -gt 0) {
    Write-Host "  ⚠️  Found $nanLoss NaN losses" -ForegroundColor Red
} else {
    Write-Host "  ✅ No NaN losses" -ForegroundColor Green
}

if ($infLoss -gt 0) {
    Write-Host "  ⚠️  Found $infLoss Inf losses" -ForegroundColor Red
} else {
    Write-Host "  ✅ No Inf losses" -ForegroundColor Green
}

# Show first and last few loss values
Write-Host "`nLoss Progression:" -ForegroundColor Yellow
$lossValues = $content | Select-String "POST-FORWARD loss=" | ForEach-Object {
    $_ -replace '.*loss=', '' -replace ' .*', ''
}

if ($lossValues.Count -gt 0) {
    $first5 = $lossValues | Select-Object -First 5
    $last5 = $lossValues | Select-Object -Last 5
    
    Write-Host "  First 5 batches: $($first5 -join ' → ')"
    if ($lossValues.Count -gt 5) {
        Write-Host "  Last 5 batches:  $($last5 -join ' → ')"
    }
}

# Command to open the log
Write-Host "`nQuick Actions:" -ForegroundColor Yellow
Write-Host "  View log:   Get-Content `"$($latest.FullName)`" | Select-String `"loss=`" | Select-Object -Last 20" -ForegroundColor Gray
Write-Host "  Edit log:   code `"$($latest.FullName)`"" -ForegroundColor Gray

Write-Host "`n" -ForegroundColor White

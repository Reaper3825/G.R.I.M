# ============================================================
# Cleanup Old Coqui TTS Installation
# ============================================================
# This script removes old Coqui TTS cache and temporary files
# Run this after upgrading to XTTS v2
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Cleaning Old Coqui TTS Data" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$cleanupPaths = @(
    "D:\G.R.I.M\resources\tts_out\temp",
    "$env:USERPROFILE\.local\share\tts"
)

$totalFreed = 0

foreach ($path in $cleanupPaths) {
    if (Test-Path $path) {
        Write-Host "Cleaning: $path" -ForegroundColor Yellow
        
        try {
            # Calculate size before deletion
            $size = (Get-ChildItem -Path $path -Recurse -File | Measure-Object -Property Length -Sum).Sum
            $sizeMB = [math]::Round($size / 1MB, 2)
            
            # Remove old model files (but keep XTTS v2)
            Get-ChildItem -Path $path -Recurse -Include "*.pth", "*.bin" | Where-Object {
                $_.FullName -notlike "*xtts_v2*"
            } | Remove-Item -Force -ErrorAction SilentlyContinue
            
            # Remove old temp files
            Get-ChildItem -Path $path -Recurse -Include "*.wav" | Where-Object {
                $_.LastWriteTime -lt (Get-Date).AddDays(-7)
            } | Remove-Item -Force -ErrorAction SilentlyContinue
            
            $totalFreed += $sizeMB
            Write-Host "  ? Cleaned $sizeMB MB" -ForegroundColor Green
            
        } catch {
            Write-Host "  ? Could not clean: $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Total space freed: $totalFreed MB" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan

# Rebuild cache index
Write-Host ""
Write-Host "Rebuilding TTS cache index..." -ForegroundColor Yellow

$cacheIndexPath = "D:\G.R.I.M\resources\tts_out\cache_index.json"
if (Test-Path $cacheIndexPath) {
    $cacheData = Get-Content $cacheIndexPath | ConvertFrom-Json
    $oldCount = ($cacheData.PSObject.Properties | Measure-Object).Count
    
    # Remove entries for missing files
    $newCache = @{}
    foreach ($key in $cacheData.PSObject.Properties.Name) {
        $entry = $cacheData.$key
        if (Test-Path $entry.file) {
            $newCache[$key] = $entry
        }
    }
    
    $newCount = $newCache.Count
    $removed = $oldCount - $newCount
    
    # Save updated cache
    $newCache | ConvertTo-Json | Set-Content $cacheIndexPath
    
    Write-Host "? Cache rebuilt: $removed entries removed, $newCount kept" -ForegroundColor Green
} else {
    Write-Host "? No cache index found (will be created on next use)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Cleanup complete!" -ForegroundColor Green

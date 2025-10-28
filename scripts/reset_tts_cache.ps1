# Clear old TTS cache and rebuild
# This fixes crashes from old cache entries pointing to moved files

Write-Host "=== TTS Cache Reset ===" -ForegroundColor Cyan
Write-Host ""

$cacheIndexPath = "D:\G.R.I.M\resources\tts_out\cache_index.json"
$tempDir = "D:\G.R.I.M\resources\tts_out\temp"
$cacheDir = "D:\G.R.I.M\resources\tts_out\cache"

# 1. Backup old cache index
if (Test-Path $cacheIndexPath) {
    $backup = "$cacheIndexPath.backup"
    Copy-Item $cacheIndexPath $backup -Force
    Write-Host "? Backed up cache index to: $backup" -ForegroundColor Green
}

# 2. Delete old cache index (will be rebuilt)
if (Test-Path $cacheIndexPath) {
    Remove-Item $cacheIndexPath -Force
    Write-Host "? Deleted old cache index" -ForegroundColor Green
}

# 3. Clean temp directory
if (Test-Path $tempDir) {
    Get-ChildItem $tempDir -Filter "*.wav" | Remove-Item -Force
    Write-Host "? Cleaned temp directory" -ForegroundColor Green
}

# 4. Verify cache directory exists
if (!(Test-Path $cacheDir)) {
    New-Item -ItemType Directory -Path $cacheDir -Force | Out-Null
    Write-Host "? Created cache directory" -ForegroundColor Green
}

Write-Host ""
Write-Host "=== Cache Reset Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "What this did:" -ForegroundColor Yellow
Write-Host "  • Backed up old cache index"
Write-Host "  • Cleared broken cache entries"
Write-Host "  • Cleaned temp directory"
Write-Host "  • Cache will rebuild automatically"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Restart GRIM"
Write-Host "  2. Pre-cache will run automatically"
Write-Host "  3. New cache entries will use correct paths"
Write-Host ""

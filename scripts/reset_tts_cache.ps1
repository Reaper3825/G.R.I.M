# Clear old TTS cache and rebuild
# This fixes crashes from old cache entries pointing to moved files

# Function to find GRIM root directory
function Get-GrimRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.PSCommandPath
    $currentDir = Get-Location
    
    foreach ($baseDir in @($scriptDir, $currentDir)) {
        $probe = $baseDir
        for ($i = 0; $i -lt 10; $i++) {
            if ((Test-Path (Join-Path $probe "control")) -and (Test-Path (Join-Path $probe "resources"))) {
                return $probe
            }
            $parent = Split-Path -Parent $probe
            if (-not $parent -or $parent -eq $probe) { break }
            $probe = $parent
        }
    }
    
    # Fallback: return script directory parent
    return (Split-Path -Parent $scriptDir)
}

Write-Host "=== TTS Cache Reset ===" -ForegroundColor Cyan
Write-Host ""

# Get GRIM root directory
$GrimRoot = Get-GrimRoot
Write-Host "GRIM root: $GrimRoot" -ForegroundColor Gray
Write-Host ""

$cacheIndexPath = Join-Path $GrimRoot "resources\tts_out\cache_index.json"
$tempDir = Join-Path $GrimRoot "resources\tts_out\temp"
$cacheDir = Join-Path $GrimRoot "resources\tts_out\cache"

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

# Clear TTS Cache for Fresh Voice Testing
# This removes all cached TTS files so new audio is generated with the correct speaker

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

# Get GRIM root directory
$GrimRoot = Get-GrimRoot

Write-Host "=== Clearing TTS Cache ===" -ForegroundColor Cyan

$paths = @(
    (Join-Path $GrimRoot "resources\tts_out\cache"),
    (Join-Path $GrimRoot "resources\tts_out\temp"),
    (Join-Path $GrimRoot "resources\tts_out\cache_index.json")
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host "Clearing: $path" -ForegroundColor Yellow
        Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Recreate directories
New-Item -ItemType Directory -Path (Join-Path $GrimRoot "resources\tts_out\cache") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $GrimRoot "resources\tts_out\temp") -Force | Out-Null

Write-Host "`n✓ TTS Cache cleared!" -ForegroundColor Green
Write-Host "Restart G.R.I.M to generate fresh audio with p226 voice`n" -ForegroundColor Cyan

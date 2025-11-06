# Clear TTS Cache for Fresh Voice Testing
# This removes all cached TTS files so new audio is generated with the correct speaker

Write-Host "=== Clearing TTS Cache ===" -ForegroundColor Cyan

$paths = @(
    "D:\G.R.I.M\resources\tts_out\cache",
    "D:\G.R.I.M\resources\tts_out\temp",
    "D:\G.R.I.M\resources\tts_out\cache_index.json"
)

foreach ($path in $paths) {
    if (Test-Path $path) {
        Write-Host "Clearing: $path" -ForegroundColor Yellow
        Remove-Item -Path $path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

# Recreate directories
New-Item -ItemType Directory -Path "D:\G.R.I.M\resources\tts_out\cache" -Force | Out-Null
New-Item -ItemType Directory -Path "D:\G.R.I.M\resources\tts_out\temp" -Force | Out-Null

Write-Host "`n✓ TTS Cache cleared!" -ForegroundColor Green
Write-Host "Restart G.R.I.M to generate fresh audio with p226 voice`n" -ForegroundColor Cyan

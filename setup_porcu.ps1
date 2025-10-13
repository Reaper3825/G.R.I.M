# ============================================================
# Correct Porcupine Windows SDK download (from official CDN)
# ============================================================

$zipUrl = "https://github.com/Picovoice/porcupine/releases/latest/download/porcupine-windows-x86_64.zip"

# Fallback official CDN mirror (in case GitHub fails)
$cdnUrl = "https://picovoice.ai/downloads/sdk/porcupine/porcupine-windows-x86_64.zip"

Write-Host "Downloading Porcupine SDK..."
try {
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -ErrorAction Stop
} catch {
    Write-Warning "GitHub download failed, trying Picovoice CDN..."
    Invoke-WebRequest -Uri $cdnUrl -OutFile $zipPath -ErrorAction Stop
}


$zipPath = "external\porcupine-$porcVersion.zip"
$extractPath = "external\porcupine"

# Create folders
New-Item -ItemType Directory -Force -Path "external" | Out-Null
New-Item -ItemType Directory -Force -Path "resources\wakeword" | Out-Null

# Download SDK
Write-Host "Downloading Porcupine SDK v$porcVersion..."
Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath

# Extract ZIP
Write-Host "Extracting SDK..."
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force
Remove-Item $zipPath

# Move expected structure
Write-Host "Standardizing folder structure..."
New-Item -ItemType Directory -Force -Path "$extractPath\lib\windows\x86_64" | Out-Null

# Move any found DLL/Libs into expected path
Get-ChildItem -Path "$extractPath" -Recurse -Include "pv_porcupine.dll","pv_porcupine.lib" |
    ForEach-Object {
        Move-Item $_.FullName "$extractPath\lib\windows\x86_64" -Force
    }

# Move .ppn wakeword if it exists in current folder
if (Test-Path ".\hey-grim_en_windows_v3_0_0.ppn") {
    Move-Item ".\hey-grim_en_windows_v3_0_0.ppn" "resources\wakeword\hey-grim_en_windows_v3_0_0.ppn" -Force
    Write-Host "Moved wakeword file to resources/wakeword/"
}

# Verify key files
$required = @(
    "$extractPath\include\pv_porcupine.h",
    "$extractPath\lib\windows\x86_64\pv_porcupine.lib",
    "$extractPath\lib\windows\x86_64\pv_porcupine.dll",
    "$extractPath\lib\common\porcupine_params.pv"
)
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "❌ Missing expected file: $f"
    } else {
        Write-Host "✅ Found $f"
    }
}

Write-Host "`nPorcupine SDK setup complete. Ready for CMake build."

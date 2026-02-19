# ============================================================
# Download Default XTTS v2 Voice Sample
# ============================================================
# This script downloads a default voice reference for XTTS v2
# ============================================================

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

$voiceDir = Join-Path $GrimRoot "resources\voices"
$defaultVoice = Join-Path $voiceDir "default.wav"

# Create directory
New-Item -ItemType Directory -Path $voiceDir -Force | Out-Null

if (Test-Path $defaultVoice) {
    Write-Host "? Default voice sample already exists" -ForegroundColor Green
    exit 0
}

Write-Host "Downloading default XTTS v2 voice sample..." -ForegroundColor Cyan

# Download a neutral English voice sample from Coqui's examples
$sampleUrl = "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav"

try {
    Invoke-WebRequest -Uri $sampleUrl -OutFile $defaultVoice
    Write-Host "? Default voice sample downloaded" -ForegroundColor Green
} catch {
    Write-Host "? Failed to download sample: $_" -ForegroundColor Red
    
    # Fallback: Create a silent WAV as placeholder
    Write-Host "Creating placeholder voice sample..." -ForegroundColor Yellow
    
    # Use Python to generate a simple WAV
    $pythonScript = @"
import numpy as np
import soundfile as sf

# Generate 1 second of silence at 22050 Hz (XTTS v2 default)
sample_rate = 22050
duration = 1.0
samples = np.zeros(int(sample_rate * duration), dtype=np.float32)

sf.write('$defaultVoice', samples, sample_rate)
print('Placeholder voice created')
"@
    
    $pythonScript | python
    
    if (Test-Path $defaultVoice) {
        Write-Host "? Placeholder voice sample created" -ForegroundColor Green
    } else {
        Write-Host "? Failed to create placeholder" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`nDefault voice path: $defaultVoice" -ForegroundColor Cyan

# ============================================================
# G.R.I.M Coqui XTTS v2 Setup Script
# ============================================================
# This script installs Coqui TTS with XTTS v2 support
# Requires: Python 3.9+ and NVIDIA GPU (recommended)
# ============================================================

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   G.R.I.M Coqui XTTS v2 Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "? Found: $pythonVersion" -ForegroundColor Green
    
    # Extract version number
    if ($pythonVersion -match "Python (\d+)\.(\d+)") {
        $major = [int]$Matches[1]
        $minor = [int]$Matches[2]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 9)) {
            Write-Host "? Python 3.9+ required (found $major.$minor)" -ForegroundColor Red
            exit 1
        }
    }
} catch {
    Write-Host "? Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Check CUDA availability (optional but recommended)
Write-Host ""
Write-Host "[2/5] Checking CUDA/GPU availability..." -ForegroundColor Yellow
try {
    $nvidiaSmi = nvidia-smi 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "? NVIDIA GPU detected - will enable GPU acceleration" -ForegroundColor Green
        $useGPU = $true
    } else {
        Write-Host "? No NVIDIA GPU detected - will use CPU (slower)" -ForegroundColor Yellow
        $useGPU = $false
    }
} catch {
    Write-Host "? nvidia-smi not found - assuming no GPU" -ForegroundColor Yellow
    $useGPU = $false
}

# Install PyTorch with CUDA support
Write-Host ""
Write-Host "[3/5] Installing PyTorch..." -ForegroundColor Yellow
if ($useGPU) {
    Write-Host "Installing PyTorch with CUDA 11.8 support..." -ForegroundColor Cyan
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Installing PyTorch (CPU only)..." -ForegroundColor Cyan
    python -m pip install torch torchvision torchaudio
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "? PyTorch installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "? PyTorch installed" -ForegroundColor Green

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

# Install Coqui TTS with dependencies
Write-Host ""
Write-Host "[4/5] Installing Coqui TTS (XTTS v2)..." -ForegroundColor Yellow
$requirementsPath = Join-Path $GrimRoot "resources\python\requirements.txt"

if (Test-Path $requirementsPath) {
    python -m pip install -r $requirementsPath
} else {
    # Fallback direct installation
    python -m pip install TTS>=0.22.0 numpy scipy soundfile pydub
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "? Coqui TTS installation failed" -ForegroundColor Red
    exit 1
}
Write-Host "? Coqui TTS installed" -ForegroundColor Green

# Download default voice reference for XTTS v2
Write-Host ""
Write-Host "[4.5/5] Setting up default voice reference..." -ForegroundColor Yellow

$voiceDir = Join-Path $GrimRoot "resources\voices"
$defaultVoice = Join-Path $voiceDir "default.wav"

New-Item -ItemType Directory -Path $voiceDir -Force | Out-Null

if (Test-Path $defaultVoice) {
    Write-Host "? Default voice sample already exists" -ForegroundColor Green
} else {
    Write-Host "Downloading default XTTS v2 voice sample..." -ForegroundColor Cyan
    
    # Download a neutral English voice sample
    $sampleUrl = "https://github.com/coqui-ai/TTS/raw/dev/tests/data/ljspeech/wavs/LJ001-0001.wav"
    
    try {
        Invoke-WebRequest -Uri $sampleUrl -OutFile $defaultVoice -ErrorAction Stop
        Write-Host "? Default voice sample downloaded" -ForegroundColor Green
    } catch {
        Write-Host "? Failed to download sample, creating placeholder..." -ForegroundColor Yellow
        
        # Create a simple WAV file using Python
        $createVoiceScript = @"
import numpy as np
import soundfile as sf

# Generate 1 second of low-volume noise (better than silence for XTTS v2)
sample_rate = 22050
duration = 1.0
samples = np.random.normal(0, 0.01, int(sample_rate * duration)).astype(np.float32)

sf.write('$defaultVoice', samples, sample_rate)
print('Placeholder voice created', flush=True)
"@
        
        $createVoiceScript | python
        
        if (Test-Path $defaultVoice) {
            Write-Host "? Placeholder voice sample created" -ForegroundColor Green
        } else {
            Write-Host "? Could not create default voice (will use first-run download)" -ForegroundColor Yellow
        }
    }
}

# Download XTTS v2 model (first run will download ~1.8GB)
Write-Host ""
Write-Host "[5/5] Pre-downloading XTTS v2 model..." -ForegroundColor Yellow
Write-Host "This will download ~1.8GB of model files on first run" -ForegroundColor Cyan

$testScript = @"
import sys
from TTS.api import TTS
try:
    print('Downloading XTTS v2 model...', file=sys.stderr)
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
    print('? Model downloaded successfully', file=sys.stderr)
except Exception as e:
    print(f'? Model download failed: {e}', file=sys.stderr)
    sys.exit(1)
"@

$testScript | python

if ($LASTEXITCODE -ne 0) {
    Write-Host "? Model download encountered issues (may download on first use)" -ForegroundColor Yellow
} else {
    Write-Host "? XTTS v2 model ready" -ForegroundColor Green
}

# Verify installation
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Verifying Installation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

$verifyScript = @"
import sys
try:
    from TTS.api import TTS
    import torch
    
    print('Python packages OK', file=sys.stderr)
    print(f'PyTorch version: {torch.__version__}', file=sys.stderr)
    print(f'CUDA available: {torch.cuda.is_available()}', file=sys.stderr)
    
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}', file=sys.stderr)
    
    sys.exit(0)
except Exception as e:
    print(f'Verification failed: {e}', file=sys.stderr)
    sys.exit(1)
"@

$verifyScript | python

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "? Installation complete!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Coqui XTTS v2 is ready to use." -ForegroundColor Cyan
    Write-Host ""
    if ($useGPU) {
        Write-Host "GPU acceleration is enabled for faster synthesis." -ForegroundColor Green
    } else {
        Write-Host "Running on CPU. For faster synthesis, install CUDA and rerun setup." -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Supported languages: en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "? Installation verification failed" -ForegroundColor Red
    Write-Host "Please check error messages above" -ForegroundColor Red
    exit 1
}

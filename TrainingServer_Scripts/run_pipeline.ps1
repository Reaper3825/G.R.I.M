# Complete GRIM-text Training Pipeline
# 1. Collect data from web sources
# 2. Verify and filter data
# 3. Train GPU model
#
# Uses local ai_config.json from GRIM root for training parameters.
# Edit ai_config.json to control hyperparameters, paths, etc.

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GRIM-text Complete Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Resolve paths relative to this script (TrainingServer_Scripts/ is under G.R.I.M root)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$grimRoot = (Get-Item $scriptDir).Parent.FullName
$exe_dir = Join-Path $grimRoot "resources\models\GRIM-text\training\build_vs_cuda\Release"
$training_dir = Join-Path $grimRoot "resources\models\GRIM-text\training"
$config_path = Join-Path $grimRoot "ai_config.json"

Set-Location $training_dir

# Step 1: Data Collection
Write-Host "[1/3] COLLECTING WEB DATA..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

if (Test-Path "$exe_dir\collect_data.exe") {
    & "$exe_dir\collect_data.exe"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Data collection complete" -ForegroundColor Green
    } else {
        Write-Host "❌ Data collection failed with code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ collect_data.exe not found" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 2: Data Verification
Write-Host "[2/3] VERIFYING DATA..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

if (Test-Path "$exe_dir\verifier.exe") {
    & "$exe_dir\verifier.exe"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Data verification complete" -ForegroundColor Green
    } else {
        Write-Host "❌ Data verification failed with code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ verifier.exe not found" -ForegroundColor Red
    exit 1
}

Write-Host ""

# Step 3: GPU Training
Write-Host "[3/3] TRAINING MODEL (GPU)..." -ForegroundColor Yellow
Write-Host "----------------------------------------" -ForegroundColor Gray

if (Test-Path "$exe_dir\train_gpu.exe") {
    if (!(Test-Path $config_path)) {
        Write-Host "❌ ai_config.json not found at: $config_path" -ForegroundColor Red
        Write-Host "   Create or copy ai_config.json to control training parameters" -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Using config: $config_path" -ForegroundColor Gray
    Set-Location $grimRoot
    & "$exe_dir\train_gpu.exe" --config $config_path
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Training complete" -ForegroundColor Green
    } else {
        Write-Host "❌ Training failed with code $LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ train_gpu.exe not found" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  PIPELINE COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan

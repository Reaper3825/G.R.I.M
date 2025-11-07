# Complete GRIM-text Training Pipeline
# 1. Collect data from web sources
# 2. Verify and filter data
# 3. Train GPU model

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GRIM-text Complete Training Pipeline" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$exe_dir = "d:\G.R.I.M\resources\models\GRIM-text\training\build_vs_cuda\Release"
$training_dir = "d:\G.R.I.M\resources\models\GRIM-text\training"

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
    & "$exe_dir\train_gpu.exe"
    
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

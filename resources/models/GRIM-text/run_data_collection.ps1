#!/usr/bin/env pwsh
# GRIM Data Collection Pipeline
# Collects web data and automatically merges it into training-ready format

param(
    [string]$Mode = "full",  # full, collect, verify, or merge
    [string]$Config = "DataCollection/source_data.json",
    [string]$OutputDir = "training/data"
)

Write-Host "`n=== GRIM DATA COLLECTION PIPELINE ===" -ForegroundColor Cyan
Write-Host "Mode: $Mode" -ForegroundColor Yellow
Write-Host "Output: $OutputDir`n" -ForegroundColor Yellow

# Ensure we're in the GRIM-text directory
Push-Location $PSScriptRoot

try {
    # Check if pipeline executable exists
    $pipelinePath = "DataCollection/build/Release/grim_data_pipeline.exe"
    if (-not (Test-Path $pipelinePath)) {
        Write-Host "❌ Pipeline not built. Building now..." -ForegroundColor Red
        
        # Build the pipeline
        Push-Location "DataCollection"
        if (-not (Test-Path "build")) {
            mkdir build | Out-Null
        }
        Push-Location "build"
        
        Write-Host "⚙️  Configuring CMake..." -ForegroundColor Yellow
        cmake .. -G "Visual Studio 17 2022" -A x64
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        Write-Host "🔨 Building pipeline..." -ForegroundColor Yellow
        cmake --build . --config Release --target grim_data_pipeline -j 16
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        Pop-Location
        Pop-Location
        
        Write-Host "✅ Pipeline built successfully`n" -ForegroundColor Green
    }
    
    # Run the pipeline
    Write-Host "🚀 Running data collection pipeline..." -ForegroundColor Cyan
    Write-Host "   This will: collect → verify → merge → tokenize`n" -ForegroundColor Gray
    
    Push-Location "DataCollection/build/Release"
    
    # Run based on mode
    switch ($Mode.ToLower()) {
        "full" {
            # Full pipeline: collect, verify, merge
            & .\grim_data_pipeline.exe full `
                --config "../../$Config" `
                --output-dir "../../../$OutputDir"
        }
        "collect" {
            # Just collect new data
            & .\grim_data_pipeline.exe collect `
                --config "../../$Config"
        }
        "verify" {
            # Just verify existing raw data
            & .\grim_data_pipeline.exe verify `
                --raw-dir "../../data/raw"
        }
        "merge" {
            # Just merge existing checkpoints
            & .\grim_data_pipeline.exe merge `
                --checkpoint-dir "../../data" `
                --verified-dir "../../../$OutputDir/verified" `
                --output-dir "../../../$OutputDir"
        }
        default {
            throw "Unknown mode: $Mode. Use: full, collect, verify, or merge"
        }
    }
    
    Pop-Location
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✅ Pipeline completed successfully!" -ForegroundColor Green
        Write-Host "`n📊 Training data ready at:" -ForegroundColor Cyan
        Write-Host "   $OutputDir/training_data.grmt" -ForegroundColor White
        Write-Host "   $OutputDir/tokenized/train.bin" -ForegroundColor White
        Write-Host "`n🎯 You can now start training from the UI" -ForegroundColor Yellow
    } else {
        Write-Host "`n❌ Pipeline failed with exit code: $LASTEXITCODE" -ForegroundColor Red
    }
    
} catch {
    Write-Host "`n❌ Error: $_" -ForegroundColor Red
    exit 1
} finally {
    Pop-Location
}

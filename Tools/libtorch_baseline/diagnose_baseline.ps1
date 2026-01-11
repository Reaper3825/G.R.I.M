#!/usr/bin/env pwsh
# Diagnostic run of PyTorch baseline with verbose error catching

$ErrorActionPreference = "Stop"

Write-Host "=== PyTorch Baseline Diagnostic Run ===" -ForegroundColor Cyan
Write-Host ""

$exe = "D:\G.R.I.M\Tools\libtorch_baseline\build\Release\grim_libtorch_baseline.exe"

if (-not (Test-Path $exe)) {
    Write-Host "ERROR: Executable not found at $exe" -ForegroundColor Red
    exit 1
}

Write-Host "Executable: $exe" -ForegroundColor Green
Write-Host ""

# Change to repo root so relative paths work
Set-Location "D:\G.R.I.M"
Write-Host "Working directory: $(Get-Location)" -ForegroundColor Green
Write-Host ""

$argsList = @(
    "--use_grmt", "1",
    "--vocab_path", "resources\models\GRIM-text\training\data\vocab.bin",
    "--grmt_path", "resources\models\GRIM-text\training\data\training_data.grmt",
    "--grmt_val_path", "resources\models\GRIM-text\training\data\validation_data.grmt",
    "--seq_len", "1024",
    "--batch_size", "6",
    "--n_layer", "12",
    "--n_head", "12",
    "--n_kv_head", "4",
    "--n_embd", "768",
    "--dropout", "0.0",
    "--lr", "0.0001",
    "--weight_decay", "0.01",
    "--epochs", "1",
    "--max_steps", "10",
    "--log_interval", "1",
    "--seed", "1",
    "--device", "cuda",
    "--use_rmsnorm", "1",
    "--tie_weights", "1",
    "--use_xavier_init", "1",
    "--max_tokens", "100000"
)

Write-Host "Running with arguments:" -ForegroundColor Yellow
for ($i = 0; $i -lt $argsList.Length; $i += 2) {
    Write-Host "  $($argsList[$i]) $($argsList[$i+1])"
}
Write-Host ""

try {
    & $exe $argsList
    $exitCode = $LASTEXITCODE
    
    if ($exitCode -eq 0) {
        Write-Host ""
        Write-Host "=== SUCCESS ===" -ForegroundColor Green
    } else {
        Write-Host ""
        Write-Host "=== FAILED (exit code: $exitCode) ===" -ForegroundColor Red
        
        if ($exitCode -eq -1073740791) {
            Write-Host "Exit code -1073740791 = 0xC0000409 = STATUS_STACK_BUFFER_OVERRUN" -ForegroundColor Yellow
            Write-Host "This indicates a buffer overflow or stack corruption." -ForegroundColor Yellow
            Write-Host ""
            Write-Host "Common causes:" -ForegroundColor Cyan
            Write-Host "  1. Fixed-size array accessed out of bounds"
            Write-Host "  2. Stack allocation too large (>1MB typically)"
            Write-Host "  3. Corrupted heap memory"
            Write-Host "  4. DLL version mismatch"
        }
    }
    
    exit $exitCode
    
} catch {
    Write-Host ""
    Write-Host "=== EXCEPTION ===" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Yellow
    exit 1
}

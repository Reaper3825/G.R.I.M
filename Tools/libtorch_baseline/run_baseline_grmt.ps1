#!/usr/bin/env pwsh
# Run PyTorch baseline with GRMT data (matching GRIM-text exactly)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "PyTorch Baseline with GRMT Data" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Paths (relative to repo root)
$GRMT_DATA = "resources\models\GRIM-text\training\data\training_data.grmt"
$VOCAB_PATH = "resources\models\GRIM-text\training\data\vocab.bin"

# Check if files exist
$REPO_ROOT = Join-Path $PSScriptRoot "..\..\"
if (-not (Test-Path (Join-Path $REPO_ROOT $GRMT_DATA))) {
    Write-Host "ERROR: GRMT data not found: $GRMT_DATA" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $REPO_ROOT $VOCAB_PATH))) {
    Write-Host "ERROR: Vocab file not found: $VOCAB_PATH" -ForegroundColor Red
    exit 1
}

Write-Host "Configuration (matching GRIM-text):" -ForegroundColor Yellow
Write-Host "  Dataset:     $GRMT_DATA" -ForegroundColor Green
Write-Host "  Vocab:       $VOCAB_PATH" -ForegroundColor Green
Write-Host "  Vocab Size:  50,376 tokens" -ForegroundColor Green
Write-Host "  Sequences:   21,161 (train)" -ForegroundColor Green
Write-Host "  Seq Length:  1024 tokens" -ForegroundColor Green
Write-Host "  Batch Size:  6" -ForegroundColor Green
Write-Host "  Epochs:      1 (for quick test)" -ForegroundColor Green
Write-Host "  Architecture: 12 layers, 768 hidden, 12 heads (GQA 12:4)" -ForegroundColor Green
Write-Host ""

# Build command arguments (matching GRIM-text config)
$CmdArgs = @(
    "--use_grmt", "1",
    "--grmt_path", $GRMT_DATA,
    "--vocab_path", $VOCAB_PATH,
    "--seq_len", "1024",
    "--batch_size", "6",
    "--n_layer", "12",
    "--n_head", "12",
    "--n_kv_head", "4",           # GQA 12:4 ratio
    "--n_embd", "768",
    "--epochs", "1",
    "--lr", "0.0003",
    "--warmup_steps", "0",        # Match GRIM-text (no warmup in recent tests)
    "--log_interval", "10",
    "--sample_interval", "100",   # Sample every 100 batches
    "--sample_tokens", "80",
    "--use_rmsnorm", "1",         # Match GRIM-text
    "--tie_weights", "1",         # Match GRIM-text
    "--seed", "1"                 # Same seed for reproducibility
)

Write-Host "Launching baseline with GRMT data..." -ForegroundColor Cyan
Write-Host ""

# Run the baseline script with our arguments
& (Join-Path $PSScriptRoot "run_baseline.ps1") @CmdArgs

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Baseline run completed" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

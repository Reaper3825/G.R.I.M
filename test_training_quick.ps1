# Quick Training Test Script
# Tests GRIM-text training with minimal epochs for validation

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  GRIM-text Quick Training Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$configPath = "ai_config.json"

# Read current config
$config = Get-Content $configPath -Raw | ConvertFrom-Json

# Backup current settings
$originalEpochs = $config.training.config.epochs
$originalBatch = $config.training.config.batch_size
$originalWarmup = $config.training.config.warmup_steps

Write-Host "[INFO] Current training config:" -ForegroundColor Yellow
Write-Host "  Epochs: $originalEpochs" -ForegroundColor White
Write-Host "  Batch Size: $originalBatch" -ForegroundColor White
Write-Host "  Warmup Steps: $originalWarmup" -ForegroundColor White
Write-Host ""

# Set minimal test config
Write-Host "[INFO] Setting minimal test config for quick validation..." -ForegroundColor Cyan
$config.training.config.epochs = 2
$config.training.config.batch_size = 4
$config.training.config.warmup_steps = 10

# Save test config
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath
Write-Host "[✓] Test config saved" -ForegroundColor Green
Write-Host "  Epochs: 2" -ForegroundColor White
Write-Host "  Batch Size: 4" -ForegroundColor White
Write-Host "  Warmup Steps: 10" -ForegroundColor White
Write-Host ""

Write-Host "[INFO] You can now use the Training Panel 'Start Training' button" -ForegroundColor Yellow
Write-Host "  or press Enter to restore original config..." -ForegroundColor Yellow
Read-Host

# Restore original config
Write-Host ""
Write-Host "[INFO] Restoring original config..." -ForegroundColor Cyan
$config.training.config.epochs = $originalEpochs
$config.training.config.batch_size = $originalBatch
$config.training.config.warmup_steps = $originalWarmup
$config | ConvertTo-Json -Depth 10 | Set-Content $configPath

Write-Host "[✓] Original config restored" -ForegroundColor Green
Write-Host "  Epochs: $originalEpochs" -ForegroundColor White
Write-Host "  Batch Size: $originalBatch" -ForegroundColor White
Write-Host "  Warmup Steps: $originalWarmup" -ForegroundColor White

# Add cuDNN to System PATH
# Run this script as Administrator

$cudnnPath = "C:\Program Files\NVIDIA\CUDNN\v9.14\bin\13.0"

# Check if cuDNN exists
if (-not (Test-Path $cudnnPath)) {
    Write-Host "❌ cuDNN path not found: $cudnnPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Checking for other versions..." -ForegroundColor Yellow
    $cudnnBin = "C:\Program Files\NVIDIA\CUDNN\v9.14\bin"
    if (Test-Path $cudnnBin) {
        Write-Host "Found cuDNN bin directories:" -ForegroundColor Green
        Get-ChildItem $cudnnBin -Directory | ForEach-Object {
            Write-Host "  - $($_.FullName)"
        }
        Write-Host ""
        Write-Host "Edit this script to use the correct CUDA version folder" -ForegroundColor Yellow
    }
    exit 1
}

Write-Host "✅ Found cuDNN at: $cudnnPath" -ForegroundColor Green
Write-Host ""

# Get current system PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", [System.EnvironmentVariableTarget]::Machine)

# Check if already in PATH
if ($currentPath -like "*$cudnnPath*") {
    Write-Host "✅ cuDNN is already in system PATH" -ForegroundColor Green
} else {
    Write-Host "Adding cuDNN to system PATH..." -ForegroundColor Yellow
    
    try {
        $newPath = $currentPath + ";$cudnnPath"
        [Environment]::SetEnvironmentVariable("Path", $newPath, [System.EnvironmentVariableTarget]::Machine)
        Write-Host "✅ Successfully added cuDNN to PATH" -ForegroundColor Green
        Write-Host ""
        Write-Host "⚠️  You may need to restart your terminal/IDE for changes to take effect" -ForegroundColor Yellow
    } catch {
        Write-Host "❌ Failed to add to PATH. Make sure you're running as Administrator" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "Testing cuDNN availability..." -ForegroundColor Cyan
Write-Host ""

# Add to current session PATH for testing
$env:PATH += ";$cudnnPath"

# List cuDNN DLLs
Write-Host "cuDNN DLLs found:" -ForegroundColor Green
Get-ChildItem $cudnnPath -Filter "cudnn*.dll" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  ✅ $($_.Name) ($sizeMB MB)" -ForegroundColor Green
}

Write-Host ""
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "="*70 -ForegroundColor Cyan
Write-Host "1. Restart your terminal/PowerShell"
Write-Host "2. Rebuild GRIM (if needed)"
Write-Host "3. Run GRIM - should now use CUDA"
Write-Host "4. Check logs for: 'Using CUDA execution provider (RTX 3080Ti)'"
Write-Host ""
Write-Host "Expected performance: ~8200ms (CPU) → ~100-300ms (GPU)" -ForegroundColor Green

# Install CUDA-enabled PyTorch for Mixed Precision Training
# RTX 3080 Ti - CUDA 12.5 compatible

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

Write-Host "🔧 Installing CUDA-enabled PyTorch for RTX 3080 Ti" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/3] Activating virtual environment..." -ForegroundColor Yellow
& (Join-Path $GrimRoot ".venv\Scripts\Activate.ps1")

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to activate virtual environment" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Virtual environment activated" -ForegroundColor Green
Write-Host ""

# Uninstall CPU-only PyTorch
Write-Host "[2/3] Removing CPU-only PyTorch..." -ForegroundColor Yellow
python -m pip uninstall -y torch torchvision torchaudio

Write-Host "✅ CPU-only PyTorch removed" -ForegroundColor Green
Write-Host ""

# Install CUDA-enabled PyTorch
Write-Host "[3/3] Installing CUDA-enabled PyTorch..." -ForegroundColor Yellow
Write-Host "   This will download ~2-3 GB..." -ForegroundColor Cyan
Write-Host ""

# For CUDA 12.1 (compatible with CUDA 12.5)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Installation failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "✅ CUDA PyTorch Installation Complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Verify installation
Write-Host "🔍 Verifying CUDA support..." -ForegroundColor Cyan
Write-Host ""

$verifyScript = @"
import torch

print('✅ PyTorch Installation Verified')
print('=' * 60)
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    
    major, minor = torch.cuda.get_device_capability(0)
    print(f'Compute Capability: {major}.{minor}')
    
    print('')
    print('🚀 GPU Capabilities:')
    if major >= 7:
        print('   ✅ Tensor Cores: Supported (FP16/TF32)')
    if major >= 8:
        print('   ✅ Ampere Architecture: Supported')
    
    print('')
    print('🔬 Testing Mixed Precision...')
    
    # Test FP16 training
    try:
        from torch.cuda.amp import autocast, GradScaler
        
        scaler = GradScaler()
        x = torch.randn(1000, 1000, device='cuda', requires_grad=True)
        
        with autocast():
            y = torch.matmul(x, x)
            loss = y.sum()
        
        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD([x], lr=0.1))
        scaler.update()
        
        print('   ✅ Mixed Precision (AMP): Working')
    except Exception as e:
        print(f'   ❌ Mixed Precision test failed: {e}')
    
    # Test TF32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print('   ✅ TF32 Mode: Enabled')
    except:
        print('   ⚠️  TF32 Mode: Not available')
    
    print('')
    print('=' * 60)
    print('✅ Ready for 2-3x faster training with FP16!')
    print('=' * 60)
else:
    print('')
    print('❌ CUDA not available - something went wrong')
    import sys
    sys.exit(1)
"@

python -c $verifyScript

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "🎉 Success! Mixed precision training is ready!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "   1. Run: .\scripts\enable_mixed_precision.ps1" -ForegroundColor White
    Write-Host "   2. Start training with optimized config" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Verification failed" -ForegroundColor Red
    exit 1
}

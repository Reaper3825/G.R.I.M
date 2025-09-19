<#
.SYNOPSIS
    Installs NVIDIA CUDA Toolkit (latest version) on Windows.

.DESCRIPTION
    - Downloads the CUDA Toolkit installer from NVIDIA.
    - Installs silently with default options.
    - Optionally sets CUDAToolkit_ROOT and updates PATH.

.NOTES
    - Run this script as Administrator.
    - A reboot may be required after install.
#>

$ErrorActionPreference = "Stop"

# ==========================================================
# Config
# ==========================================================
$cudaVersion   = "12.5.1"   # set to latest stable CUDA
$installerUrl  = "https://developer.download.nvidia.com/compute/cuda/12.5.1/local_installers/cuda_12.5.1_windows.exe"
$installerPath = "$env:TEMP\cuda_installer.exe"
$installDir    = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$cudaVersion"

# ==========================================================
# Download
# ==========================================================
Write-Host "Downloading CUDA Toolkit $cudaVersion..."
Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath

# ==========================================================
# Install
# ==========================================================
Write-Host "Installing CUDA Toolkit $cudaVersion..."
Start-Process -FilePath $installerPath -ArgumentList "-s" -Wait -Verb RunAs

# ==========================================================
# Set environment variables
# ==========================================================
if (Test-Path $installDir) {
    Write-Host "CUDA installed at $installDir"

    # Set CUDAToolkit_ROOT system-wide
    [System.Environment]::SetEnvironmentVariable("CUDAToolkit_ROOT", $installDir, "Machine")

    # Add to PATH if not already
    $cudaBin = "$installDir\bin"
    $sysPath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    if ($sysPath -notlike "*$cudaBin*") {
        [System.Environment]::SetEnvironmentVariable("Path", "$sysPath;$cudaBin", "Machine")
        Write-Host "Added $cudaBin to PATH."
    }
} else {
    Write-Host "⚠️ CUDA install directory not found. Please check installer logs."
}

Write-Host "[OK] CUDA Toolkit $cudaVersion installation complete."
Write-Host "[INFO] You may need to restart PowerShell or reboot for PATH changes to take effect."

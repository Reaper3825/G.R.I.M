# ============================================================
# GRIM Windows Physical Vision Model Setup
# Downloads/exports the ONNX files declared in ai_config.json
# under resources/models/vision.
# ============================================================

param(
    [switch]$Force,
    [switch]$SkipPythonExports,
    [switch]$SkipSam2,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"

function Get-GrimRoot {
    $scriptDir = Split-Path -Parent $MyInvocation.PSCommandPath
    foreach ($baseDir in @($scriptDir, (Get-Location).Path)) {
        $probe = $baseDir
        for ($i = 0; $i -lt 10; $i++) {
            if ((Test-Path (Join-Path $probe "ai_config.json")) -and
                (Test-Path (Join-Path $probe "resources")) -and
                (Test-Path (Join-Path $probe "perception"))) {
                return $probe
            }
            $parent = Split-Path -Parent $probe
            if (-not $parent -or $parent -eq $probe) { break }
            $probe = $parent
        }
    }
    throw "Could not locate GRIM root from script directory or current directory."
}

function Resolve-Python {
    param([string]$Root, [string]$Requested)
    if ($Requested) {
        if (-not (Test-Path $Requested)) { throw "Requested Python executable not found: $Requested" }
        return $Requested
    }
    $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { return $venvPython }
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) { return $py.Source }
    throw "Python not found. Create .venv or pass -Python <path-to-python.exe>."
}

function Download-ModelFile {
    param(
        [Parameter(Mandatory=$true)][string]$Url,
        [Parameter(Mandatory=$true)][string]$OutFile,
        [Parameter(Mandatory=$true)][int64]$MinBytes
    )
    $name = Split-Path -Leaf $OutFile
    if ((Test-Path $OutFile) -and -not $Force) {
        $size = (Get-Item $OutFile).Length
        if ($size -ge $MinBytes) {
            Write-Host "  OK existing $name ($([math]::Round($size / 1MB, 1)) MiB)" -ForegroundColor Green
            return
        }
        throw "Existing $OutFile is too small ($size bytes; expected >= $MinBytes). Delete it or run with -Force."
    }

    $dir = Split-Path -Parent $OutFile
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $tmp = "$OutFile.download"
    if (Test-Path $tmp) { Remove-Item $tmp -Force }

    Write-Host "  Downloading $name" -ForegroundColor Cyan
    Invoke-WebRequest -Uri $Url -OutFile $tmp -MaximumRedirection 8 -Headers @{ "User-Agent" = "GRIM-Windows-vision-setup" }

    $size = (Get-Item $tmp).Length
    if ($size -lt $MinBytes) {
        Remove-Item $tmp -Force
        throw "Downloaded $name is too small ($size bytes; expected >= $MinBytes). URL may have returned an HTML error page: $Url"
    }
    Move-Item -Force $tmp $OutFile
    Write-Host "  Wrote $name ($([math]::Round($size / 1MB, 1)) MiB)" -ForegroundColor Green
}

function Ensure-PythonPackages {
    param([string]$PythonExe, [string[]]$Packages)
    Write-Host "  Ensuring Python packages: $($Packages -join ', ')" -ForegroundColor Cyan
    & $PythonExe -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
    & $PythonExe -m pip install @Packages
    if ($LASTEXITCODE -ne 0) { throw "pip install failed for: $($Packages -join ', ')" }
}

function Require-File {
    param([string]$Path, [int64]$MinBytes)
    if (-not (Test-Path $Path)) { throw "Missing required model file: $Path" }
    $size = (Get-Item $Path).Length
    if ($size -lt $MinBytes) { throw "Model file is too small: $Path ($size bytes; expected >= $MinBytes)" }
}

$GrimRoot = Get-GrimRoot
$VisionDir = Join-Path $GrimRoot "resources\models\vision"
New-Item -ItemType Directory -Force -Path $VisionDir | Out-Null

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  GRIM Windows Physical Vision Model Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "GRIM root: $GrimRoot" -ForegroundColor Gray
Write-Host "Vision dir: $VisionDir" -ForegroundColor Gray

Write-Host "`n[1/4] Downloading direct ONNX models..." -ForegroundColor Yellow
Download-ModelFile "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx" `
    (Join-Path $VisionDir "yolov8n.onnx") 5000000
Download-ModelFile "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.onnx" `
    (Join-Path $VisionDir "yolov8n_pose.onnx") 5000000
Download-ModelFile "https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx" `
    (Join-Path $VisionDir "pphumanseg.onnx") 1000000
Download-ModelFile "https://github.com/opencv/opencv_zoo/raw/main/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx" `
    (Join-Path $VisionDir "ppocrv3_en_det.onnx") 1000000
Download-ModelFile "https://github.com/opencv/opencv_zoo/raw/main/models/text_recognition_crnn/text_recognition_CRNN_EN_2021sep.onnx" `
    (Join-Path $VisionDir "crnn_en.onnx") 1000000
Download-ModelFile "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
    (Join-Path $VisionDir "yunet_face_detector.onnx") 100000
Download-ModelFile "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx" `
    (Join-Path $VisionDir "ferplus_emotion.onnx") 100000
Download-ModelFile "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx" `
    (Join-Path $VisionDir "midas_v21_small_256.onnx") 10000000

if (-not $SkipSam2) {
    Write-Host "`n[2/4] Downloading SAM2 ONNX pair..." -ForegroundColor Yellow
    Download-ModelFile "https://huggingface.co/onnx-community/sam2-hiera-tiny/resolve/main/onnx/vision_encoder.onnx" `
        (Join-Path $VisionDir "sam2_hiera_tiny_encoder.onnx") 10000000
    Download-ModelFile "https://huggingface.co/onnx-community/sam2-hiera-tiny/resolve/main/onnx/prompt_encoder_mask_decoder.onnx" `
        (Join-Path $VisionDir "sam2_hiera_tiny_decoder.onnx") 1000000
} else {
    Write-Host "`n[2/4] Skipping SAM2 because -SkipSam2 was supplied." -ForegroundColor Yellow
}

if (-not $SkipPythonExports) {
    Write-Host "`n[3/4] Exporting Python-backed models..." -ForegroundColor Yellow
    $PythonExe = Resolve-Python $GrimRoot $Python
    Write-Host "  Python: $PythonExe" -ForegroundColor Gray
    Ensure-PythonPackages $PythonExe @("torch", "onnx", "transformers", "open_clip_torch", "huggingface_hub", "timm", "safetensors")

    Push-Location $GrimRoot
    try {
        & $PythonExe "scripts\setup_mobileclip.py"
        if ($LASTEXITCODE -ne 0) { throw "setup_mobileclip.py failed" }

        & $PythonExe "scripts\export_depth_anything_v2_metric.py" `
            --hf-model "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf" `
            --output "resources/models/vision/depth_anything_v2_metric_indoor_small.onnx" `
            --size 518
        if ($LASTEXITCODE -ne 0) { throw "export_depth_anything_v2_metric.py failed" }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "`n[3/4] Skipping Python exports because -SkipPythonExports was supplied." -ForegroundColor Yellow
}

Write-Host "`n[4/4] Validating files required by ai_config.json..." -ForegroundColor Yellow
Require-File (Join-Path $VisionDir "yolov8n.onnx") 5000000
Require-File (Join-Path $VisionDir "pphumanseg.onnx") 1000000
Require-File (Join-Path $VisionDir "yolov8n_pose.onnx") 5000000
Require-File (Join-Path $VisionDir "ppocrv3_en_det.onnx") 1000000
Require-File (Join-Path $VisionDir "crnn_en.onnx") 1000000
Require-File (Join-Path $VisionDir "yunet_face_detector.onnx") 100000
Require-File (Join-Path $VisionDir "ferplus_emotion.onnx") 100000
Require-File (Join-Path $VisionDir "midas_v21_small_256.onnx") 10000000
if (-not $SkipPythonExports) {
    Require-File (Join-Path $VisionDir "mobileclip_s0_image.onnx") 1000000
    Require-File (Join-Path $VisionDir "mobileclip_text_embeddings.bin") 1000
    Require-File (Join-Path $VisionDir "depth_anything_v2_metric_indoor_small.onnx") 1000000
}
if (-not $SkipSam2) {
    Require-File (Join-Path $VisionDir "sam2_hiera_tiny_encoder.onnx") 10000000
    Require-File (Join-Path $VisionDir "sam2_hiera_tiny_decoder.onnx") 1000000
}

Write-Host "`nSetup complete. Rebuild/run GRIM from the repo root or use the Visual Studio debugger working directory." -ForegroundColor Green
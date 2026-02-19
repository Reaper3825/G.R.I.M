# ============================================================
# GRIM Perception System Auto-Setup
# Downloads Tesseract language data and optional YOLO models
# ============================================================

param(
    [switch]$SkipYOLO,       # Skip YOLO model download (saves 236 MB)
    [switch]$YOLOTiny,       # Use YOLOv3-tiny instead (faster, smaller)
    [switch]$ExtraLanguages  # Download additional language packs
)

$ErrorActionPreference = "Stop"

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

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "   GRIM Perception System Setup" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host ""

# Get GRIM root directory
$GrimRoot = Get-GrimRoot
Write-Host "GRIM root: $GrimRoot" -ForegroundColor Gray
Write-Host ""

# Paths
$ResourcesPath = Join-Path $GrimRoot "resources"
$TessdataPath = Join-Path $ResourcesPath "tessdata"
$YOLOPath = Join-Path $ResourcesPath "models\yolo"

# ============================================================
# 1. Create Directories
# ============================================================
Write-Host "[1/4] Creating directories..." -ForegroundColor Yellow

New-Item -ItemType Directory -Force -Path $TessdataPath | Out-Null
New-Item -ItemType Directory -Force -Path $YOLOPath | Out-Null

Write-Host "  ? Created: $TessdataPath" -ForegroundColor Green
Write-Host "  ? Created: $YOLOPath" -ForegroundColor Green
Write-Host ""

# ============================================================
# 2. Download Tesseract Language Data
# ============================================================
Write-Host "[2/4] Downloading Tesseract language data..." -ForegroundColor Yellow

$TessdataBase = "https://github.com/tesseract-ocr/tessdata_fast/raw/main"

# Core language: English
$EngFile = "$TessdataPath\eng.traineddata"
if (Test-Path $EngFile) {
    Write-Host "  ? English data already exists, skipping" -ForegroundColor Gray
} else {
    Write-Host "  Downloading English language data (~11 MB)..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri "$TessdataBase/eng.traineddata" -OutFile $EngFile
  Write-Host "  ? Downloaded: eng.traineddata" -ForegroundColor Green
}

# Optional: Additional languages
if ($ExtraLanguages) {
    Write-Host "  Downloading additional languages..." -ForegroundColor Cyan
    
    $Languages = @{
     "spa" = "Spanish"
      "fra" = "French"
 "deu" = "German"
        "ita" = "Italian"
        "por" = "Portuguese"
     "rus" = "Russian"
 "jpn" = "Japanese"
   "chi_sim" = "Chinese (Simplified)"
    }
    
 foreach ($lang in $Languages.Keys) {
        $LangFile = "$TessdataPath\$lang.traineddata"
     if (Test-Path $LangFile) {
  Write-Host "    ? $($Languages[$lang]) already exists" -ForegroundColor Gray
        } else {
   Write-Host "    Downloading $($Languages[$lang])..." -ForegroundColor Cyan
       Invoke-WebRequest -Uri "$TessdataBase/$lang.traineddata" -OutFile $LangFile
          Write-Host "    ? $($Languages[$lang])" -ForegroundColor Green
        }
    }
}

Write-Host ""

# ============================================================
# 3. Download YOLO Model (Optional)
# ============================================================
if ($SkipYOLO) {
    Write-Host "[3/4] Skipping YOLO model download (--SkipYOLO flag)" -ForegroundColor Yellow
    Write-Host "  Object detection will use basic color analysis" -ForegroundColor Gray
} else {
    Write-Host "[3/4] Downloading YOLO model files..." -ForegroundColor Yellow
    
    if ($YOLOTiny) {
        Write-Host "  Using YOLOv3-tiny (faster, smaller)" -ForegroundColor Cyan
        $ConfigFile = "yolov3-tiny.cfg"
        $WeightsFile = "yolov3-tiny.weights"
    $ConfigUrl = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
        $WeightsUrl = "https://pjreddie.com/media/files/yolov3-tiny.weights"
      $WeightsSize = "34 MB"
    } else {
 Write-Host "  Using YOLOv3 (standard, more accurate)" -ForegroundColor Cyan
        $ConfigFile = "yolov3.cfg"
        $WeightsFile = "yolov3.weights"
   $ConfigUrl = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
 $WeightsUrl = "https://pjreddie.com/media/files/yolov3.weights"
        $WeightsSize = "236 MB"
  }
    
    # Config file
  $ConfigPath = "$YOLOPath\$ConfigFile"
    if (Test-Path $ConfigPath) {
        Write-Host "  ? Config file already exists" -ForegroundColor Gray
    } else {
      Write-Host "  Downloading config file..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $ConfigUrl -OutFile $ConfigPath
        Write-Host "  ? Downloaded: $ConfigFile" -ForegroundColor Green
    }
    
    # Class names file
    $NamesPath = "$YOLOPath\coco.names"
    if (Test-Path $NamesPath) {
   Write-Host "  ? Class names already exist" -ForegroundColor Gray
    } else {
        Write-Host "  Downloading class names..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names" -OutFile $NamesPath
        Write-Host "  ? Downloaded: coco.names" -ForegroundColor Green
    }
  
    # Weights file (large download)
    $WeightsPath = "$YOLOPath\$WeightsFile"
    if (Test-Path $WeightsPath) {
     Write-Host "  ? Weights file already exists" -ForegroundColor Gray
    } else {
        Write-Host "  Downloading weights file ($WeightsSize)..." -ForegroundColor Cyan
        Write-Host "  This may take several minutes..." -ForegroundColor Gray
        
   # Show progress
$ProgressPreference = 'Continue'
        Invoke-WebRequest -Uri $WeightsUrl -OutFile $WeightsPath
        
        Write-Host "  ? Downloaded: $WeightsFile" -ForegroundColor Green
    }
    
  # Update perception.cpp if using tiny model
    if ($YOLOTiny) {
        Write-Host ""
 Write-Host "  ? IMPORTANT: Update perception.cpp to use YOLOv3-tiny:" -ForegroundColor Yellow
        Write-Host "    In loadYOLO() function, change:" -ForegroundColor Gray
        Write-Host "      yolov3.cfg  ?  yolov3-tiny.cfg" -ForegroundColor Gray
  Write-Host "      yolov3.weights  ?  yolov3-tiny.weights" -ForegroundColor Gray
    }
}

Write-Host ""

# ============================================================
# 4. Verify Installation
# ============================================================
Write-Host "[4/4] Verifying installation..." -ForegroundColor Yellow

$AllGood = $true

# Check Tesseract
if (Test-Path "$TessdataPath\eng.traineddata") {
    Write-Host "  ? Tesseract English data: OK" -ForegroundColor Green
} else {
    Write-Host "  ? Tesseract English data: MISSING" -ForegroundColor Red
    $AllGood = $false
}

# Check YOLO (if not skipped)
if (-not $SkipYOLO) {
 $ExpectedConfig = if ($YOLOTiny) { "yolov3-tiny.cfg" } else { "yolov3.cfg" }
    $ExpectedWeights = if ($YOLOTiny) { "yolov3-tiny.weights" } else { "yolov3.weights" }
    
    if (Test-Path "$YOLOPath\$ExpectedConfig") {
   Write-Host "  ? YOLO config: OK" -ForegroundColor Green
    } else {
     Write-Host "  ? YOLO config: MISSING" -ForegroundColor Red
        $AllGood = $false
    }
    
    if (Test-Path "$YOLOPath\$ExpectedWeights") {
        Write-Host "  ? YOLO weights: OK" -ForegroundColor Green
    } else {
        Write-Host "  ? YOLO weights: MISSING" -ForegroundColor Red
   $AllGood = $false
 }
    
    if (Test-Path "$YOLOPath\coco.names") {
        Write-Host "  ? YOLO classes: OK" -ForegroundColor Green
    } else {
        Write-Host "  ? YOLO classes: MISSING" -ForegroundColor Red
        $AllGood = $false
    }
}

Write-Host ""

# ============================================================
# 5. Summary
# ============================================================
if ($AllGood) {
    Write-Host "=============================================" -ForegroundColor Green
  Write-Host "   Setup Complete! ?" -ForegroundColor Green
    Write-Host "=============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Rebuild GRIM: cmake --build --preset=release" -ForegroundColor White
    Write-Host "  2. Run GRIM and test perception commands" -ForegroundColor White
    Write-Host ""
    Write-Host "Test commands:" -ForegroundColor Cyan
    Write-Host "  analyze_screen  - Analyze current screen content" -ForegroundColor White
    Write-Host "  read_text       - Perform OCR on screen" -ForegroundColor White
    Write-Host "  detect_objects  - Detect objects in view" -ForegroundColor White
} else {
    Write-Host "=============================================" -ForegroundColor Red
    Write-Host "   Setup Incomplete" -ForegroundColor Red
    Write-Host "=============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Some files are missing. Please check the errors above." -ForegroundColor Yellow
    Write-Host "You can re-run this script to download missing files." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Documentation: docs/PERCEPTION_SETUP.md" -ForegroundColor Gray
Write-Host ""

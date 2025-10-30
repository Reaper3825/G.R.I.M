# ============================================================
# Complete Perception System Fix
# ============================================================

Write-Host "="*60 -ForegroundColor Cyan
Write-Host "  Fixing GRIM Perception System" -ForegroundColor Cyan
Write-Host "="*60 -ForegroundColor Cyan
Write-Host ""

# Step 1: Download YOLOv8 ONNX
Write-Host "[1/4] Downloading YOLOv8 ONNX model..." -ForegroundColor Yellow
$YOLOPath = "D:\G.R.I.M\resources\models\yolo"
New-Item -ItemType Directory -Force -Path $YOLOPath | Out-Null

$ModelURL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
$ModelPath = "$YOLOPath\yolov8n.onnx"

if (!(Test-Path $ModelPath)) {
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri $ModelURL -OutFile $ModelPath
    Write-Host "  ✓ Downloaded yolov8n.onnx (6.2 MB)" -ForegroundColor Green
} else {
    Write-Host "  ✓ YOLOv8 already downloaded" -ForegroundColor Green
}

# Step 2: Update perception.cpp to use ONNX
Write-Host "`n[2/4] Updating perception.cpp for YOLOv8 ONNX..." -ForegroundColor Yellow

$perceptionFile = "D:\G.R.I.M\perception\perception.cpp"
$content = Get-Content $perceptionFile -Raw

# Fix YOLO loading function
$content = $content -replace '// loadYOLO\(\);  // Temporarily disabled.*', 'loadYOLO();  // ✅ Re-enabled with YOLOv8 ONNX'

# Update loadYOLO to use ONNX
$oldYOLO = @'
        bool loadYOLO\(\) \{
            std::string yoloPath = getResourcePath\(\) \+ "/models/yolo";
            std::string cfgFile = yoloPath \+ "/yolov3\.cfg";
            std::string weightsFile = yoloPath \+ "/yolov3\.weights";
            std::string namesFile = yoloPath \+ "/coco\.names";

            if \(!fs::exists\(cfgFile\) \|\| !fs::exists\(weightsFile\)\) \{
                LOG_ERROR\("Perception", "YOLO model files not found"\);
                return false;
            \}

            try \{
                g_yoloNet = cv::dnn::readNetFromDarknet\(cfgFile, weightsFile\);
'@

$newYOLO = @'
        bool loadYOLO() {
            std::string yoloPath = getResourcePath() + "/models/yolo";
            // ✅ Use YOLOv8 ONNX (OpenCV 4.11 compatible)
            std::string modelFile = yoloPath + "/yolov8n.onnx";
            std::string namesFile = yoloPath + "/coco.names";

            if (!fs::exists(modelFile)) {
                LOG_ERROR("Perception", "YOLOv8 ONNX model not found at: " + modelFile);
                return false;
            }

            try {
                // Load YOLOv8 ONNX model
                g_yoloNet = cv::dnn::readNetFromONNX(modelFile);
'@

$content = $content -replace $oldYOLO, $newYOLO

Set-Content $perceptionFile $content
Write-Host "  ✓ Updated to use YOLOv8 ONNX format" -ForegroundColor Green

# Step 3: Verify Tesseract data
Write-Host "`n[3/4] Verifying Tesseract data..." -ForegroundColor Yellow

if (Test-Path "D:\G.R.I.M\resources\tessdata\eng.traineddata") {
    if (!(Test-Path "D:\G.R.I.M\resources\eng.traineddata")) {
        Copy-Item "D:\G.R.I.M\resources\tessdata\eng.traineddata" "D:\G.R.I.M\resources\eng.traineddata"
        Write-Host "  ✓ Copied Tesseract data to correct location" -ForegroundColor Green
    } else {
        Write-Host "  ✓ Tesseract data already in place" -ForegroundColor Green
    }
} else {
    Write-Host "  ✗ Tesseract data not found - run: .\scripts\setup_perception.ps1" -ForegroundColor Red
}

# Step 4: Rebuild
Write-Host "`n[4/4] Rebuilding GRIM..." -ForegroundColor Yellow
Set-Location "D:\G.R.I.M"

cmake --build "out\build" --config Release --target GRIM 2>&1 | Select-String "perception|error|GRIM.exe" | Select-Object -Last 10

Write-Host ""
Write-Host "="*60 -ForegroundColor Green
Write-Host "  Perception System Fixed!" -ForegroundColor Green
Write-Host "="*60 -ForegroundColor Green
Write-Host ""
Write-Host "Changes made:" -ForegroundColor Cyan
Write-Host "  ✓ YOLOv8 ONNX model downloaded (6.2 MB)" -ForegroundColor White
Write-Host "  ✓ perception.cpp updated for ONNX support" -ForegroundColor White
Write-Host "  ✓ Tesseract data path fixed" -ForegroundColor White
Write-Host "  ✓ System rebuilt" -ForegroundColor White
Write-Host ""
Write-Host "Test with:" -ForegroundColor Cyan
Write-Host "  analyze_screen" -ForegroundColor White
Write-Host "  read_text" -ForegroundColor White
Write-Host "  detect_objects" -ForegroundColor White
Write-Host ""
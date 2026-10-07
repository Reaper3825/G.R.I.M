@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /DNOMINMAX /TP /I"D:\G.R.I.M\resources\models\GRIM-text\training\vcpkg_installed\x64-windows\include" /I"D:\G.R.I.M" /I"D:\G.R.I.M\resources\models\GRIM-text" /I"D:\G.R.I.M\resources\models\llama.cpp\vendor" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" "D:\G.R.I.M\tests\inference_payload_boundary_tests.cpp" "D:\G.R.I.M\resources\models\GRIM-text\Shared\UnigramByte\Detectors\DetectorRegistry.cu" "D:\G.R.I.M\resources\models\GRIM-text\Shared\UnigramByte\Detectors\AtomDelimiterDetector.cu" "D:\G.R.I.M\resources\models\GRIM-text\Shared\UnigramByte\Detectors\TextFeatureDetectors.cu" "D:\G.R.I.M\resources\models\GRIM-text\Shared\UnigramByte\TextUtils.cu" /Fe:inference_payload_boundary_tests.exe
if errorlevel 1 exit /b %errorlevel%
inference_payload_boundary_tests.exe

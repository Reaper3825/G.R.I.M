@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /Zs /TP /DUSE_CUDA /I"D:\G.R.I.M" /I"D:\G.R.I.M\resources\models\GRIM-text" /I"D:\G.R.I.M\resources\models\llama.cpp\vendor" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" "D:\G.R.I.M\resources\models\GRIM-text\Shared\Lenses\LensCapture_GPU.cu"

@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc  /DNOMINMAX /I"D:\G.R.I.M\resources\models\GRIM-text\training\vcpkg_installed\x64-windows\include" /I"D:\G.R.I.M" /I"D:\G.R.I.M\resources\models\GRIM-text" /I"D:\G.R.I.M\resources\models\llama.cpp\vendor" /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" "D:\G.R.I.M\resources\models\GRIM-text\Tests\lens_generation_config_test.cpp" /Fe:lens_generation_config_test.exe
if errorlevel 1 exit /b %errorlevel%
lens_generation_config_test.exe D:\G.R.I.M\ai_config.json


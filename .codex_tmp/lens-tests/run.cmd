@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
cl /nologo /std:c++17 /EHsc /W4 /I"D:\G.R.I.M\resources\models\llama.cpp\vendor" "D:\G.R.I.M\resources\models\GRIM-text\Tests\lens_metadata_test.cpp" "D:\G.R.I.M\resources\models\GRIM-text\Shared\Lenses\LensMetadata.cpp" /Fe:lens_metadata_test.exe
if errorlevel 1 exit /b %errorlevel%
lens_metadata_test.exe

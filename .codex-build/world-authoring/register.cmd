@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
cl @register.rsp
if errorlevel 1 exit /b %errorlevel%
register_world.exe D:\G.R.I.M\resources\models\games\grim_internal.grimworld.json

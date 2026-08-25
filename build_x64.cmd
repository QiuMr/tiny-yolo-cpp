@echo off
rem x64: 静态CRT零依赖 + UPX压缩
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /c /O2 /GL /Gy /EHsc /MT /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 tiny_yolo.cpp /Fotiny_yolo64.obj
if errorlevel 1 exit /b 1
link /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X64 tiny_yolo64.obj /OUT:tiny_yolo.dll
if errorlevel 1 exit /b 1
where upx >nul 2>nul && upx -9 -q tiny_yolo.dll
echo BUILD OK

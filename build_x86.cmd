@echo off
rem x86 (易语言用): 静态CRT零依赖 + UPX压缩
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsamd64_x86.bat" >nul
if errorlevel 1 exit /b 1
cl /c /O2 /GL /Gy /EHsc /MT /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 tiny_yolo.cpp /Fotiny_yolo.obj
if errorlevel 1 exit /b 1
link /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X86 /LARGEADDRESSAWARE tiny_yolo.obj /OUT:tiny_yolo_x86.dll
if errorlevel 1 exit /b 1
where upx >nul 2>nul && upx -9 -q tiny_yolo_x86.dll
echo BUILD X86 OK

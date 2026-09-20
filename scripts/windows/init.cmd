@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\.."
echo ==========================================================
echo   DiffusionProject - install everything and get CIFAR-10
echo ==========================================================
if "%~1"=="" (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Yes -Datasets cifar10
) else (
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -Yes %*
)
if errorlevel 1 exit /b 1
echo INITIALIZATION COMPLETE. Run scripts\windows\train.cmd next.

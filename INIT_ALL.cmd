@echo off
setlocal
cd /d "%~dp0"
echo ==========================================================
echo   DiffusionProject - install everything and get datasets
echo ==========================================================
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -Yes -Datasets all
if errorlevel 1 (
    echo.
    echo INITIALIZATION FAILED. Read the error above, then run this file again.
    pause
    exit /b 1
)
echo.
echo INITIALIZATION COMPLETE. You can now double-click TRAIN.cmd.
pause

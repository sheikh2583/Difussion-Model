@echo off
setlocal
set "PROJECT_ROOT=%~dp0"

echo ==========================================================
echo   DiffusionProject - complete Windows initialization
echo ==========================================================

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%PROJECT_ROOT%scripts\windows\init_all.ps1" %*
set "INIT_EXIT=%ERRORLEVEL%"

if not "%INIT_EXIT%"=="0" (
    echo.
    echo INITIALIZATION FAILED with exit code %INIT_EXIT%.
    exit /b %INIT_EXIT%
)

echo.
echo INITIALIZATION COMPLETE.
echo Preview training with: scripts\windows\train_all_datasets.ps1 -DryRun
exit /b 0

@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%PROJECT_ROOT%\scripts\windows\train_all_datasets.ps1" %*
exit /b %ERRORLEVEL%
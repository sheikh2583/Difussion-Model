@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass ^
  -File "%PROJECT_ROOT%\scripts\windows\setup.ps1" -Yes %*
exit /b %ERRORLEVEL%
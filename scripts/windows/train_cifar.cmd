@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\.."
set "RUNNER=%PROJECT_ROOT%\scripts\windows\train_all.ps1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%RUNNER%" -Dataset cifar10 -Mode continue %*
exit /b %ERRORLEVEL%

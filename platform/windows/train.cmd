@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\.."
call "%PROJECT_ROOT%\TRAIN.cmd" %*
exit /b %ERRORLEVEL%


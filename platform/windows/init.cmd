@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\.."
call "%PROJECT_ROOT%\INIT_ALL.cmd" %*
exit /b %ERRORLEVEL%


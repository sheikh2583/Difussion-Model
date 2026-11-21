@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\.."
set "PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo ERROR: project environment not found. Run scripts\windows\init.cmd first. 1>&2
  exit /b 1
)
cd /d "%PROJECT_ROOT%"
"%PYTHON%" scripts\interactive_train.py %*
exit /b %ERRORLEVEL%

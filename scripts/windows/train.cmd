@echo off
setlocal
set "PROJECT_ROOT=%~dp0..\.."
set "PROJECT_PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo Project environment not found. Run scripts\windows\init.cmd first.
    exit /b 1
)
cd /d "%PROJECT_ROOT%"
"%PROJECT_PYTHON%" "scripts\interactive_train.py" %*
exit /b %ERRORLEVEL%

@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
    echo Project environment not found.
    echo Double-click INIT_ALL.cmd first.
    pause
    exit /b 1
)
"%PROJECT_PYTHON%" "%~dp0scripts\interactive_train.py" %*
set "TRAIN_EXIT=%ERRORLEVEL%"
if not "%TRAIN_EXIT%"=="0" echo Training stopped with exit code %TRAIN_EXIT%.
if "%~1"=="" pause
exit /b %TRAIN_EXIT%

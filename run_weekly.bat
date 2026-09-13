@echo off
REM  FPL Assistant - weekly run (Windows Task Scheduler points here)
REM  Schedule: daily. The notifier decides for itself whether a reminder is due.
setlocal
cd /d "%~dp0"

REM  The package lives in src\, so it is not importable without this.
set "PYTHONPATH=%~dp0src"
REM  A stock Windows console is cp1252 and cannot print Thai.
set "PYTHONIOENCODING=utf-8"

if exist ".venv\Scripts\python.exe" (
    set "PY=%~dp0.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

"%PY%" -m fplbot build
if errorlevel 1 (
    echo(
    echo Build failed - skipping the reminder so it retries on the next run.
    exit /b 1
)
"%PY%" -m fplbot notify
endlocal

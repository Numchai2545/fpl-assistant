@echo off
REM  FPL Assistant - weekly run (Windows Task Scheduler points here)
REM  Schedule: daily. The notifier decides for itself whether the alert is due.
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
    set PY=.venv\Scripts\python.exe
) else (
    set PY=python
)
%PY% -m fplbot build
%PY% -m fplbot notify

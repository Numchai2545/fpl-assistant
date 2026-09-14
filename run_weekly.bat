@echo off
REM  FPL Assistant - start the local dashboard silently and open it.
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    set "PYW=%~dp0.venv\Scripts\pythonw.exe"
) else (
    set "PYW=pythonw"
)

start "" /b "%PYW%" "%~dp0start_fpl_server.py"
endlocal

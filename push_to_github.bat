@echo off
REM Push the prepared mobile release using Windows' normal network and Git login.
setlocal
cd /d "%~dp0"

echo Pushing FPL Assistant to GitHub...
git push -u origin master
if errorlevel 1 (
    echo.
    echo Push failed. Complete the GitHub sign-in window, then run this file again.
    pause
    exit /b 1
)

echo.
echo Push complete.
echo Next: GitHub repository Settings ^> Pages ^> Source ^> GitHub Actions.
pause
endlocal

@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\firecrawl.ps1" -Action Start
set "RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %RESULT%

@echo off
chcp 65001 >nul

REM Ensure we are in this script's directory
cd /d %~dp0

REM Activate local venv if exists
if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

REM Check python availability
where python >nul 2>nul
if errorlevel 1 (
  echo Python not found in PATH. Please install/activate environment.
  pause
  exit /b 1
)

REM Run scheduler
python scheduler.py

pause
exit /b

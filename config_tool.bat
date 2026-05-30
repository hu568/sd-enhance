@echo off
title SD Enhance Config Tool

echo ============================================
echo   SD Enhance Config Tool
echo ============================================
echo(

cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Failed to change directory
    pause
    exit /b 1
)

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] venv not found. Please run:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

echo Starting configuration tool...
echo(
echo Open browser at: http://127.0.0.1:7861
echo Press Ctrl+C to stop.
echo(

venv\Scripts\python config_tool.py

set "EXIT_CODE=%errorlevel%"
if %EXIT_CODE% NEQ 0 (
    echo [ERROR] Config tool exited with error code %EXIT_CODE%
    pause
)

pause

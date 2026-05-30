@echo off
title SD Enhance

echo ============================================
echo   SD Enhance - Image Post-Processing Tool
echo ============================================
echo(

cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Failed to change directory
    pause
    exit /b 1
)

rem --- Try to read python_path from config file ---
set "PYTHON_CMD=venv\Scripts\python.exe"
if exist "sd_enhance_config.json" (
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command "try { $c = Get-Content 'sd_enhance_config.json' -Raw | ConvertFrom-Json; if ($c.python_path) { Write-Output $c.python_path } } catch {}" 2^>nul') do (
        if not "%%i"=="" set "PYTHON_CMD=%%i"
    )
)

echo Using Python: %PYTHON_CMD%

rem --- Validate Python ---
if "%PYTHON_CMD%"=="venv\Scripts\python.exe" (
    if not exist "venv\Scripts\python.exe" (
        echo [ERROR] venv not found. Please run:
        echo   python -m venv venv
        echo   venv\Scripts\pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo(
echo Starting SD Enhance...
echo Device: auto (edit in config_tool.py)
echo Port: 7860
echo(
echo Open browser at: http://127.0.0.1:7860
echo For configuration: python config_tool.py
echo Press Ctrl+C to stop.
echo(

%PYTHON_CMD% main.py

if errorlevel 1 (
    echo [ERROR] SD Enhance exited with error code %errorlevel%
    pause
)

echo(
echo Server stopped.
pause

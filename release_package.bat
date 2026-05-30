@echo off
title SD Enhance - Release Package Builder

echo ============================================
echo   SD Enhance Release Package Builder
echo ============================================
echo(

cd /d "%~dp0"
if errorlevel 1 (
    echo [ERROR] Failed to change directory
    pause
    exit /b 1
)

set "RELEASE_NAME=sd-enhance-v1.0.0"
set "RELEASE_DIR=%TEMP%\%RELEASE_NAME%"

echo Destination: %RELEASE_DIR%
echo(

rem --- Clean temp dir ---
if exist "%RELEASE_DIR%" rmdir /s /q "%RELEASE_DIR%"
mkdir "%RELEASE_DIR%"

rem --- Copy source files ---
echo Copying source files...

mkdir "%RELEASE_DIR%\sd_enhance"
mkdir "%RELEASE_DIR%\sd_enhance\models"
mkdir "%RELEASE_DIR%\sd_enhance\postprocessing"
mkdir "%RELEASE_DIR%\sd_enhance\utils"

copy main.py                   "%RELEASE_DIR%\"          >nul
copy config_tool.py            "%RELEASE_DIR%\"          >nul
copy start.bat                 "%RELEASE_DIR%\"          >nul
copy config_tool.bat           "%RELEASE_DIR%\"          >nul
copy requirements.txt          "%RELEASE_DIR%\"          >nul
copy README.md                 "%RELEASE_DIR%\"          >nul
copy LICENSE                   "%RELEASE_DIR%\"          >nul

echo f | xcopy "sd_enhance\*.py"          "%RELEASE_DIR%\sd_enhance\"          /s /y >nul
echo f | xcopy "sd_enhance\models\*.py"   "%RELEASE_DIR%\sd_enhance\models\"   /s /y >nul
echo f | xcopy "sd_enhance\postprocessing\*.py" "%RELEASE_DIR%\sd_enhance\postprocessing\" /s /y >nul
echo f | xcopy "sd_enhance\utils\*.py"    "%RELEASE_DIR%\sd_enhance\utils\"    /s /y >nul

rem --- Copy model files ---
echo Copying model files...

mkdir "%RELEASE_DIR%\models\ESRGAN"
mkdir "%RELEASE_DIR%\models\RealESRGAN"
mkdir "%RELEASE_DIR%\models\SwinIR"

copy "models\ESRGAN\ESRGAN_4x.pth"             "%RELEASE_DIR%\models\ESRGAN\"   >nul
copy "models\ESRGAN\BSRGAN.pth"                "%RELEASE_DIR%\models\ESRGAN\"   >nul
copy "models\ESRGAN\4x-AnimeSharp.pth"         "%RELEASE_DIR%\models\ESRGAN\"   >nul
copy "models\RealESRGAN\RealESRGAN_x4plus.pth" "%RELEASE_DIR%\models\RealESRGAN\" >nul
copy "models\RealESRGAN\RealESRGAN_x4plus_anime_6B.pth" "%RELEASE_DIR%\models\RealESRGAN\" >nul
copy "models\SwinIR\SwinIR_4x.pth"             "%RELEASE_DIR%\models\SwinIR\"   >nul

rem --- Create zip ---
echo(
echo Creating zip archive...

set "ZIP_NAME=%RELEASE_NAME%.zip"

rem Try PowerShell compression first (Windows 10+)
powershell -NoProfile -Command "Compress-Archive -Path '%RELEASE_DIR%\*' -DestinationPath '%CD%\%ZIP_NAME%' -Force" >nul 2>&1

if exist "%ZIP_NAME%" (
    echo Successfully created: %ZIP_NAME%
    rem Get size
    for %%f in ("%ZIP_NAME%") do echo Size: %%~zf bytes
) else (
    echo [ERROR] Failed to create zip. Make sure PowerShell is available.
    pause
    exit /b 1
)

rem --- Clean temp ---
rmdir /s /q "%RELEASE_DIR%"

echo(
echo Done! Release package ready: %ZIP_NAME%
pause

@echo off
setlocal

cd /d "%~dp0\.."

if "%~1"=="--in-place" (
    python format_hotel_like_twitter.py --in_place --overwrite
) else (
    python format_hotel_like_twitter.py --overwrite
)

if errorlevel 1 (
    echo Hotel formatting failed.
    exit /b 1
)

echo.
echo Default output: datasets\hotel_twitter
echo For training with --dataset_type hotel, run: .\scripts\format_hotel_like_twitter.bat --in-place
echo Rows with missing image files are skipped by default.
endlocal

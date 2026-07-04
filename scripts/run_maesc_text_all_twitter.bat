@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set TASK_NAME=dualc

python -c "import accelerate" >nul 2>nul
if errorlevel 1 (
    echo Missing Python package: accelerate
    echo Run: python -m pip install -r requirements.txt
    exit /b 1
)

for %%D in (2017) do (
    if not exist "datasets\finetune\%TASK_NAME%\%%D\input.pt" (
        echo Missing processed input: datasets\finetune\%TASK_NAME%\%%D\input.pt
        echo Run .\scripts\run_twitter_train.bat once, or generate inputs with utils\TrainInputProcess.py.
        exit /b 1
    )
)

for %%M in (bert roberta) do (
    if "%%M"=="bert" set MODEL_DIR=models\bert-base-uncased
    if "%%M"=="roberta" set MODEL_DIR=models\roberta-base

    if not exist "!MODEL_DIR!" (
        echo Missing model directory: !MODEL_DIR!
        echo Run: python download_pretrained_model.py
        exit /b 1
    )
)

for %%D in (2017) do (
    for %%M in (bert roberta) do (
        echo.
        echo ========================================
        echo MAESC text baseline: Twitter %%D / %%M
        echo ========================================

        call .\scripts\run_maesc_text.bat %%D %%M

        if errorlevel 1 (
            echo Failed: Twitter %%D / %%M
            exit /b 1
        )
    )
)

echo.
echo Done. All MAESC text baselines finished.
endlocal

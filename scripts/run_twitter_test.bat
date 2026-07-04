@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set TEXT_MODEL=bert
set IMAGE_MODEL=vit
set TASK_NAME=dualc
set BATCH_SIZE=4

python -c "import accelerate" >nul 2>nul
if errorlevel 1 (
    echo Missing Python package: accelerate
    echo Run: python -m pip install -r requirements.txt
    exit /b 1
)

for %%D in (2015 2017) do (
    set MODEL_FILE=results\twitter%%D_%TASK_NAME%_%TEXT_MODEL%_%IMAGE_MODEL%\final_model.pt
    set EVAL_DIR=results\twitter%%D_%TASK_NAME%_%TEXT_MODEL%_%IMAGE_MODEL%\test_eval

    echo.
    echo ========================================
    echo Testing DTCA on Twitter %%D
    echo ========================================

    if not exist "!MODEL_FILE!" (
        echo Missing trained model: !MODEL_FILE!
        echo Re-run .\scripts\run_twitter_train.bat first. main.py now saves final_model.pt.
        exit /b 1
    )

    python eval_dtca_test.py ^
        --dataset_type %%D ^
        --task_name %TASK_NAME% ^
        --text_model_name %TEXT_MODEL% ^
        --image_model_name %IMAGE_MODEL% ^
        --model_file "!MODEL_FILE!" ^
        --output_dir "!EVAL_DIR!" ^
        --batch_size %BATCH_SIZE%

    if errorlevel 1 (
        echo Testing failed for Twitter %%D
        exit /b 1
    )
)

echo.
echo Done.
endlocal

@echo off
setlocal enabledelayedexpansion

cd /d "%~dp0\.."

set TEXT_MODEL=bert
set IMAGE_MODEL=vit
set TASK_NAME=dualc
set BATCH_SIZE=4
set EPOCHS=10
set LR=2e-5

if not exist "models\bert-base-uncased" (
    echo Missing models\bert-base-uncased
    echo Run: python download_pretrained_model.py
    exit /b 1
)

if not exist "models\vit-base-patch16-224-in21k" (
    echo Missing models\vit-base-patch16-224-in21k
    echo Run: python download_pretrained_model.py
    exit /b 1
)

python -c "import accelerate" >nul 2>nul
if errorlevel 1 (
    echo Missing Python package: accelerate
    echo Run: conda install -c conda-forge accelerate
    exit /b 1
)

for %%D in (2017) do (
    echo.
    echo ========================================
    echo Generating inputs for Twitter %%D
    echo ========================================
    python utils\TrainInputProcess.py ^
        --dataset_type %%D ^
        --text_model_type %TEXT_MODEL% ^
        --image_model_type %IMAGE_MODEL% ^
        --train_type 0 ^
        --finetune_task %TASK_NAME%

    if errorlevel 1 (
        echo Failed to generate inputs for Twitter %%D
        exit /b 1
    )

    echo.
    echo ========================================
    echo Training DTCA on Twitter %%D
    echo ========================================
    python main.py ^
        --dataset_type %%D ^
        --task_name %TASK_NAME% ^
        --text_model_name %TEXT_MODEL% ^
        --image_model_name %IMAGE_MODEL% ^
        --batch_size %BATCH_SIZE% ^
        --epochs %EPOCHS% ^
        --lr %LR% ^
        --output_dir results\twitter%%D_%TASK_NAME%_%TEXT_MODEL%_%IMAGE_MODEL% ^
        --output_result_file result.txt

    if errorlevel 1 (
        echo Training failed for Twitter %%D
        exit /b 1
    )
)

echo.
echo Done. Results appended to result.txt
endlocal

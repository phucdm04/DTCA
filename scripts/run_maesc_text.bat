@echo off
setlocal

cd /d "%~dp0\.."

set DATASET=%~1
set TEXT_MODEL=%~2
if "%DATASET%"=="" set DATASET=2015
if "%TEXT_MODEL%"=="" set TEXT_MODEL=bert

set TASK_NAME=dualc
set BATCH_SIZE=8
set EPOCHS=10
set LR=2e-5
set OUT_DIR=results\maesc_%DATASET%_%TEXT_MODEL%

python -c "import accelerate" >nul 2>nul
if errorlevel 1 (
    echo Missing Python package: accelerate
    echo Run: python -m pip install -r requirements.txt
    exit /b 1
)

if "%TEXT_MODEL%"=="bert" set MODEL_DIR=models\bert-base-uncased
if "%TEXT_MODEL%"=="roberta" set MODEL_DIR=models\roberta-base

if not exist "%MODEL_DIR%" (
    echo Missing model directory: %MODEL_DIR%
    echo Run: python download_pretrained_model.py
    exit /b 1
)

if not exist "datasets\finetune\%TASK_NAME%\%DATASET%\input.pt" (
    echo Missing processed input: datasets\finetune\%TASK_NAME%\%DATASET%\input.pt
    echo Generate it first with utils\TrainInputProcess.py.
    exit /b 1
)

python train_maesc_text.py ^
    --dataset_type %DATASET% ^
    --task_name %TASK_NAME% ^
    --text_model_name %TEXT_MODEL% ^
    --batch_size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --output_dir "%OUT_DIR%" ^
    --output_result_file result_maesc_text.txt

if errorlevel 1 (
    echo MAESC text baseline failed.
    exit /b 1
)

echo.
echo Done. Best model: %OUT_DIR%\best_model
echo Report: %OUT_DIR%\maesc_pred_vs_gold.tsv
endlocal

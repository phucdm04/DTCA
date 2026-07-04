@echo off
setlocal

cd /d "%~dp0\.."

set TEXT_MODEL=%~1
if "%TEXT_MODEL%"=="" set TEXT_MODEL=bert

set DATA_DIR=datasets\hotel
set IMAGE_DIR=datasets\hotel_images
set IMAGE_MODEL=models\vit-base-patch16-224-in21k
set OUT_DIR=results\hotel_macsa_dtca_%TEXT_MODEL%_vit
set BATCH_SIZE=8
set EPOCHS=10
set LR=2e-5
set KMP_DUPLICATE_LIB_OK=TRUE

if "%TEXT_MODEL%"=="bert" set TEXT_DIR=models\bert-base-uncased
if "%TEXT_MODEL%"=="roberta" set TEXT_DIR=models\roberta-base

if not exist "%TEXT_DIR%" (
    echo Missing text model directory: %TEXT_DIR%
    exit /b 1
)

if not exist "%IMAGE_MODEL%" (
    echo Missing image model directory: %IMAGE_MODEL%
    exit /b 1
)

if exist "%OUT_DIR%\best_model\pytorch_model.bin" (
    echo Found existing MACSA checkpoint: %OUT_DIR%\best_model
    echo Running evaluation only.
    python eval_macsa_dtca_hotel.py ^
        --data_dir %DATA_DIR% ^
        --image_dir %IMAGE_DIR% ^
        --text_model_name %TEXT_MODEL% ^
        --image_model_path %IMAGE_MODEL% ^
        --model_dir "%OUT_DIR%\best_model" ^
        --output_dir "%OUT_DIR%" ^
        --batch_size %BATCH_SIZE%
) else (
    echo Training and evaluating DTCA MACSA: %TEXT_MODEL% + ViT
    python train_macsa_dtca_hotel.py ^
        --data_dir %DATA_DIR% ^
        --image_dir %IMAGE_DIR% ^
        --text_model_name %TEXT_MODEL% ^
        --image_model_path %IMAGE_MODEL% ^
        --output_dir "%OUT_DIR%" ^
        --batch_size %BATCH_SIZE% ^
        --epochs %EPOCHS% ^
        --lr %LR%
)

if errorlevel 1 (
    echo Hotel DTCA MACSA run failed.
    exit /b 1
)

echo.
echo Done. MACSA reports:
echo %OUT_DIR%\macsa_pred_vs_gold.tsv
echo %OUT_DIR%\macsa_summary_counts.json
endlocal

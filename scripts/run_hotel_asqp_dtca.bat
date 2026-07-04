@echo off
setlocal

cd /d "%~dp0\.."

set DATA_DIR=datasets\hotel
set IMAGE_DIR=datasets\hotel_images
set BART_MODEL=models\bart-base
set IMAGE_MODEL=models\vit-base-patch16-224-in21k
set OUT_DIR=results\hotel_asqp_dtca_bart_vit
set BATCH_SIZE=4
set EPOCHS=10
set LR=2e-5
set KMP_DUPLICATE_LIB_OK=TRUE

if not exist "%BART_MODEL%" (
    echo Missing BART model directory: %BART_MODEL%
    echo Add facebook/bart-base to models\bart-base first.
    exit /b 1
)

if not exist "%IMAGE_MODEL%" (
    echo Missing image model directory: %IMAGE_MODEL%
    exit /b 1
)

if exist "%OUT_DIR%\best_model\pytorch_model.bin" (
    echo Found existing ASQP checkpoint: %OUT_DIR%\best_model
    echo Running evaluation only.
    python eval_asqp_dtca_hotel.py ^
        --data_dir %DATA_DIR% ^
        --image_dir %IMAGE_DIR% ^
        --bart_model_path %BART_MODEL% ^
        --image_model_path %IMAGE_MODEL% ^
        --model_dir "%OUT_DIR%\best_model" ^
        --output_dir "%OUT_DIR%" ^
        --batch_size %BATCH_SIZE%
) else (
    echo Training and evaluating DTCA-style ASQP: BART + ViT
    python train_asqp_dtca_hotel.py ^
        --data_dir %DATA_DIR% ^
        --image_dir %IMAGE_DIR% ^
        --bart_model_path %BART_MODEL% ^
        --image_model_path %IMAGE_MODEL% ^
        --output_dir "%OUT_DIR%" ^
        --batch_size %BATCH_SIZE% ^
        --epochs %EPOCHS% ^
        --lr %LR%
)

if errorlevel 1 (
    echo Hotel ASQP run failed.
    exit /b 1
)

echo.
echo Done. ASQP reports:
echo %OUT_DIR%\asqp_pred_vs_gold.tsv
echo %OUT_DIR%\asqp_summary_counts.json
endlocal

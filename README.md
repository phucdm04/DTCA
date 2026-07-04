# DTCA Hotel Experiments

This repository is based on **DTCA/DCTA** for multimodal aspect-sentiment analysis: one transformer encoder for text and one transformer encoder for images. The original model implementation is in:

```text
model/modeling_dtca.py
```

This fork keeps the original DTCA fine-tuning flow and adds hotel-dataset experiments for:

```text
MAESC  = aspect + sentiment extraction
ACC    = gold aspect -> category classification
MACSA  = extract (category, sentiment)
ASQP   = generate (aspect span, category, opinion span, sentiment)
```

## Repository Layout

Expected local layout:

```text
DTCA\
  datasets\
    twitter2015\
    twitter2015_images\
    twitter2017\
    twitter2017_images\
    hotel\
    hotel_images\
  models\
    bert-base-uncased\
    roberta-base\
    vit-base-patch16-224-in21k\
    bart-base\
  scripts\
  model\
  utils\
  main.py
```

`datasets/`, `models/`, `logs/`, and `results/` are ignored by git.

## Setup

Create and activate your Python environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If `pip.exe` is blocked by Windows policy, use:

```powershell
python -m pip install -r requirements.txt
```

or install missing packages with conda where possible.

## Datasets

### Twitter 2015 / 2017

Download Twitter 2015 and Twitter 2017 and place them under:

```text
datasets\twitter2015
datasets\twitter2015_images
datasets\twitter2017
datasets\twitter2017_images
```

The original README points to the dataset links from the paper **Joint Multi-modal Aspect-Sentiment Analysis with Auxiliary Cross-modal Relation Detection** with source code at https://github.com/windforfurture/DTCA.

### Hotel

Hotel data should be placed under:

```text
datasets\hotel
datasets\hotel_images
```

Expected raw hotel files:

```text
datasets\hotel\train.json
datasets\hotel\val.json
datasets\hotel\test.json
```

The hotel dataset will be provided via Google Drive. Link will be updated later.

The raw hotel JSON should contain fields like:

```json
{
  "review_photo": ".../456841602.jpg?...",
  "review": "We couldn’t shower because the shower head was above the toilet...",
  "extraction": [
    {
      "Aspect": "shower head",
      "Opinion": "above the toilet",
      "Polarity": "Negative",
      "Category": "Facility",
      "Aspect_span": [5, 7],
      "Opinion_span": [8, 11]
    }
  ]
}
```

For the original DTCA MAESC pipeline, convert hotel JSON to Twitter-style text files:

```powershell
.\scripts\format_hotel_like_twitter.bat --in-place
```

This creates:

```text
datasets\hotel\train.txt
datasets\hotel\dev.txt
datasets\hotel\test.txt
```

Raw JSON files are backed up as `*.raw.bak`.

## Pretrained Models

Place pretrained models under `models/`.

Required for DTCA/MAESC/ACC/MACSA:

```text
models\bert-base-uncased
models\roberta-base
models\vit-base-patch16-224-in21k
```

Required additionally for ASQP:

```text
models\bart-base
```

You can use `download_pretrained_model.py` for the provided model downloader. If BART is not included in your local downloader, download `facebook/bart-base` manually into:

```text
models\bart-base
```

## Original DTCA Fine-Tuning

Generate input for a dataset:

```powershell
python utils\TrainInputProcess.py --dataset_type hotel --text_model_type bert --image_model_type vit --train_type 0 --finetune_task dualc
```

Train DTCA:

```powershell
python main.py --dataset_type hotel --task_name dualc --text_model_name bert --image_model_name vit --batch_size 4 --epochs 10 --output_dir results\hotel_dtca_bert_vit --output_result_file result.txt
```

`main.py` saves the best/final model to:

```text
results\<run_name>\final_model.pt
```

## Current Training / Evaluation Scripts

Run scripts from the repository root.

### Twitter DTCA

Train Twitter 2015 and 2017 with BERT + ViT:

```powershell
.\scripts\run_twitter_train.bat
```

Evaluate saved Twitter DTCA models:

```powershell
.\scripts\run_twitter_test.bat
```

### Text-Only MAESC Baselines

Run one dataset/model:

```powershell
.\scripts\run_maesc_text.bat 2015 bert
.\scripts\run_maesc_text.bat 2017 roberta
```

Run all Twitter text-only baselines:

```powershell
.\scripts\run_maesc_text_all_twitter.bat
```

Reports include:

```text
maesc_pred_vs_gold.tsv
mate_pred_vs_gold.tsv
masc_classification_report.txt
summary_counts.json
```

## Hotel Tasks

### ACC

ACC is **Aspect Category Classification**:

```text
gold aspect span -> category
```

Run BERT/RoBERTa + ViT:

```powershell
.\scripts\run_hotel_acsa_dtca.bat bert
.\scripts\run_hotel_acsa_dtca.bat roberta
```

Outputs:

```text
results\hotel_acsa_dtca_bert_vit\
results\hotel_acsa_dtca_roberta_vit\
```

Reports:

```text
acc_category_report.txt
sentiment_report.txt
acsa_pred_vs_gold.tsv
acsa_summary_counts.json
```

`acc_category_report.txt` is the detailed per-class category report.

### MACSA

MACSA here means extracting:

```text
(category, sentiment)
```

Run:

```powershell
.\scripts\run_hotel_macsa_dtca.bat bert
.\scripts\run_hotel_macsa_dtca.bat roberta
```

Reports:

```text
macsa_pred_vs_gold.tsv
macsa_summary_counts.json
```

### ASQP

ASQP uses a BART decoder to generate a target sequence:

```text
aspect_start aspect_end <cat_CATEGORY> opinion_start opinion_end <sent_SENTIMENT>
```

Example:

```text
5 7 <cat_FACILITY> 8 11 <sent_NEG>
```

Run:

```powershell
.\scripts\run_hotel_asqp_dtca.bat
```

Reports:

```text
asqp_pred_vs_gold.tsv
asqp_summary_counts.json
training_log.json
best_model\
```

ASQP requires:

```text
models\bart-base
models\vit-base-patch16-224-in21k
```

## Output Metrics

Most extraction tasks report:

```text
TP
FP
FN
precision
recall
f1
```

Classification reports use `sklearn.metrics.classification_report` and include per-class precision, recall, F1, and support.

## Notes

- Use `bert` when `models\roberta-base` is not available.
- Use `roberta` only after downloading `models\roberta-base`.
- ASQP requires BART. It will not run without `models\bart-base`.
- Existing scripts skip training and run evaluation only when a checkpoint already exists in the expected output directory.

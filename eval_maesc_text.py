import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

from train_maesc_text import (
    counts,
    keep_text_fields,
    load_json_rows,
    span_only,
    prf,
    write_pred_vs_gold,
    SENTIMENT_NAMES,
)
from utils.MyDataSet import MyDataSet2
from utils.metrics import cal_f1
from sklearn.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser(description="Report-only evaluator for text-only MAESC checkpoints.")
    parser.add_argument("--dataset_type", type=str, default="hotel")
    parser.add_argument("--task_name", type=str, default="dualc")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    data_input_file = os.path.join("datasets", "finetune", args.task_name, args.dataset_type, "input.pt")
    data_inputs = torch.load(data_input_file, weights_only=False)
    test_pairs = data_inputs["test"].pop("pairs")
    keep_text_fields(data_inputs["test"])
    test_dataset = MyDataSet2(inputs=data_inputs["test"])

    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "trainer_tmp"),
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        label_names=["labels"],
    )
    trainer = Trainer(model=model, args=training_args)
    outputs = trainer.predict(test_dataset)
    pred_labels = np.argmax(outputs.predictions, axis=-1)
    maesc_p, maesc_r, maesc_f1, pred_pairs = cal_f1(
        pred_labels, data_inputs["test"], test_pairs, is_result=True
    )
    maesc_correct = sum(len(p & g) for p, g in zip(pred_pairs, test_pairs))
    maesc_pred = sum(len(p) for p in pred_pairs)
    maesc_gold = sum(len(g) for g in test_pairs)
    maesc_counts = counts(maesc_correct, maesc_pred, maesc_gold)

    mate_correct = sum(len(span_only(p) & span_only(g)) for p, g in zip(pred_pairs, test_pairs))
    mate_pred = sum(len(span_only(p)) for p in pred_pairs)
    mate_gold = sum(len(span_only(g)) for g in test_pairs)
    mate_p, mate_r, mate_f1 = prf(mate_correct, mate_pred, mate_gold)
    mate_counts = counts(mate_correct, mate_pred, mate_gold)

    pred_by_span = [{span: sent for span, sent in pairs} for pairs in pred_pairs]
    y_true = []
    y_pred = []
    for pred_map, gold in zip(pred_by_span, test_pairs):
        for span, gold_sent in gold:
            if span in pred_map:
                y_true.append(SENTIMENT_NAMES[gold_sent])
                y_pred.append(SENTIMENT_NAMES.get(pred_map[span], "UNK"))

    masc_report = "No exact aspect span matches; MASC classification report cannot be computed."
    if y_true:
        masc_report = classification_report(
            y_true,
            y_pred,
            labels=["NEG", "NEU", "POS"],
            digits=4,
            zero_division=0,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    json_rows = load_json_rows(args.dataset_type)
    maesc_path = os.path.join(args.output_dir, "maesc_pred_vs_gold.tsv")
    mate_path = os.path.join(args.output_dir, "mate_pred_vs_gold.tsv")
    masc_path = os.path.join(args.output_dir, "masc_classification_report.txt")
    write_pred_vs_gold(maesc_path, pred_pairs, test_pairs, json_rows, include_sentiment=True)
    write_pred_vs_gold(mate_path, pred_pairs, test_pairs, json_rows, include_sentiment=False)
    with open(masc_path, "w", encoding="utf-8") as f:
        f.write(masc_report)
        f.write("\n")
    summary_path = os.path.join(args.output_dir, "summary_counts.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "maesc": {
                    **maesc_counts,
                    "precision": maesc_p,
                    "recall": maesc_r,
                    "f1": maesc_f1,
                },
                "mate": {
                    **mate_counts,
                    "precision": mate_p,
                    "recall": mate_r,
                    "f1": mate_f1,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    print(f"MAESC pred vs gold saved to {maesc_path}")
    print(f"MATE pred vs gold saved to {mate_path}")
    print(f"MASC classification report saved to {masc_path}")
    print(f"Summary counts saved to {summary_path}")
    print(f"Test MAESC TP/FP/FN: {maesc_counts['tp']} / {maesc_counts['fp']} / {maesc_counts['fn']}")
    print(f"Test MAESC P/R/F1: {maesc_p:.2f} / {maesc_r:.2f} / {maesc_f1:.2f}")
    print(f"Test MATE TP/FP/FN: {mate_counts['tp']} / {mate_counts['fp']} / {mate_counts['fn']}")
    print(f"Test MATE P/R/F1: {mate_p:.2f} / {mate_r:.2f} / {mate_f1:.2f}")
    print("Test MASC classification report:")
    print(masc_report)


if __name__ == "__main__":
    main()

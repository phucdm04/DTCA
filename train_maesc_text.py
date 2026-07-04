import argparse
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForTokenClassification, Trainer, TrainingArguments

from utils.MyDataSet import MyDataSet2
from utils.metrics import cal_f1


LABELS = ["O", "I", "B-NEG", "B-NEU", "B-POS"]
SENTIMENT_NAMES = {0: "NEG", 1: "NEU", 2: "POS"}


def parse_args():
    parser = argparse.ArgumentParser(description="Text-only BERT/RoBERTa baseline for MAESC.")
    parser.add_argument("--dataset_type", type=str, default="2015")
    parser.add_argument("--task_name", type=str, default="dualc")
    parser.add_argument("--text_model_name", type=str, default="bert", choices=["bert", "roberta"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--random_seed", type=int, default=2022)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_result_file", type=str, default="result_maesc_text.txt")
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def model_path(text_model_name):
    if text_model_name == "bert":
        return "./models/bert-base-uncased"
    if text_model_name == "roberta":
        return "./models/roberta-base"
    raise ValueError(f"Unsupported text model: {text_model_name}")


def keep_text_fields(batch_encoding):
    remove_keys = [k for k in batch_encoding.keys() if k not in {"input_ids", "attention_mask", "token_type_ids", "labels"}]
    for key in remove_keys:
        batch_encoding.pop(key)
    return batch_encoding


def build_compute_metrics_fn(inputs, pairs):
    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        pred_labels = np.argmax(logits, axis=-1)
        precision, recall, f1 = cal_f1(pred_labels, inputs, pairs)
        return {"precision": precision, "recall": recall, "f1": f1}

    return compute_metrics


def load_json_rows(dataset_type):
    if dataset_type in {"2015", "2017"}:
        path = os.path.join("datasets", f"twitter{dataset_type}", "test.json")
    else:
        path = os.path.join("datasets", dataset_type, "test.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def term_for_span(words, span):
    start, end = [int(x) for x in span.split("-")]
    if not words or start >= len(words):
        return span
    end = min(end, len(words) - 1)
    return " ".join(words[start : end + 1])


def span_only(pairs):
    return {span for span, _ in pairs}


def prf(correct, predicted, gold):
    precision = 0.0 if predicted == 0 else correct / predicted * 100
    recall = 0.0 if gold == 0 else correct / gold * 100
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def counts(correct, predicted, gold):
    return {
        "tp": correct,
        "fp": predicted - correct,
        "fn": gold - correct,
    }


def write_pred_vs_gold(path, pred_pairs, gold_pairs, json_rows, include_sentiment=True):
    with open(path, "w", encoding="utf-8") as f:
        f.write("sample_id\tpred\tgold\n")
        for i, (pred, gold) in enumerate(zip(pred_pairs, gold_pairs)):
            words = json_rows[i]["words"] if i < len(json_rows) else []
            if include_sentiment:
                pred_items = [
                    f"{term_for_span(words, span)}[{span}]/{SENTIMENT_NAMES.get(sent, sent)}"
                    for span, sent in sorted(pred)
                ]
                gold_items = [
                    f"{term_for_span(words, span)}[{span}]/{SENTIMENT_NAMES.get(sent, sent)}"
                    for span, sent in sorted(gold)
                ]
            else:
                pred_items = [f"{term_for_span(words, span)}[{span}]" for span in sorted(span_only(pred))]
                gold_items = [f"{term_for_span(words, span)}[{span}]" for span in sorted(span_only(gold))]
            f.write(f"{i}\t{' | '.join(pred_items)}\t{' | '.join(gold_items)}\n")


def main():
    args = parse_args()
    set_random_seed(args.random_seed)

    output_dir = args.output_dir or f"results/maesc_{args.dataset_type}_{args.text_model_name}"
    data_input_file = os.path.join("datasets", "finetune", args.task_name, args.dataset_type, "input.pt")
    if not os.path.exists(data_input_file):
        raise FileNotFoundError(
            f"Missing {data_input_file}. Generate inputs first with utils/TrainInputProcess.py."
        )

    path = model_path(args.text_model_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run download_pretrained_model.py or choose another text model.")

    data_inputs = torch.load(data_input_file, weights_only=False)
    train_pairs = data_inputs["train"].pop("pairs")
    dev_pairs = data_inputs["dev"].pop("pairs")
    test_pairs = data_inputs["test"].pop("pairs")

    keep_text_fields(data_inputs["train"])
    keep_text_fields(data_inputs["dev"])
    keep_text_fields(data_inputs["test"])

    train_dataset = MyDataSet2(inputs=data_inputs["train"])
    dev_dataset = MyDataSet2(inputs=data_inputs["dev"])
    test_dataset = MyDataSet2(inputs=data_inputs["test"])

    model = AutoModelForTokenClassification.from_pretrained(
        path,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=build_compute_metrics_fn(data_inputs["dev"], dev_pairs),
    )
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(os.path.join(output_dir, "best_model"))

    trainer.compute_metrics = None
    test_outputs = trainer.predict(test_dataset)
    pred_labels = np.argmax(test_outputs.predictions, axis=-1)
    precision, recall, f1, pred_pairs = cal_f1(pred_labels, data_inputs["test"], test_pairs, is_result=True)
    maesc_correct = sum(len(p & g) for p, g in zip(pred_pairs, test_pairs))
    maesc_pred = sum(len(p) for p in pred_pairs)
    maesc_gold = sum(len(g) for g in test_pairs)
    maesc_counts = counts(maesc_correct, maesc_pred, maesc_gold)
    maesc_metric = {**maesc_counts, "precision": precision, "recall": recall, "f1": f1}

    mate_correct = sum(len(span_only(p) & span_only(g)) for p, g in zip(pred_pairs, test_pairs))
    mate_pred = sum(len(span_only(p)) for p in pred_pairs)
    mate_gold = sum(len(span_only(g)) for g in test_pairs)
    mate_precision, mate_recall, mate_f1 = prf(mate_correct, mate_pred, mate_gold)
    mate_counts = counts(mate_correct, mate_pred, mate_gold)
    mate_metric = {**mate_counts, "precision": mate_precision, "recall": mate_recall, "f1": mate_f1}

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

    json_rows = load_json_rows(args.dataset_type)
    maesc_path = os.path.join(output_dir, "maesc_pred_vs_gold.tsv")
    mate_path = os.path.join(output_dir, "mate_pred_vs_gold.tsv")
    masc_path = os.path.join(output_dir, "masc_classification_report.txt")
    write_pred_vs_gold(maesc_path, pred_pairs, test_pairs, json_rows, include_sentiment=True)
    write_pred_vs_gold(mate_path, pred_pairs, test_pairs, json_rows, include_sentiment=False)
    with open(masc_path, "w", encoding="utf-8") as f:
        f.write(masc_report)
        f.write("\n")
    summary_path = os.path.join(output_dir, "summary_counts.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "maesc": maesc_metric,
                "mate": mate_metric,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    log_file = os.path.join(output_dir, "training_log.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        for log_entry in trainer.state.log_history:
            f.write(str(log_entry) + "\n")

    with open(args.output_result_file, "a", encoding="utf-8") as f:
        f.write(
            "Parameter:"
            + str(
                {
                    "task": "MAESC-text-only",
                    "dataset_type": args.dataset_type,
                    "text_model": args.text_model_name,
                    "batch_size": args.batch_size,
                    "epochs": args.epochs,
                    "lr": args.lr,
                }
            )
            + "\n"
        )
        f.write("test_maesc: " + str(maesc_metric) + "\n")
        f.write("test_mate: " + str(mate_metric) + "\n")
        f.write("test_masc:\n" + masc_report + "\n\n")

    print(f"Best model saved to {os.path.join(output_dir, 'best_model')}")
    print(f"MAESC pred vs gold saved to {maesc_path}")
    print(f"MATE pred vs gold saved to {mate_path}")
    print(f"MASC classification report saved to {masc_path}")
    print(f"Summary counts saved to {summary_path}")
    print(f"Test MAESC TP/FP/FN: {maesc_counts['tp']} / {maesc_counts['fp']} / {maesc_counts['fn']}")
    print(f"Test MAESC P/R/F1: {precision:.2f} / {recall:.2f} / {f1:.2f}")
    print(f"Test MATE TP/FP/FN: {mate_counts['tp']} / {mate_counts['fp']} / {mate_counts['fn']}")
    print(f"Test MATE P/R/F1: {mate_precision:.2f} / {mate_recall:.2f} / {mate_f1:.2f}")
    print("Test MASC classification report:")
    print(masc_report)


if __name__ == "__main__":
    main()

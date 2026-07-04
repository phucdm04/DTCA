import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report
from transformers import (
    AlbertForTokenClassification,
    AutoConfig,
    BertForTokenClassification,
    ConvNextForImageClassification,
    DeiTModel,
    RobertaForTokenClassification,
    SwinForImageClassification,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
)

from model import DTCAModel
from utils.MyDataSet import MyDataSet2


SENTIMENT_NAMES = {0: "NEG", 1: "NEU", 2: "POS"}
SENTIMENT_TO_ID = {"NEG": 0, "NEU": 1, "POS": 2}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, required=True, choices=["2015", "2017", "hotel", "hotel_twitter"])
    parser.add_argument("--task_name", type=str, default="dualc")
    parser.add_argument("--text_model_name", type=str, default="bert")
    parser.add_argument("--image_model_name", type=str, default="vit")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.6)
    return parser.parse_args()


def load_text_config_and_weights(text_model_name):
    if text_model_name == "bert":
        model_path = "./models/bert-base-uncased"
        return AutoConfig.from_pretrained(model_path), BertForTokenClassification.from_pretrained(model_path).state_dict()
    if text_model_name == "roberta":
        model_path = "./models/roberta-base"
        return AutoConfig.from_pretrained(model_path), RobertaForTokenClassification.from_pretrained(model_path).state_dict()
    if text_model_name == "albert":
        model_path = "./models/albert-base-v2"
        return AutoConfig.from_pretrained(model_path), AlbertForTokenClassification.from_pretrained(model_path).state_dict()
    if text_model_name == "electra":
        model_path = "./models/electra-base-discriminator"
        return AutoConfig.from_pretrained(model_path), AlbertForTokenClassification.from_pretrained(model_path).state_dict()
    raise ValueError(f"Unsupported text model: {text_model_name}")


def load_image_config_and_weights(image_model_name):
    if image_model_name == "vit":
        model_path = "./models/vit-base-patch16-224-in21k"
        return AutoConfig.from_pretrained(model_path), ViTForImageClassification.from_pretrained(model_path).state_dict()
    if image_model_name == "swin":
        model_path = "./models/swin-tiny-patch4-window7-224"
        return AutoConfig.from_pretrained(model_path), SwinForImageClassification.from_pretrained(model_path).state_dict()
    if image_model_name == "deit":
        model_path = "./models/deit-base-patch16-224"
        return AutoConfig.from_pretrained(model_path), DeiTModel.from_pretrained(model_path).state_dict()
    if image_model_name == "convnext":
        model_path = "./models/convnext-tiny-224"
        return AutoConfig.from_pretrained(model_path), ConvNextForImageClassification.from_pretrained(model_path).state_dict()
    raise ValueError(f"Unsupported image model: {image_model_name}")


def build_model(args):
    config1, text_pretrained_dict = load_text_config_and_weights(args.text_model_name)
    config2, image_pretrained_dict = load_image_config_and_weights(args.image_model_name)
    model = DTCAModel(
        config1,
        config2,
        text_num_labels=5,
        text_model_name=args.text_model_name,
        image_model_name=args.image_model_name,
        alpha=args.alpha,
        beta=args.beta,
    )

    model_dict = model.state_dict()
    for k, v in image_pretrained_dict.items():
        if model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            model_dict[k] = v
    for k, v in text_pretrained_dict.items():
        if model_dict.get(k) is not None and k not in {"classifier.bias", "classifier.weight"}:
            model_dict[k] = v
    model.load_state_dict(model_dict)

    if not os.path.exists(args.model_file):
        raise FileNotFoundError(
            f"Missing trained model: {args.model_file}. Re-run training first so main.py saves final_model.pt."
        )
    state_dict = torch.load(args.model_file, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def decode_pairs(pred_labels, tokenized_inputs):
    all_pairs = []
    for i, pred_label in enumerate(pred_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        pairs = set()
        active = False
        start_pos = 0
        end_pos = 0
        sentiment = 0
        prev_word_idx = None
        for j, label in enumerate(pred_label):
            word_idx = word_ids[j]
            if word_idx is None:
                if active:
                    pairs.add((f"{start_pos}-{end_pos}", sentiment))
                    active = False
                prev_word_idx = word_idx
                continue
            if word_idx != prev_word_idx:
                if label > 1:
                    if active:
                        pairs.add((f"{start_pos}-{end_pos}", sentiment))
                    start_pos = word_idx
                    end_pos = word_idx
                    sentiment = int(label) - 2
                    active = True
                elif label == 1 and active:
                    end_pos = word_idx
                else:
                    if active:
                        pairs.add((f"{start_pos}-{end_pos}", sentiment))
                    active = False
            prev_word_idx = word_idx
        if active:
            pairs.add((f"{start_pos}-{end_pos}", sentiment))
        all_pairs.append(pairs)
    return all_pairs


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


def term_for_span(words, span):
    start, end = [int(x) for x in span.split("-")]
    if not words or start >= len(words):
        return span
    end = min(end, len(words) - 1)
    return " ".join(words[start : end + 1])


def load_json_rows(dataset_type):
    if dataset_type in {"2015", "2017"}:
        path = os.path.join("datasets", f"twitter{dataset_type}", "test.json")
    else:
        path = os.path.join("datasets", dataset_type, "test.json")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_pair_report(path, pred_pairs, gold_pairs, json_rows, include_sentiment):
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
    os.makedirs(args.output_dir, exist_ok=True)

    data_input_file = os.path.join("datasets", "finetune", args.task_name, args.dataset_type, "input.pt")
    data_inputs = torch.load(data_input_file, weights_only=False)
    test_pairs = [set((span, int(sent)) for span, sent in pairs) for pairs in data_inputs["test"]["pairs"]]
    data_inputs["test"].pop("pairs")
    test_dataset = MyDataSet2(inputs=data_inputs["test"])

    model = build_model(args)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "trainer_tmp"),
        per_device_eval_batch_size=args.batch_size,
        do_train=False,
        do_eval=True,
        label_names=["labels", "cross_labels"],
    )
    trainer = Trainer(model=model, args=training_args)
    outputs = trainer.predict(test_dataset)
    _, cross_logits = outputs.predictions
    pred_labels = np.argmax(cross_logits, axis=-1)
    pred_pairs = decode_pairs(pred_labels, data_inputs["test"])

    maesc_correct = sum(len(p & g) for p, g in zip(pred_pairs, test_pairs))
    maesc_pred = sum(len(p) for p in pred_pairs)
    maesc_gold = sum(len(g) for g in test_pairs)
    maesc_p, maesc_r, maesc_f1 = prf(maesc_correct, maesc_pred, maesc_gold)
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

    json_rows = load_json_rows(args.dataset_type)
    maesc_path = os.path.join(args.output_dir, "maesc_pred_vs_gold.tsv")
    mate_path = os.path.join(args.output_dir, "mate_pred_vs_gold.tsv")
    masc_path = os.path.join(args.output_dir, "masc_classification_report.txt")
    write_pair_report(maesc_path, pred_pairs, test_pairs, json_rows, include_sentiment=True)
    write_pair_report(mate_path, pred_pairs, test_pairs, json_rows, include_sentiment=False)
    with open(masc_path, "w", encoding="utf-8") as f:
        f.write(masc_report)
        f.write("\n")
    summary_path = os.path.join(args.output_dir, "summary_counts.json")
    with open(summary_path, "w", encoding="utf-8") as f:
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

    print(f"Dataset: Twitter {args.dataset_type}")
    print(f"MAESC pred vs gold: {maesc_path}")
    print(f"MAESC TP/FP/FN: {maesc_counts['tp']} / {maesc_counts['fp']} / {maesc_counts['fn']}")
    print(f"MAESC P/R/F1: {maesc_p:.2f} / {maesc_r:.2f} / {maesc_f1:.2f}")
    print(f"MATE pred vs gold: {mate_path}")
    print(f"MATE TP/FP/FN: {mate_counts['tp']} / {mate_counts['fp']} / {mate_counts['fn']}")
    print(f"MATE P/R/F1: {mate_p:.2f} / {mate_r:.2f} / {mate_f1:.2f}")
    print(f"Summary counts: {summary_path}")
    print("MASC classification report:")
    print(masc_report)


if __name__ == "__main__":
    main()

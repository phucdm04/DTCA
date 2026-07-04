import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, ViTModel

from train_quad_hotel import build_label_maps, load_raw, model_path, norm_sentiment, set_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="DTCA-style MACSA: extract (category, sentiment) pairs.")
    parser.add_argument("--data_dir", default="datasets/hotel")
    parser.add_argument("--image_dir", default="datasets/hotel_images")
    parser.add_argument("--text_model_name", default="bert", choices=["bert", "roberta"])
    parser.add_argument("--image_model_path", default="./models/vit-base-patch16-224-in21k")
    parser.add_argument("--output_dir", default="results/hotel_macsa_dtca_bert_vit")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


def image_name(record):
    url = str(record.get("review_photo", ""))
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if name:
        return name
    return os.path.splitext(str(record.get("json_file", "missing.json")))[0] + ".jpg"


def extract_category_sentiment_pairs(record):
    pairs = set()
    for item in record.get("extraction", []):
        category = str(item.get("Category", "")).strip().upper()
        if not category:
            continue
        sentiment = norm_sentiment(item.get("Polarity"))
        pairs.add((category, sentiment))
    return pairs


def pair_labels(category2id, sentiment2id):
    categories = [c for c, _ in sorted(category2id.items(), key=lambda x: x[1])]
    sentiments = [s for s, _ in sorted(sentiment2id.items(), key=lambda x: x[1])]
    return [(c, s) for c in categories for s in sentiments]


def prf(tp, fp, fn):
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp) * 100
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn) * 100
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


class MACSADataset(Dataset):
    def __init__(self, records, tokenizer, image_processor, pair2id, max_length, image_dir):
        self.samples = []
        self.skipped = 0
        for idx, record in enumerate(records):
            words = str(record.get("review", "")).strip().split()
            pairs = extract_category_sentiment_pairs(record)
            img_path = os.path.join(image_dir, image_name(record))
            if not words or not pairs or not os.path.exists(img_path):
                self.skipped += 1
                continue

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((224, 224))
                pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

            labels = torch.zeros(len(pair2id), dtype=torch.float)
            kept_pairs = set()
            for pair in pairs:
                if pair in pair2id:
                    labels[pair2id[pair]] = 1.0
                    kept_pairs.add(pair)
            if not kept_pairs:
                self.skipped += 1
                continue

            self.samples.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
                    "pixel_values": pixel_values,
                    "labels": labels,
                    "gold_pairs": kept_pairs,
                    "sample_id": idx,
                    "review": " ".join(words),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    result = {}
    for key in ["input_ids", "attention_mask", "token_type_ids", "pixel_values", "labels"]:
        result[key] = torch.stack([x[key] for x in batch])
    result["meta"] = batch
    return result


class DTCAMACSAClassifier(nn.Module):
    def __init__(self, text_path, image_path, num_pairs):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_path)
        self.image_encoder = ViTModel.from_pretrained(image_path)
        hidden = self.text_encoder.config.hidden_size
        self.image_text_cross = nn.MultiheadAttention(hidden, 8, dropout=0.1, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(getattr(self.text_encoder.config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_pairs),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "roberta" not in self.text_encoder.__class__.__name__.lower():
            kwargs["token_type_ids"] = token_type_ids
        text_hidden = self.text_encoder(**kwargs).last_hidden_state
        image_hidden = self.image_encoder(pixel_values=pixel_values).last_hidden_state
        cross_hidden, _ = self.image_text_cross(text_hidden, image_hidden, image_hidden)
        fused = self.cross_norm(text_hidden + cross_hidden)
        return self.classifier(self.dropout(fused[:, 0]))


def batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fct = nn.BCEWithLogitsLoss()
    total = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()
        logits = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"), batch["pixel_values"])
        loss = loss_fct(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def decode_pairs(logits, id2pair, threshold):
    probs = torch.sigmoid(logits).cpu()
    decoded = []
    for row in probs:
        indices = [i for i, score in enumerate(row.tolist()) if score >= threshold]
        if not indices:
            indices = [int(row.argmax().item())]
        decoded.append({id2pair[i] for i in indices})
    return decoded


@torch.no_grad()
def evaluate(model, loader, device, id2pair, threshold, output_dir=None):
    model.eval()
    pred_all = set()
    gold_all = set()
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        meta = batch["meta"]
        batch = batch_to_device(batch, device)
        logits = model(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"), batch["pixel_values"])
        pred_sets = decode_pairs(logits, id2pair, threshold)
        for item, pred_pairs in zip(meta, pred_sets):
            pred_all |= {(item["sample_id"], *pair) for pair in pred_pairs}
            gold_all |= {(item["sample_id"], *pair) for pair in item["gold_pairs"]}
            if output_dir is not None:
                rows.append((item["sample_id"], item["review"], pred_pairs, item["gold_pairs"]))

    tp = len(pred_all & gold_all)
    fp = len(pred_all - gold_all)
    fn = len(gold_all - pred_all)
    precision, recall, f1 = prf(tp, fp, fn)
    metrics = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "macsa_pred_vs_gold.tsv"), "w", encoding="utf-8") as f:
            f.write("sample_id\tpred\tgold\treview\n")
            for sample_id, review, pred, gold in rows:
                pred_text = " | ".join(f"C={c} S={s}" for c, s in sorted(pred))
                gold_text = " | ".join(f"C={c} S={s}" for c, s in sorted(gold))
                f.write(f"{sample_id}\t{pred_text}\t{gold_text}\t{review}\n")
        with open(os.path.join(output_dir, "macsa_summary_counts.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return metrics


def save_checkpoint(model, tokenizer, output_dir, pair2id, category2id, sentiment2id, threshold):
    model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "macsa_label_maps.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "pair2id": {f"{c}|||{s}": i for (c, s), i in pair2id.items()},
                "category2id": category2id,
                "sentiment2id": sentiment2id,
                "threshold": threshold,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    text_path = model_path(args.text_model_name)
    train_records = load_raw(args.data_dir, "train")
    dev_records = load_raw(args.data_dir, "dev")
    test_records = load_raw(args.data_dir, "test")
    category2id, sentiment2id = build_label_maps(train_records)
    labels = pair_labels(category2id, sentiment2id)
    pair2id = {pair: i for i, pair in enumerate(labels)}
    id2pair = {i: pair for pair, i in pair2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(text_path, add_prefix_space=(args.text_model_name == "roberta"))
    image_processor = AutoImageProcessor.from_pretrained(args.image_model_path)
    train_ds = MACSADataset(train_records, tokenizer, image_processor, pair2id, args.max_length, args.image_dir)
    dev_ds = MACSADataset(dev_records, tokenizer, image_processor, pair2id, args.max_length, args.image_dir)
    test_ds = MACSADataset(test_records, tokenizer, image_processor, pair2id, args.max_length, args.image_dir)
    print(f"train/dev/test records: {len(train_ds)}/{len(dev_ds)}/{len(test_ds)}")
    print(f"skipped train/dev/test: {train_ds.skipped}/{dev_ds.skipped}/{test_ds.skipped}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTCAMACSAClassifier(text_path, args.image_model_path, len(pair2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    logs = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, dev_loader, device, id2pair, args.threshold)
        logs.append({"epoch": epoch, "loss": loss, "dev_macsa": dev_metrics})
        print(f"epoch={epoch} loss={loss:.4f} dev_f1={dev_metrics['f1']:.2f}")
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            save_checkpoint(model, tokenizer, args.output_dir, pair2id, category2id, sentiment2id, args.threshold)

    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    best_model = DTCAMACSAClassifier(text_path, args.image_model_path, len(pair2id)).to(device)
    best_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model", "pytorch_model.bin"), map_location=device))
    test_metrics = evaluate(best_model, test_loader, device, id2pair, args.threshold, output_dir=args.output_dir)
    print(
        "test MACSA TP/FP/FN/P/R/F1: "
        f"{test_metrics['tp']} / {test_metrics['fp']} / {test_metrics['fn']} / "
        f"{test_metrics['precision']:.2f} / {test_metrics['recall']:.2f} / {test_metrics['f1']:.2f}"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, ViTModel

from train_quad_hotel import build_label_maps, load_raw, model_path, norm_sentiment, set_seed, valid_span


ImageFile.LOAD_TRUNCATED_IMAGES = True


def image_name(record):
    url = str(record.get("review_photo", ""))
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if name:
        return name
    return os.path.splitext(str(record.get("json_file", "missing.json")))[0] + ".jpg"


def parse_args():
    parser = argparse.ArgumentParser(description="DTCA-style ACC/ACSA classifier for hotel.")
    parser.add_argument("--data_dir", default="datasets/hotel")
    parser.add_argument("--image_dir", default="datasets/hotel_images")
    parser.add_argument("--text_model_name", default="bert", choices=["bert", "roberta"])
    parser.add_argument("--image_model_path", default="./models/vit-base-patch16-224-in21k")
    parser.add_argument("--output_dir", default="results/hotel_acsa_dtca_bert_vit")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


class ACSADataset(Dataset):
    def __init__(self, records, tokenizer, image_processor, category2id, sentiment2id, max_length, image_dir):
        self.samples = []
        self.skipped = 0
        for record_idx, record in enumerate(records):
            words = str(record.get("review", "")).strip().split()
            img_path = os.path.join(image_dir, image_name(record))
            if not words or not os.path.exists(img_path):
                self.skipped += len(record.get("extraction", []))
                continue

            encoding = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            word_ids = encoding.word_ids(batch_index=0)
            first_token_by_word = {}
            prev = None
            for tok_idx, word_idx in enumerate(word_ids):
                if word_idx is not None and word_idx != prev:
                    first_token_by_word[word_idx] = tok_idx
                prev = word_idx

            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((224, 224))
                pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

            for ext_idx, item in enumerate(record.get("extraction", [])):
                aspect_span = item.get("Aspect_span")
                if not valid_span(aspect_span, words):
                    self.skipped += 1
                    continue
                token_positions = [
                    first_token_by_word.get(i)
                    for i in range(aspect_span[0], aspect_span[1])
                    if first_token_by_word.get(i) is not None
                ]
                if not token_positions:
                    self.skipped += 1
                    continue
                category = str(item.get("Category", "")).strip().upper()
                sentiment = norm_sentiment(item.get("Polarity"))
                if category not in category2id:
                    self.skipped += 1
                    continue

                self.samples.append(
                    {
                        "input_ids": encoding["input_ids"].squeeze(0),
                        "attention_mask": encoding["attention_mask"].squeeze(0),
                        "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
                        "pixel_values": pixel_values,
                        "aspect_token_span": torch.tensor([min(token_positions), max(token_positions)], dtype=torch.long),
                        "category_label": torch.tensor(category2id[category], dtype=torch.long),
                        "sentiment_label": torch.tensor(sentiment2id[sentiment], dtype=torch.long),
                        "sample_id": f"{record_idx}:{ext_idx}",
                        "words": words,
                        "aspect_span": aspect_span,
                        "aspect": str(item.get("Aspect", " ".join(words[aspect_span[0] : aspect_span[1]]))),
                        "category": category,
                        "sentiment": sentiment,
                    }
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    result = {}
    for key in ["input_ids", "attention_mask", "token_type_ids", "pixel_values", "aspect_token_span"]:
        result[key] = torch.stack([x[key] for x in batch])
    result["category_label"] = torch.stack([x["category_label"] for x in batch])
    result["sentiment_label"] = torch.stack([x["sentiment_label"] for x in batch])
    result["meta"] = batch
    return result


class DTCAACSAClassifier(nn.Module):
    def __init__(self, text_path, image_path, num_categories, num_sentiments):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_path)
        self.image_encoder = ViTModel.from_pretrained(image_path)
        hidden = self.text_encoder.config.hidden_size
        self.image_text_cross = nn.MultiheadAttention(hidden, 8, dropout=0.1, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(getattr(self.text_encoder.config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.category_classifier = nn.Linear(hidden, num_categories)
        self.sentiment_classifier = nn.Linear(hidden, num_sentiments)

    def forward(self, input_ids, attention_mask, token_type_ids, pixel_values, aspect_token_span):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "roberta" not in self.text_encoder.__class__.__name__.lower():
            kwargs["token_type_ids"] = token_type_ids
        text_hidden = self.text_encoder(**kwargs).last_hidden_state
        image_hidden = self.image_encoder(pixel_values=pixel_values).last_hidden_state
        cross_hidden, _ = self.image_text_cross(text_hidden, image_hidden, image_hidden)
        text_hidden = self.cross_norm(text_hidden + cross_hidden)

        reps = []
        for i, (start, end) in enumerate(aspect_token_span.tolist()):
            aspect_rep = text_hidden[i, start : end + 1].mean(dim=0)
            cls_rep = text_hidden[i, 0]
            reps.append(torch.cat([cls_rep, aspect_rep], dim=-1))
        reps = self.dropout(torch.stack(reps))
        reps = self.classifier(reps)
        return self.category_classifier(reps), self.sentiment_classifier(reps)


def batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fct = nn.CrossEntropyLoss()
    total = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()
        category_logits, sentiment_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
            batch["pixel_values"],
            batch["aspect_token_span"],
        )
        loss = loss_fct(category_logits, batch["category_label"])
        loss = loss + loss_fct(sentiment_logits, batch["sentiment_label"])
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, id2category, id2sentiment, output_dir=None):
    model.eval()
    gold_cat = []
    pred_cat = []
    gold_sent = []
    pred_sent = []
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        meta = batch["meta"]
        batch = batch_to_device(batch, device)
        category_logits, sentiment_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
            batch["pixel_values"],
            batch["aspect_token_span"],
        )
        cat_pred_ids = category_logits.argmax(dim=-1).cpu().tolist()
        sent_pred_ids = sentiment_logits.argmax(dim=-1).cpu().tolist()
        cat_gold_ids = batch["category_label"].cpu().tolist()
        sent_gold_ids = batch["sentiment_label"].cpu().tolist()
        pred_cat.extend(cat_pred_ids)
        pred_sent.extend(sent_pred_ids)
        gold_cat.extend(cat_gold_ids)
        gold_sent.extend(sent_gold_ids)
        for item, pc, ps, gc, gs in zip(meta, cat_pred_ids, sent_pred_ids, cat_gold_ids, sent_gold_ids):
            rows.append(
                {
                    "sample_id": item["sample_id"],
                    "aspect": item["aspect"],
                    "aspect_span": item["aspect_span"],
                    "pred_category": id2category[pc],
                    "gold_category": id2category[gc],
                    "pred_sentiment": id2sentiment[ps],
                    "gold_sentiment": id2sentiment[gs],
                }
            )

    gold_pairs = list(zip(gold_cat, gold_sent))
    pred_pairs = list(zip(pred_cat, pred_sent))
    acc = accuracy_score(gold_cat, pred_cat) * 100
    sentiment_acc = accuracy_score(gold_sent, pred_sent) * 100
    pair_tp = sum(1 for g, p in zip(gold_pairs, pred_pairs) if g == p)
    acsa_acc = 0.0 if not gold_pairs else pair_tp / len(gold_pairs) * 100
    metrics = {
        "acc_category_accuracy": acc,
        "sentiment_accuracy": sentiment_acc,
        "acsa_category_sentiment_accuracy": acsa_acc,
        "acsa_tp": pair_tp,
        "acsa_fp": len(pred_pairs) - pair_tp,
        "acsa_fn": len(gold_pairs) - pair_tp,
        "total": len(gold_pairs),
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "acc_category_report.txt"), "w", encoding="utf-8") as f:
            f.write(
                classification_report(
                    [id2category[i] for i in gold_cat],
                    [id2category[i] for i in pred_cat],
                    labels=[id2category[i] for i in sorted(id2category)],
                    digits=4,
                    zero_division=0,
                )
            )
            f.write("\n")
        with open(os.path.join(output_dir, "sentiment_report.txt"), "w", encoding="utf-8") as f:
            f.write(
                classification_report(
                    [id2sentiment[i] for i in gold_sent],
                    [id2sentiment[i] for i in pred_sent],
                    labels=[id2sentiment[i] for i in sorted(id2sentiment)],
                    digits=4,
                    zero_division=0,
                )
            )
            f.write("\n")
        with open(os.path.join(output_dir, "acsa_pred_vs_gold.tsv"), "w", encoding="utf-8") as f:
            f.write("sample_id\taspect\taspect_span\tpred_category\tgold_category\tpred_sentiment\tgold_sentiment\n")
            for row in rows:
                f.write(
                    f"{row['sample_id']}\t{row['aspect']}\t{row['aspect_span']}\t"
                    f"{row['pred_category']}\t{row['gold_category']}\t"
                    f"{row['pred_sentiment']}\t{row['gold_sentiment']}\n"
                )
        with open(os.path.join(output_dir, "acsa_summary_counts.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return metrics


def save_checkpoint(model, tokenizer, output_dir, category2id, sentiment2id):
    model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "acsa_label_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"category2id": category2id, "sentiment2id": sentiment2id}, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    text_path = model_path(args.text_model_name)
    train_records = load_raw(args.data_dir, "train")
    dev_records = load_raw(args.data_dir, "dev")
    test_records = load_raw(args.data_dir, "test")
    category2id, sentiment2id = build_label_maps(train_records)
    id2category = {v: k for k, v in category2id.items()}
    id2sentiment = {v: k for k, v in sentiment2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(text_path, add_prefix_space=(args.text_model_name == "roberta"))
    image_processor = AutoImageProcessor.from_pretrained(args.image_model_path)
    train_ds = ACSADataset(train_records, tokenizer, image_processor, category2id, sentiment2id, args.max_length, args.image_dir)
    dev_ds = ACSADataset(dev_records, tokenizer, image_processor, category2id, sentiment2id, args.max_length, args.image_dir)
    test_ds = ACSADataset(test_records, tokenizer, image_processor, category2id, sentiment2id, args.max_length, args.image_dir)
    print(f"train/dev/test instances: {len(train_ds)}/{len(dev_ds)}/{len(test_ds)}")
    print(f"skipped train/dev/test: {train_ds.skipped}/{dev_ds.skipped}/{test_ds.skipped}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTCAACSAClassifier(text_path, args.image_model_path, len(category2id), len(sentiment2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = -1.0
    logs = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, dev_loader, device, id2category, id2sentiment)
        logs.append({"epoch": epoch, "loss": loss, "dev": dev_metrics})
        print(
            f"epoch={epoch} loss={loss:.4f} "
            f"dev_acc={dev_metrics['acc_category_accuracy']:.2f} "
            f"dev_acsa={dev_metrics['acsa_category_sentiment_accuracy']:.2f}"
        )
        if dev_metrics["acsa_category_sentiment_accuracy"] > best:
            best = dev_metrics["acsa_category_sentiment_accuracy"]
            save_checkpoint(model, tokenizer, args.output_dir, category2id, sentiment2id)

    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    best_model = DTCAACSAClassifier(text_path, args.image_model_path, len(category2id), len(sentiment2id)).to(device)
    best_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model", "pytorch_model.bin"), map_location=device))
    test_metrics = evaluate(best_model, test_loader, device, id2category, id2sentiment, output_dir=args.output_dir)
    print(
        f"test ACC={test_metrics['acc_category_accuracy']:.2f} "
        f"sentiment_acc={test_metrics['sentiment_accuracy']:.2f} "
        f"ACSA={test_metrics['acsa_category_sentiment_accuracy']:.2f}"
    )


if __name__ == "__main__":
    main()

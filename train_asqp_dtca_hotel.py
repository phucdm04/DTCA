import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizerFast, ViTModel, AutoImageProcessor
from transformers.modeling_outputs import BaseModelOutput

from train_quad_hotel import build_label_maps, load_raw, norm_sentiment, set_seed, valid_span


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="DTCA-style ASQP seq2seq model for hotel.")
    parser.add_argument("--data_dir", default="datasets/hotel")
    parser.add_argument("--image_dir", default="datasets/hotel_images")
    parser.add_argument("--bart_model_path", default="./models/bart-base")
    parser.add_argument("--image_model_path", default="./models/vit-base-patch16-224-in21k")
    parser.add_argument("--output_dir", default="results/hotel_asqp_dtca_bart_vit")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


def image_name(record):
    url = str(record.get("review_photo", ""))
    name = url.split("?")[0].rstrip("/").split("/")[-1]
    if name:
        return name
    return os.path.splitext(str(record.get("json_file", "missing.json")))[0] + ".jpg"


def category_token(category):
    return f"<cat_{category.upper()}>"


def sentiment_token(sentiment):
    return f"<sent_{sentiment.upper()}>"


def asqp_gold(record):
    words = str(record.get("review", "")).strip().split()
    quads = []
    for item in record.get("extraction", []):
        asp = item.get("Aspect_span")
        opn = item.get("Opinion_span")
        if not valid_span(asp, words) or not valid_span(opn, words):
            continue
        category = str(item.get("Category", "")).strip().upper()
        sentiment = norm_sentiment(item.get("Polarity"))
        quads.append((asp[0], asp[1], category, opn[0], opn[1], sentiment))
    return words, sorted(set(quads))


def serialize_quads(quads):
    parts = []
    for a0, a1, category, o0, o1, sentiment in quads:
        parts.extend([str(a0), str(a1), category_token(category), str(o0), str(o1), sentiment_token(sentiment)])
    return " ".join(parts)


def parse_generated(text):
    toks = text.strip().split()
    quads = set()
    i = 0
    while i + 5 < len(toks):
        try:
            a0 = int(toks[i])
            a1 = int(toks[i + 1])
            cat_tok = toks[i + 2]
            o0 = int(toks[i + 3])
            o1 = int(toks[i + 4])
            sent_tok = toks[i + 5]
        except ValueError:
            i += 1
            continue
        if cat_tok.startswith("<cat_") and cat_tok.endswith(">") and sent_tok.startswith("<sent_") and sent_tok.endswith(">"):
            category = cat_tok[5:-1]
            sentiment = sent_tok[6:-1]
            if a0 < a1 and o0 < o1:
                quads.add((a0, a1, category, o0, o1, sentiment))
            i += 6
        else:
            i += 1
    return quads


def prf(tp, fp, fn):
    p = 0.0 if tp + fp == 0 else tp / (tp + fp) * 100
    r = 0.0 if tp + fn == 0 else tp / (tp + fn) * 100
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


class ASQPDataset(Dataset):
    def __init__(self, records, tokenizer, image_processor, max_source_length, max_target_length, image_dir):
        self.samples = []
        self.skipped = 0
        for idx, record in enumerate(records):
            words, quads = asqp_gold(record)
            img_path = os.path.join(image_dir, image_name(record))
            if not words or not quads or not os.path.exists(img_path):
                self.skipped += 1
                continue

            source = " ".join(words)
            target = serialize_quads(quads)
            source_enc = tokenizer(
                source,
                truncation=True,
                padding="max_length",
                max_length=max_source_length,
                return_tensors="pt",
            )
            target_enc = tokenizer(
                target,
                truncation=True,
                padding="max_length",
                max_length=max_target_length,
                return_tensors="pt",
            )
            labels = target_enc["input_ids"].squeeze(0)
            labels[labels == tokenizer.pad_token_id] = -100

            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((224, 224))
                pixel_values = image_processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

            self.samples.append(
                {
                    "input_ids": source_enc["input_ids"].squeeze(0),
                    "attention_mask": source_enc["attention_mask"].squeeze(0),
                    "pixel_values": pixel_values,
                    "labels": labels,
                    "gold_quads": quads,
                    "words": words,
                    "sample_id": idx,
                    "target": target,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    result = {}
    for key in ["input_ids", "attention_mask", "pixel_values", "labels"]:
        result[key] = torch.stack([x[key] for x in batch])
    result["meta"] = batch
    return result


class DTCABartASQP(nn.Module):
    def __init__(self, bart_path, image_path):
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(bart_path)
        self.image_encoder = ViTModel.from_pretrained(image_path)
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, self.bart.config.d_model)
        self.fusion_norm = nn.LayerNorm(self.bart.config.d_model)

    def resize_token_embeddings(self, n):
        self.bart.resize_token_embeddings(n)

    def fused_encoder_outputs(self, input_ids, attention_mask, pixel_values):
        text_outputs = self.bart.model.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        image_hidden = self.image_proj(self.image_encoder(pixel_values=pixel_values).last_hidden_state)
        hidden = torch.cat([text_outputs.last_hidden_state, image_hidden], dim=1)
        hidden = self.fusion_norm(hidden)
        image_mask = torch.ones(pixel_values.size(0), image_hidden.size(1), dtype=attention_mask.dtype, device=attention_mask.device)
        fused_mask = torch.cat([attention_mask, image_mask], dim=1)
        return BaseModelOutput(last_hidden_state=hidden), fused_mask

    def forward(self, input_ids, attention_mask, pixel_values, labels=None):
        encoder_outputs, fused_mask = self.fused_encoder_outputs(input_ids, attention_mask, pixel_values)
        return self.bart(encoder_outputs=encoder_outputs, attention_mask=fused_mask, labels=labels, return_dict=True)

    def generate(self, input_ids, attention_mask, pixel_values, **kwargs):
        encoder_outputs, fused_mask = self.fused_encoder_outputs(input_ids, attention_mask, pixel_values)
        return self.bart.generate(encoder_outputs=encoder_outputs, attention_mask=fused_mask, **kwargs)


def batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()
        loss = model(batch["input_ids"], batch["attention_mask"], batch["pixel_values"], labels=batch["labels"]).loss
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def format_quad(words, quad):
    a0, a1, category, o0, o1, sentiment = quad
    return f"A={' '.join(words[a0:a1])}[{a0}-{a1}] C={category} O={' '.join(words[o0:o1])}[{o0}-{o1}] S={sentiment}"


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_target_length, num_beams, output_dir=None):
    model.eval()
    pred_all = set()
    gold_all = set()
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        meta = batch["meta"]
        batch = batch_to_device(batch, device)
        generated = model.generate(
            batch["input_ids"],
            batch["attention_mask"],
            batch["pixel_values"],
            max_length=max_target_length,
            num_beams=num_beams,
        )
        texts = tokenizer.batch_decode(generated, skip_special_tokens=False)
        for item, text in zip(meta, texts):
            pred = parse_generated(text)
            gold = set(item["gold_quads"])
            pred_all |= {(item["sample_id"], *q) for q in pred}
            gold_all |= {(item["sample_id"], *q) for q in gold}
            if output_dir is not None:
                rows.append((item["sample_id"], item["words"], pred, gold, text))

    tp = len(pred_all & gold_all)
    fp = len(pred_all - gold_all)
    fn = len(gold_all - pred_all)
    p, r, f1 = prf(tp, fp, fn)
    metrics = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "asqp_pred_vs_gold.tsv"), "w", encoding="utf-8") as f:
            f.write("sample_id\tpred\tgold\tgenerated\n")
            for sample_id, words, pred, gold, text in rows:
                pred_text = " | ".join(format_quad(words, q) for q in sorted(pred))
                gold_text = " | ".join(format_quad(words, q) for q in sorted(gold))
                f.write(f"{sample_id}\t{pred_text}\t{gold_text}\t{text}\n")
        with open(os.path.join(output_dir, "asqp_summary_counts.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return metrics


def save_checkpoint(model, tokenizer, output_dir, categories, sentiments):
    model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "asqp_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"categories": categories, "sentiments": sentiments}, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.exists(args.bart_model_path):
        raise FileNotFoundError(f"Missing BART model: {args.bart_model_path}")
    if not os.path.exists(args.image_model_path):
        raise FileNotFoundError(f"Missing image model: {args.image_model_path}")

    train_records = load_raw(args.data_dir, "train")
    dev_records = load_raw(args.data_dir, "dev")
    test_records = load_raw(args.data_dir, "test")
    category2id, sentiment2id = build_label_maps(train_records)
    categories = [c for c, _ in sorted(category2id.items(), key=lambda x: x[1])]
    sentiments = [s for s, _ in sorted(sentiment2id.items(), key=lambda x: x[1])]

    tokenizer = BartTokenizerFast.from_pretrained(args.bart_model_path)
    special_tokens = [category_token(c) for c in categories] + [sentiment_token(s) for s in sentiments]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    image_processor = AutoImageProcessor.from_pretrained(args.image_model_path)

    train_ds = ASQPDataset(train_records, tokenizer, image_processor, args.max_source_length, args.max_target_length, args.image_dir)
    dev_ds = ASQPDataset(dev_records, tokenizer, image_processor, args.max_source_length, args.max_target_length, args.image_dir)
    test_ds = ASQPDataset(test_records, tokenizer, image_processor, args.max_source_length, args.max_target_length, args.image_dir)
    print(f"train/dev/test records: {len(train_ds)}/{len(dev_ds)}/{len(test_ds)}")
    print(f"skipped train/dev/test: {train_ds.skipped}/{dev_ds.skipped}/{test_ds.skipped}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTCABartASQP(args.bart_model_path, args.image_model_path).to(device)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    logs = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, dev_loader, tokenizer, device, args.max_target_length, args.num_beams)
        logs.append({"epoch": epoch, "loss": loss, "dev_asqp": dev_metrics})
        print(f"epoch={epoch} loss={loss:.4f} dev_f1={dev_metrics['f1']:.2f}")
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            save_checkpoint(model, tokenizer, args.output_dir, categories, sentiments)

    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    best_model = DTCABartASQP(args.bart_model_path, args.image_model_path).to(device)
    best_model.resize_token_embeddings(len(tokenizer))
    best_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model", "pytorch_model.bin"), map_location=device))
    test_metrics = evaluate(best_model, test_loader, tokenizer, device, args.max_target_length, args.num_beams, output_dir=args.output_dir)
    print(
        "test ASQP TP/FP/FN/P/R/F1: "
        f"{test_metrics['tp']} / {test_metrics['fp']} / {test_metrics['fn']} / "
        f"{test_metrics['precision']:.2f} / {test_metrics['recall']:.2f} / {test_metrics['f1']:.2f}"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import os
import random
from collections import Counter

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


TAG_O = 0
TAG_B = 1
TAG_I = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Simple text-only Quad baseline for hotel ABSA.")
    parser.add_argument("--data_dir", default="datasets/hotel")
    parser.add_argument("--text_model_name", default="bert", choices=["bert", "roberta"])
    parser.add_argument("--output_dir", default="results/hotel_quad_bert")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_eval_pairs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2022)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def model_path(name):
    if name == "bert":
        return "./models/bert-base-uncased"
    if name == "roberta":
        return "./models/roberta-base"
    raise ValueError(name)


def raw_path(data_dir, split):
    candidates = []
    if split == "dev":
        candidates = ["val.json.raw.bak", "val.json", "dev.json.raw.bak", "dev.json"]
    else:
        candidates = [f"{split}.json.raw.bak", f"{split}.json"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No raw JSON found for {split} in {data_dir}")


def load_raw(data_dir, split):
    with open(raw_path(data_dir, split), "r", encoding="utf-8") as f:
        return json.load(f)


def norm_sentiment(value):
    key = str(value).strip().lower()
    if key in {"negative", "neg", "-1"}:
        return "NEG"
    if key in {"neutral", "neu", "0"}:
        return "NEU"
    if key in {"positive", "pos", "1"}:
        return "POS"
    raise ValueError(f"Unsupported polarity: {value}")


def valid_span(span, words):
    return (
        isinstance(span, list)
        and len(span) == 2
        and isinstance(span[0], int)
        and isinstance(span[1], int)
        and 0 <= span[0] < span[1] <= len(words)
    )


def extract_quads(record):
    words = str(record.get("review", "")).strip().split()
    quads = []
    skipped = 0
    for item in record.get("extraction", []):
        asp = item.get("Aspect_span")
        opn = item.get("Opinion_span")
        if not valid_span(asp, words) or not valid_span(opn, words):
            skipped += 1
            continue
        cat = str(item.get("Category", "")).strip().upper()
        sent = norm_sentiment(item.get("Polarity"))
        quads.append((asp[0], asp[1], opn[0], opn[1], cat, sent))
    return words, sorted(set(quads)), skipped


def build_label_maps(records):
    cats = sorted({q[4] for r in records for q in extract_quads(r)[1]})
    sents = ["NEG", "NEU", "POS"]
    return {c: i for i, c in enumerate(cats)}, {s: i for i, s in enumerate(sents)}


def token_span(first_token_by_word, start, end):
    token_positions = [first_token_by_word.get(i) for i in range(start, end)]
    token_positions = [p for p in token_positions if p is not None]
    if not token_positions:
        return None
    return min(token_positions), max(token_positions)


class QuadDataset(Dataset):
    def __init__(self, records, tokenizer, category2id, sentiment2id, max_length):
        self.samples = []
        self.skipped = 0
        for idx, record in enumerate(records):
            words, quads, skipped = extract_quads(record)
            self.skipped += skipped
            if not words:
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

            aspect_labels = torch.full((max_length,), -100, dtype=torch.long)
            opinion_labels = torch.full((max_length,), -100, dtype=torch.long)
            for tok_idx, word_idx in enumerate(word_ids):
                if word_idx is not None and first_token_by_word.get(word_idx) == tok_idx:
                    aspect_labels[tok_idx] = TAG_O
                    opinion_labels[tok_idx] = TAG_O

            pair_indices = []
            category_labels = []
            sentiment_labels = []
            gold_quads = set()
            for a0, a1, o0, o1, cat, sent in quads:
                a_tok = token_span(first_token_by_word, a0, a1)
                o_tok = token_span(first_token_by_word, o0, o1)
                if a_tok is None or o_tok is None:
                    self.skipped += 1
                    continue

                aspect_labels[a_tok[0]] = TAG_B
                for pos in range(a_tok[0] + 1, a_tok[1] + 1):
                    if aspect_labels[pos] != -100:
                        aspect_labels[pos] = TAG_I

                opinion_labels[o_tok[0]] = TAG_B
                for pos in range(o_tok[0] + 1, o_tok[1] + 1):
                    if opinion_labels[pos] != -100:
                        opinion_labels[pos] = TAG_I

                pair_indices.append([a_tok[0], a_tok[1], o_tok[0], o_tok[1]])
                category_labels.append(category2id[cat])
                sentiment_labels.append(sentiment2id[sent])
                gold_quads.add((a0, a1, o0, o1, cat, sent))

            if not gold_quads:
                continue
            self.samples.append(
                {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
                    "aspect_labels": aspect_labels,
                    "opinion_labels": opinion_labels,
                    "pair_indices": torch.tensor(pair_indices, dtype=torch.long),
                    "category_labels": torch.tensor(category_labels, dtype=torch.long),
                    "sentiment_labels": torch.tensor(sentiment_labels, dtype=torch.long),
                    "word_ids": word_ids,
                    "words": words,
                    "gold_quads": gold_quads,
                    "sample_id": idx,
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate(batch):
    max_pairs = max(x["pair_indices"].shape[0] for x in batch)
    result = {}
    for key in ["input_ids", "attention_mask", "token_type_ids", "aspect_labels", "opinion_labels"]:
        result[key] = torch.stack([x[key] for x in batch])
    result["pair_indices"] = torch.zeros(len(batch), max_pairs, 4, dtype=torch.long)
    result["pair_mask"] = torch.zeros(len(batch), max_pairs, dtype=torch.bool)
    result["category_labels"] = torch.full((len(batch), max_pairs), -100, dtype=torch.long)
    result["sentiment_labels"] = torch.full((len(batch), max_pairs), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["pair_indices"].shape[0]
        result["pair_indices"][i, :n] = item["pair_indices"]
        result["pair_mask"][i, :n] = True
        result["category_labels"][i, :n] = item["category_labels"]
        result["sentiment_labels"][i, :n] = item["sentiment_labels"]
    result["meta"] = batch
    return result


class QuadModel(nn.Module):
    def __init__(self, base_path, num_categories, num_sentiments):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_path)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(getattr(self.encoder.config, "hidden_dropout_prob", 0.1))
        self.aspect_classifier = nn.Linear(hidden, 3)
        self.opinion_classifier = nn.Linear(hidden, 3)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.category_classifier = nn.Linear(hidden, num_categories)
        self.sentiment_classifier = nn.Linear(hidden, num_sentiments)

    def pair_representations(self, hidden, pair_indices):
        reps = []
        for b in range(hidden.size(0)):
            cur = []
            for a0, a1, o0, o1 in pair_indices[b].tolist():
                aspect = hidden[b, a0 : a1 + 1].mean(dim=0)
                opinion = hidden[b, o0 : o1 + 1].mean(dim=0)
                cls = hidden[b, 0]
                cur.append(torch.cat([cls, aspect, opinion], dim=-1))
            reps.append(torch.stack(cur))
        return torch.stack(reps)

    def forward(self, input_ids, attention_mask, token_type_ids=None, pair_indices=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and "roberta" not in self.encoder.__class__.__name__.lower():
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs).last_hidden_state
        hidden = self.dropout(hidden)
        aspect_logits = self.aspect_classifier(hidden)
        opinion_logits = self.opinion_classifier(hidden)
        category_logits = None
        sentiment_logits = None
        if pair_indices is not None and pair_indices.numel() > 0:
            pair_repr = self.pair_mlp(self.pair_representations(hidden, pair_indices))
            category_logits = self.category_classifier(pair_repr)
            sentiment_logits = self.sentiment_classifier(pair_repr)
        return aspect_logits, opinion_logits, category_logits, sentiment_logits


def batch_to_device(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def decode_spans(labels, word_ids):
    spans = []
    active = False
    start = None
    end = None
    prev_word = None
    for tok_idx, label in enumerate(labels):
        word = word_ids[tok_idx]
        if word is None or word == prev_word:
            prev_word = word
            continue
        if label == TAG_B:
            if active:
                spans.append((start, end + 1))
            start = word
            end = word
            active = True
        elif label == TAG_I and active:
            end = word
        else:
            if active:
                spans.append((start, end + 1))
            active = False
        prev_word = word
    if active:
        spans.append((start, end + 1))
    return spans


def word_span_to_token_span(word_ids, span):
    positions = [i for i, w in enumerate(word_ids) if w is not None and span[0] <= w < span[1]]
    if not positions:
        return None
    return min(positions), max(positions)


def score(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = 0.0 if tp + fp == 0 else tp / (tp + fp) * 100
    r = 0.0 if tp + fn == 0 else tp / (tp + fn) * 100
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1}


@torch.no_grad()
def evaluate(model, loader, device, id2category, id2sentiment, output_dir=None, max_eval_pairs=64):
    model.eval()
    all_pred = []
    all_gold = []
    rows = []
    for batch in tqdm(loader, desc="eval", leave=False):
        meta = batch["meta"]
        batch = batch_to_device(batch, device)
        aspect_logits, opinion_logits, _, _ = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
            None,
        )
        aspect_preds = aspect_logits.argmax(dim=-1).cpu().numpy()
        opinion_preds = opinion_logits.argmax(dim=-1).cpu().numpy()

        pred_pair_indices = []
        pred_pair_meta = []
        for i, item in enumerate(meta):
            asp_spans = decode_spans(aspect_preds[i], item["word_ids"])
            opn_spans = decode_spans(opinion_preds[i], item["word_ids"])
            candidates = []
            candidate_meta = []
            for asp in asp_spans:
                for opn in opn_spans:
                    a_tok = word_span_to_token_span(item["word_ids"], asp)
                    o_tok = word_span_to_token_span(item["word_ids"], opn)
                    if a_tok and o_tok:
                        candidates.append([a_tok[0], a_tok[1], o_tok[0], o_tok[1]])
                        candidate_meta.append((asp, opn))
            candidates = candidates[:max_eval_pairs]
            candidate_meta = candidate_meta[:max_eval_pairs]
            if not candidates:
                candidates = [[0, 0, 0, 0]]
                candidate_meta = []
            pred_pair_indices.append(torch.tensor(candidates, dtype=torch.long))
            pred_pair_meta.append(candidate_meta)

        max_pairs = max(x.shape[0] for x in pred_pair_indices)
        pair_tensor = torch.zeros(len(pred_pair_indices), max_pairs, 4, dtype=torch.long, device=device)
        for i, pairs in enumerate(pred_pair_indices):
            pair_tensor[i, : pairs.shape[0]] = pairs.to(device)

        _, _, cat_logits, sent_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
            pair_tensor,
        )
        cat_pred = cat_logits.argmax(dim=-1).cpu().numpy()
        sent_pred = sent_logits.argmax(dim=-1).cpu().numpy()

        for i, item in enumerate(meta):
            pred_quads = set()
            for j, (asp, opn) in enumerate(pred_pair_meta[i]):
                pred_quads.add(
                    (
                        asp[0],
                        asp[1],
                        opn[0],
                        opn[1],
                        id2category[int(cat_pred[i][j])],
                        id2sentiment[int(sent_pred[i][j])],
                    )
                )
            all_pred.append(pred_quads)
            all_gold.append(item["gold_quads"])
            if output_dir is not None:
                rows.append((item["sample_id"], item["words"], pred_quads, item["gold_quads"]))

    pred_set = set()
    gold_set = set()
    for idx, (pred, gold) in enumerate(zip(all_pred, all_gold)):
        pred_set |= {(idx, *q) for q in pred}
        gold_set |= {(idx, *q) for q in gold}
    metrics = score(pred_set, gold_set)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        pred_path = os.path.join(output_dir, "quad_pred_vs_gold.tsv")
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write("sample_id\tpred\tgold\n")
            for sample_id, words, pred, gold in rows:
                pred_text = " | ".join(format_quad(words, q) for q in sorted(pred))
                gold_text = " | ".join(format_quad(words, q) for q in sorted(gold))
                f.write(f"{sample_id}\t{pred_text}\t{gold_text}\n")
        with open(os.path.join(output_dir, "quad_summary_counts.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
            f.write("\n")
    return metrics


def format_quad(words, quad):
    a0, a1, o0, o1, cat, sent = quad
    aspect = " ".join(words[a0:a1])
    opinion = " ".join(words[o0:o1])
    return f"A={aspect}[{a0}-{a1}] O={opinion}[{o0}-{o1}] C={cat} S={sent}"


def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    total = 0.0
    steps = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = batch_to_device(batch, device)
        optimizer.zero_grad()
        aspect_logits, opinion_logits, category_logits, sentiment_logits = model(
            batch["input_ids"],
            batch["attention_mask"],
            batch.get("token_type_ids"),
            batch["pair_indices"],
        )
        loss = loss_fct(aspect_logits.view(-1, 3), batch["aspect_labels"].view(-1))
        loss = loss + loss_fct(opinion_logits.view(-1, 3), batch["opinion_labels"].view(-1))
        loss = loss + loss_fct(category_logits.view(-1, category_logits.size(-1)), batch["category_labels"].view(-1))
        loss = loss + loss_fct(sentiment_logits.view(-1, sentiment_logits.size(-1)), batch["sentiment_labels"].view(-1))
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        steps += 1
    return total / max(steps, 1)


def save_checkpoint(model, tokenizer, output_dir, category2id, sentiment2id):
    model_dir = os.path.join(output_dir, "best_model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(model_dir)
    with open(os.path.join(model_dir, "quad_label_maps.json"), "w", encoding="utf-8") as f:
        json.dump({"category2id": category2id, "sentiment2id": sentiment2id}, f, indent=2)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    base_path = model_path(args.text_model_name)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing local model: {base_path}")

    train_records = load_raw(args.data_dir, "train")
    dev_records = load_raw(args.data_dir, "dev")
    test_records = load_raw(args.data_dir, "test")
    category2id, sentiment2id = build_label_maps(train_records)
    id2category = {v: k for k, v in category2id.items()}
    id2sentiment = {v: k for k, v in sentiment2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(base_path, add_prefix_space=(args.text_model_name == "roberta"))
    train_ds = QuadDataset(train_records, tokenizer, category2id, sentiment2id, args.max_length)
    dev_ds = QuadDataset(dev_records, tokenizer, category2id, sentiment2id, args.max_length)
    test_ds = QuadDataset(test_records, tokenizer, category2id, sentiment2id, args.max_length)

    print(f"train/dev/test records: {len(train_ds)}/{len(dev_ds)}/{len(test_ds)}")
    print(f"skipped train/dev/test extractions: {train_ds.skipped}/{dev_ds.skipped}/{test_ds.skipped}")
    print(f"categories: {category2id}")
    print(f"sentiments: {sentiment2id}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QuadModel(base_path, len(category2id), len(sentiment2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    logs = []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        dev_metrics = evaluate(model, dev_loader, device, id2category, id2sentiment, max_eval_pairs=args.max_eval_pairs)
        logs.append({"epoch": epoch, "loss": loss, "dev_quad": dev_metrics})
        print(f"epoch={epoch} loss={loss:.4f} dev_quad_f1={dev_metrics['f1']:.2f}")
        if dev_metrics["f1"] > best_f1:
            best_f1 = dev_metrics["f1"]
            save_checkpoint(model, tokenizer, args.output_dir, category2id, sentiment2id)

    with open(os.path.join(args.output_dir, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    best_model = QuadModel(base_path, len(category2id), len(sentiment2id)).to(device)
    best_model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model", "pytorch_model.bin"), map_location=device))
    test_metrics = evaluate(
        best_model,
        test_loader,
        device,
        id2category,
        id2sentiment,
        output_dir=args.output_dir,
        max_eval_pairs=args.max_eval_pairs,
    )
    print(
        "test_quad TP/FP/FN/P/R/F1: "
        f"{test_metrics['tp']} / {test_metrics['fp']} / {test_metrics['fn']} / "
        f"{test_metrics['precision']:.2f} / {test_metrics['recall']:.2f} / {test_metrics['f1']:.2f}"
    )


if __name__ == "__main__":
    main()

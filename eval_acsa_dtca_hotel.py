import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from train_acsa_dtca_hotel import ACSADataset, DTCAACSAClassifier, collate, evaluate
from train_quad_hotel import load_raw, model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved DTCA ACC/ACSA hotel checkpoint.")
    parser.add_argument("--data_dir", default="datasets/hotel")
    parser.add_argument("--image_dir", default="datasets/hotel_images")
    parser.add_argument("--text_model_name", default="bert", choices=["bert", "roberta"])
    parser.add_argument("--image_model_path", default="./models/vit-base-patch16-224-in21k")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    text_path = model_path(args.text_model_name)
    with open(os.path.join(args.model_dir, "acsa_label_maps.json"), "r", encoding="utf-8") as f:
        maps = json.load(f)
    category2id = maps["category2id"]
    sentiment2id = maps["sentiment2id"]
    id2category = {v: k for k, v in category2id.items()}
    id2sentiment = {v: k for k, v in sentiment2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=(args.text_model_name == "roberta"))
    image_processor = AutoImageProcessor.from_pretrained(args.image_model_path)
    records = load_raw(args.data_dir, "test")
    dataset = ACSADataset(records, tokenizer, image_processor, category2id, sentiment2id, args.max_length, args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTCAACSAClassifier(text_path, args.image_model_path, len(category2id), len(sentiment2id)).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location=device))
    metrics = evaluate(model, loader, device, id2category, id2sentiment, output_dir=args.output_dir)
    print(
        f"test ACC={metrics['acc_category_accuracy']:.2f} "
        f"sentiment_acc={metrics['sentiment_accuracy']:.2f} "
        f"ACSA={metrics['acsa_category_sentiment_accuracy']:.2f}"
    )


if __name__ == "__main__":
    main()

import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from train_macsa_dtca_hotel import DTCAMACSAClassifier, MACSADataset, collate, evaluate
from train_quad_hotel import load_raw, model_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DTCA MACSA checkpoint.")
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
    with open(os.path.join(args.model_dir, "macsa_label_maps.json"), "r", encoding="utf-8") as f:
        maps = json.load(f)
    pair2id = {tuple(k.split("|||")): v for k, v in maps["pair2id"].items()}
    id2pair = {v: k for k, v in pair2id.items()}
    threshold = float(maps.get("threshold", 0.5))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, add_prefix_space=(args.text_model_name == "roberta"))
    image_processor = AutoImageProcessor.from_pretrained(args.image_model_path)
    records = load_raw(args.data_dir, "test")
    dataset = MACSADataset(records, tokenizer, image_processor, pair2id, args.max_length, args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTCAMACSAClassifier(text_path, args.image_model_path, len(pair2id)).to(device)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location=device))
    metrics = evaluate(model, loader, device, id2pair, threshold, output_dir=args.output_dir)
    print(
        "test MACSA TP/FP/FN/P/R/F1: "
        f"{metrics['tp']} / {metrics['fp']} / {metrics['fn']} / "
        f"{metrics['precision']:.2f} / {metrics['recall']:.2f} / {metrics['f1']:.2f}"
    )


if __name__ == "__main__":
    main()

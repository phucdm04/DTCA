import argparse
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, BartTokenizerFast

from train_asqp_dtca_hotel import ASQPDataset, DTCABartASQP, collate, evaluate
from train_quad_hotel import load_raw


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DTCA-style ASQP checkpoint.")
    parser.add_argument("--data_dir", default="datasets/hotel")
    parser.add_argument("--image_dir", default="datasets/hotel_images")
    parser.add_argument("--bart_model_path", default="./models/bart-base")
    parser.add_argument("--image_model_path", default="./models/vit-base-patch16-224-in21k")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = BartTokenizerFast.from_pretrained(args.model_dir)
    image_processor = AutoImageProcessor.from_pretrained(args.image_model_path)
    records = load_raw(args.data_dir, "test")
    dataset = ASQPDataset(records, tokenizer, image_processor, args.max_source_length, args.max_target_length, args.image_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTCABartASQP(args.bart_model_path, args.image_model_path).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "pytorch_model.bin"), map_location=device))
    metrics = evaluate(model, loader, tokenizer, device, args.max_target_length, args.num_beams, output_dir=args.output_dir)
    print(
        "test ASQP TP/FP/FN/P/R/F1: "
        f"{metrics['tp']} / {metrics['fp']} / {metrics['fn']} / "
        f"{metrics['precision']:.2f} / {metrics['recall']:.2f} / {metrics['f1']:.2f}"
    )


if __name__ == "__main__":
    main()

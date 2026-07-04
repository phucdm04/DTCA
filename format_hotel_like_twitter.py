import argparse
import json
import os
from urllib.parse import urlparse


POLARITY_TO_TWITTER = {
    "negative": -1,
    "neutral": 0,
    "positive": 1,
    "neg": -1,
    "neu": 0,
    "pos": 1,
}

POLARITY_TO_JSON = {
    -1: "NEG",
    0: "NEU",
    1: "POS",
}

JSON_TO_TWITTER = {
    "NEG": -1,
    "NEU": 0,
    "POS": 1,
}

SPLIT_MAP = [
    ("train", "train"),
    ("val", "dev"),
    ("dev", "dev"),
    ("test", "test"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert hotel JSON splits to the Twitter MABSA txt/json format used by this repo."
    )
    parser.add_argument("--input_dir", default="datasets/hotel")
    parser.add_argument("--image_dir", default="datasets/hotel_images")
    parser.add_argument("--output_dir", default="datasets/hotel_twitter")
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Write converted files back into --input_dir. Existing JSON files are backed up first.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in the output directory.",
    )
    parser.add_argument(
        "--allow_missing_images",
        action="store_true",
        help="Keep rows even when the referenced image file is not present in --image_dir.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def image_name(review_photo, fallback, image_dir):
    candidates = []
    if review_photo:
        path = urlparse(review_photo).path
        name = os.path.basename(path)
        if name:
            candidates.append(name)
    candidates.append(fallback)

    for candidate in candidates:
        if os.path.exists(os.path.join(image_dir, candidate)):
            return candidate, True
    return candidates[0], False


def normalize_polarity(value):
    if value is None:
        return 0
    key = str(value).strip().lower()
    if key not in POLARITY_TO_TWITTER:
        raise ValueError(f"Unsupported polarity: {value}")
    return POLARITY_TO_TWITTER[key]


def span_is_valid(span, words):
    return (
        isinstance(span, list)
        and len(span) == 2
        and isinstance(span[0], int)
        and isinstance(span[1], int)
        and 0 <= span[0] < span[1] <= len(words)
    )


def convert_record(record, index, image_dir):
    review = str(record.get("review", "")).strip()
    words = review.split()
    fallback_image = os.path.splitext(str(record.get("json_file", f"{index}.json")))[0] + ".jpg"
    img, image_exists = image_name(record.get("review_photo"), fallback_image, image_dir)

    unique = {}
    skipped = 0
    for extraction in record.get("extraction", []):
        span = extraction.get("Aspect_span")
        if not span_is_valid(span, words):
            skipped += 1
            continue
        polarity = normalize_polarity(extraction.get("Polarity"))
        key = (span[0], span[1], polarity)
        if key not in unique:
            term = extraction.get("Aspect")
            aspect_words = words[span[0] : span[1]]
            if term:
                aspect_words = str(term).split()
            unique[key] = {
                "from": span[0],
                "to": span[1],
                "polarity": POLARITY_TO_JSON[polarity],
                "term": aspect_words,
            }

    aspects = sorted(unique.values(), key=lambda x: (x["from"], x["to"], x["polarity"]))
    json_row = {
        "words": words,
        "image_id": img,
        "aspects": aspects,
        "opinions": [{"term": []}],
    }

    txt_rows = []
    for aspect in aspects:
        start = aspect["from"]
        end = aspect["to"]
        masked_words = words[:start] + ["$T$"] + words[end:]
        txt_rows.append(
            (
                " ".join(masked_words),
                " ".join(aspect["term"]),
                str(JSON_TO_TWITTER[aspect["polarity"]]),
                img,
            )
        )
    return json_row, txt_rows, skipped, image_exists


def backup_if_needed(path):
    if not os.path.exists(path):
        return
    backup_path = path + ".raw.bak"
    if os.path.exists(backup_path):
        return
    os.replace(path, backup_path)


def write_txt(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for text, aspect, polarity, img in rows:
            f.write(text + "\n")
            f.write(aspect + "\n")
            f.write(polarity + "\n")
            f.write(img + "\n")


def main():
    args = parse_args()
    output_dir = args.input_dir if args.in_place else args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    total_records = 0
    total_aspects = 0
    total_skipped = 0
    total_missing_images = 0

    saw_val_split = False
    for source_name, target_name in SPLIT_MAP:
        if source_name == "dev" and saw_val_split:
            continue
        input_path = os.path.join(args.input_dir, source_name + ".json")
        if not os.path.exists(input_path):
            continue
        if args.in_place and os.path.exists(input_path + ".raw.bak"):
            input_path = input_path + ".raw.bak"
        if source_name == "val":
            saw_val_split = True

        output_json_path = os.path.join(output_dir, target_name + ".json")
        output_txt_path = os.path.join(output_dir, target_name + ".txt")
        if not args.overwrite and not args.in_place:
            for path in (output_json_path, output_txt_path):
                if os.path.exists(path):
                    raise FileExistsError(f"{path} exists. Use --overwrite to replace it.")

        records = load_json(input_path)
        converted_json = []
        txt_rows = []
        split_skipped = 0
        split_missing_images = 0
        for i, record in enumerate(records):
            json_row, rows, skipped, image_exists = convert_record(record, i, args.image_dir)
            if not image_exists:
                split_missing_images += 1
                if not args.allow_missing_images:
                    continue
            if rows:
                converted_json.append(json_row)
                txt_rows.extend(rows)
            split_skipped += skipped

        if args.in_place:
            backup_if_needed(output_json_path)

        write_json(output_json_path, converted_json)
        write_txt(output_txt_path, txt_rows)

        total_records += len(converted_json)
        total_aspects += len(txt_rows)
        total_skipped += split_skipped
        total_missing_images += split_missing_images
        print(
            f"{source_name}.json -> {target_name}: "
            f"{len(converted_json)} records, {len(txt_rows)} aspect rows, "
            f"{split_skipped} skipped extractions, {split_missing_images} missing images"
        )

    print(
        f"Done. Wrote Twitter-style files to {output_dir}. "
        f"Total: {total_records} records, {total_aspects} aspect rows, "
        f"{total_skipped} skipped extractions, {total_missing_images} missing images."
    )


if __name__ == "__main__":
    main()

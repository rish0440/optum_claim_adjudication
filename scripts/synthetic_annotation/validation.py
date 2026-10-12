"""Validate synthetic image/CSV pairs and the existing Qwen SFT input contract."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image

from src.config.settings import get_settings
from src.components.inference.parsing import CSV_FIELDS
from src.schema import InvoiceLine
from trainer.sft_qlora_invoice_extraction.data_prep import SOURCE_INSTRUCTION as INSTRUCTION, make_hf_dataset, read_records


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_dataset(base: Path | None = None, *, expected_count: int | None = None,
                   expected_train: int | None = None, expected_validation: int | None = None,
                   check_hf: bool = False, manifests: dict[str, Path] | None = None) -> dict:
    """Validate actual files, manifests, arithmetic, split isolation and trainer inputs.

    Expected counts are optional for the reusable generator. The Phase 4 CLI
    enforces ten images with an 8/2 split independently of dataset metadata.
    """
    settings = get_settings()
    base = Path(base or settings.paths.synthetic_dataset).resolve()
    info = json.loads((base / "dataset_info.json").read_text(encoding="utf-8"))
    original_hashes = {_digest(p) for p in settings.paths.optum_invoices.glob("*.jpg")}
    original_labels = {p.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
                       for p in settings.paths.evaluation_ground_truth.glob("*.csv")}
    images, labels, ids, hashes, targets = set(), set(), set(), set(), set()
    counts, row_counts, layouts = Counter(), Counter(), Counter()
    missing_rates = Counter()
    dimensions = set()
    all_records = {}
    _require(info.get("csv_columns") == list(CSV_FIELDS), "Dataset CSV columns metadata differs")
    for split in ("train", "validation"):
        # Use exactly the same reader as the QLoRA trainer, with unchanged schema.
        manifest = (manifests or {}).get(split, base / split / "annotations.jsonl")
        records = read_records(base, split, manifest=manifest)
        all_records[split] = records
        raw_records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for record, raw in zip(records, raw_records):
            image_path, csv_path = record["image_path"], record["csv_path"]
            image_file, csv_file = image_path.resolve(), csv_path.resolve()
            _require(image_file.is_relative_to(base / split / "images"), "Image escapes its split (including symlinks)")
            _require(csv_file.is_relative_to(base / split / "labels"), "CSV escapes its split (including symlinks)")
            _require(image_path.suffix.lower() == ".jpg", "Synthetic images must be JPG files")
            _require(csv_path.suffix.lower() == ".csv", "Labels must be CSV files")
            invoice_id = image_path.stem
            _require(re.fullmatch(r"invoice_\d{5}", invoice_id), "Invalid synthetic invoice ID")
            _require(invoice_id not in ids, f"Duplicate invoice ID: {invoice_id}")
            _require(image_file not in images and csv_file not in labels, "Duplicate image or annotation path")
            image_hash = _digest(image_file)
            _require(image_hash not in hashes, "Duplicate image content across records/splits")
            _require(image_hash not in original_hashes, "An original Optum image leaked into synthetic data")
            with Image.open(image_file) as image:
                _require(image.format == "JPEG" and image.mode == "RGB", "Expected an RGB JPEG image")
                _require(image.width >= 800 and image.height >= 400, "Invoice resolution is too small")
                dimensions.add(image.size)
                image.verify()
            with Image.open(image_file) as image:
                image.load()  # Detect truncated data, not only a valid JPEG header.
            text = csv_file.read_text(encoding="utf-8")
            _require(text.replace("\r\n", "\n").strip() not in original_labels,
                     "An original Optum invoice label was copied into synthetic data")
            _require(text not in targets, "Duplicate invoice annotation content")
            reader = csv.reader(io.StringIO(text, newline=""), strict=True)
            _require(next(reader, None) == list(CSV_FIELDS), "CSV header must be exactly Item,Quantity,Rate,Amount")
            lines = []
            for row in reader:
                _require(len(row) == 4, "CSV row must contain exactly four fields")
                item, quantity, rate, amount = row
                _require(bool(item.strip()), "Blank item name")
                _require(re.fullmatch(r"[1-9]\d*", quantity), "Quantity must be a positive integer")
                money = [Decimal(value) for value in (rate, amount) if value != ""]
                _require(all(value.is_finite() and value >= 0 for value in money), "Invalid monetary value")
                lines.append(InvoiceLine(item=item, quantity=int(quantity),
                                         rate=Decimal(rate) if rate else None, amount=Decimal(amount)))
            _require(bool(lines), "Invoice has no table items")
            _require(type(raw["rows"]) is int and len(lines) == raw["rows"], "Row count mismatch")
            _require(type(raw["rate_column_visible"]) is bool, "Invalid rate visibility flag")
            _require(all((line.rate is not None) == raw["rate_column_visible"] for line in lines),
                     "Missing-rate visibility and labels disagree")
            _require(sum((line.amount for line in lines), Decimal(0)) == Decimal(raw["billed_total"]),
                     "Billed total mismatch")
            _require(raw["layout"] in {"bordered", "semi_bordered", "borderless"}, "Unknown invoice layout")
            ids.add(invoice_id)
            images.add(image_file)
            labels.add(csv_file)
            hashes.add(image_hash)
            targets.add(text)
            counts[split] += 1
            row_counts[split] += len(lines)
            layouts[raw["layout"]] += 1
            missing_rates[split] += int(not raw["rate_column_visible"])

    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    actual_images = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in image_suffixes]
    actual_labels = [p for p in base.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"]
    _require(len(actual_images) == len(images) and {p.resolve() for p in actual_images} == images,
             "Image inventory differs from manifests (extra/missing images)")
    _require(len(actual_labels) == len(labels) and {p.resolve() for p in actual_labels} == labels,
             "Annotation inventory differs from manifests (extra/missing CSVs)")
    total = sum(counts.values())
    for key, observed in (("total", total), ("train", counts["train"]), ("validation", counts["validation"])):
        _require(type(info.get(key)) is int and info[key] == observed, f"Dataset {key} count mismatch")
    for expected, observed, name in ((expected_count, total, "total"),
                                    (expected_train, counts["train"], "train"),
                                    (expected_validation, counts["validation"], "validation")):
        _require(expected is None or observed == expected, f"Expected {expected} {name} images, found {observed}")
    _require(info["layouts"] == dict(layouts), "Layout counts mismatch")
    _require(info["missing_rate_column_images"] == sum(missing_rates.values()), "Missing-rate count mismatch")

    hf_counts = {}
    if check_hf:
        for split, records in all_records.items():
            dataset = make_hf_dataset(records)
            _require(len(dataset) == len(records), "HF dataset dropped invoice records")
            _require(set(dataset.column_names) == {"images", "prompt", "completion"}, "Unexpected HF dataset columns")
            for row, record in zip(dataset, records):
                _require(len(row["images"]) == 1 and isinstance(row["images"][0], Image.Image),
                         "HF images must decode to a one-image list")
                row["images"][0].load()
                _require(len(row["prompt"]) == 1 and row["prompt"][0]["role"] == "user", "Invalid prompt role")
                content = row["prompt"][0]["content"]
                _require(len(content) == 2 and content[0]["type"] == "image"
                         and content[1]["type"] == "text" and content[1]["text"] == INSTRUCTION,
                         "Prompt must contain the image marker and original extraction instruction")
                _require(len(row["completion"]) == 1 and row["completion"][0]["role"] == "assistant",
                         "Invalid completion role")
                completion = row["completion"][0]["content"]
                _require(len(completion) == 1 and completion[0]["type"] == "text"
                         and completion[0]["text"] == record["target_csv"], "Completion differs from ground truth")
            hf_counts[split] = len(dataset)

    return {
        "status": "PASS", "total_images": total, "train": counts["train"], "validation": counts["validation"],
        "total_rows": sum(row_counts.values()), "rows_by_split": dict(row_counts),
        "layout_counts": dict(layouts), "missing_rate_column_images": sum(missing_rates.values()),
        "missing_rate_by_split": dict(missing_rates), "unique_invoice_ids": len(ids),
        "unique_image_hashes": len(hashes), "image_sizes": [list(size) for size in sorted(dimensions)],
        "csv_columns": list(CSV_FIELDS), "trainer_record_validation": "PASS",
        "hf_multimodal_validation": "PASS" if check_hf else "not_requested",
        "hf_examples": hf_counts, "optum_image_overlap": 0, "optum_label_copies": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=get_settings().paths.synthetic_dataset)
    parser.add_argument("--expected-count", type=int, default=10)
    parser.add_argument("--expected-train", type=int, default=8)
    parser.add_argument("--expected-validation", type=int, default=2)
    parser.add_argument("--check-hf", action="store_true", help="Decode all examples with the installed HF dataset builder; no model download")
    args = parser.parse_args()
    try:
        report = verify_dataset(args.dataset_root, expected_count=args.expected_count,
                                expected_train=args.expected_train, expected_validation=args.expected_validation,
                                check_hf=args.check_hf)
    except (ValueError, AssertionError, KeyError, TypeError, AttributeError, OSError, ArithmeticError, ImportError) as exc:
        print(json.dumps({"status": "FAIL", "error": str(exc)}, indent=2))
        raise SystemExit(1) from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

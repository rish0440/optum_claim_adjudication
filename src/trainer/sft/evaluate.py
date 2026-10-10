"""Evaluate actual Qwen image->CSV predictions with transparent denominators.

Example:
  python -m src.trainer.sft.evaluate --dataset-root data/synthetic --split validation \
      --output reports/training/baseline.json
  python -m src.trainer.sft.evaluate --dataset-root data/synthetic --split validation \
      --adapter models/qwen25-vl-invoice-lora --output reports/training/finetuned.json

This evaluates SYNTHETIC validation only. Do not use the five Optum test invoices
for model selection, hyperparameter tuning or training.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from .training_data import read_records
from src.inference.parsing import CSV_FIELDS, read_rows, clean_csv_text, get_predicted_rows
from .config import MODEL_ID, MAX_PIXELS, dataset_root as default_dataset_root


def evaluate_pairs(pairs: list[dict]) -> dict:
    """Pure-Python metrics so they can be unit tested without model/GPU."""
    field_correct = field_total = reference_rows = predicted_rows = row_matches = 0
    valid_csv_count = invoice_exact_count = 0
    sample_details = []
    for sample in pairs:
        expected = sample["expected_rows"]
        raw = sample["prediction"]
        error = None
        try:
            predicted = get_predicted_rows(raw)
            valid_csv_count += 1
        except ValueError as exc:
            predicted = []
            error = str(exc)
        total = len(expected)
        reference_rows += total
        predicted_rows += len(predicted)
        field_total += 4 * total
        for index in range(min(len(expected), len(predicted))):
            for key in CSV_FIELDS:
                field_correct += int(expected[index][key].strip() == predicted[index][key].strip())
        exp_counter = Counter(tuple(row[key] for key in CSV_FIELDS) for row in expected)
        pred_counter = Counter(tuple(row[key] for key in CSV_FIELDS) for row in predicted)
        row_matches += sum((exp_counter & pred_counter).values())
        exact = predicted == expected
        invoice_exact_count += int(exact)
        sample_details.append({"image": sample["image"], "reference_rows": total,
                               "predicted_rows": len(predicted), "exact_csv": exact,
                               "parse_error": error, "raw_prediction": raw})
    n = len(pairs)
    return {
        "samples_evaluated": n,
        "valid_csv": {"numerator": valid_csv_count, "denominator": n},
        "invoice_exact_match": {"numerator": invoice_exact_count, "denominator": n},
        "field_accuracy_by_row_position": {"numerator": field_correct, "denominator": field_total},
        "exact_row_precision": {"numerator": row_matches, "denominator": predicted_rows},
        "exact_row_recall": {"numerator": row_matches, "denominator": reference_rows},
        "samples": sample_details,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=default_dataset_root())
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = evaluate entire split")
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = read_records(args.dataset_root, args.split)
    if args.max_samples:
        data = data[:args.max_samples]
    from src.inference.local import extract_csv, load_model
    model, processor = load_model(args.model, args.adapter, args.max_pixels)
    predictions = []
    for i, record in enumerate(data, 1):
        raw = extract_csv(model, processor, record["image_path"])
        predictions.append({"image": record["image_path"].name,
                            "expected_rows": record["rows"], "prediction": raw})
        print(f"Prediction {i}/{len(data)}: {record['image_path'].name}", flush=True)
    report = evaluate_pairs(predictions)
    report["model"] = args.model
    report["adapter"] = args.adapter
    report["split"] = args.split
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k != "samples"}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

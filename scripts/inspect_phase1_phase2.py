"""Lightweight read-only inspection of policy, evaluation annotations and data."""

import csv
import json
from pathlib import Path

from src.utils.policy import load_policy, summarize_policy
from src.schema import InvoiceGroundTruth
from src.config.settings import get_settings
from scripts.synthetic_annotation.generator import verify_dataset


def main() -> None:
    settings = get_settings()
    print("POLICY", json.dumps(summarize_policy(load_policy()), indent=2))
    base = settings.paths.evaluation_ground_truth
    documents = [InvoiceGroundTruth.model_validate_json(p.read_text(encoding="utf-8"))
                 for p in sorted(base.glob("invoice_*.json"))]
    print(f"EVALUATION SET: {len(documents)} Optum invoices, {sum(len(d.lines) for d in documents)} lines")
    for doc in documents:
        csv_path=base/f"{doc.invoice_id}.csv"
        print(csv_path)
        with csv_path.open(encoding="utf-8",newline="") as f:
            parsed = list(csv.DictReader(f))
        if len(parsed)!=len(doc.lines):
            raise AssertionError(f"CSV mismatch for {doc.invoice_id}")
        print(f"  {doc.invoice_id}: {len(doc.lines)} rows; total INR {doc.billed_total}; rate-column={doc.rate_column_visible}")
    print("SYNTHETIC DATASET",json.dumps(verify_dataset(),indent=2))
    print("All inspection checks passed.")


if __name__ == "__main__":
    main()

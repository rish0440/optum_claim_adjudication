"""Write manually transcribed labels for the FIVE HELD-OUT Optum invoice images.

The values below were transcribed from the original provided JPGs. This script
DOES NOT use OCR, a VLM, or generated synthetic examples. Keep these five images
and their annotations out of every training and synthetic validation split.
"""

import csv
import json
from decimal import Decimal
from pathlib import Path

from src.policy import load_policy
from src.schemas import InvoiceGroundTruth
from src.settings import get_settings

# Item, quantity, visible rate (None means no Rate column), visible amount,
# policy rule ID, expected eligible amount.
GOLD = [
    {
        "id": "invoice_01_standard", "date": "2024-01-15", "total": 10500,
        "lines": [
            ("Room Charges", 2, 3500, 7000, "COV-001", 6000),
            ("Consultation", 1, 1500, 1500, "COV-003", 1500),
            ("Medicines", 5, 240, 1200, "COV-005", 1200),
            ("Blood Test", 1, 800, 800, "COV-010", 800),
        ],
    },
    {
        "id": "invoice_02_exclusions", "date": "2024-02-10", "total": 6750,
        "lines": [
            ("Room Charges", 1, 2800, 2800, "COV-001", 2800),
            ("Consultation", 1, 1200, 1200, "COV-003", 1200),
            ("Medicines", 3, 350, 1050, "COV-005", 1050),
            ("Personal Care Kit", 1, 800, 800, "EXC-001", 0),
            ("Health Supplement", 2, 450, 900, "EXC-004", 0),
        ],
    },
    {
        "id": "invoice_03_icu_mixed", "date": "2024-03-05", "total": 28100,
        "lines": [
            ("ICU Charges", 2, 7500, 15000, "COV-002", 12000),
            ("Room Charges", 1, 3200, 3200, "COV-001", 3000),
            ("Consultation", 2, 1500, 3000, "COV-003", 3000),
            ("Medicines", 8, 300, 2400, "COV-005", 2400),
            ("CT Scan", 1, 4500, 4500, "COV-008", 4500),
        ],
    },
    {
        "id": "invoice_04_diagnostics", "date": "2024-04-18", "total": 13060,
        "notes": "No unit-rate column is printed; every Rate field is intentionally null/blank.",
        "lines": [
            ("MRI Scan", 1, None, 8000, "COV-007", 8000),
            ("Blood Test (CBC)", 1, None, 600, "COV-010", 600),
            ("Blood Test (Lipid Profile)", 1, None, 1200, "COV-010", 1200),
            ("X-Ray", 1, None, 800, "COV-009", 800),
            ("Consultation", 1, None, 1500, "COV-003", 1500),
            ("Medicines", 4, None, 960, "COV-005", 960),
        ],
    },
    {
        "id": "invoice_05_nonmedical", "date": "2024-05-30", "total": 10250,
        "notes": "Surgical Gloves mapped provisionally to COV-019 (Medical Consumables). Confirm if alternative interpretation is requested.",
        "lines": [
            ("Room Charges", 2, 2500, 5000, "COV-001", 5000),
            ("Consultation", 1, 1200, 1200, "COV-003", 1200),
            ("Medicines", 6, 400, 2400, "COV-005", 2400),
            ("Diet Meal", 3, 300, 900, "EXC-006", 0),
            ("Television Charges", 2, 200, 400, "EXC-007", 0),
            ("Surgical Gloves", 1, 350, 350, "COV-019", 350),
        ],
    },
]


def write_gold() -> None:
    settings = get_settings()
    out = settings.paths.evaluation_ground_truth
    out.mkdir(parents=True, exist_ok=True)
    policy = load_policy()
    valid_rule_ids = {r.rule_id for r in policy.coverage_rules + policy.exclusion_rules}
    manifest = []
    adjudication = []

    for spec in GOLD:
        name = spec["id"]
        image_path = settings.paths.optum_invoices / f"{name}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        raw_lines = spec["lines"]
        gold = InvoiceGroundTruth.model_validate({
            "invoice_id": name,
            "image_file": str(Path("data/invoices") / f"{name}.jpg"),
            "invoice_date": spec["date"],
            "lines": [
                {"item": item, "quantity": qty, "rate": rate, "amount": amount}
                for item, qty, rate, amount, _rule, _allowed in raw_lines
            ],
            "billed_total": spec["total"],
            "rate_column_visible": all(row[2] is not None for row in raw_lines),
            "notes": spec.get("notes", ""),
        })
        (out / f"{name}.json").write_text(
            gold.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        with (out / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Item", "Quantity", "Rate", "Amount"])
            for line in gold.lines:
                writer.writerow([line.item, line.quantity, "" if line.rate is None else line.rate, line.amount])

        eligible = sum((Decimal(str(row[5])) for row in raw_lines), Decimal(0))
        billed = gold.billed_total
        co_pay = eligible * policy.reimbursement_rules.copay.rate_percent / Decimal(100)
        decisions = []
        for item, qty, rate, amount, rule_id, allowed in raw_lines:
            if rule_id not in valid_rule_ids:
                raise ValueError(f"Unknown policy rule ID {rule_id}")
            decisions.append({
                "item": item,
                "rule_id": rule_id,
                "billed": str(amount),
                "eligible": str(allowed),
                "status": "denied" if not allowed else "partial" if allowed < amount else "approved",
            })
        adjudication.append({
            "invoice_id": name,
            "billed_total": str(billed),
            "eligible_total": str(eligible),
            "copay_total": str(co_pay),
            "reimbursement_total": str(eligible - co_pay),
            "item_decisions": decisions,
            "annotation_note": spec.get("notes", ""),
        })
        manifest.append({
            "invoice_id": name,
            "image": f"data/invoices/{name}.jpg",
            "csv": f"data/evaluation/ground_truth/{name}.csv",
            "json": f"data/evaluation/ground_truth/{name}.json",
            "rows": len(raw_lines),
            "split": "final_evaluation_only",
        })

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "expected_adjudication.json").write_text(json.dumps(adjudication, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(GOLD)} evaluation invoices / {sum(len(x['lines']) for x in GOLD)} line items")
    for a in adjudication:
        print(f"  {a['invoice_id']}: billed={a['billed_total']}, eligible={a['eligible_total']}, reimbursed={a['reimbursement_total']}")
    print(f"Saved evaluation-only CSV/JSON labels to {out}")


if __name__ == "__main__":
    write_gold()

"""Phase 7 offline claim adjudication and reference verification.

Examples:
  python -m scripts.run_adjudication --invoice-json data/assignment_dataset/ground_truth/invoice_01_standard.json
  python -m scripts.run_adjudication --csv my_invoice.csv --date 2024-06-01
  python -m scripts.run_adjudication --all-reference

Add --with-milvus to retrieve policy candidates using the live Phase 6 index.
This command does not invoke a VLM, train on, or alter the evaluation images.
"""

import argparse
import json
from pathlib import Path

from src.utils.calculator import adjudicate_invoice
from src.utils.rules import parse_invoice_csv
from src.utils.policy import load_policy
from src.components.rag.resolver import PolicyRuleResolver
from src.schema import InvoiceGroundTruth
from src.config.settings import get_settings


def reference_evaluation(policy, resolver, retriever=None) -> dict:
    """Evaluate structured-data adjudication against the five supplied labels."""
    root = get_settings().paths.evaluation_ground_truth
    expected = json.loads((root / "expected_adjudication.json").read_text(encoding="utf-8"))
    if len(expected) != 5:
        raise ValueError("Expected five held-out reference records")
    count = {"invoice_total_exact": 0, "line_rule_id": 0,
             "line_status": 0, "line_eligible_exact": 0}
    summary = []
    total_lines = 0
    for reference in expected:
        file = root / f"{reference['invoice_id']}.json"
        invoice = InvoiceGroundTruth.model_validate_json(file.read_text(encoding="utf-8"))
        actual = adjudicate_invoice(invoice, policy=policy, resolver=resolver, retriever=retriever)
        truth = reference["item_decisions"]
        if len(actual.decisions) != len(truth):
            raise AssertionError(f"Row count differs: {invoice.invoice_id}")
        # Compare monetary amounts numerically, not their presentation precision.
        from decimal import Decimal
        all_amounts = (actual.final_reimbursement == Decimal(reference["reimbursement_total"])
                       and actual.eligible_total == Decimal(reference["eligible_total"])
                       and actual.copay_amount == Decimal(reference["copay_total"])
                       and actual.billed_total == Decimal(reference["billed_total"]))
        count["invoice_total_exact"] += int(all_amounts)
        details = []
        for decision, label in zip(actual.decisions, truth):
            if decision.item != label["item"]:
                raise AssertionError(f"Line ordering differs for {invoice.invoice_id}")
            matched = {
                "rule_id": decision.rule_id == label["rule_id"],
                "status": decision.status == label["status"],
                "eligible_amount": decision.approved_amount == Decimal(label["eligible"]),
            }
            count["line_rule_id"] += int(matched["rule_id"])
            count["line_status"] += int(matched["status"])
            count["line_eligible_exact"] += int(matched["eligible_amount"])
            total_lines += 1
            details.append({"item": label["item"], "checks": matched})
        summary.append({"invoice_id": invoice.invoice_id,
                        "totals_match": all_amounts,
                        "final_reimbursement": (str(actual.final_reimbursement)
                                                if actual.final_reimbursement is not None else None),
                        "lines": details})
    return {
        "evaluation_type": "oracle_structured_csv_only_no_vlm_no_image_inference",
        "invoice_count": len(expected), "line_count": total_lines,
        "metrics": {
            metric: {"correct": success, "total": len(expected) if metric == "invoice_total_exact" else total_lines}
            for metric, success in count.items()
        },
        "results": summary,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    options.add_argument("--invoice-json", type=Path, help="Verified invoice JSON or schema-compatible input")
    options.add_argument("--csv", type=Path, help="Extracted invoice CSV")
    options.add_argument("--all-reference", action="store_true", help="Compare five structured reference invoices")
    parser.add_argument("--date", help="Date (YYYY-MM-DD), required for --csv")
    parser.add_argument("--invoice-id", help="Optional ID for --csv")
    parser.add_argument("--with-milvus", action="store_true", help="Query the real Milvus index before rule resolution")
    parser.add_argument("--output", type=Path, help="Save JSON report")
    args = parser.parse_args()
    policy = load_policy()
    resolver = PolicyRuleResolver(policy)
    retriever = None
    if args.with_milvus:
        from src.components.rag.retriever import PolicyRetriever
        retriever = PolicyRetriever()
    if args.all_reference:
        payload = reference_evaluation(policy, resolver, retriever)
        for name, metric in payload["metrics"].items():
            print(f"{name}: {metric['correct']}/{metric['total']}")
    else:
        if args.csv:
            if not args.date:
                parser.error("--date YYYY-MM-DD is required with --csv")
            invoice = parse_invoice_csv(args.csv, invoice_date=args.date, invoice_id=args.invoice_id)
        else:
            invoice = InvoiceGroundTruth.model_validate_json(args.invoice_json.read_text(encoding="utf-8"))
        result = adjudicate_invoice(invoice, policy=policy, resolver=resolver, retriever=retriever)
        payload = result.model_dump(mode="json")
        print(json.dumps({
            "invoice_id": payload["invoice_id"],
            "status": payload["claim_status"],
            "billed": payload["billed_total"],
            "eligible": payload["eligible_total"],
            "co_pay": payload["copay_amount"],
            "final_reimbursement": payload["final_reimbursement"],
            "rule_ids": [entry["rule_id"] for entry in payload["decisions"]],
        }, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

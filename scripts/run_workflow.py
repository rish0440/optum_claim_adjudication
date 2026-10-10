"""Phase 8 CLI: run the actual LangGraph against saved invoice inputs.

Examples:
  python -m scripts.run_workflow --invoice-json data/evaluation/ground_truth/invoice_01_standard.json
  python -m scripts.run_workflow --csv data/evaluation/ground_truth/invoice_04_diagnostics.csv --date 2024-04-18
  python -m scripts.run_workflow --invoice-json data/evaluation/ground_truth/invoice_01_standard.json --with-milvus

Use --image and --remote-vlm only when a real authenticated GPU VLM is online.
"""
import argparse
import json
from pathlib import Path

from src.orchestration.workflow import run_claim_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--invoice-json", type=Path)
    input_group.add_argument("--csv", type=Path)
    input_group.add_argument("--image", type=Path)
    parser.add_argument("--date", help="YYYY-MM-DD for CSV or image input")
    parser.add_argument("--invoice-id", help="Optional invoice identifier")
    parser.add_argument("--with-milvus", action="store_true", help="Search your local Phase 6 Milvus index")
    parser.add_argument("--strict-rag", action="store_true", help="Fail instead of policy-only fallback on Milvus errors")
    parser.add_argument("--remote-vlm", action="store_true", help="Use configured GPU inference API for image input")
    parser.add_argument("--output", type=Path, help="Save the JSON result")
    args = parser.parse_args()
    if args.strict_rag and not args.with_milvus:
        parser.error("--strict-rag requires --with-milvus")
    if args.remote_vlm and not args.image:
        parser.error("--remote-vlm requires --image")
    if (args.csv or args.image) and not args.date:
        parser.error("--date YYYY-MM-DD is required for --csv and --image")
    result = run_claim_workflow(
        invoice_json=args.invoice_json, csv_file=args.csv, image=args.image,
        invoice_date=args.date, invoice_id=args.invoice_id,
        with_milvus=args.with_milvus, strict_rag=args.strict_rag,
        remote_vlm=args.remote_vlm,
    )
    claim = result.get("claim") or {}
    print(json.dumps({
        "status": result["status"], "invoice_id": claim.get("invoice_id"),
        "extraction_source": result["extraction_source"],
        "rag_status": result["rag_status"],
        "billed_total": claim.get("billed_total"),
        "eligible_total": claim.get("eligible_total"),
        "final_reimbursement": claim.get("final_reimbursement"),
        "errors": result.get("errors", []),
        "warnings": result.get("warnings", []),
        "trace": result["trace"],
    }, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Saved: {args.output}")
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

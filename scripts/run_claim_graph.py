"""Manual Phase 7 demonstration against live Milvus Lite and the existing vLLM."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from src.orchestration.workflow import run_claim_graph
from src.utils.paths import project_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--csv", type=Path)
    parser.add_argument("--date", required=True, help="Actual invoice date YYYY-MM-DD; policy is dated")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/claim_graph"))
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-retries", type=int, choices=(0, 1), default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        report = run_claim_graph(image=project_path(args.image) if args.image else None,
            csv_file=project_path(args.csv) if args.csv else None, invoice_date=args.date,
            endpoint=args.endpoint, timeout=args.timeout, request_timeout=args.request_timeout,
            max_tokens=args.max_tokens, max_retries=args.max_retries, verbose=args.verbose)
    except ValueError as exc:
        parser.error(str(exc))
    directory = project_path(args.output_dir) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + report["request_id"])
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in (("final_report.json", report), ("policy_evidence.json", report["evidence"]),
                        ("agent_1.json", report["agent_1"]), ("deterministic_claim.json", report["claim"]),
                        ("agent_2.json", report["agent_2"]), ("trace.json", report["trace"])):
        (directory / name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    if report["extracted_csv"]:
        (directory / "validated_invoice.csv").write_text(report["extracted_csv"], encoding="utf-8")
    if report["metadata"].get("agent_1_rejection"):
        (directory / "agent_1_rejection.json").write_text(
            json.dumps(report["metadata"]["agent_1_rejection"], indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"request_id": report["request_id"], "status": report["status"],
        "reviewed": report["reviewed"], "artifacts": str(directory)}, indent=2), flush=True)
    raise SystemExit(0 if report["status"] == "COMPLETED" else 2 if report["status"] == "REVIEW_REQUIRED" else 1)


if __name__ == "__main__":
    main()

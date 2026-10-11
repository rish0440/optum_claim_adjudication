"""Readable reports from actual paired predictions; no fabricated outcomes."""
import csv
import json
from pathlib import Path

from src.trainer.sft.train.train_qlora import write_json
from .evaluation_metrics import summarize
from .schema import sha

LIMITATIONS = (
    "19 held-out Indonesian retail/restaurant receipts form a small test set, not a healthcare-invoice benchmark. "
    "Smoke reports contain one image and are not the full evaluation. "
    "Test results must not be used to tune prompts, select checkpoints or train. "
    "Names are normalized for whitespace only; numeric equality uses Decimal without currency conversion. "
    "Row alignment greedily prioritizes exact field agreement, then exact names; numeric-only alignment requires "
    "all three numeric fields to agree. Ties use reference/prediction position. This is deterministic, not semantic matching. "
    "Field/complete-row recall denominators include all reference rows; row precision separately penalizes extras. "
    "Truncated/error outputs are unsuccessful and receive no extraction credit. "
    "The baseline run did not necessarily record its original training base revision; both evaluation modes "
    "use the same pinned CORD base revision. Training losses do not establish extraction improvement."
)


def write_reports(directory, comparisons, metadata):
    directory = Path(directory)
    count = metadata["evaluated_documents"]
    if len(comparisons) != count:
        raise ValueError("Incomplete comparison would change the document denominator")
    verified = [row for row in comparisons if row["ground_truth_csv"] is not None]
    if metadata.get("dataset_name") != "optum" and len(verified) != count:
        raise ValueError("CORD evaluation requires verified labels for every test record")
    summaries = {mode: summarize([row[mode]["metrics"] for row in verified], len(verified))
                 for mode in ("baseline", "cord_sft")}
    for mode, summary in summaries.items():
        from .evaluation_metrics import ratio
        for key in ("csv_valid", "eligible_completion"):
            summary[key] = ratio(sum(row[mode]["metrics"][key] for row in comparisons), count)
        for key in ("parse_failures", "inference_errors", "truncations", "invalid_numeric_fields", "arithmetic_errors"):
            field = {"parse_failures": "parse_error", "inference_errors": "inference_error",
                     "truncations": "truncated", "invalid_numeric_fields": "invalid_numeric_fields",
                     "arithmetic_errors": "arithmetic_error_indices"}[key]
            summary[key] = sum(len(row[mode]["metrics"][field]) if isinstance(row[mode]["metrics"][field], list)
                               else bool(row[mode]["metrics"][field]) for row in comparisons)
        if not verified:
            for key in ("document_exact", "order_insensitive_exact", "row_precision", "row_recall",
                        "complete_row_exact", "complete_row_precision"):
                summary[key] = {"numerator": None, "denominator": 0, "value": None}
            for field in summary["fields"]:
                summary["fields"][field] = {"numerator": None, "denominator": 0, "value": None}
            for key in ("row_f1", "missing_rows", "extra_rows", "incorrect_rate", "incorrect_amount"):
                summary[key] = None
        summary["total_inference_latency_seconds"] = sum(row[mode]["latency_seconds"] for row in comparisons)
        summary["average_inference_latency_seconds"] = summary["total_inference_latency_seconds"] / count if count else 0
        summary["generated_tokens"] = sum(row[mode]["generated_tokens"] for row in comparisons)
    optum = metadata.get("dataset_name") == "optum"
    limitations = ("Five original Optum invoices; accuracy uses existing documented manual transcription, not model predictions. "
                  "Missing labels are NOT ASSESSED and excluded only from accuracy denominators. "
                  "Intentionally absent rates remain empty; arithmetic is not asserted where rates are absent. "
                  "Both adapters use the same general prompt; production-prompt compatibility must be tested separately. "
                  "This small final healthcare demo set does not establish generalization. "
                  + LIMITATIONS[LIMITATIONS.index("Names are normalized"):]) if optum else LIMITATIONS
    summary = {"status": metadata["status"], "sample_count": count, "models": summaries,
               "verified_ground_truth_documents": len(verified), "accuracy_status": "ASSESSED" if verified else "NOT ASSESSED",
               "limitations": limitations}
    write_json(directory / "per_image_comparison.json", comparisons)
    with (directory / "per_image_comparison.csv").open("x", encoding="utf-8", newline="") as stream:
        fields = ["image", "ground_truth_csv"] + [f"{mode}_{field}" for mode in ("baseline", "cord_sft")
                 for field in ("raw_response", "parsed_csv", "document_exact", "missing_rows", "extra_rows",
                               "mismatches", "parse_error", "error", "truncated", "latency_seconds")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in comparisons:
            output = {"image": row["image"], "ground_truth_csv": row["ground_truth_csv"]}
            for mode in ("baseline", "cord_sft"):
                prediction, metrics = row[mode], row[mode]["metrics"]
                for field in ("raw_response", "error", "truncated", "latency_seconds"):
                    output[f"{mode}_{field}"] = prediction[field]
                for field in ("parsed_csv", "document_exact", "parse_error"):
                    output[f"{mode}_{field}"] = "NOT ASSESSED" if field == "document_exact" and metrics[field] is None else metrics[field]
                output[f"{mode}_missing_rows"] = json.dumps(metrics["missing_reference_indices"])
                output[f"{mode}_extra_rows"] = json.dumps(metrics["extra_prediction_indices"])
                output[f"{mode}_mismatches"] = json.dumps(metrics["matched_rows"])
            writer.writerow(output)
    write_json(directory / "summary.json", summary)
    text = ["# Optum final extraction comparison" if optum else "# CORD extraction comparison", "",
            f"Status: {metadata['status']}; samples: {count}; verified labels: {len(verified)}",
            f"Base: {metadata['model_id']} @ {metadata['settings']['revision']}",
            f"Prompt SHA256: {metadata['settings']['prompt_sha256']}",
            f"Test fingerprint: {metadata['dataset']['test_split_sha256']}", ""]
    for mode, adapter in metadata["adapters"].items():
        text.append(f"{mode}: `{adapter['path']}`; weights SHA256 `{adapter['adapter_sha256']}`")
    text += ["", "| Metric | Baseline | CORD SFT | Measured result |", "|---|---:|---:|---|"]
    def comparison(name, a, b, higher=True):
        if a is None or b is None:
            text.append(f"| {name} | NOT ASSESSED | NOT ASSESSED | Insufficient verified labels |")
            return
        winner = "Tie" if a == b else ("CORD SFT better" if (b > a) == higher else "Baseline better")
        text.append(f"| {name} | {a:.4f} | {b:.4f} | {winner} |")
    a, b = summaries["baseline"], summaries["cord_sft"]
    for metric in ("csv_valid", "eligible_completion", "document_exact", "order_insensitive_exact",
                   "row_precision", "row_recall", "complete_row_exact", "complete_row_precision"):
        comparison(metric, a[metric]["value"], b[metric]["value"])
    comparison("row_f1", a["row_f1"], b["row_f1"])
    for field in a["fields"]:
        comparison(f"{field} accuracy", a["fields"][field]["value"], b["fields"][field]["value"])
    for metric in ("missing_rows", "extra_rows", "parse_failures", "invalid_numeric_fields", "arithmetic_errors",
                   "incorrect_rate", "incorrect_amount", "truncations", "inference_errors", "average_inference_latency_seconds"):
        comparison(metric, a[metric], b[metric], higher=False)
    text += ["", "## Representative outcomes", ""]
    for category, row in representatives(comparisons):
        text.append(f"- {category}: `{row['image']}`; baseline error={row['baseline']['error']}; "
                    f"CORD error={row['cord_sft']['error']}")
    text += ["", "## Limits", "", limitations,
             "", "No deployment recommendation is inferred from training loss or this small test set alone."]
    (directory / "summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def representatives(comparisons):
    categories = {}
    for row in comparisons:
        a, b = (row[mode]["metrics"]["document_exact"] for mode in ("baseline", "cord_sft"))
        category = "NOT ASSESSED" if a is None or b is None else "Both succeed" if a and b else "Baseline wins" if a else "CORD SFT wins" if b else "Both fail"
        categories.setdefault(category, row)
    return list(categories.items())[:4]


def inspect_report(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "evaluation_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") not in {"complete", "complete_smoke"}:
        raise ValueError("Evaluation did not complete; inspect its manifest/partial predictions instead")
    expected_files = {"baseline_predictions.jsonl", "cord_sft_predictions.jsonl", "per_image_comparison.csv",
                      "per_image_comparison.json", "summary.json", "summary.md"}
    if manifest.get("dataset_name") == "optum":
        expected_files.add("comparison.csv")
    if set(manifest.get("report_sha256", {})) != expected_files:
        raise ValueError("Incomplete evaluation report fingerprints")
    for filename, checksum in manifest["report_sha256"].items():
        if sha(directory / filename) != checksum:
            raise ValueError("Saved evaluation report checksum mismatch")
    print((directory / "summary.md").read_text(encoding="utf-8"), flush=True)
    comparisons = json.loads((directory / "per_image_comparison.json").read_text(encoding="utf-8"))
    for category, row in representatives(comparisons):
        print(f"\n{category}: {row['image_path']}\nGROUND TRUTH:\n{row['ground_truth_csv'] if row['ground_truth_csv'] is not None else 'NOT ASSESSED'}", flush=True)
        for mode in ("baseline", "cord_sft"):
            print(f"{mode}:\n{row[mode]['raw_response']}", flush=True)

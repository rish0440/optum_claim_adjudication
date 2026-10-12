"""Invoice extraction evaluation, metrics and reports."""


# Evaluation metrics
import csv
from decimal import Decimal, InvalidOperation
import io
import re

from src.components.inference.parsing import CSV_FIELDS, read_rows


def normalize(row, *, allow_empty_rates=False):
    result = [" ".join(row["Item"].split())]
    for field in CSV_FIELDS[1:]:
        value = row[field].strip()
        if field == "Rate" and not value and allow_empty_rates:
            result.append("")  # A visibly absent rate is a value, never a derived price.
            continue
        try:
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
                raise ValueError("not a plain nonnegative decimal")
            number = Decimal(value)
            if field == "Quantity" and (number <= 0 or number != number.to_integral_value()):
                raise ValueError("quantity must be a positive integer")
        except (ValueError, InvalidOperation):
            number = None
        result.append(number)
    return tuple(result)


def parse_prediction(raw):
    if not isinstance(raw, str) or not raw.strip() or "```" in raw:
        raise ValueError("Empty output or forbidden Markdown")
    table = list(csv.reader(io.StringIO(raw.strip()), strict=True))
    if not table or tuple(table[0]) != CSV_FIELDS:
        raise ValueError("Exact Item,Quantity,Rate,Amount header required")
    if len(table) < 2 or any(len(row) != 4 for row in table[1:]):
        raise ValueError("CSV must contain nonempty four-column item rows only")
    # Reuse the runtime structural parser without its Markdown-cleanup helper.
    return read_rows(raw)


def align_rows(expected, predicted):
    """Greedy best agreement; tie break by reference then prediction position.

    Eligible pairs share the whitespace-normalized item name OR all three valid
    numeric fields. Exact four-field rows win first. Each row is used at most once.
    No fuzzy/semantic name rewriting, rate derivation or positional assumption.
    """
    candidates = []
    for i, truth in enumerate(expected):
        for j, actual in enumerate(predicted):
            matches = [a is not None and a == b for a, b in zip(truth, actual)]
            if matches[0] or all(matches[1:]):
                candidates.append((-sum(matches), -int(matches[0]), i, j))
    used_expected, used_predicted, matches = set(), set(), []
    for _, _, i, j in sorted(candidates):
        if i not in used_expected and j not in used_predicted:
            matches.append((i, j))
            used_expected.add(i)
            used_predicted.add(j)
    return sorted(matches), sorted(set(range(len(expected))) - used_expected), sorted(
        set(range(len(predicted))) - used_predicted)


def score(expected_csv, prediction, *, allow_empty_rates=False):
    assessed = expected_csv is not None
    truth_rows = parse_prediction(expected_csv) if assessed else []
    expected = [normalize(row, allow_empty_rates=allow_empty_rates) for row in truth_rows]
    if any(value is None for row in expected for value in row):
        raise ValueError("Invalid ground-truth numeric fields")
    raw, error = prediction.get("raw_response", ""), prediction.get("error")
    parsed, parse_error = [], None
    try:
        parsed = parse_prediction(raw)
    except (ValueError, csv.Error) as exc:
        parse_error = str(exc)
    normalized = [normalize(row, allow_empty_rates=allow_empty_rates) for row in parsed]
    invalid = [{"prediction_index": i, "field": field} for i, row in enumerate(normalized)
               for field, value in zip(CSV_FIELDS[1:], row[1:]) if value is None]
    format_valid = parse_error is None and not invalid
    eligible = format_valid and not error and not prediction.get("truncated", False)
    # Incomplete generations/errors score as failures, even if their prefix is CSV.
    scoring_rows = normalized if not error and not prediction.get("truncated", False) else []
    matched, missing, extra = align_rows(expected, scoring_rows)
    field_correct = {field: 0 for field in CSV_FIELDS}
    details, complete = [], 0
    for i, j in matched:
        mismatches = [field for field, a, b in zip(CSV_FIELDS, expected[i], scoring_rows[j])
                      if b is None or a != b]
        for field in CSV_FIELDS:
            field_correct[field] += int(field not in mismatches)
        complete += int(not mismatches)
        details.append({"reference_index": i, "prediction_index": j, "mismatched_fields": mismatches})
    arithmetic = []
    for i, row in enumerate(normalized):
        q, rate, amount = row[1:]
        if None not in (q, rate, amount) and rate != "" and q * rate != amount:
            arithmetic.append(i)
    return {"csv_valid": format_valid, "eligible_completion": eligible, "accuracy_assessed": assessed,
            "document_exact": bool(eligible and normalized == expected) if assessed else None,
            "order_insensitive_exact": bool(eligible and complete == len(expected) == len(normalized)) if assessed else None,
            "reference_rows": len(expected) if assessed else None, "predicted_rows": len(normalized),
            "detected_rows": len(matched) if assessed else None, "complete_rows": complete if assessed else None,
            "field_correct": field_correct if assessed else None, "matched_rows": details if assessed else None,
            "missing_reference_indices": missing if assessed else None, "extra_prediction_indices": extra if assessed else None,
            "invalid_numeric_fields": invalid, "arithmetic_error_indices": arithmetic,
            "parse_error": parse_error, "inference_error": error,
            "truncated": bool(prediction.get("truncated", False)),
            "parsed_rows": parsed, "parsed_csv": raw if parse_error is None else None}


def ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": numerator / denominator if denominator else 0.0}


def summarize(details, expected_documents):
    if len(details) != expected_documents:
        raise ValueError("Missing evaluations would change the fixed document denominator")
    references = sum(row["reference_rows"] for row in details)
    predictions = sum(row["predicted_rows"] for row in details)
    detected = sum(row["detected_rows"] for row in details)
    complete = sum(row["complete_rows"] for row in details)
    precision = detected / predictions if predictions else 0.0
    recall = detected / references if references else 0.0
    return {
        "documents": expected_documents,
        **{name: ratio(sum(bool(row[name]) for row in details), expected_documents)
           for name in ("csv_valid", "eligible_completion", "document_exact", "order_insensitive_exact")},
        "fields": {field: ratio(sum(row["field_correct"][field] for row in details), references)
                   for field in CSV_FIELDS},
        "row_precision": ratio(detected, predictions), "row_recall": ratio(detected, references),
        "row_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "complete_row_exact": ratio(complete, references),
        "complete_row_precision": ratio(complete, predictions),
        "missing_rows": sum(len(row["missing_reference_indices"]) for row in details),
        "extra_rows": sum(len(row["extra_prediction_indices"]) for row in details),
        "parse_failures": sum(row["parse_error"] is not None for row in details),
        "invalid_numeric_fields": sum(len(row["invalid_numeric_fields"]) for row in details),
        "arithmetic_errors": sum(len(row["arithmetic_error_indices"]) for row in details),
        "incorrect_rate": references - sum(row["field_correct"]["Rate"] for row in details),
        "incorrect_amount": references - sum(row["field_correct"]["Amount"] for row in details),
        "truncations": sum(row["truncated"] for row in details),
        "inference_errors": sum(bool(row["inference_error"]) for row in details),
    }


# Evaluation data
import json
from pathlib import Path
import re

from trainer.sft_qlora_invoice_extraction.train import latest_adapter, verify_adapter
from src.utils.paths import project_path
from trainer.sft_qlora_invoice_extraction.data_prep import inspect_image
from trainer.sft_qlora_invoice_extraction.data_prep import FILES, REPO_ID, SOURCE_COUNTS, convert, fingerprint, messages, read_jsonl, sha
from trainer.sft_qlora_invoice_extraction.train import MODEL_ID, OUTPUT, verify_saved
from trainer.sft_qlora_invoice_extraction.data_prep import EXPECTED_COUNTS
from trainer.sft_qlora_invoice_extraction.data_prep import within


def resolve_checkpoints(baseline_root=None, cord_root=None):
    cord_root = project_path(cord_root or OUTPUT).resolve()
    paths = {}
    if baseline_root is not None:
        paths["baseline"] = latest_adapter(Path(baseline_root).resolve())
    paths["cord_sft"] = latest_adapter(cord_root)
    metadata = {}
    for mode, path in paths.items():
        checked = verify_adapter(path, MODEL_ID)
        config_name = "training_config.json" if mode == "cord_sft" else "run_config.json"
        config = json.loads((path.parent / config_name).read_text(encoding="utf-8"))
        result = json.loads((path.parent / "training_result.json").read_text(encoding="utf-8"))
        metrics = result["metrics"]
        if config.get("model_id") != MODEL_ID or metrics["global_steps_completed"] < 1:
            raise ValueError("Adapter backbone/completed optimizer steps mismatch")
        import math
        if not math.isfinite(metrics["training_loss"]) or not math.isfinite(metrics["evaluation"]["eval_loss"]):
            raise ValueError("Adapter lacks finite training/validation loss")
        if mode == "cord_sft":
            verified = verify_saved(path, cord_root)
            if verified["smoke"] or result.get("smoke"):
                raise ValueError("Smoke adapter cannot be evaluated as the CORD trained model")
            if metrics["global_steps_completed"] != config["expected_optimizer_steps"]:
                raise ValueError("CORD training did not complete its requested optimizer steps")
        metadata[mode] = {"path": str(path), "model_id": MODEL_ID, "run_id": path.parent.name,
                          "adapter_sha256": checked["sha256"], "tensor_count": checked["tensor_count"],
                          "training_config": config}
    if "baseline" in paths and (paths["baseline"] == paths["cord_sft"]
            or metadata["baseline"]["adapter_sha256"] == metadata["cord_sft"]["adapter_sha256"]):
        raise ValueError("Comparison requires two different verified adapters")
    config = metadata["cord_sft"]["training_config"]
    instruction = config["instruction"]
    if hashlib_prompt(instruction) != config["instruction_sha256"]:
        raise ValueError("Recorded training prompt checksum mismatch")
    revision = config.get("resolved_model_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CORD training must record a resolved base revision for fair evaluation")
    if "baseline" not in metadata:
        metadata["baseline"] = {"path": None, "model_id": MODEL_ID, "run_id": None,
                                "adapter_sha256": None, "tensor_count": 0,
                                "kind": "pretrained_base", "training_config": {}}
    baseline_revision = metadata["baseline"]["training_config"].get("resolved_model_revision")
    if baseline_revision and baseline_revision != revision:
        raise ValueError("Training metadata records conflicting base revisions")
    return metadata, instruction, revision


def hashlib_prompt(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_test(root, adapters):
    root = Path(root).resolve()
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    source = manifest["source"]
    if (manifest["counts"] != EXPECTED_COUNTS or source["repo_id"] != REPO_ID
            or set(source["files"]) != set(FILES)
            or not re.fullmatch(r"[a-f0-9]{40}", source["revision"])):
        raise ValueError("Wrong CORD source/split identity")
    training = adapters["cord_sft"]["training_config"]["dataset"]
    if (training["dataset_manifest_sha256"] != sha(manifest_file)
            or training["counts"] != manifest["counts"] or training["source"] != source
            or training.get("test_artifacts_opened") is not False):
        raise ValueError("CORD training provenance does not match the held-out dataset")
    # Metadata comparison only: never open train/validation samples or labels.
    for split, filename in (("train", "training.jsonl"), ("validation", "validation.jsonl")):
        if training["split_sha256"][split] != manifest["file_sha256"][f"{split}/{filename}"]:
            raise ValueError("Train/validation fingerprint differs from recorded training")
    for key in ("source_id", "image_sha256", "pixel_sha256"):
        if len({entry[key] for entry in manifest["entries"]}) != sum(EXPECTED_COUNTS.values()):
            raise ValueError("Duplicate/missing dataset identity or image fingerprint")
    for split, count in EXPECTED_COUNTS.items():
        if sum(entry["source_split"] == split for entry in manifest["entries"]) != count:
            raise ValueError("Manifest split separation/count mismatch")
    files = [(root / "test" / name).resolve() for name in ("test.jsonl", "annotations.jsonl")]
    for file in files:
        if not file.is_relative_to(root / "test") or sha(file) != manifest["file_sha256"][f"test/{file.name}"]:
            raise ValueError("Test manifest/annotation checksum mismatch")
    if sha(files[0]) in adapters["baseline"]["training_config"].get("manifest_sha256", {}).values():
        raise ValueError("Baseline training metadata includes the held-out test manifest")
    records, annotations = (read_jsonl(file) for file in files)
    if len(records) != EXPECTED_COUNTS["test"] or len(annotations) != len(records):
        raise ValueError("Exactly 19 held-out test pairs are required")
    entries = {entry["source_id"]: entry for entry in manifest["entries"]}
    result, seen = [], set()
    for record, annotation in zip(records, annotations):
        name, provenance = record["source_id"], record["provenance"]
        entry = entries[name]
        if name in seen or record["source_split"] != "test" or entry["source_split"] != "test":
            raise ValueError("Duplicate/wrong-split test record")
        seen.add(name)
        if (annotation["image_file"] != name or record["source_filename"] != name
                or fingerprint(annotation) != provenance["annotation_sha256"]
                or provenance["annotation_sha256"] != entry["annotation_sha256"]
                or provenance["source_file"] != "cord-test.jsonl.gz"
                or type(provenance["source_line"]) is not int
                or not 1 <= provenance["source_line"] <= SOURCE_COUNTS["test"]
                or provenance["source_line"] != entry["source_line"]):
            raise ValueError("Test source provenance mismatch")
        image = within(root, record["image"], ("test", "images"))
        csv = within(root, record["csv"], ("test", "labels"))
        text, rows = convert(annotation)
        hashes = inspect_image(image.read_bytes(), annotation)
        if (image.name != name or image.stem != csv.stem
                or entry["image"] != record["image"] or entry["csv"] != record["csv"]
                or hashes != (entry["image_sha256"], entry["pixel_sha256"])
                or sha(csv) != entry["csv_sha256"] or csv.read_text(encoding="utf-8") != text
                or record["target_csv"] != text or record["rows"] != len(rows) or entry["rows"] != len(rows)
                or any(record[key] != value for key, value in messages(record["image"], text).items())):
            raise ValueError("Test image/CSV/label fingerprint mismatch")
        result.append({"image_path": image, "image": name, "ground_truth_csv": text,
                       "image_sha256": hashes[0], "pixel_sha256": hashes[1]})
    return result, {"dataset_manifest_sha256": sha(manifest_file), "test_split_sha256": sha(files[0]),
                    "source": source, "counts": manifest["counts"], "train_samples_opened": False}


# Evaluation report
import csv
import json
from pathlib import Path

from trainer.sft_qlora_invoice_extraction.train import write_json
from trainer.sft_qlora_invoice_extraction.data_prep import sha

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
        if adapter.get("kind") == "pretrained_base":
            text.append(f"{mode}: pretrained base at the common revision, with LoRA disabled")
        else:
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


# Optum evaluation
import json
from pathlib import Path

from PIL import Image

from src.utils.paths import project_path
from trainer.sft_qlora_invoice_extraction.train import write_json
from trainer.sft_qlora_invoice_extraction.data_prep import fingerprint, sha

INVOICE_IDS = ("invoice_01_standard", "invoice_02_exclusions", "invoice_03_icu_mixed",
               "invoice_04_diagnostics", "invoice_05_nonmedical")


def load_optum(root=None):
    from src.utils.paths import assignment_annotation_path
    root = project_path(".").resolve() if root is None else Path(root).resolve()
    label_root = root / "data/assignment_dataset/ground_truth"
    manifest_file = label_root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if len(manifest) != 5 or {entry["invoice_id"] for entry in manifest} != set(INVOICE_IDS):
        raise ValueError("Expected exactly the five original Optum invoice identities")
    # Existing documentation and source explicitly attest manual transcription,
    # not model predictions. Record their hashes; do not regenerate annotations.
    provenance = root / "docs/data_evaluation_README.md"
    if not provenance.is_file():
        provenance = root / "docs/sft_qlora_invoice_extraction_OPTUM_FINAL.md"
    if not provenance.is_file():
        # Standalone evaluation datasets may still carry their own provenance.
        provenance = root / "data/assignment_dataset/README.md"
    documentation = provenance.read_text(encoding="utf-8")
    if "manually transcribed" not in documentation:
        raise ValueError("Optum annotations lack the existing manual-transcription provenance")
    records, inputs, seen = [], [], set()
    entries = {entry["invoice_id"]: entry for entry in manifest}
    for invoice_id in INVOICE_IDS:
        entry = entries[invoice_id]
        expected_paths = {"image": f"data/assignment_dataset/invoices/{invoice_id}.jpg",
                          "csv": f"data/assignment_dataset/ground_truth/{invoice_id}.csv",
                          "json": f"data/assignment_dataset/ground_truth/{invoice_id}.json"}
        recorded_paths = {key: assignment_annotation_path(entry[key]).as_posix() for key in expected_paths}
        if entry.get("split") != "final_evaluation_only" or recorded_paths != expected_paths:
            raise ValueError("Optum image/label pairing or evaluation provenance mismatch")
        image, csv, annotation = (root / recorded_paths[key] for key in ("image", "csv", "json"))
        for file in (image, csv, annotation):
            if not file.resolve().is_relative_to(root / ("data/assignment_dataset/invoices" if file == image else "data/assignment_dataset/ground_truth")):
                raise ValueError("Unsafe Optum image/label path")
        with Image.open(image) as source:
            if source.format != "JPEG" or source.width * source.height > 20_000_000:
                raise ValueError("Invalid/oversized original Optum image")
            source.load()
            rgb = source.convert("RGB")
            import hashlib
            pixel_hash = hashlib.sha256(f"{rgb.size}:".encode() + rgb.tobytes()).hexdigest()
        if pixel_hash in seen:
            raise ValueError("Duplicate Optum images")
        seen.add(pixel_hash)
        text, label_hashes = None, {}
        if csv.is_file() and annotation.is_file():
            gold = json.loads(annotation.read_text(encoding="utf-8"))
            if (gold["invoice_id"] != invoice_id
                    or assignment_annotation_path(gold["image_file"]).as_posix() != recorded_paths["image"]):
                raise ValueError("Optum JSON annotation belongs to another image")
            text = csv.read_text(encoding="utf-8")
            rows = parse_prediction(text)
            canonical = [{"Item": line["item"], "Quantity": str(line["quantity"]),
                          "Rate": "" if line["rate"] is None else str(line["rate"]), "Amount": str(line["amount"])}
                         for line in gold["lines"]]
            actual, expected = ([normalize(row, allow_empty_rates=True) for row in values]
                                for values in (rows, canonical))
            if (actual != expected or len(rows) != entry["rows"]
                    or any(value is None for row in expected for value in row)
                    or any(row[2] != "" and row[1]*row[2] != row[3] for row in expected)):
                raise ValueError("Manually transcribed Optum CSV/JSON disagreement or invalid arithmetic")
            label_hashes = {"csv_sha256": sha(csv), "json_sha256": sha(annotation)}
        records.append({"image": image.name, "image_path": image.resolve(), "ground_truth_csv": text,
                        "image_sha256": sha(image), "pixel_sha256": pixel_hash})
        inputs.append({"invoice_id": invoice_id, "image_sha256": sha(image), "pixel_sha256": pixel_hash,
                       "accuracy_status": "VERIFIED_EXISTING_MANUAL_TRANSCRIPTION" if text is not None else "NOT ASSESSED",
                       **label_hashes})
    return records, {"test_split_sha256": fingerprint(inputs), "label_manifest_sha256": sha(manifest_file),
                     "manual_provenance_sha256": sha(provenance), "label_provenance": "Existing manual transcription; no new human review performed",
                     "verified_label_count": sum(row["ground_truth_csv"] is not None for row in records), "inputs": inputs}


def add_recommendation(directory, summary):
    a, b = summary["models"]["baseline"], summary["models"]["cord_sft"]
    recommendation, reason = "INCONCLUSIVE", "All five verified labels and an unambiguous measured improvement are required."
    if summary["verified_ground_truth_documents"] == 5:
        comparisons = [(a[key]["value"], b[key]["value"]) for key in
                       ("csv_valid", "document_exact", "complete_row_exact", "row_precision", "row_recall")]
        comparisons += [(a["fields"][field]["value"], b["fields"][field]["value"])
                        for field in ("Item", "Quantity", "Rate", "Amount")]
        if any(new < old for old, new in comparisons):
            recommendation, reason = "ORIGINAL SFT", "CORD regressed on at least one verified extraction metric; retain the original for the demo."
        elif any(new > old for old, new in comparisons):
            recommendation, reason = "NEW CORD SFT", "CORD improved verified extraction metrics without regression in the compared accuracy/CSV metrics."
        else:
            reason = "Verified extraction metrics tied; latency alone does not establish better healthcare extraction."
    summary.update(recommendation=recommendation, recommendation_reason=reason)
    write_json(Path(directory) / "summary.json", summary)
    with (Path(directory) / "summary.md").open("a", encoding="utf-8") as stream:
        stream.write(f"\n## Demo recommendation\n\n{recommendation}: {reason}\n\n"
                     "No adapter was promoted. Confirm the existing production prompt and end-to-end workflow manually.\n")
    (Path(directory) / "comparison.csv").write_bytes((Path(directory) / "per_image_comparison.csv").read_bytes())


# Evaluate
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

from trainer.sft_qlora_invoice_extraction.train import versions, write_json
from src.utils.paths import project_path
from trainer.sft_qlora_invoice_extraction.data_prep import fingerprint, sha, write_jsonl
from trainer.sft_qlora_invoice_extraction.train import MODEL_ID


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", choices=("cord", "optum"), default="cord")
    cli.add_argument("--dataset-root", type=Path, default=Path("data/cord_sft_qlora"))
    cli.add_argument("--baseline-root", type=Path,
                     help="Optional verified comparison adapter root; defaults to the pretrained base with LoRA disabled")
    cli.add_argument("--cord-root", type=Path, default=Path("models/qwen25_finetune_sf_qlora"))
    cli.add_argument("--output-dir", type=Path, help="Defaults to the selected dataset's isolated report directory")
    cli.add_argument("--min-pixels", type=int, default=50176)
    cli.add_argument("--max-pixels", type=int, default=200704)
    cli.add_argument("--max-context-length", type=int, default=2048)
    cli.add_argument("--max-new-tokens", type=int, default=1024)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--local-files-only", action="store_true")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="CPU artifact/fingerprint checks, no model loading")
    mode.add_argument("--smoke", action="store_true", help="First held-out image in both evaluation modes; clearly partial report")
    mode.add_argument("--inspect", type=Path, help="Inspect an existing completed report and representative cases")
    return cli


def common_settings(args, instruction, revision):
    if (args.min_pixels < 1 or args.max_pixels < args.min_pixels or args.max_new_tokens < 1
            or args.max_context_length <= args.max_new_tokens or args.seed < 0):
        raise ValueError("Invalid image/context/generation limits or seed")
    return {"model_id": MODEL_ID, "revision": revision, "instruction": instruction,
            "prompt_sha256": hashlib_prompt(instruction), "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels, "max_context_length": args.max_context_length,
            "max_new_tokens": args.max_new_tokens, "do_sample": False, "num_beams": 1,
            "dtype": "bfloat16", "quantization": "4-bit NF4 double quantization",
            "attention": "sdpa", "seed": args.seed, "local_files_only": args.local_files_only,
            "allow_empty_rates": args.dataset == "optum"}


def assert_adapter_active(model, mode):
    if model.active_adapters != [mode] or mode not in model.peft_config:
        raise RuntimeError("Wrong/missing active PEFT adapter; no base-model fallback is allowed")
    if model.peft_config[mode].base_model_name_or_path != MODEL_ID:
        raise RuntimeError("Active adapter backbone differs from the shared model")


def verify_loaded_weights(model, mode, adapter, expected_sha256):
    """Verify PEFT's loaded tensors against the saved adapter, not just its name."""
    import torch
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    weights = Path(adapter) / "adapter_model.safetensors"
    if sha(weights) != expected_sha256:
        raise RuntimeError("Adapter weights changed after CPU verification")
    actual = get_peft_model_state_dict(model, adapter_name=mode, save_embedding_layers=False)
    expected = load_file(str(weights), device="cpu")
    if set(actual) != set(expected) or not expected:
        raise RuntimeError("PEFT adapter tensor mapping differs from the verified saved checkpoint")
    if any(not torch.equal(actual[key].detach().float().cpu(), value.float()) for key, value in expected.items()):
        raise RuntimeError("Active model adapter tensors differ from the saved weights")


class AdapterComparison:
    """One quantized base, two small frozen adapters, evaluated sequentially."""
    def __init__(self, adapters, settings):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, set_seed
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError("One visible BF16 CUDA GPU required; activate ../.venv and use CUDA_VISIBLE_DEVICES=0")
        free, total = torch.cuda.mem_get_info(0)
        if free / 2**30 < 18:
            raise RuntimeError("Less than 18 GiB free; manually stop BOTH vLLM servers before evaluation")
        set_seed(settings["seed"])
        torch.cuda.reset_peak_memory_stats(0)
        self.settings = settings
        self.gpu = {"name": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                    "total_gib": total/2**30, "initial_free_gib": free/2**30}
        print("Loading one shared NF4 Qwen-VL base/processor; no text reasoning model", flush=True)
        self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=settings["revision"],
            use_fast=False, min_pixels=settings["min_pixels"], max_pixels=settings["max_pixels"],
            local_files_only=settings["local_files_only"], trust_remote_code=False)
        if not self.processor.chat_template:
            raise RuntimeError("Shared Qwen processor is missing its chat template")
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, revision=settings["revision"],
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
            device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=settings["local_files_only"], trust_remote_code=False)
        if getattr(base.config, "_commit_hash", None) != settings["revision"]:
            raise RuntimeError("Loaded base revision differs from the common evaluation revision")
        self.base_baseline = adapters["baseline"]["path"] is None
        first_mode = "cord_sft" if self.base_baseline else "baseline"
        self.model = PeftModel.from_pretrained(base, adapters[first_mode]["path"],
                                              adapter_name=first_mode, is_trainable=False)
        if not self.base_baseline:
            self.model.load_adapter(adapters["cord_sft"]["path"], adapter_name="cord_sft", is_trainable=False)
        self._freeze_for_inference()
        self.model.config.use_cache = True
        for mode in ("baseline", "cord_sft"):
            if adapters[mode]["path"] is not None:
                verify_loaded_weights(self.model, mode, adapters[mode]["path"], adapters[mode]["adapter_sha256"])
        self.vision_calls = 0
        self.hook = self.model.get_base_model().visual.register_forward_hook(self._vision_used)

    def _vision_used(self, module, inputs, output):
        self.vision_calls += 1

    def _freeze_for_inference(self):
        # Loading another adapter or set_adapter can re-enable active LoRA grads.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("Evaluation model unexpectedly has trainable parameters")

    def select(self, mode):
        if mode not in {"baseline", "cord_sft"}:
            raise ValueError("Unknown extraction evaluation mode")
        self.selected_mode = mode
        active = "cord_sft" if getattr(self, "base_baseline", False) else mode
        self.model.set_adapter(active)
        self._freeze_for_inference()
        assert_adapter_active(self.model, active)

    def _adapter_context(self):
        if getattr(self, "base_baseline", False) and self.selected_mode == "baseline":
            return self.model.disable_adapter()
        return nullcontext()

    def generate(self, image_path, instruction):
        """Only the image/instruction enter generation; no labels or reference CSV."""
        import torch
        from PIL import Image
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        # Same Qwen message/image path as the working src.components.inference.local implementation.
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": instruction}]}]
        template = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[template], images=[image], padding=True,
                                truncation=False, return_tensors="pt")
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens + self.settings["max_new_tokens"] > self.settings["max_context_length"]:
            return {"raw_response": "", "error": "context_budget_exceeded_no_truncation",
                    "prompt_tokens": prompt_tokens, "generated_tokens": 0, "truncated": False}
        if any(key not in inputs or inputs[key].numel() == 0 for key in ("pixel_values", "image_grid_thw")):
            raise RuntimeError("Missing actual multimodal image tensors")
        inputs = inputs.to("cuda")
        previous = self.vision_calls
        with torch.inference_mode(), self._adapter_context():
            output = self.model.generate(**inputs, max_new_tokens=self.settings["max_new_tokens"],
                                         do_sample=False, num_beams=1, use_cache=True)
        tokens = output[0][prompt_tokens:]
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        eos = {value for value in eos if value is not None}
        raw = self.processor.tokenizer.decode(tokens.tolist(), skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        stopped = len(tokens) > 0 and tokens[-1].item() in eos
        vision_verified = self.vision_calls > previous
        return {"raw_response": raw, "error": None if vision_verified else "vision_forward_not_verified", "prompt_tokens": prompt_tokens,
                "generated_tokens": len(tokens), "truncated": not stopped,
                "vision_forward_verified": vision_verified}

    def close(self):
        self.hook.remove()
        del self.model, self.processor
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def evaluate_predictions(records, predictions, settings):
    """Check common image/settings identities before any paired scoring."""
    comparisons = []
    for mode in ("baseline", "cord_sft"):
        if len(predictions[mode]) != len(records):
            raise ValueError("Incomplete model comparison; fixed denominators cannot be reduced")
    for index, record in enumerate(records):
        row = {"image": record["image"], "image_path": str(record["image_path"]),
               "ground_truth_csv": record["ground_truth_csv"]}
        for mode in ("baseline", "cord_sft"):
            prediction = predictions[mode][index]
            if (prediction["image"] != record["image"] or prediction["image_sha256"] != record["image_sha256"]
                    or prediction["common_settings_sha256"] != fingerprint(settings) or prediction["mode"] != mode):
                raise ValueError("Model/input/config identity differs between paired evaluations")
            row[mode] = {**prediction, "metrics": score(record["ground_truth_csv"], prediction,
                        allow_empty_rates=settings.get("allow_empty_rates", False))}
        comparisons.append(row)
    return comparisons


def run(args):
    if args.inspect:
        inspect_report(project_path(args.inspect))
        return None
    if Path(sys.prefix).name != ".venv" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate the existing application/training ../.venv")
    adapters, instruction, revision = resolve_checkpoints(
        project_path(args.baseline_root) if args.baseline_root else None, project_path(args.cord_root))
    settings = common_settings(args, instruction, revision)
    if args.dataset == "optum":
        records, dataset = load_optum()
    else:
        records, dataset = load_test(project_path(args.dataset_root), adapters)
    libraries = versions()
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "Pillow", "safetensors"):
        if libraries[package] == "MISSING":
            raise RuntimeError(f"Missing existing evaluation dependency: {package}; no installation performed")
    metadata = {"model_id": MODEL_ID, "adapters": adapters, "settings": settings, "dataset_name": args.dataset,
                "common_settings_sha256": fingerprint(settings), "dataset": dataset, "versions": libraries,
                "baseline_training_revision_recorded": bool(adapters["baseline"]["training_config"].get("resolved_model_revision")),
                "interpreter": sys.executable, "environment": sys.prefix}
    if args.preflight:
        print(json.dumps({"status": "CPU_PREFLIGHT_PASS_GPU_UNVERIFIED", **metadata}, indent=2), flush=True)
        return None
    report_root = "reports/cord/optum_final" if args.dataset == "optum" else "reports/cord/evaluation"
    output = project_path(args.output_dir or report_root).resolve()
    if not output.is_relative_to(project_path(report_root).resolve()):
        raise ValueError(f"Evaluation outputs must remain under {report_root}")
    records = records[:1] if args.smoke else records
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    directory = output / run_id
    directory.mkdir(parents=True, exist_ok=False)
    metadata.update(run_id=run_id, smoke=args.smoke, evaluated_documents=len(records), status="running",
                    inputs=[{key: record[key] for key in ("image", "image_sha256", "pixel_sha256")} for record in records])
    write_json(directory / "evaluation_manifest.json", metadata)
    engine, predictions, started = None, {}, time.monotonic()
    try:
        engine = AdapterComparison(adapters, settings)
        metadata.update(gpu=engine.gpu, chat_template_sha256=hashlib_prompt(engine.processor.chat_template))
        for mode in ("baseline", "cord_sft"):
            engine.select(mode)
            predictions[mode] = []
            for index, record in enumerate(records, 1):
                if sha(record["image_path"]) != record["image_sha256"]:
                    raise ValueError("Test image changed during evaluation")
                started_image = time.monotonic()
                try:
                    prediction = engine.generate(record["image_path"], instruction)
                except Exception as exc:
                    prediction = {"raw_response": "", "error": type(exc).__name__, "truncated": False,
                                  "generated_tokens": 0, "prompt_tokens": None}
                    import torch
                    torch.cuda.empty_cache()
                prediction.update(mode=mode, image=record["image"], image_sha256=record["image_sha256"],
                                  pixel_sha256=record["pixel_sha256"],
                                  common_settings_sha256=fingerprint(settings), latency_seconds=time.monotonic()-started_image)
                predictions[mode].append(prediction)
                print(f"{mode} {index}/{len(records)} {record['image']} tokens={prediction['generated_tokens']} "
                      f"truncated={prediction['truncated']} error={prediction['error']}", flush=True)
            write_jsonl(directory / f"{mode}_predictions.jsonl", predictions[mode])
        comparisons = evaluate_predictions(records, predictions, settings)
        import torch
        metadata.update(status="complete_smoke" if args.smoke else "complete",
                        runtime_seconds=time.monotonic()-started,
                        peak_cuda_allocated_gib=torch.cuda.max_memory_allocated(0)/2**30,
                        peak_cuda_reserved_gib=torch.cuda.max_memory_reserved(0)/2**30)
        summary = write_reports(directory, comparisons, metadata)
        if args.dataset == "optum":
            add_recommendation(directory, summary)
        metadata["report_sha256"] = {name: sha(directory / name) for name in (
            "baseline_predictions.jsonl", "cord_sft_predictions.jsonl", "per_image_comparison.csv",
            "per_image_comparison.json", "summary.json", "summary.md")}
        if args.dataset == "optum":
            metadata["report_sha256"]["comparison.csv"] = sha(directory / "comparison.csv")
        write_json(directory / "evaluation_manifest.json", metadata)
        print(f"Report: {directory}", flush=True)
        return directory
    except BaseException as exc:
        metadata.update(status="failed", error_type=type(exc).__name__, runtime_seconds=time.monotonic()-started)
        write_json(directory / "evaluation_manifest.json", metadata)
        print(f"Evaluation failed ({type(exc).__name__}); partial artifacts retained at {directory}", flush=True)
        raise
    finally:
        if engine is not None:
            engine.close()


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()

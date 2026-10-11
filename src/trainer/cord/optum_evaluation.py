"""Five original Optum fixtures: existing manual labels only, no label generation."""
import json
from pathlib import Path

from PIL import Image

from src.utils.paths import project_path
from src.trainer.sft.train.train_qlora import write_json
from .evaluation_metrics import normalize, parse_prediction
from .schema import fingerprint, sha

INVOICE_IDS = ("invoice_01_standard", "invoice_02_exclusions", "invoice_03_icu_mixed",
               "invoice_04_diagnostics", "invoice_05_nonmedical")


def load_optum(root=None):
    root = project_path(".").resolve() if root is None else Path(root).resolve()
    label_root = root / "data/evaluation/ground_truth"
    manifest_file = label_root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if len(manifest) != 5 or {entry["invoice_id"] for entry in manifest} != set(INVOICE_IDS):
        raise ValueError("Expected exactly the five original Optum invoice identities")
    # Existing documentation and source explicitly attest manual transcription,
    # not model predictions. Record their hashes; do not regenerate annotations.
    provenance = root / "data/evaluation/README.md"
    documentation = provenance.read_text(encoding="utf-8")
    if "manually transcribed" not in documentation:
        raise ValueError("Optum annotations lack the existing manual-transcription provenance")
    records, inputs, seen = [], [], set()
    entries = {entry["invoice_id"]: entry for entry in manifest}
    for invoice_id in INVOICE_IDS:
        entry = entries[invoice_id]
        expected_paths = {"image": f"data/invoices/{invoice_id}.jpg",
                          "csv": f"data/evaluation/ground_truth/{invoice_id}.csv",
                          "json": f"data/evaluation/ground_truth/{invoice_id}.json"}
        if entry.get("split") != "final_evaluation_only" or any(entry[key] != path for key, path in expected_paths.items()):
            raise ValueError("Optum image/label pairing or evaluation provenance mismatch")
        image, csv, annotation = (root / entry[key] for key in ("image", "csv", "json"))
        for file in (image, csv, annotation):
            if not file.resolve().is_relative_to(root / ("data/invoices" if file == image else "data/evaluation/ground_truth")):
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
            if gold["invoice_id"] != invoice_id or gold["image_file"] != entry["image"]:
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

"""Deterministically select complete CORD receipts; write only selected images."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import statistics
from uuid import uuid4
import zipfile

from PIL import Image

from src.utils.paths import project_path
from .download import verify_source
from .schema import (COUNTS, SOURCE_COUNTS, MANIFESTS, MAX_IMAGE_BYTES, MAX_PIXELS,
                     LabelError, configured_counts, convert, fingerprint, messages, sha, write_json, write_jsonl)


def inspect_image(data, record):
    if len(data) > MAX_IMAGE_BYTES:
        raise LabelError("image_too_large", "image_file")
    if any(type(record.get(key)) is not int or record[key] <= 0 for key in ("width", "height")):
        raise LabelError("invalid_source_dimensions", "width/height")
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.width * image.height > MAX_PIXELS or min(image.size) < 1:
            raise LabelError("invalid_image_format_or_dimensions", "image_file")
        if image.size != (record.get("width"), record.get("height")):
            raise LabelError("source_dimension_mismatch", "width/height")
        image.verify()
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        rgb = image.convert("RGB")
        pixel_hash = hashlib.sha256(f"{rgb.size}:".encode() + rgb.tobytes()).hexdigest()
    return hashlib.sha256(data).hexdigest(), pixel_hash


def eligible_candidates(source, *, counts=None):
    counts = configured_counts(counts)
    source = Path(source).resolve()
    provenance = verify_source(source)
    candidates, rejected, seen_ids, seen_images, seen_pixels = {}, [], set(), set(), set()
    observed = {}
    for split in COUNTS:
        with gzip.open(source / f"cord-{split}.jsonl.gz", "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        observed[split] = len(records)
        if len(records) != SOURCE_COUNTS[split]:
            raise ValueError(f"Source {split} must contain {SOURCE_COUNTS[split]} records; found {len(records)}")
        candidates[split] = []
        with zipfile.ZipFile(source / f"cord-{split}-images.zip") as archive:
            entries = archive.infolist()
            names = [entry.filename for entry in entries]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate ZIP member names in {split}")
            for line_number, record in enumerate(records, 1):
                name = record.get("image_file") if isinstance(record, dict) else None
                try:
                    if not isinstance(name, str) or Path(name).name != name or "\\" in name or not name.endswith(".jpg"):
                        raise LabelError("unsafe_or_invalid_filename", "image_file")
                    if name in seen_ids:
                        raise LabelError("duplicate_source_id", "image_file")
                    seen_ids.add(name)
                    text, lines = convert(record)
                    if name not in names or archive.getinfo(name).file_size > MAX_IMAGE_BYTES:
                        raise LabelError("missing_or_oversized_image", "image_file")
                    data = archive.read(name)  # In memory only; no ZIP extraction to disk.
                    image_hash, pixel_hash = inspect_image(data, record)
                    if image_hash in seen_images or pixel_hash in seen_pixels:
                        raise LabelError("duplicate_image_or_cross_split_leakage", "image_file")
                    seen_images.add(image_hash)
                    seen_pixels.add(pixel_hash)
                    candidates[split].append({"source_id": name, "source_split": split,
                        "source_line": line_number, "annotation": record, "target_csv": text,
                        "rows": len(lines), "image_sha256": image_hash, "pixel_sha256": pixel_hash,
                        "annotation_sha256": fingerprint(record),
                        "quantities": [line.quantity for line in lines],
                        "longest_item": max(len(line.item) for line in lines)})
                except (LabelError, OSError, ValueError, KeyError, zipfile.BadZipFile, Image.DecompressionBombError) as exc:
                    rejected.append({"source_filename": name, "source_split": split, "source_line": line_number,
                        "reason": exc.reason if isinstance(exc, LabelError) else "image_decode_or_source_error",
                        "field": exc.field if isinstance(exc, LabelError) else "image_file",
                        "row_index": exc.row_index if isinstance(exc, LabelError) else None,
                        "error_type": type(exc).__name__})
        print(f"{split}: source={observed[split]}, eligible={len(candidates[split])}, required={counts[split]}", flush=True)
    return candidates, rejected, provenance, observed


def select(candidates, count, seed):
    if len(candidates) < count:
        raise ValueError(f"Insufficient strict candidates: required {count}, eligible {len(candidates)}")
    # Prefer multi-row receipts; stratify annotation/image proxies for diversity.
    # No rule is relaxed to reach the count. Seeded sorting avoids source-order bias.
    rng, groups = random.Random(seed), defaultdict(list)
    for candidate in sorted(candidates, key=lambda row: row["source_id"]):
        annotation = candidate["annotation"]
        key = (candidate["rows"] > 1, candidate["rows"] >= 5,
               annotation["width"] >= 800, annotation["height"] >= 2 * annotation["width"],
               candidate["longest_item"] >= 25)
        groups[key].append(candidate)
    for group in groups.values():
        rng.shuffle(group)
    chosen = []
    for multiple in (True, False):
        keys = sorted(key for key in groups if key[0] == multiple)
        rng.shuffle(keys)
        while len(chosen) < count and any(groups[key] for key in keys):
            for key in keys:
                if groups[key] and len(chosen) < count:
                    chosen.append(groups[key].pop())
    return chosen


def quality_report(candidates, rejected, selected, provenance, observed, seed, counts):
    chosen = [row for split in COUNTS for row in selected.get(split, [])]
    sizes = [row["rows"] for row in chosen]
    return {"source": provenance, "seed": seed, "required_counts": dict(counts),
        "source_counts": observed, "eligible_counts": {split: len(rows) for split, rows in candidates.items()},
        "selected_counts": {split: len(selected.get(split, [])) for split in COUNTS},
        "total_line_items": sum(sizes), "mean_rows": statistics.mean(sizes) if sizes else None,
        "median_rows": statistics.median(sizes) if sizes else None,
        "multi_row_receipts": sum(size > 1 for size in sizes),
        "quantity_distribution": dict(Counter(str(q) for row in chosen for q in row["quantities"])),
        "rejected_receipts": len(rejected), "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        "examples": [{"source_id": row["source_id"], "split": row["source_split"], "csv": row["target_csv"]}
                     for split in COUNTS for row in selected.get(split, [])[:2]],
        "selection_method": "seeded multi-row-first stratified round-robin; size/aspect/name-length proxies",
        "arithmetic": "exact Decimal; explicit whole-rupiah rates only; zero price retained",
        "human_verified": False, "currency_conversion": False, "derived_rates": False,
        "limitations": ["Indonesian retail/restaurant domain, not healthcare policy reasoning",
            "Strict complete-row filtering biases towards receipts with explicit rates",
            "Annotation/decode eligibility does not establish visual readability; inspect selected images",
            "Original hospital-worded production extraction prompt is preserved",
            "Current SFT CLI has synthetic-only path/metadata guards; Phase 10B must adapt its preflight"]}


def eligibility(source, seed=42, *, counts=None):
    counts = configured_counts(counts)
    if seed < 0:
        raise ValueError("Seed must be nonnegative")
    candidates, rejected, provenance, observed = eligible_candidates(source, counts=counts)
    sufficient = all(len(candidates[split]) >= count for split, count in counts.items())
    selected = {split: select(candidates[split], count, seed + offset) for offset, (split, count) in enumerate(counts.items())} if sufficient else {}
    report = quality_report(candidates, rejected, selected, provenance, observed, seed, counts)
    report["status"] = "ELIGIBLE" if sufficient else "INSUFFICIENT_ELIGIBLE_RECEIPTS"
    for split in COUNTS:
        if observed[split] != len(candidates[split]) + sum(row["source_split"] == split for row in rejected):
            raise ValueError("Unaccounted source record")
    return selected, report, rejected


def save_eligibility(report, rejected, directory=None):
    directory = Path(directory) if directory else project_path("reports/cord") / (
        "eligibility-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "selection_report.json", report)
    write_json(directory / "rejection_report.json", rejected)
    print(f"Eligibility reports: {directory}", flush=True)


def prepare(source, output, seed=42, *, preflight=False, report_dir=None, counts=None):
    counts = configured_counts(counts)
    source, output = Path(source).resolve(), Path(output).resolve()
    if not preflight and output.exists():
        raise ValueError("Output already exists; choose a new --output-dir. No overwrite or deletion supported")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError("CORD source and output directories must be separate")
    selected, report, rejected = eligibility(source, seed, counts=counts)
    if preflight or report["status"] != "ELIGIBLE":
        save_eligibility(report, rejected, report_dir)
        if report["status"] != "ELIGIBLE":
            raise ValueError(f"Not enough complete, explicit-rate receipts for {counts}; see eligibility reports. No dataset written")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    needed = sum((source / name).stat().st_size for name in report["source"]["files"] if name.endswith(".zip"))
    if shutil.disk_usage(output.parent).free < needed + 256 * 1024 * 1024:
        raise ValueError("Insufficient preparation disk space; reserve all archive bytes plus 256 MiB")
    output.mkdir(exist_ok=False)
    entries, file_hashes = [], {}
    for split in COUNTS:
        split_dir = output / split
        (split_dir / "images").mkdir(parents=True)
        (split_dir / "labels").mkdir()
        records = []
        with zipfile.ZipFile(source / f"cord-{split}-images.zip") as archive:
            for candidate in selected[split]:
                name, text = candidate["source_id"], candidate["target_csv"]
                image_rel, csv_rel = f"{split}/images/{name}", f"{split}/labels/{Path(name).stem}.csv"
                with (output / image_rel).open("xb") as stream:
                    stream.write(archive.read(name))  # Only selected members ever written.
                with (output / csv_rel).open("x", encoding="utf-8", newline="") as stream:
                    stream.write(text)
                provenance = {"source_file": f"cord-{split}.jsonl.gz", "source_line": candidate["source_line"],
                              "annotation_sha256": candidate["annotation_sha256"],
                              "label_source": "gt_parse.menu explicit nm/cnt/unitprice/price; no derivation"}
                record = {"image": image_rel, "csv": csv_rel, "target_csv": text, "rows": candidate["rows"],
                          "source_filename": name, "source_id": name, "source_split": split,
                          "provenance": provenance, **messages(image_rel, text)}
                records.append(record)
                entries.append({"source_id": name, "source_split": split, "image": image_rel, "csv": csv_rel,
                    "source_line": candidate["source_line"], "rows": candidate["rows"],
                    "annotation_sha256": candidate["annotation_sha256"], "image_sha256": candidate["image_sha256"],
                    "pixel_sha256": candidate["pixel_sha256"], "csv_sha256": sha(output / csv_rel)})
        write_jsonl(split_dir / "annotations.jsonl", [row["annotation"] for row in selected[split]])
        write_jsonl(split_dir / MANIFESTS[split], records)
        for name in ("annotations.jsonl", MANIFESTS[split]):
            file_hashes[f"{split}/{name}"] = sha(split_dir / name)
    write_json(output / "selection_report.json", report)
    write_json(output / "rejection_report.json", rejected)
    (output / "README.md").write_text(
        "# CORD 200\n\nSource: Heliosoph/CORD; original CORD authors and NAVER Clova; CC BY 4.0.\n"
        f"{counts['train']} train / {counts['validation']} validation / {counts['test']} test. No currency conversion or inferred rates.\n"
        "Original annotations retained. Inspect selection_report.json and rejection_report.json.\n"
        "See src/trainer/cord/README.md for schema compatibility and Phase 10B restrictions.\n", encoding="utf-8")
    for name in ("selection_report.json", "rejection_report.json", "README.md"):
        file_hashes[name] = sha(output / name)
    write_json(output / "manifest.json", {"source": report["source"], "seed": seed, "counts": dict(counts),
        "entries": entries, "file_sha256": file_hashes, "human_verified": False,
        "raw_schema": "existing SFT image/csv/target_csv/rows + provenance",
        "hf_schema": "images + user(image marker, shared instruction) + assistant(CSV text)"})
    from .validate import validate_dataset
    validation = validate_dataset(output, expected_counts=counts)
    write_json(output / "validation_report.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/cord_source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cord_200"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=COUNTS["train"])
    parser.add_argument("--validation-count", type=int, default=COUNTS["validation"])
    parser.add_argument("--test-count", type=int, default=COUNTS["test"])
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    args = parser.parse_args()
    prepare(project_path(args.source_dir), project_path(args.output_dir), args.seed,
            preflight=args.preflight, report_dir=project_path(args.report_dir) if args.report_dir else None,
            counts={"train": args.train_count, "validation": args.validation_count, "test": args.test_count})


if __name__ == "__main__":
    main()

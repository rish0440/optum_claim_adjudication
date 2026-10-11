"""CPU validation of selected CORD pairs, provenance and shared HF input format."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

from PIL import Image, ImageDraw, ImageFont

from src.utils.paths import project_path
from .prepare import inspect_image
from .schema import (COUNTS, SOURCE_COUNTS, FILES, MANIFESTS, REPO_ID, convert,
                     configured_counts, fingerprint, messages, read_jsonl, sha)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(root, relative, prefix):
    require(isinstance(relative, str), "Image/CSV path must be a string")
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts and "\\" not in relative,
            "Unsafe relative dataset path")
    require(path.parts[:2] == prefix and len(path.parts) == 3, "Path belongs to wrong split or directory")
    resolved = (root / path).resolve()
    require(resolved.is_relative_to(root / Path(*prefix)) and resolved.is_file(), "Missing/unsafe image or CSV path")
    return resolved


def hf_messages(value):
    """Arrow fills optional struct fields (e.g. image-marker text) with None."""
    if isinstance(value, dict):
        return {key: hf_messages(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [hf_messages(item) for item in value]
    return value


def validate_dataset(root, *, source_dir=None, check_hf=False, expected_counts=None):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    source = manifest["source"]
    require(source.get("repo_id") == REPO_ID and set(source.get("files", {})) == set(FILES), "Wrong source identity/files")
    require(bool(re.fullmatch(r"[a-f0-9]{40}", source.get("revision", ""))), "Source revision must be a resolved full commit SHA")
    required_counts = configured_counts(manifest["counts"])
    if expected_counts is not None:
        require(required_counts == configured_counts(expected_counts), "Configured split counts differ from manifest")
    entries = manifest["entries"]
    require(len(entries) == sum(required_counts.values()), "Exact selected entry count required")
    expected_files = {f"{split}/{name}" for split in COUNTS for name in ("annotations.jsonl", MANIFESTS[split])}
    expected_files |= {"selection_report.json", "rejection_report.json", "README.md"}
    require(set(manifest["file_sha256"]) == expected_files, "Incomplete metadata file fingerprints")
    for relative, digest in manifest["file_sha256"].items():
        file = (root / relative).resolve()
        require(file.is_relative_to(root) and file.is_file() and sha(file) == digest, "Metadata file checksum mismatch")
    expected = {}
    for entry in entries:
        require(entry["source_id"] not in expected, "Duplicate source ID in manifest")
        expected[entry["source_id"]] = entry
    originals = {}
    if source_dir is not None:
        source_dir = Path(source_dir).resolve()
        for split in COUNTS:
            name = f"cord-{split}.jsonl.gz"
            file = (source_dir / name).resolve()
            require(file.is_relative_to(source_dir) and sha(file) == source["files"][name]["sha256"], "Source annotation checksum mismatch")
            with gzip.open(file, "rt", encoding="utf-8") as stream:
                originals[split] = [json.loads(line) for line in stream]
            require(len(originals[split]) == SOURCE_COUNTS[split], "Source split count mismatch")
    paths, images, pixels, ids, counts, row_counts, hf_counts = set(), set(), set(), set(), {}, {}, {}
    source_lines = set()
    for split, count in required_counts.items():
        records = read_jsonl(root / split / MANIFESTS[split])
        annotations = read_jsonl(root / split / "annotations.jsonl")
        require(len(records) == count and len(annotations) == count, "Exact image/annotation split count required")
        hf_records = []
        counts[split], row_counts[split] = len(records), 0
        for record, annotation in zip(records, annotations):
            name = annotation.get("image_file")
            require(isinstance(name, str) and name == Path(name).name and name.endswith(".jpg")
                    and "\\" not in name, "Invalid source filename")
            require(name not in ids, "Duplicate source ID or cross-split leakage")
            ids.add(name)
            require(name in expected, "Unmanifested source ID")
            entry = expected[name]
            require(entry["source_split"] == record["source_split"] == split, "Original source split not preserved")
            require(record["source_id"] == record["source_filename"] == name, "Image/annotation identity mismatch")
            provenance = record["provenance"]
            require(provenance["source_file"] == f"cord-{split}.jsonl.gz", "Wrong original annotation reference")
            require(type(provenance["source_line"]) is int and 1 <= provenance["source_line"] <= SOURCE_COUNTS[split], "Invalid source line reference")
            require(entry["source_line"] == provenance["source_line"], "Source line mismatch")
            locator = (split, provenance["source_line"])
            require(locator not in source_lines, "Duplicate annotation reference")
            source_lines.add(locator)
            if originals:
                require(originals[split][provenance["source_line"] - 1] == annotation, "Original ground_truth changed")
            require(fingerprint(annotation) == provenance["annotation_sha256"] == entry["annotation_sha256"], "Source annotation fingerprint mismatch")
            text, lines = convert(annotation)  # Recheck explicit labels and exact arithmetic.
            require(type(record["rows"]) is int and record["rows"] == entry["rows"] == len(lines), "Missing or invented item row")
            image_path = within(root, record["image"], (split, "images"))
            csv_path = within(root, record["csv"], (split, "labels"))
            require(image_path.name == name and csv_path.stem == image_path.stem, "Image/CSV filename mismatch")
            require(entry["image"] == record["image"] and entry["csv"] == record["csv"], "Pair manifest path mismatch")
            require(image_path not in paths and csv_path not in paths, "Duplicate pair path")
            paths.update((image_path, csv_path))
            image_hash, pixel_hash = inspect_image(image_path.read_bytes(), annotation)
            require(image_hash not in images and pixel_hash not in pixels, "Duplicate image content or cross-split leakage")
            images.add(image_hash)
            pixels.add(pixel_hash)
            require(image_hash == entry["image_sha256"] and pixel_hash == entry["pixel_sha256"], "Image/source fingerprint mismatch")
            require(sha(csv_path) == entry["csv_sha256"], "CSV fingerprint mismatch")
            actual_text = csv_path.read_text(encoding="utf-8")
            require(actual_text == record["target_csv"] == text, "CSV differs from explicit canonical source rows")
            require(bool(text) and text.startswith("Item,Quantity,Rate,Amount\n"), "Missing header/empty completion/Markdown")
            rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
            require(len(rows) == len(lines) + 1 and all(len(row) == 4 for row in rows), "CSV row count/escaping invalid")
            for key, value in messages(record["image"], text).items():
                require(record[key] == value, "Training multimodal message/image contract mismatch")
            hf_records.append({"image_path": image_path, "csv_path": csv_path, "target_csv": text, "rows": lines})
            row_counts[split] += len(lines)
        if check_hf:
            from src.trainer.sft.training_data import make_hf_dataset
            dataset = make_hf_dataset(hf_records)  # Existing corrected image feature handling.
            require(len(dataset) == count and set(dataset.column_names) == {"images", "prompt", "completion"}, "HF schema/count mismatch")
            for example, record in zip(dataset, records):
                require(len(example["images"]) == 1 and isinstance(example["images"][0], Image.Image), "HF image was not decoded")
                example["images"][0].load()
                rgb = example["images"][0].convert("RGB")
                decoded_hash = hashlib.sha256(f"{rgb.size}:".encode() + rgb.tobytes()).hexdigest()
                require(decoded_hash == expected[record["source_id"]]["pixel_sha256"], "HF image paired with the wrong CSV")
                require(hf_messages(example["prompt"]) == record["prompt"]
                        and hf_messages(example["completion"]) == record["completion"], "HF prompt/CSV target changed")
            hf_counts[split] = len(dataset)
    actual_pairs = {path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in
                    {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".csv"}}
    require(actual_pairs == paths, "Image/CSV inventory differs from paired manifests")
    report = json.loads((root / "selection_report.json").read_text())
    rejected = json.loads((root / "rejection_report.json").read_text())
    require(report["source"] == source and report["source_counts"] == SOURCE_COUNTS
            and report["required_counts"] == required_counts and report["selected_counts"] == required_counts,
            "Selection report source/count mismatch")
    rejected_counts = Counter(row["source_split"] for row in rejected)
    require(all(report["eligible_counts"][split] + rejected_counts[split] == SOURCE_COUNTS[split] for split in COUNTS), "Unaccounted rejected source records")
    return {"status": "PASS", "images": sum(counts.values()), "annotations": len(ids), "counts": counts,
        "total_line_items": sum(row_counts.values()), "rows_by_split": row_counts,
        "unique_image_hashes": len(images), "cross_split_duplicates": 0,
        "arithmetic": "exact Decimal PASS; zero-priced rows preserved; explicit rates only",
        "source_revision": source["revision"], "dataset_manifest_sha256": sha(root / "manifest.json"),
        "original_compressed_annotations_checked": bool(originals), "hf_examples": hf_counts,
        "hf_builder_compatibility": "PASS" if check_hf else "not_requested",
        "current_training_cli_compatibility": "BLOCKED: synthetic-only guards; Phase 10B adaptation required",
        "human_visual_review": "pending"}


def contact_sheet(root, destination):
    destination = Path(destination).resolve()
    require(destination.is_relative_to(project_path("reports/cord").resolve()), "Contact sheet must be under reports/cord")
    records = [record for split in COUNTS for record in read_jsonl(root / split / MANIFESTS[split])[:4]]
    sheet = Image.new("RGB", (1050, 450 * ((len(records) + 2) // 3)), "white")
    draw, font = ImageDraw.Draw(sheet), ImageFont.load_default(size=13)
    for index, record in enumerate(records):
        x, y = (index % 3) * 350, (index // 3) * 450
        with Image.open(root / record["image"]) as image:
            image.thumbnail((330, 330))
            sheet.paste(image.convert("RGB"), (x + 10, y + 10))
        sample = record["target_csv"].splitlines()[:3]
        draw.text((x + 10, y + 345), record["source_filename"] + "\n" + "\n".join(line[:45] for line in sample),
                  font=font, fill="black")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        sheet.save(stream, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cord_200"))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--check-hf", action="store_true", help="Use installed datasets/Pillow builder; no tokenizer/model download")
    parser.add_argument("--examples", type=int, default=0, help="Print this many actual image/CSV pairs per split")
    parser.add_argument("--contact-sheet", type=Path, help="Optional first four images per split, under reports/cord; no overwrite")
    args = parser.parse_args()
    if args.examples < 0:
        parser.error("--examples must be nonnegative")
    root = project_path(args.dataset_root).resolve()
    result = validate_dataset(root, source_dir=project_path(args.source_dir) if args.source_dir else None, check_hf=args.check_hf)
    print(json.dumps(result, indent=2), flush=True)
    for split in COUNTS:
        for record in read_jsonl(root / split / MANIFESTS[split])[:args.examples]:
            print(f"{split}: {root / record['image']}\n{record['target_csv']}", flush=True)
    if args.contact_sheet:
        contact_sheet(root, project_path(args.contact_sheet))


if __name__ == "__main__":
    main()

"""Receipt preparation, dataset validation and training split access."""


# Schema
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import re

from src.components.inference.parsing import CSV_FIELDS
from src.components.prompts.extraction import INSTRUCTION as SOURCE_INSTRUCTION
from src.schema import InvoiceLine

REPO_ID = "Heliosoph/CORD"
DEFAULT_REVISION = "51b7e93"  # Explicit published commit prefix; resolved to full SHA on download.
SOURCE_COUNTS = {"train": 800, "validation": 100, "test": 100}
COUNTS = {"train": 161, "validation": 20, "test": 19}
TOTAL_COUNT = 200
FILES = tuple(name for split in COUNTS for name in
              (f"cord-{split}-images.zip", f"cord-{split}.jsonl.gz"))
MANIFESTS = {"train": "training.jsonl", "validation": "validation.jsonl", "test": "test.jsonl"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 20_000_000


def configured_counts(counts=None):
    """Validate requested split sizes; production always requires exactly 200."""
    counts = COUNTS if counts is None else counts
    if (not isinstance(counts, dict) or set(counts) != set(SOURCE_COUNTS)
            or any(type(value) is not int or value <= 0 for value in counts.values())):
        raise ValueError("Split counts must be positive integers for train, validation and test")
    if sum(counts.values()) != TOTAL_COUNT:
        raise ValueError(f"Split counts must total exactly {TOTAL_COUNT}")
    return {split: counts[split] for split in SOURCE_COUNTS}


class LabelError(ValueError):
    def __init__(self, reason, field, row_index=None):
        super().__init__(f"{reason}: {field}")
        self.reason, self.field, self.row_index = reason, field, row_index


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, values):
    with Path(path).open("x", encoding="utf-8") as stream:
        for value in values:
            stream.write(canonical(value) + "\n")


def read_jsonl(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record at line {index}")
            value = json.loads(line, object_pairs_hook=unique,
                               parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
            if not isinstance(value, dict):
                raise ValueError(f"JSONL record must be an object at line {index}")
            result.append(value)
    return result


def quantity(value):
    if not isinstance(value, str):
        raise LabelError("invalid_quantity", "cnt")
    match = re.fullmatch(r"(?:[xX]\s*)?([1-9][0-9]*)", value.strip()) or re.fullmatch(
        r"([1-9][0-9]*)\s*[xX]", value.strip())
    if not match:
        raise LabelError("invalid_quantity", "cnt")
    return int(match.group(1))


def money(value, field):
    """Only explicit whole-rupiah labels: . and , are grouping separators.

    Decimal-looking/mixed notation is rejected, never guessed or converted.
    A single terminal period is accepted after otherwise valid grouping.
    """
    if not isinstance(value, str):
        raise LabelError("invalid_money", field)
    text = value.strip()
    if text.endswith("."):
        text = text[:-1]
    if re.fullmatch(r"[0-9]+", text):
        digits = text
    elif re.fullmatch(r"[0-9]{1,3}(?:,[0-9]{3})+", text):
        digits = text.replace(",", "")
    elif re.fullmatch(r"[0-9]{1,3}(?:\.[0-9]{3})+", text):
        digits = text.replace(".", "")
    else:
        raise LabelError("ambiguous_or_invalid_money", field)
    if len(digits) > 12:
        raise LabelError("money_out_of_range", field)
    return Decimal(digits).quantize(Decimal("0.01"))


def convert(record):
    """Reject the entire receipt if any menu row is incomplete; never drop rows."""
    try:
        menu = record["ground_truth"]["gt_parse"]["menu"]
    except (KeyError, TypeError):
        raise LabelError("missing_menu", "ground_truth.gt_parse.menu") from None
    menu = [menu] if isinstance(menu, dict) else menu
    if not isinstance(menu, list) or not menu:
        raise LabelError("invalid_menu", "menu")
    lines = []
    for index, row in enumerate(menu):
        try:
            if not isinstance(row, dict):
                raise LabelError("invalid_menu_row", "menu")
            for field in ("nm", "cnt", "unitprice", "price"):
                if field not in row or row[field] is None or row[field] == "":
                    raise LabelError("missing_field", field)
            if any(isinstance(value, (dict, list)) for value in row.values()):
                raise LabelError("unsupported_nested_menu", "menu")
            item = row["nm"]
            if not isinstance(item, str) or not item.strip() or any(ord(c) < 32 and c not in "\n\r\t" for c in item):
                raise LabelError("invalid_item_name", "nm")
            count, rate, amount = quantity(row["cnt"]), money(row["unitprice"], "unitprice"), money(row["price"], "price")
            if Decimal(count) * rate != amount:
                raise LabelError("arithmetic_mismatch", "cnt/unitprice/price")
            lines.append(InvoiceLine(item=item, quantity=count, rate=rate, amount=amount))
        except LabelError as exc:
            exc.row_index = index
            raise
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    writer.writerows((line.item, line.quantity, f"{line.rate:.2f}", f"{line.amount:.2f}") for line in lines)
    return stream.getvalue(), lines


def messages(image, csv_text):
    """The exact corrected multimodal representation used by make_hf_dataset."""
    return {"images": [image], "prompt": [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": SOURCE_INSTRUCTION}]}],
        "completion": [{"role": "assistant", "content": [{"type": "text", "text": csv_text}]}]}


# Download
import argparse
import gzip
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

from src.utils.paths import project_path


def verify_source(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "source_manifest.json").read_text())
    if manifest.get("repo_id") != REPO_ID or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", "")):
        raise ValueError("Invalid CORD source identity/revision")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Source manifest must identify exactly six files")
    for name in FILES:
        path, info = source / name, manifest["files"][name]
        if not path.resolve().is_relative_to(source) or not path.is_file():
            raise ValueError(f"Missing or unsafe source file: {name}")
        if path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
            raise ValueError(f"Source checksum/size mismatch: {name}")
    return manifest


def download(source, revision, retries=3):
    # No imports or network activity occur until this manual command is called.
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import enable_progress_bars
    if retries < 1:
        raise ValueError("Retries must be positive")
    enable_progress_bars()
    info = HfApi(token=False).dataset_info(REPO_ID, revision=revision, files_metadata=True, timeout=60)
    if not re.fullmatch(r"[a-f0-9]{40}", info.sha or ""):
        raise ValueError("Hub did not resolve a full commit SHA")
    siblings = {file.rfilename: file for file in info.siblings}
    metadata = {}
    for name in FILES:
        file = siblings.get(name)
        if file is None or file.lfs is None or file.size is None:
            raise ValueError(f"Missing file or cryptographic Hub metadata: {name}")
        # BlobLfsInfo is an attribute object in supported huggingface_hub releases.
        digest = file.lfs.sha256
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Missing Hub SHA256: {name}")
        metadata[name] = {"size": file.size, "sha256": digest,
                          "url": f"https://huggingface.co/datasets/{REPO_ID}/resolve/{info.sha}/{name}"}
    source = Path(source).resolve()
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".cache").resolve().is_relative_to(source):
        raise ValueError("Source cache path escapes the download directory")
    manifest = {"repo_id": REPO_ID, "requested_revision": revision, "revision": info.sha, "files": metadata}
    manifest_path = source / "source_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("revision") != info.sha or old.get("files") != metadata:
            raise ValueError("Source directory belongs to another revision; choose a new --source-dir")
    else:
        write_json(manifest_path, manifest)
    needed = sum(value["size"] for name, value in metadata.items() if not (source / name).exists())
    if shutil.disk_usage(source).free < 2 * needed + 512 * 1024 * 1024:
        raise ValueError("Insufficient disk space: reserve twice remaining download bytes plus 512 MiB")
    print(f"CORD revision {info.sha}; source {source}; remaining {needed:,} bytes", flush=True)
    for index, name in enumerate(FILES, 1):
        target = source / name
        print(f"[{index}/6] {name}", flush=True)
        if target.is_symlink():
            raise ValueError(f"Refusing a symlinked source download: {name}")
        if target.exists():
            if not target.resolve().is_relative_to(source) or sha(target) != metadata[name]["sha256"]:
                raise ValueError(f"Existing source differs; refusing overwrite: {name}")
            print("  verified existing file; reused", flush=True)
            continue
        for attempt in range(retries):
            try:
                downloaded = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=name,
                    revision=info.sha, local_dir=source, cache_dir=source / ".cache/hub", token=False))
                if downloaded.resolve() != target.resolve():
                    raise ValueError("Unexpected Hub download destination")
                break
            except Exception as exc:
                if attempt + 1 == retries:
                    raise RuntimeError(f"Download failed for {name} ({type(exc).__name__}); rerun to resume") from exc
                time.sleep(min(2 ** attempt, 8))
        if target.stat().st_size != metadata[name]["size"] or sha(target) != metadata[name]["sha256"]:
            raise ValueError(f"Downloaded integrity failure: {name}; no automatic deletion")
        if name.endswith(".zip"):
            with zipfile.ZipFile(target) as archive:
                if not archive.infolist():
                    raise ValueError(f"Empty ZIP: {name}")
        else:
            with gzip.open(target, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass  # Validate the gzip footer/CRC without extracting files.
    verify_source(source)
    print("All six source files verified; no receipt images extracted", flush=True)


def cleanup(source, dataset):
    """Explicit user-invoked deletion of our three source ZIPs only, after validation."""
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    manifest = verify_source(source)
    validate_dataset(dataset)
    selected = json.loads((dataset / "manifest.json").read_text())
    if selected["source"] != manifest:
        raise ValueError("Prepared dataset does not belong to this source manifest")
    for name in FILES:
        if name.endswith(".zip"):
            (source / name).unlink()
            print(f"Removed verified temporary archive: {name}", flush=True)
    print("Annotations, source manifest and all caches retained. Source preparation now requires re-download of ZIPs.", flush=True)


def download_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(".cache/cord_source"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cleanup-archives", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cord_sft_qlora"))
    args = parser.parse_args()
    if args.cleanup_archives:
        cleanup(project_path(args.source_dir), project_path(args.dataset_root))
    else:
        download(project_path(args.source_dir), args.revision, args.retries)


# Prepare
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
        "See docs/trainer_sft_qlora_invoice_extraction_README.md for schema compatibility and Phase 10B restrictions.\n", encoding="utf-8")
    for name in ("selection_report.json", "rejection_report.json", "README.md"):
        file_hashes[name] = sha(output / name)
    write_json(output / "manifest.json", {"source": report["source"], "seed": seed, "counts": dict(counts),
        "entries": entries, "file_sha256": file_hashes, "human_verified": False,
        "raw_schema": "existing SFT image/csv/target_csv/rows + provenance",
        "hf_schema": "images + user(image marker, shared instruction) + assistant(CSV text)"})
    validation = validate_dataset(output, expected_counts=counts)
    write_json(output / "validation_report.json", validation)
    print(json.dumps(validation, indent=2), flush=True)
    return validation


def prepare_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(".cache/cord_source"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cord_sft_qlora"))
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


# Validate
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
        allowed_root = root
        if relative == "README.md" and not file.is_file():
            repository = project_path(".").resolve()
            if root.is_relative_to(repository / "data"):
                allowed_root = repository / "docs"
                document = "_".join(root.relative_to(repository).parts) + "_README.md"
                file = (allowed_root / document).resolve()
            # Documentation may be moved or removed during repository cleanup.
            # Every annotation, split manifest and selection report stays mandatory.
            if not file.is_file():
                continue
        require(file.is_relative_to(allowed_root) and file.is_file() and sha(file) == digest, "Metadata file checksum mismatch")
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


def validate_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cord_sft_qlora"))
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


# Dataset utils
import json
from pathlib import Path

from src.components.inference.parsing import CSV_FIELDS, read_rows
from src.components.prompts.extraction import INSTRUCTION as SOURCE_INSTRUCTION


def read_records(dataset_root: str | Path, split: str, *, manifest: str | Path | None = None) -> list[dict]:
    """Read one synthetic split and check image, CSV and JSONL agree exactly.

    The Optum evaluation images are never read or copied by this routine.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Split must be train or validation")
    root = Path(dataset_root).resolve()
    if root.name != "synthetic":
        raise ValueError("Dataset root must be a directory named 'synthetic'")
    manifest = Path(manifest).resolve() if manifest is not None else root / split / "annotations.jsonl"
    if manifest.parent != root / split:
        raise ValueError("Manifest must be inside its synthetic dataset split")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    records = []
    for lineno, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        img_rel = Path(record["image"])
        csv_rel = Path(record["csv"])
        if (img_rel.is_absolute() or csv_rel.is_absolute() or
            img_rel.parts[:2] != (split, "images") or
            csv_rel.parts[:2] != (split, "labels") or
            ".." in img_rel.parts or ".." in csv_rel.parts):
            raise ValueError(f"Unsafe/non-synthetic path at {manifest}:{lineno}")
        img_path, label_path = root / img_rel, root / csv_rel
        if not img_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing synthetic pair for {img_rel}")
        text = label_path.read_text(encoding="utf-8")
        if text.replace("\r\n", "\n") != record["target_csv"].replace("\r\n", "\n"):
            raise ValueError(f"JSONL vs CSV label mismatch: {csv_rel}")
        csv_rows = read_rows(text)
        if len(csv_rows) != record["rows"]:
            raise ValueError(f"Row count mismatch: {csv_rel}")
        if img_path.stem != label_path.stem:
            raise ValueError(f"Image and label stems differ: {img_rel}")
        records.append({"image_path": img_path, "csv_path": label_path, "target_csv": text, "rows": csv_rows})
    if not records:
        raise ValueError(f"Empty {split} dataset")
    return records



def make_hf_dataset(records: list[dict]):
    """Convert invoice images and CSV labels into a TRL VLM dataset."""
    from datasets import Dataset, Image, Sequence

    data = []

    for row in records:
        data.append({
            "images": [str(row["image_path"])],
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": SOURCE_INSTRUCTION},
                    ],
                }
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": row["target_csv"]},
                    ],
                }
            ],
        })

    return Dataset.from_list(data).cast_column(
        "images", Sequence(Image())
    )


# Training data
import json
from pathlib import Path
import re


EXPECTED_COUNTS = {"train": 161, "validation": 20, "test": 19}
INSTRUCTION_VERSION = "cord-general-csv-v1"
INSTRUCTION = (
    "Read the receipt or invoice image and extract all itemized rows into CSV with exactly "
    "these columns: Item,Quantity,Rate,Amount. Return only CSV text. Preserve the item names "
    "and their original order. Do not include subtotals, taxes or grand totals as item rows. "
    "Do not invent values, include explanations or convert currencies."
)


def read_split(root, split, manifest):
    # Reject test BEFORE resolving or opening any requested manifest.
    if split not in {"train", "validation"}:
        raise ValueError("CORD training can only read train or validation")
    root = Path(root).resolve()
    filename = "training.jsonl" if split == "train" else "validation.jsonl"
    file = (root / split / filename).resolve()
    if not file.is_relative_to(root / split):
        raise ValueError("Unsafe split manifest")
    if sha(file) != manifest["file_sha256"][f"{split}/{filename}"]:
        raise ValueError("CORD split manifest checksum mismatch")
    annotations_file = (root / split / "annotations.jsonl").resolve()
    if not annotations_file.is_relative_to(root / split):
        raise ValueError("Unsafe annotations path")
    if sha(annotations_file) != manifest["file_sha256"][f"{split}/annotations.jsonl"]:
        raise ValueError("CORD source annotation checksum mismatch")
    raw, annotations = read_jsonl(file), read_jsonl(annotations_file)
    if len(raw) != EXPECTED_COUNTS[split] or len(annotations) != len(raw):
        raise ValueError("Incorrect CORD training/validation count")
    entries = {row["source_id"]: row for row in manifest["entries"]}
    result = []
    for record, annotation in zip(raw, annotations):
        entry = entries[record["source_id"]]
        if entry["source_split"] != split or record["source_split"] != split:
            raise ValueError("Original CORD split not preserved")
        name = annotation["image_file"]
        if record["source_id"] != name or record["source_filename"] != name:
            raise ValueError("CORD annotation/image identity mismatch")
        image = within(root, record["image"], (split, "images"))
        csv = within(root, record["csv"], (split, "labels"))
        if image.name != name or image.stem != csv.stem:
            raise ValueError("CORD image/CSV pairing mismatch")
        if entry["image"] != record["image"] or entry["csv"] != record["csv"]:
            raise ValueError("CORD manifest pair mismatch")
        provenance = record["provenance"]
        if (provenance["source_file"] != f"cord-{split}.jsonl.gz"
                or type(provenance["source_line"]) is not int
                or not 1 <= provenance["source_line"] <= SOURCE_COUNTS[split]
                or provenance["source_line"] != entry["source_line"]
                or fingerprint(annotation) != entry["annotation_sha256"]
                or provenance["annotation_sha256"] != entry["annotation_sha256"]):
            raise ValueError("CORD provenance mismatch")
        text, rows = convert(annotation)  # Existing explicit-rate/Decimal rules, unchanged.
        if (not rows or text != csv.read_text(encoding="utf-8") or text != record["target_csv"]
                or len(rows) != record["rows"] or len(rows) != entry["rows"]):
            raise ValueError("Empty/changed CSV or missing item rows")
        if any(record[key] != value for key, value in messages(record["image"], text).items()):
            raise ValueError("Changed Phase 10A multimodal messages")
        image_hash, pixel_hash = inspect_image(image.read_bytes(), annotation)
        if (image_hash != entry["image_sha256"] or pixel_hash != entry["pixel_sha256"]
                or sha(csv) != entry["csv_sha256"]):
            raise ValueError("CORD image/label fingerprint mismatch")
        result.append({"image_path": image, "csv_path": csv, "target_csv": text,
                       "rows": rows, "image_sha256": image_hash, "pixel_sha256": pixel_hash,
                       "source_id": name})
    return result


def load_training_data(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    source = manifest["source"]
    if (source.get("repo_id") != REPO_ID or set(source.get("files", {})) != set(FILES)
            or not re.fullmatch(r"[a-f0-9]{40}", source.get("revision", ""))):
        raise ValueError("Invalid canonical CORD source identity/revision")
    if configured_counts(manifest["counts"]) != EXPECTED_COUNTS:
        raise ValueError("Phase 10B requires the approved 161/20/19 split")
    entries = manifest["entries"]
    ids = [entry["source_id"] for entry in entries]
    if len(ids) != sum(EXPECTED_COUNTS.values()) or len(set(ids)) != len(ids):
        raise ValueError("Duplicate/incomplete CORD manifest inventory")
    for split, count in EXPECTED_COUNTS.items():
        if sum(entry["source_split"] == split for entry in entries) != count:
            raise ValueError("CORD manifest split count mismatch")
    for field in ("image_sha256", "pixel_sha256"):
        if len({entry[field] for entry in entries}) != len(entries):
            raise ValueError("Duplicate image or cross-split leakage in manifest")
    records = {split: read_split(root, split, manifest) for split in ("train", "validation")}
    metadata = {"counts": manifest["counts"], "source": manifest["source"],
                "dataset_manifest_sha256": sha(root / "manifest.json"),
                "split_sha256": {split: sha(root / split / filename) for split, filename in
                                 (("train", "training.jsonl"), ("validation", "validation.jsonl"))},
                "test_artifacts_opened": False}
    return records, metadata


def make_dataset(records, instruction):
    # Retain the corrected HF Image feature and original assistant CSV verbatim.
    dataset = make_hf_dataset(records)
    return dataset.map(lambda example: {"prompt": [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": instruction}]}]})


class BoundedCollator:
    """Audit through the successful SFT collator; reject rather than truncate."""
    def __init__(self, processor, max_tokens, delegate=None):
        if delegate is None:
            from trainer.sft_qlora_invoice_extraction.runtime import CheckedInvoiceCollator
            delegate = CheckedInvoiceCollator(processor)
        self.delegate, self.max_tokens = delegate, max_tokens

    def __call__(self, examples):
        batch = self.delegate(examples)
        if batch["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Receipt plus full CSV exceeds --max-sequence-length; reduce --max-pixels "
                             "or raise the token budget. No image/CSV truncation is allowed")
        return batch


def main():
    import sys
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=('download', 'prepare', 'validate'))
    args = cli.parse_args(sys.argv[1:2])
    remaining = sys.argv[2:]
    original = sys.argv
    try:
        sys.argv = [original[0], *remaining]
        commands = {'download': 'download_main', 'prepare': 'prepare_main', 'validate': 'validate_main'}
        globals()[commands[args.action]]()
    finally:
        sys.argv = original



if __name__ == "__main__":
    main()

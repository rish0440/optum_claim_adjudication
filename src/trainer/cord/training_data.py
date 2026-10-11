"""CORD train/validation access only; never open held-out split artifacts."""
import json
from pathlib import Path
import re

from .prepare import inspect_image
from .schema import (FILES, REPO_ID, SOURCE_COUNTS, configured_counts, convert,
                     fingerprint, messages, read_jsonl, sha)
from .validate import within

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
    from src.trainer.sft.training_data import make_hf_dataset
    # Retain the corrected HF Image feature and original assistant CSV verbatim.
    dataset = make_hf_dataset(records)
    return dataset.map(lambda example: {"prompt": [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": instruction}]}]})


class BoundedCollator:
    """Audit through the successful SFT collator; reject rather than truncate."""
    def __init__(self, processor, max_tokens, delegate=None):
        if delegate is None:
            from src.trainer.sft.train.runtime import CheckedInvoiceCollator
            delegate = CheckedInvoiceCollator(processor)
        self.delegate, self.max_tokens = delegate, max_tokens

    def __call__(self, examples):
        batch = self.delegate(examples)
        if batch["input_ids"].shape[1] > self.max_tokens:
            raise ValueError("Receipt plus full CSV exceeds --max-sequence-length; reduce --max-pixels "
                             "or raise the token budget. No image/CSV truncation is allowed")
        return batch

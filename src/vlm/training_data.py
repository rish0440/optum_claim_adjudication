"""Phase 3: safe synthetic-only image/CSV training-data access.

This module intentionally imports no Torch/Transformers so validation and
packaging work in the normal CPU-only Windows virtual environment.
"""

import csv
import io
import json
from pathlib import Path

CSV_FIELDS = ("Item", "Quantity", "Rate", "Amount")
INSTRUCTION = (
    "Extract every line item visible in this hospital invoice. "
    "Return ONLY CSV text with the exact header Item,Quantity,Rate,Amount. "
    "Preserve row order and item names. Copy numeric amounts without currency "
    "symbols or thousands separators. If the rate column is not printed, "
    "leave Rate empty; do not calculate or guess missing rates. "
    "Do not include totals, explanations, or Markdown fences."
)


def read_rows(text: str) -> list[dict[str, str]]:
    """Parse strict four-column CSV, retaining intentionally empty Rate cells."""
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff").strip()))
    if reader.fieldnames is None or tuple(reader.fieldnames) != CSV_FIELDS:
        raise ValueError(f"CSV must start with {','.join(CSV_FIELDS)}")
    rows = []
    for line in reader:
        if None in line or any(key not in line for key in CSV_FIELDS):
            raise ValueError("CSV row has extra/missing columns")
        if not line["Item"].strip() or not line["Quantity"].strip() or not line["Amount"].strip():
            raise ValueError("Missing Item, Quantity or Amount")
        rows.append({k: line[k].strip() for k in CSV_FIELDS})
    if not rows:
        raise ValueError("CSV has no invoice items")
    return rows


def read_records(dataset_root: str | Path, split: str) -> list[dict]:
    """Read one synthetic split and check image, CSV and JSONL agree exactly.

    The Optum evaluation images are never read or copied by this routine.
    """
    if split not in {"train", "validation"}:
        raise ValueError("Split must be train or validation")
    root = Path(dataset_root).resolve()
    if root.name != "synthetic":
        raise ValueError("Dataset root must be a directory named 'synthetic'")
    manifest = root / split / "annotations.jsonl"
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
                        {"type": "text", "text": INSTRUCTION},
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


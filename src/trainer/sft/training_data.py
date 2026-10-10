"""Phase 3: safe synthetic-only image/CSV training-data access.

This module intentionally imports no Torch/Transformers so validation and
packaging work in the normal CPU-only Windows virtual environment.
"""

import json
from pathlib import Path

from src.inference.parsing import CSV_FIELDS, read_rows
from src.prompts.extraction import INSTRUCTION


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


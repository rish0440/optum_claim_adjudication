"""Build one uploadable ZIP containing only Phase 3 code and synthetic data.

Usage: python -m scripts.prepare_colab_dataset
The ZIP explicitly excludes original Optum images, evaluation labels and CORD.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from src.vlm.training_data import read_records

ROOT = Path(__file__).resolve().parents[1]
CODE = (
    "src/__init__.py", "src/vlm/__init__.py", "src/vlm/training_data.py",
    "src/vlm/train_lora.py", "src/vlm/inference.py", "src/vlm/evaluate.py",
    "requirements-colab.txt",
)


def package(output: Path, root: Path = ROOT) -> dict:
    data_root = root / "data" / "synthetic"
    splits = {name: read_records(data_root, name) for name in ("train", "validation")}
    train = {record["image_path"].name for record in splits["train"]}
    valid = {record["image_path"].name for record in splits["validation"]}
    if train & valid:
        raise ValueError("Training and validation image overlap")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    count = 0
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for rel in CODE:
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.write(path, rel)
            count += 1
        for split, records in splits.items():
            manifest = data_root / split / "annotations.jsonl"
            archive.write(manifest, manifest.relative_to(root).as_posix())
            count += 1
            for item in records:
                for path in (item["image_path"], item["csv_path"]):
                    archive.write(path, path.relative_to(root).as_posix())
                    count += 1
        info = data_root / "dataset_info.json"
        if info.exists():
            archive.write(info, info.relative_to(root).as_posix())
            count += 1
    with ZipFile(output) as archive:
        names = archive.namelist()
        forbidden = [n for n in names if n.startswith(("data/invoices/", "data/evaluation/", "data/cord/"))]
        if forbidden:
            raise ValueError(f"Held-out/third-party dataset leaked into ZIP: {forbidden}")
        if archive.testzip():
            raise ValueError("Corrupted ZIP contents")
    result = {"bundle": str(output), "train_images": len(train), "validation_images": len(valid),
              "files": count, "size_mb": round(output.stat().st_size / 1024**2, 2), "test_data_included": False}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "colab_training_bundle.zip")
    args = parser.parse_args()
    package(args.output)


if __name__ == "__main__":
    main()

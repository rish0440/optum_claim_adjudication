"""Build one uploadable ZIP containing only Phase 3 code and synthetic data.

Usage: python -m scripts.prepare_colab_dataset
The ZIP explicitly excludes original Optum images, evaluation labels and CORD.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from src.trainer.sft.training_data import read_records
from src.utils.paths import ROOT

CODE = (
    "src/__init__.py", "src/trainer/__init__.py", "src/trainer/sft/__init__.py",
    "src/trainer/sft/training_data.py", "src/trainer/sft/config.py",
    "src/trainer/sft/train_lora.py", "src/trainer/sft/evaluate.py",
    "src/trainer/sft/train/__init__.py", "src/trainer/sft/train/train_qlora.py",
    "src/trainer/sft/train/runtime.py", "src/trainer/sft/train/checkpoints.py",
    "src/trainer/sft/annotation/__init__.py", "src/trainer/sft/annotation/validation.py",
    "src/schemas.py", "src/config/settings.py", "src/config/default.yaml",
    "src/inference/__init__.py", "src/inference/local.py", "src/inference/parsing.py",
    "src/prompts/__init__.py", "src/prompts/extraction.py",
    "src/config/__init__.py", "src/config/models.py",
    "src/utils/__init__.py", "src/utils/paths.py",
    # Retain old notebook/CLI paths, backed by the canonical implementations.
    "src/vlm/__init__.py", "src/vlm/training_data.py", "src/vlm/train_lora.py",
    "src/vlm/inference.py", "src/vlm/evaluate.py",
    "requirements-colab.txt",
    "requirements-sft.txt", "requirements-sft-constraints.txt",
)


def package(output: Path, root: Path = ROOT, *, dataset_root: Path | None = None) -> dict:
    data_root = dataset_root or root / "data" / "synthetic"
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
            archive.write(manifest, (Path("data/synthetic") / manifest.relative_to(data_root)).as_posix())
            count += 1
            for item in records:
                for path in (item["image_path"], item["csv_path"]):
                    archive.write(path, (Path("data/synthetic") / path.relative_to(data_root)).as_posix())
                    count += 1
        info = data_root / "dataset_info.json"
        if info.exists():
            archive.write(info, "data/synthetic/dataset_info.json")
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
    from src.config.settings import get_settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "colab_training_bundle.zip")
    parser.add_argument("--dataset-root", type=Path, default=get_settings().paths.synthetic_dataset)
    args = parser.parse_args()
    package(args.output, dataset_root=args.dataset_root)


if __name__ == "__main__":
    main()

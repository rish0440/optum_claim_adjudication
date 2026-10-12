"""Package the active CORD LoRA trainer with train/validation data only.

Usage: python -m scripts.prepare_colab_dataset
Original Optum invoices and held-out CORD test files are never included.
"""
import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from trainer.sft_qlora_invoice_extraction.data_prep import load_training_data
from src.utils.paths import ROOT, project_path

CODE = (
    "src/__init__.py", "src/components/__init__.py", "trainer/__init__.py",
    "src/schema/__init__.py", "src/config/__init__.py", "src/config/settings.py",
    "src/config/default.yaml", "src/config/models.py",
    "src/components/inference/__init__.py", "src/components/inference/local.py", "src/components/inference/parsing.py",
    "src/components/prompts/__init__.py", "src/components/prompts/extraction.py",
    "src/utils/__init__.py", "src/utils/paths.py", "requirements.txt",
) + tuple(path.relative_to(ROOT).as_posix()
          for path in sorted((ROOT / "trainer/sft_qlora_invoice_extraction").glob("*.py")))


def package(output: Path, root: Path = ROOT, *, dataset_root: Path | None = None) -> dict:
    data_root = Path(dataset_root or root / "data/cord_sft_qlora").resolve()
    splits, audit = load_training_data(data_root)
    train = {record["image_path"].name for record in splits["train"]}
    valid = {record["image_path"].name for record in splits["validation"]}
    if train & valid or audit["test_artifacts_opened"]:
        raise ValueError("Training/validation overlap or held-out test access")
    files = [(root / name, name) for name in CODE]
    files.append((data_root / "manifest.json", "data/cord_sft_qlora/manifest.json"))
    for split, records in splits.items():
        name = "training.jsonl" if split == "train" else "validation.jsonl"
        for metadata in ("annotations.jsonl", name):
            files.append((data_root / split / metadata, f"data/cord_sft_qlora/{split}/{metadata}"))
        for record in records:
            for path in (record["image_path"], record["csv_path"]):
                files.append((path, "data/cord_sft_qlora/" + path.relative_to(data_root).as_posix()))
    for source, name in files:
        if not source.is_file():
            raise FileNotFoundError(source)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for source, name in files:
            archive.write(source, name)
    with ZipFile(output) as archive:
        if any(name.startswith(("data/assignment_dataset/", "data/cord_sft_qlora/test/"))
               for name in archive.namelist()):
            raise ValueError("Held-out evaluation files leaked into ZIP")
        if archive.testzip():
            raise ValueError("Corrupted ZIP contents")
    result = {"bundle": str(output), "train_images": len(train), "validation_images": len(valid),
              "files": len(files), "size_mb": round(output.stat().st_size / 1024**2, 2),
              "test_data_included": False}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "colab_training_bundle.zip")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/cord_sft_qlora")
    args = parser.parse_args()
    package(args.output, dataset_root=project_path(args.dataset_root))


if __name__ == "__main__":
    main()

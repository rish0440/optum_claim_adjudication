"""Read held-out CORD only and reuse existing CPU adapter verification."""
import json
from pathlib import Path
import re

from src.config.models import sft_output_root
from src.trainer.sft.train.checkpoints import latest_adapter, verify_adapter
from src.utils.paths import project_path
from .prepare import inspect_image
from .schema import FILES, REPO_ID, SOURCE_COUNTS, convert, fingerprint, messages, read_jsonl, sha
from .train import MODEL_ID, OUTPUT, verify_saved
from .training_data import EXPECTED_COUNTS
from .validate import within


def resolve_checkpoints(baseline_root=None, cord_root=None):
    baseline_root = Path(baseline_root or sft_output_root()).resolve()
    cord_root = project_path(cord_root or OUTPUT).resolve()
    paths = {"baseline": latest_adapter(baseline_root), "cord_sft": latest_adapter(cord_root)}
    metadata = {}
    for mode, path in paths.items():
        checked = verify_adapter(path, MODEL_ID)
        config_name = "training_config.json" if mode == "cord_sft" else "run_config.json"
        config = json.loads((path.parent / config_name).read_text(encoding="utf-8"))
        result = json.loads((path.parent / "training_result.json").read_text(encoding="utf-8"))
        metrics = result["metrics"]
        if config.get("model_id") != MODEL_ID or metrics["global_steps_completed"] < 1:
            raise ValueError("Adapter backbone/completed optimizer steps mismatch")
        import math
        if not math.isfinite(metrics["training_loss"]) or not math.isfinite(metrics["evaluation"]["eval_loss"]):
            raise ValueError("Adapter lacks finite training/validation loss")
        if mode == "cord_sft":
            verified = verify_saved(path, cord_root)
            if verified["smoke"] or result.get("smoke"):
                raise ValueError("Smoke adapter cannot be evaluated as the CORD trained model")
            if metrics["global_steps_completed"] != config["expected_optimizer_steps"]:
                raise ValueError("CORD training did not complete its requested optimizer steps")
        metadata[mode] = {"path": str(path), "model_id": MODEL_ID, "run_id": path.parent.name,
                          "adapter_sha256": checked["sha256"], "tensor_count": checked["tensor_count"],
                          "training_config": config}
    if paths["baseline"] == paths["cord_sft"] or metadata["baseline"]["adapter_sha256"] == metadata["cord_sft"]["adapter_sha256"]:
        raise ValueError("Comparison requires two different verified adapters")
    config = metadata["cord_sft"]["training_config"]
    instruction = config["instruction"]
    if hashlib_prompt(instruction) != config["instruction_sha256"]:
        raise ValueError("Recorded training prompt checksum mismatch")
    revision = config.get("resolved_model_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("CORD training must record a resolved base revision for fair evaluation")
    baseline_revision = metadata["baseline"]["training_config"].get("resolved_model_revision")
    if baseline_revision and baseline_revision != revision:
        raise ValueError("Training metadata records conflicting base revisions")
    return metadata, instruction, revision


def hashlib_prompt(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_test(root, adapters):
    root = Path(root).resolve()
    manifest_file = root / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    source = manifest["source"]
    if (manifest["counts"] != EXPECTED_COUNTS or source["repo_id"] != REPO_ID
            or set(source["files"]) != set(FILES)
            or not re.fullmatch(r"[a-f0-9]{40}", source["revision"])):
        raise ValueError("Wrong CORD source/split identity")
    training = adapters["cord_sft"]["training_config"]["dataset"]
    if (training["dataset_manifest_sha256"] != sha(manifest_file)
            or training["counts"] != manifest["counts"] or training["source"] != source
            or training.get("test_artifacts_opened") is not False):
        raise ValueError("CORD training provenance does not match the held-out dataset")
    # Metadata comparison only: never open train/validation samples or labels.
    for split, filename in (("train", "training.jsonl"), ("validation", "validation.jsonl")):
        if training["split_sha256"][split] != manifest["file_sha256"][f"{split}/{filename}"]:
            raise ValueError("Train/validation fingerprint differs from recorded training")
    for key in ("source_id", "image_sha256", "pixel_sha256"):
        if len({entry[key] for entry in manifest["entries"]}) != sum(EXPECTED_COUNTS.values()):
            raise ValueError("Duplicate/missing dataset identity or image fingerprint")
    for split, count in EXPECTED_COUNTS.items():
        if sum(entry["source_split"] == split for entry in manifest["entries"]) != count:
            raise ValueError("Manifest split separation/count mismatch")
    files = [(root / "test" / name).resolve() for name in ("test.jsonl", "annotations.jsonl")]
    for file in files:
        if not file.is_relative_to(root / "test") or sha(file) != manifest["file_sha256"][f"test/{file.name}"]:
            raise ValueError("Test manifest/annotation checksum mismatch")
    if sha(files[0]) in adapters["baseline"]["training_config"].get("manifest_sha256", {}).values():
        raise ValueError("Baseline training metadata includes the held-out test manifest")
    records, annotations = (read_jsonl(file) for file in files)
    if len(records) != EXPECTED_COUNTS["test"] or len(annotations) != len(records):
        raise ValueError("Exactly 19 held-out test pairs are required")
    entries = {entry["source_id"]: entry for entry in manifest["entries"]}
    result, seen = [], set()
    for record, annotation in zip(records, annotations):
        name, provenance = record["source_id"], record["provenance"]
        entry = entries[name]
        if name in seen or record["source_split"] != "test" or entry["source_split"] != "test":
            raise ValueError("Duplicate/wrong-split test record")
        seen.add(name)
        if (annotation["image_file"] != name or record["source_filename"] != name
                or fingerprint(annotation) != provenance["annotation_sha256"]
                or provenance["annotation_sha256"] != entry["annotation_sha256"]
                or provenance["source_file"] != "cord-test.jsonl.gz"
                or type(provenance["source_line"]) is not int
                or not 1 <= provenance["source_line"] <= SOURCE_COUNTS["test"]
                or provenance["source_line"] != entry["source_line"]):
            raise ValueError("Test source provenance mismatch")
        image = within(root, record["image"], ("test", "images"))
        csv = within(root, record["csv"], ("test", "labels"))
        text, rows = convert(annotation)
        hashes = inspect_image(image.read_bytes(), annotation)
        if (image.name != name or image.stem != csv.stem
                or entry["image"] != record["image"] or entry["csv"] != record["csv"]
                or hashes != (entry["image_sha256"], entry["pixel_sha256"])
                or sha(csv) != entry["csv_sha256"] or csv.read_text(encoding="utf-8") != text
                or record["target_csv"] != text or record["rows"] != len(rows) or entry["rows"] != len(rows)
                or any(record[key] != value for key, value in messages(record["image"], text).items())):
            raise ValueError("Test image/CSV/label fingerprint mismatch")
        result.append({"image_path": image, "image": name, "ground_truth_csv": text,
                       "image_sha256": hashes[0], "pixel_sha256": hashes[1]})
    return result, {"dataset_manifest_sha256": sha(manifest_file), "test_split_sha256": sha(files[0]),
                    "source": source, "counts": manifest["counts"], "train_samples_opened": False}

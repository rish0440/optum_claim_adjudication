"""Dataset integrity and reproducibility checks; no GPU or model downloads."""
import csv
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.config.settings import get_settings
from src.schemas import InvoiceLine
from src.trainer.sft.annotation.generator import generate_dataset, lines_as_csv
from src.trainer.sft.annotation.validation import verify_dataset


class SyntheticAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name) / "synthetic"
        shutil.copytree(get_settings().paths.synthetic_dataset, self.base)

    def records(self, split):
        return [json.loads(line) for line in (self.base / split / "annotations.jsonl").read_text().splitlines()]

    def write_records(self, split, records):
        (self.base / split / "annotations.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    def change_csv(self, change):
        records = self.records("train")
        rows = list(csv.reader(io.StringIO(records[0]["target_csv"])))
        change(rows)
        stream = io.StringIO(newline="")
        csv.writer(stream, lineterminator="\n").writerows(rows)
        records[0]["target_csv"] = stream.getvalue()
        (self.base / records[0]["csv"]).write_text(stream.getvalue(), encoding="utf-8")
        self.write_records("train", records)

    def test_exact_phase4_counts_and_hf_decoding(self):
        report = verify_dataset(self.base, expected_count=10, expected_train=8,
                                expected_validation=2, check_hf=True)
        self.assertEqual(report["hf_examples"], {"train": 8, "validation": 2})
        self.assertEqual(report["unique_invoice_ids"], 10)
        self.assertEqual(report["unique_image_hashes"], 10)
        self.assertEqual(report["total_rows"], 64)

    def test_counts_are_enforced_independently_of_metadata(self):
        with self.assertRaisesRegex(ValueError, "Expected 11 total"):
            verify_dataset(self.base, expected_count=11)

    def test_duplicate_invoice_id_across_splits_is_rejected(self):
        train = self.records("train")[0]
        records = self.records("validation")
        duplicate = {**train, "image": train["image"].replace("train/", "validation/"),
                     "csv": train["csv"].replace("train/", "validation/")}
        shutil.copyfile(self.base / train["image"], self.base / duplicate["image"])
        shutil.copyfile(self.base / train["csv"], self.base / duplicate["csv"])
        records[0] = duplicate
        self.write_records("validation", records)
        with self.assertRaisesRegex(ValueError, "Duplicate invoice ID"):
            verify_dataset(self.base)

    def test_identical_pixels_under_different_names_are_rejected(self):
        train = self.records("train")[0]
        valid = self.records("validation")[0]
        shutil.copyfile(self.base / train["image"], self.base / valid["image"])
        with self.assertRaisesRegex(ValueError, "Duplicate image content"):
            verify_dataset(self.base)

    def test_orphan_image_is_rejected(self):
        image = self.base / self.records("train")[0]["image"]
        shutil.copyfile(image, image.with_name("orphan.jpg"))
        with self.assertRaisesRegex(ValueError, "Image inventory"):
            verify_dataset(self.base)

    def test_orphan_csv_is_rejected(self):
        label = self.base / self.records("train")[0]["csv"]
        shutil.copyfile(label, label.with_name("orphan.csv"))
        with self.assertRaisesRegex(ValueError, "Annotation inventory"):
            verify_dataset(self.base)

    def test_missing_image_is_rejected(self):
        (self.base / self.records("train")[0]["image"]).unlink()
        with self.assertRaises(FileNotFoundError):
            verify_dataset(self.base)

    def test_corrupt_image_is_rejected(self):
        (self.base / self.records("train")[0]["image"]).write_bytes(b"not a JPEG")
        with self.assertRaises(OSError):
            verify_dataset(self.base)

    def test_wrong_header_is_rejected(self):
        self.change_csv(lambda rows: rows[0].__setitem__(0, "Service"))
        with self.assertRaisesRegex(ValueError, "CSV must start"):
            verify_dataset(self.base)

    def test_arithmetic_errors_are_rejected_even_if_manifest_matches(self):
        self.change_csv(lambda rows: rows[1].__setitem__(3, str(Decimal(rows[1][3]) + 1)))
        with self.assertRaises(ValueError):
            verify_dataset(self.base)

    def test_invalid_jsonl_is_rejected(self):
        (self.base / "train/annotations.jsonl").write_text("{not json}\n")
        with self.assertRaises(json.JSONDecodeError):
            verify_dataset(self.base)

    def test_csv_escaping_preserves_commas_quotes_and_missing_rates(self):
        text = lines_as_csv([InvoiceLine(item='Dressing, "sterile"', quantity=2,
                                        rate=Decimal("12.50"), amount=Decimal("25.00")),
                             InvoiceLine(item="Lab test", quantity=1, rate=None, amount=Decimal("100"))])
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(rows[0]["Item"], 'Dressing, "sterile"')
        self.assertEqual(rows[0]["Rate"], "12.50")
        self.assertEqual(rows[1]["Rate"], "")

    def test_generator_rejects_heldout_overlap_before_writing(self):
        settings = get_settings().model_copy(deep=True)
        settings.paths.synthetic_dataset = settings.paths.evaluation_ground_truth / "synthetic"
        with patch("src.trainer.sft.annotation.generator.get_settings", return_value=settings):
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                generate_dataset(overwrite=True)

    def test_generation_is_byte_reproducible_with_fixed_seed(self):
        snapshots = []
        for run in ("first", "second"):
            settings = get_settings().model_copy(deep=True)
            output = Path(self.tmp.name) / run / "synthetic"
            settings.paths.synthetic_dataset = output
            with patch("src.trainer.sft.annotation.generator.get_settings", return_value=settings):
                report = generate_dataset(count=10, seed=42)
            self.assertEqual((report["train"], report["validation"]), (8, 2))
            snapshots.append({str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in output.rglob("*") if p.is_file()})
        self.assertEqual(snapshots[0], snapshots[1])


if __name__ == "__main__":
    unittest.main()

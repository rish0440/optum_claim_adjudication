"""CPU-only CORD fixtures: no dataset downloads, models or services."""
import copy
from decimal import Decimal
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from src.trainer.cord import schema
from src.trainer.cord.download import verify_source
from src.trainer.cord.prepare import eligible_candidates, eligibility, prepare, select
from src.trainer.cord.schema import (COUNTS, SOURCE_COUNTS, FILES, MANIFESTS, REPO_ID,
                                     LabelError, configured_counts, convert, messages, money, quantity, sha)
from src.trainer.cord.validate import validate_dataset


def receipt(menu):
    return {"image_file": "example.jpg", "width": 20, "height": 30,
            "ground_truth": {"gt_parse": {"menu": menu, "total": {"total_price": "999,999"}}}}


def item(name="Meal", qty="x 2", rate="10,000", amount="20,000"):
    return {"nm": name, "cnt": qty, "unitprice": rate, "price": amount}


class NormalizationTests(unittest.TestCase):
    def test_single_dictionary_and_total_not_a_row(self):
        text, rows = convert(receipt(item()))
        self.assertEqual(text, "Item,Quantity,Rate,Amount\nMeal,2,10000.00,20000.00\n")
        self.assertEqual(len(rows), 1)

    def test_repeated_rows_and_csv_escaping_preserved(self):
        name = 'Nasi, "special"\nlarge'
        text, rows = convert(receipt([item(name), item(name)]))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].item, name)
        self.assertIn('"Nasi, ""special""\nlarge"', text)
        self.assertEqual(text.count("20000.00"), 2)

    def test_quantities(self):
        for text in ("x 2", "2", "2X", "X2", " 2 x "):
            self.assertEqual(quantity(text), 2)
        for text in ("0", "-2", "2.5", "2x3", "x2x", "", 2):
            with self.assertRaises(LabelError):
                quantity(text)

    def test_indonesian_grouping_is_not_decimal(self):
        for text in ("75,000", "75.000", "75000"):
            self.assertEqual(money(text, "price"), Decimal("75000.00"))
        self.assertEqual(money("40,000.", "price"), Decimal("40000.00"))
        self.assertEqual(money("1.234.567", "price"), Decimal("1234567.00"))
        for text in ("75.00", "1.234,56", "Rp 75000", "NaN", "-1", "1e3", "12,34", "", 75000):
            with self.assertRaises(LabelError):
                money(text, "price")

    def test_missing_rate_never_derived_and_entire_receipt_rejected(self):
        bad = item()
        del bad["unitprice"]
        with self.assertRaises(LabelError) as rejected:
            convert(receipt([item(), bad]))
        self.assertEqual((rejected.exception.reason, rejected.exception.field, rejected.exception.row_index),
                         ("missing_field", "unitprice", 1))

    def test_inconsistent_amount_rejected(self):
        with self.assertRaises(LabelError) as rejected:
            convert(receipt(item(amount="10,000")))
        self.assertEqual(rejected.exception.reason, "arithmetic_mismatch")

    def test_zero_priced_item_and_original_annotation_preserved(self):
        source = receipt([item(), item("Complimentary", qty="1", rate="0", amount="0")])
        original = copy.deepcopy(source)
        text, rows = convert(source)
        self.assertEqual(source, original)
        self.assertEqual(rows[1].amount, Decimal("0.00"))
        self.assertIn("Complimentary,1,0.00,0.00", text)

    def test_nested_menu_not_silently_omitted(self):
        row = {**item(), "sub": [{"nm": "extra", "price": "1000"}]}
        with self.assertRaises(LabelError):
            convert(receipt(row))


class SplitCountTests(unittest.TestCase):
    def test_approved_target_is_exactly_200(self):
        approved = {"train": 161, "validation": 20, "test": 19}
        self.assertEqual(configured_counts(approved), approved)
        self.assertEqual(configured_counts(), approved)

    def test_invalid_counts_cannot_lower_total_or_remove_splits(self):
        for counts in ({"train": 160, "validation": 20, "test": 19},
                       {"train": 162, "validation": 20, "test": 19},
                       {"train": 180, "validation": 20, "test": 0},
                       {"train": 161, "validation": 20},
                       {"train": True, "validation": 20, "test": 179}):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                configured_counts(counts)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.output = self.root / "source", self.root / "cord_200"
        self.source.mkdir()
        # Small isolated fixtures; the production total remains exactly 200.
        total = patch.object(schema, "TOTAL_COUNT", 4)
        total.start()
        self.addCleanup(total.stop)
        for mapping, values in ((COUNTS, {"train": 2, "validation": 1, "test": 1}),
                                (SOURCE_COUNTS, {"train": 4, "validation": 2, "test": 2})):
            patched = patch.dict(mapping, values, clear=True)
            patched.start()
            self.addCleanup(patched.stop)
        self.write_source()

    def write_source(self):
        for offset, (split, count) in enumerate(SOURCE_COUNTS.items()):
            records = []
            with zipfile.ZipFile(self.source / f"cord-{split}-images.zip", "w") as archive:
                for index in range(count):
                    row = receipt([item(f"Meal {index}"), item("Complimentary", "1", "0", "0")])
                    row.update(image_file=f"cord-{split}-{index:04d}.jpg", width=20, height=30+index)
                    image = Image.new("RGB", (20, 30+index), (offset * 70, (index * 40) % 256, 120))
                    stream = io.BytesIO()
                    image.save(stream, format="JPEG")
                    archive.writestr(row["image_file"], stream.getvalue())
                    records.append(row)
            data = "".join(json.dumps(row) + "\n" for row in records).encode()
            (self.source / f"cord-{split}.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
        self.refresh_source()

    def refresh_source(self):
        manifest = {"repo_id": REPO_ID, "requested_revision": "fixture", "revision": "a" * 40,
                    "files": {name: {"size": (self.source / name).stat().st_size,
                                     "sha256": sha(self.source / name)} for name in FILES}}
        (self.source / "source_manifest.json").write_text(json.dumps(manifest))

    def rewrite_records(self, split, change):
        path = self.output / split / MANIFESTS[split]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        change(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["file_sha256"][str(path.relative_to(self.output))] = sha(path)
        (self.output / "manifest.json").write_text(json.dumps(manifest))

    def test_selection_reproducibility_without_changing_candidates(self):
        candidates, _, _, _ = eligible_candidates(self.source)
        original = copy.deepcopy(candidates)
        first = select(candidates["train"], 2, 42)
        second = select(list(reversed(candidates["train"])), 2, 42)
        self.assertEqual(first, second)
        self.assertEqual(candidates, original)

    def test_pairing_exact_counts_zero_rows_and_source_provenance(self):
        prepare(self.source, self.output)
        result = validate_dataset(self.output, source_dir=self.source)
        self.assertEqual(result["counts"], COUNTS)
        self.assertEqual(result["images"], 4)
        self.assertEqual(result["total_line_items"], 8)
        self.assertTrue(result["original_compressed_annotations_checked"])
        self.assertEqual(len(list(self.output.rglob("*.jpg"))), 4)
        with self.assertRaises(ValueError):
            prepare(self.source, self.output)

    def test_manifest_drives_validation_for_explicit_count_override(self):
        counts = {"train": 1, "validation": 2, "test": 1}
        prepare(self.source, self.output, counts=counts)
        result = validate_dataset(self.output, source_dir=self.source)
        self.assertEqual(result["counts"], counts)
        manifest = json.loads((self.output / "manifest.json").read_text())
        report = json.loads((self.output / "selection_report.json").read_text())
        self.assertEqual(manifest["counts"], counts)
        self.assertEqual(report["required_counts"], counts)
        self.assertEqual(report["selected_counts"], counts)
        with self.assertRaisesRegex(ValueError, "Configured split counts"):
            validate_dataset(self.output, expected_counts=dict(COUNTS))

    def test_161_20_19_preparation_pairs_exactly_200_without_split_leakage(self):
        counts = {"train": 161, "validation": 20, "test": 19}
        with patch.object(schema, "TOTAL_COUNT", 200), \
             patch.dict(SOURCE_COUNTS, {"train": 185, "validation": 22, "test": 19}, clear=True):
            self.write_source()  # Synthetic tiny JPEG/annotation fixtures only.
            prepare(self.source, self.output, seed=42, counts=counts)
            result = validate_dataset(self.output, source_dir=self.source)
            self.assertEqual(result["counts"], counts)
            self.assertEqual(result["images"], 200)
            self.assertEqual(result["annotations"], 200)
            self.assertEqual(result["cross_split_duplicates"], 0)
            self.assertEqual(len(list(self.output.rglob("*.jpg"))), 200)
            for split, count in counts.items():
                records = [json.loads(line) for line in (self.output / split / MANIFESTS[split]).read_text().splitlines()]
                self.assertEqual(len(records), count)
                self.assertTrue(all(row["source_split"] == split
                                    and row["source_filename"].startswith(f"cord-{split}-") for row in records))

    def test_sft_raw_and_multimodal_contract_uses_existing_builder(self):
        prepare(self.source, self.output)
        class Dataset(list):
            column_names = ["images", "prompt", "completion"]
        def builder(records):
            values = []
            for record in records:
                with Image.open(record["image_path"]) as image:
                    example = messages(str(record["image_path"]), record["target_csv"])
                    example["images"] = [image.convert("RGB").copy()]
                    example["prompt"][0]["content"][0]["text"] = None  # Arrow optional struct field.
                values.append(example)
            return Dataset(values)
        with patch("src.trainer.sft.training_data.make_hf_dataset", side_effect=builder) as shared:
            result = validate_dataset(self.output, check_hf=True)
        self.assertEqual(shared.call_count, 3)
        self.assertEqual(result["hf_examples"], COUNTS)
        records = [json.loads(line) for line in (self.output / "train/training.jsonl").read_text().splitlines()]
        self.assertTrue({"image", "csv", "target_csv", "rows", "images", "prompt", "completion"} <= records[0].keys())

    def test_count_and_split_leakage_rejected(self):
        prepare(self.source, self.output)
        self.rewrite_records("train", lambda rows: rows[0].update(source_split="test"))
        with self.assertRaisesRegex(ValueError, "split"):
            validate_dataset(self.output)
        self.rewrite_records("train", lambda rows: rows[0].update(source_split="train"))
        self.rewrite_records("train", lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "count"):
            validate_dataset(self.output)

    def test_duplicate_image_and_missing_pair_rejected(self):
        prepare(self.source, self.output)
        training = next((self.output / "train/images").glob("*.jpg"))
        validation = next((self.output / "validation/images").glob("*.jpg"))
        original = validation.read_bytes()
        validation.write_bytes(training.read_bytes())
        with self.assertRaises(ValueError):
            validate_dataset(self.output)
        validation.write_bytes(original)
        validation.unlink()
        with self.assertRaises(ValueError):
            validate_dataset(self.output)

    def test_source_integrity_and_rejected_record_accounting(self):
        path = self.source / "cord-train.jsonl.gz"
        records = [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]
        del records[0]["ground_truth"]["gt_parse"]["menu"][0]["unitprice"]
        path.write_bytes(gzip.compress("".join(json.dumps(row) + "\n" for row in records).encode(), mtime=0))
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_source(self.source)
        self.refresh_source()
        _, report, rejected = eligibility(self.source)
        self.assertEqual(report["eligible_counts"]["train"], 3)
        self.assertEqual(rejected[0]["source_filename"], "cord-train-0000.jpg")
        self.assertEqual((rejected[0]["reason"], rejected[0]["field"]), ("missing_field", "unitprice"))

    def test_duplicate_source_image_across_splits_is_reported(self):
        with zipfile.ZipFile(self.source / "cord-train-images.zip") as archive:
            duplicate = archive.read("cord-train-0000.jpg")
        target = self.source / "cord-validation-images.zip"
        with zipfile.ZipFile(target) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        files["cord-validation-0000.jpg"] = duplicate
        with zipfile.ZipFile(target, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        self.refresh_source()
        candidates, rejected, _, _ = eligible_candidates(self.source)
        self.assertEqual(len(candidates["validation"]), 1)
        self.assertTrue(any(row["source_filename"] == "cord-validation-0000.jpg"
                            and row["reason"] == "duplicate_image_or_cross_split_leakage" for row in rejected))

    def test_insufficient_candidates_stop_with_report_without_partial_dataset(self):
        with patch.dict(COUNTS, {"train": 10}), patch.object(schema, "TOTAL_COUNT", 12):
            with self.assertRaisesRegex(ValueError, "Not enough"):
                prepare(self.source, self.output, report_dir=self.root / "eligibility_report")
        self.assertFalse(self.output.exists())
        self.assertTrue((self.root / "eligibility_report/rejection_report.json").is_file())

    def test_preflight_extracts_no_images_and_does_not_write_dataset(self):
        prepare(self.source, self.output, preflight=True, report_dir=self.root / "preflight")
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.source.glob("*.jpg")), [])


if __name__ == "__main__":
    unittest.main()

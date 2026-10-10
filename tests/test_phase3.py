"""CPU-only checks for Phase 3 packaging, dataset safety and scoring."""
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.prepare_colab_dataset import ROOT, package
from src.trainer.sft.evaluate import evaluate_pairs, get_predicted_rows
from src.trainer.sft.training_data import INSTRUCTION, read_records


class Phase3Tests(unittest.TestCase):
    def test_training_splits_load_without_optum(self):
        root = ROOT / "data" / "synthetic"
        train = read_records(root, "train")
        val = read_records(root, "validation")
        self.assertEqual((len(train), len(val)), (8, 2))
        self.assertFalse({r['image_path'].name for r in train} & {r['image_path'].name for r in val})
        self.assertIn("leave Rate empty", INSTRUCTION)

    def test_reject_non_synthetic_dataset(self):
        with self.assertRaises(ValueError):
            read_records(ROOT / "data" / "invoices", "train")

    def test_exact_metrics_have_denominators(self):
        csv = "Item,Quantity,Rate,Amount\nRoom Charges,2,3500,7000\nConsultation,1,1500,1500\n"
        expected = get_predicted_rows(csv)
        results = evaluate_pairs([{"image": "1.jpg", "expected_rows": expected, "prediction": csv}])
        self.assertEqual(results["field_accuracy_by_row_position"], {"numerator": 8, "denominator": 8})
        self.assertEqual(results["exact_row_recall"], {"numerator": 2, "denominator": 2})
        self.assertEqual(results["invoice_exact_match"], {"numerator": 1, "denominator": 1})

    def test_invalid_csv_scores_as_failure(self):
        expected = get_predicted_rows("Item,Quantity,Rate,Amount\nECG,1,500,500\n")
        results = evaluate_pairs([{"image": "1.jpg", "expected_rows": expected,
                                   "prediction": "Sorry, here is the result:"}])
        self.assertEqual(results["valid_csv"]["numerator"], 0)
        self.assertEqual(results["field_accuracy_by_row_position"]["numerator"], 0)

    def test_missing_rate_is_preserved(self):
        rows = get_predicted_rows("Item,Quantity,Rate,Amount\nMRI Scan,1,,8500\n")
        self.assertEqual(rows[0]["Rate"], "")

    def test_bundle_excludes_optum_and_cord(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "bundle.zip"
            info = package(out)
            self.assertEqual((info["train_images"], info["validation_images"]), (8, 2))
            with ZipFile(out) as z:
                names = z.namelist()
                self.assertIn("src/trainer/sft/train_lora.py", names)
                self.assertIn("src/vlm/train_lora.py", names)  # legacy notebook compatibility
                self.assertIn("data/synthetic/train/annotations.jsonl", names)
                self.assertFalse(any(x.startswith(("data/invoices/", "data/evaluation/", "data/cord/")) for x in names))


if __name__ == "__main__":
    unittest.main()

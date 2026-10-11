"""Phase 10D CPU fixtures only; Phase 10C tests cover shared model execution."""
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.trainer.cord.evaluation_metrics import score, summarize
from src.trainer.cord.evaluation_report import write_reports
from src.trainer.cord.optum_evaluation import INVOICE_IDS, add_recommendation, load_optum

CSV = "Item,Quantity,Rate,Amount\nTest Item,2,10,20\n"


class OptumExtractionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.labels = self.root / "data/evaluation/ground_truth"
        self.labels.mkdir(parents=True)
        (self.root / "data/invoices").mkdir()
        (self.labels.parent / "README.md").write_text("CPU fixture labels are manually transcribed; not live annotations.")
        manifest = []
        for i, invoice_id in enumerate(INVOICE_IDS):
            image = f"data/invoices/{invoice_id}.jpg"
            Image.new("RGB", (20, 30+i), (i*40, 80, 120)).save(self.root / image)
            (self.labels / f"{invoice_id}.csv").write_text(CSV)
            (self.labels / f"{invoice_id}.json").write_text(json.dumps({"invoice_id": invoice_id, "image_file": image,
                "lines": [{"item": "Test Item", "quantity": 2, "rate": "10", "amount": "20"}]}))
            manifest.append({"invoice_id": invoice_id, "image": image,
                "csv": f"data/evaluation/ground_truth/{invoice_id}.csv",
                "json": f"data/evaluation/ground_truth/{invoice_id}.json", "rows": 1, "split": "final_evaluation_only"})
        (self.labels / "manifest.json").write_text(json.dumps(manifest))

    def test_five_existing_manual_pairs_and_hashes(self):
        records, metadata = load_optum(self.root)
        self.assertEqual(len(records), 5)
        self.assertEqual(metadata["verified_label_count"], 5)
        self.assertEqual(len({row["image_sha256"] for row in records}), 5)
        self.assertTrue(all(row["ground_truth_csv"] == CSV for row in records))

    def test_missing_label_is_unassessed_not_fabricated(self):
        (self.labels / f"{INVOICE_IDS[0]}.csv").unlink()
        records, metadata = load_optum(self.root)
        self.assertIsNone(records[0]["ground_truth_csv"])
        self.assertEqual(metadata["verified_label_count"], 4)
        result = score(None, {"raw_response": CSV, "error": None}, allow_empty_rates=True)
        self.assertTrue(result["csv_valid"])
        self.assertIsNone(result["document_exact"])
        self.assertIsNone(result["field_correct"])
        self.assertIsNone(result["extra_prediction_indices"])

    def test_wrong_image_annotation_pair_rejected(self):
        file = self.labels / f"{INVOICE_IDS[0]}.json"
        annotation = json.loads(file.read_text())
        annotation["image_file"] = f"data/invoices/{INVOICE_IDS[1]}.jpg"
        file.write_text(json.dumps(annotation))
        with self.assertRaisesRegex(ValueError, "another image"):
            load_optum(self.root)

    def test_absent_rate_preserved_and_invented_rate_penalized(self):
        absent = "Item,Quantity,Rate,Amount\nTest Item,2,,20\n"
        self.assertTrue(score(absent, {"raw_response": absent}, allow_empty_rates=True)["document_exact"])
        invented = score(absent, {"raw_response": CSV}, allow_empty_rates=True)
        self.assertFalse(invented["document_exact"])
        self.assertEqual(invented["field_correct"]["Rate"], 0)
        # Phase 10C's strict explicit-rate behavior is unchanged.
        self.assertFalse(score(CSV, {"raw_response": absent})["csv_valid"])

    def test_recommendations_from_measured_metrics_only(self):
        correct = score(CSV, {"raw_response": CSV})
        wrong = score(CSV, {"raw_response": ""})
        def result(a, b, count=5):
            return {"verified_ground_truth_documents": count, "models": {
                "baseline": summarize([a]*5, 5), "cord_sft": summarize([b]*5, 5)}}
        for baseline, cord, count, expected in ((wrong, correct, 5, "NEW CORD SFT"),
                (correct, wrong, 5, "ORIGINAL SFT"), (correct, correct, 5, "INCONCLUSIVE"),
                (wrong, correct, 4, "INCONCLUSIVE")):
            directory = self.root / expected.replace(" ", "_") / str(count)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "summary.md").write_text("Measured comparison\n")
            (directory / "per_image_comparison.csv").write_text("image\nfixture.jpg\n")
            summary = result(baseline, cord, count)
            add_recommendation(directory, summary)
            self.assertEqual(summary["recommendation"], expected)
            self.assertTrue((directory / "comparison.csv").is_file())

    def test_unlabelled_report_has_no_accuracy_scores(self):
        metrics = score(None, {"raw_response": CSV}, allow_empty_rates=True)
        pred = {"raw_response": CSV, "error": None, "truncated": False, "latency_seconds": 1,
                "generated_tokens": 10, "metrics": metrics}
        comparisons = [{"image": "fixture.jpg", "image_path": "fixture.jpg", "ground_truth_csv": None,
                        "baseline": pred, "cord_sft": pred} for _ in range(5)]
        directory = self.root / "report"
        directory.mkdir()
        metadata = {"dataset_name": "optum", "evaluated_documents": 5, "status": "complete", "model_id": "fixture",
                    "settings": {"revision": "fixture", "prompt_sha256": "fixture"},
                    "dataset": {"test_split_sha256": "fixture"}, "adapters": {}}
        summary = write_reports(directory, comparisons, metadata)
        add_recommendation(directory, summary)
        self.assertEqual(summary["recommendation"], "INCONCLUSIVE")
        self.assertIsNone(summary["models"]["baseline"]["document_exact"]["value"])
        self.assertIn("NOT ASSESSED", (directory / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main()

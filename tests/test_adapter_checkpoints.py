"""CPU checks for dataset split isolation and saved-adapter selection."""
import json
from pathlib import Path
import tempfile
import unittest

from trainer.sft_qlora_invoice_extraction.train import latest_adapter
from trainer.sft_qlora_invoice_extraction.data_prep import read_records
from src.utils.paths import ROOT


class AdapterCheckpointTests(unittest.TestCase):


    def test_manifest_override_keeps_split_boundary(self):
        root = ROOT / "data/assignment_dataset/synthetic"
        self.assertEqual(len(read_records(root, "train", manifest=root / "train/annotations.jsonl")), 8)
        with self.assertRaises(ValueError):
            read_records(root, "train", manifest=root / "validation/annotations.jsonl")

    def test_latest_pointer_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "latest_successful.json").write_text(json.dumps({"adapter_path": "../adapter"}))
            with self.assertRaisesRegex(RuntimeError, "Unsafe adapter path"):
                latest_adapter(root)

    def test_failed_run_cannot_be_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "runs/test/model_weights"
            adapter.mkdir(parents=True)
            (root / "latest_successful.json").write_text(json.dumps({
                "adapter_path": "runs/test/model_weights", "model_id": "example", "run_id": "test"}))
            (adapter.parent / "training_result.json").write_text('{"status":"failed"}')
            with self.assertRaisesRegex(RuntimeError, "not successful"):
                latest_adapter(root)


if __name__ == "__main__":
    unittest.main()

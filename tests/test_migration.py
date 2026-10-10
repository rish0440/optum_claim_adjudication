"""Regression checks for package moves; no model downloads or training."""
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from fastapi.testclient import TestClient
import yaml

from app.main import app
from scripts.prepare_colab_dataset import package
from src.config.settings import get_settings
from src.utils.paths import ROOT


class MigrationTests(unittest.TestCase):
    def run_python(self, *args, cwd=ROOT, env=None):
        result = subprocess.run(
            [sys.executable, "-B", *args], cwd=cwd, env=env,
            text=True, capture_output=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_legacy_imports_share_canonical_modules(self):
        aliases = {
            "src.main": "app.main",
            "src.settings": "src.config.settings",
            "src.policy": "src.adjudication.policy",
            "src.vlm.dataset": "src.trainer.sft.dataset",
            "src.vlm.training_data": "src.trainer.sft.training_data",
            "src.vlm.train_lora": "src.trainer.sft.train_lora",
            "src.vlm.evaluate": "src.trainer.sft.evaluate",
            "src.vlm.inference": "src.inference.local",
            "src.vlm.remote_client": "src.inference.remote_client",
        }
        for name in ("graph", "state", "workflow"):
            aliases[f"src.agentic_workflow.{name}"] = f"src.orchestration.{name}"
        for name in ("invoice_extraction", "csv_validation", "policy_retrieval",
                     "rule_resolution", "adjudication", "response_builder"):
            aliases[f"src.agentic_workflow.nodes.{name}.node"] = f"src.orchestration.nodes.{name}.node"
        for old, new in aliases.items():
            with self.subTest(module=old):
                self.assertIs(importlib.import_module(old), importlib.import_module(new))
        with patch("src.main._run_graph") as mocked:
            self.assertIs(importlib.import_module("app.main")._run_graph, mocked)

    def test_inference_imports_do_not_load_training_or_torch(self):
        self.run_python("-c", """
import sys
import src.inference.local
import src.inference.remote_client
assert 'torch' not in sys.modules
assert not any(name.startswith('src.trainer') for name in sys.modules)
""")

    def test_configuration_and_sft_paths_outside_repo_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = yaml.safe_load((ROOT / "src/config/default.yaml").read_text())
            config["paths"]["policy"] = "data/policy.json"
            custom = Path(tmp) / "config.yaml"
            custom.write_text(yaml.safe_dump(config))
            env = {**os.environ, "PYTHONPATH": str(ROOT), "OPTUM_PROJECT_ROOT": str(ROOT),
                   "OPTUM_CONFIG_FILE": str(custom),
                   "OPTUM_SYNTHETIC_DATASET": str(Path(tmp) / "synthetic"),
                   "OPTUM_SFT_OUTPUT_DIR": str(Path(tmp) / "model-run")}
            self.run_python("-c", """
import os
from pathlib import Path
from src.config.settings import get_settings
from src.trainer.sft.config import dataset_root, output_dir
from src.adjudication.policy import load_policy
assert load_policy().policy_id == 'POL-CLM-001'
assert get_settings().paths.synthetic_dataset == dataset_root() == Path(os.environ['OPTUM_SYNTHETIC_DATASET'])
assert output_dir() == Path(os.environ['OPTUM_SFT_OUTPUT_DIR'])
""", cwd=tmp, env=env)

    def test_bundle_is_self_contained_with_new_and_legacy_entrypoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "bundle.zip"
            package(output)
            unpacked = Path(tmp) / "unpacked"
            with ZipFile(output) as archive:
                self.assertEqual(archive.read("src/trainer/sft/training_data.py"),
                                 (ROOT / "src/trainer/sft/training_data.py").read_bytes())
                self.assertFalse(any(name.startswith(("data/invoices/", "data/evaluation/", "data/cord/"))
                                     for name in archive.namelist()))
                archive.extractall(unpacked)
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("OPTUM_") and key != "PYTHONPATH"}
            # -S proves CLI imports need only the packaged code and standard library.
            for module in ("src.trainer.sft.train_lora", "src.trainer.sft.evaluate",
                           "src.inference.local", "src.vlm.train_lora",
                           "src.vlm.evaluate", "src.vlm.inference"):
                with self.subTest(module=module):
                    self.run_python("-S", "-m", module, "--help", cwd=unpacked, env=env)
            self.run_python("-S", "-c", """
from src.trainer.sft.training_data import read_records
from src.trainer.sft.config import dataset_root
assert len(read_records(dataset_root(), 'train')) == 8
assert len(read_records(dataset_root(), 'validation')) == 2
""", cwd=unpacked, env=env)

    @unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is not installed")
    def test_corrected_hf_images_sequence_is_preserved(self):
        from datasets import Image, Sequence
        from src.trainer.sft.training_data import make_hf_dataset, read_records
        records = read_records(get_settings().paths.synthetic_dataset, "train")
        dataset = make_hf_dataset(records)
        # Inspect image paths without requiring optional image-decoding backends.
        sample = dataset.cast_column("images", Sequence(Image(decode=False)))[0]
        self.assertEqual(len(dataset), 8)
        self.assertEqual(len(sample["images"]), 1)
        self.assertEqual(sample["images"][0]["path"], str(records[0]["image_path"]))
        self.assertEqual(sample["prompt"][0]["content"][0]["type"], "image")
        self.assertEqual(sample["completion"][0]["content"][0]["text"], records[0]["target_csv"])

    def test_api_reference_demos_use_real_compiled_graph(self):
        expected = json.loads((get_settings().paths.evaluation_ground_truth / "expected_adjudication.json").read_text())
        from decimal import Decimal
        with TestClient(app) as client:
            for reference in expected:
                response = client.post("/api/adjudicate/demo", json={"example_id": reference["invoice_id"]})
                self.assertEqual(response.status_code, 200, response.text)
                result = response.json()
                self.assertEqual(Decimal(result["claim"]["final_reimbursement"]), Decimal(reference["reimbursement_total"]))
                self.assertEqual(result["trace"], ["invoice_extraction", "csv_validation", "policy_retrieval",
                                                  "rule_resolution", "adjudication", "response_builder"])

    def test_api_reference_csv_uses_real_graph_and_preserves_blank_rates(self):
        path = get_settings().paths.evaluation_ground_truth / "invoice_04_diagnostics.csv"
        with TestClient(app) as client:
            response = client.post("/api/adjudicate", data={"invoice_date": "2024-04-18"},
                                   files={"file": (path.name, path.read_bytes(), "text/csv")})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["claim"]["final_reimbursement"], "11754.00")
        self.assertEqual(result["extraction_source"], "provided_csv")
        self.assertIn(",,", result["extracted_csv"])

    def test_api_bad_csv_short_circuits_real_graph(self):
        with TestClient(app) as client:
            response = client.post("/api/adjudicate", data={"invoice_date": "2024-04-18"},
                                   files={"file": ("bad.csv", b"Bad,Header\nx,y\n", "text/csv")})
        self.assertEqual(response.status_code, 422)
        workflow = response.json()["detail"]["workflow"]
        self.assertIsNone(workflow["claim"])
        self.assertEqual(workflow["trace"], ["invoice_extraction", "csv_validation", "response_builder"])


if __name__ == "__main__":
    unittest.main()

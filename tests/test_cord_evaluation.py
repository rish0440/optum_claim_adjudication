"""CPU scoring, provenance and mocked inference tests; no GPU/model downloads."""
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.trainer.cord import evaluation_data
from src.trainer.cord.evaluate import (AdapterComparison, assert_adapter_active, common_settings, evaluate_predictions,
                                      parser, run)
from src.trainer.cord.evaluation_data import hashlib_prompt, load_test, resolve_checkpoints
from src.trainer.cord.evaluation_metrics import normalize, parse_prediction, score, summarize
from src.trainer.cord.evaluation_report import inspect_report, representatives
from src.trainer.cord.prepare import inspect_image
from src.trainer.cord.schema import FILES, REPO_ID, convert, fingerprint, messages, sha
from src.trainer.cord.train import MODEL_ID
from src.trainer.cord.training_data import INSTRUCTION
from src.trainer.sft.train.checkpoints import latest_adapter
from tests.test_cord_preparation import item, receipt

HEADER = "Item,Quantity,Rate,Amount\n"
CSV = HEADER + "Meal,2,10.00,20.00\nTea,1,5.00,5.00\n"


def prediction(raw=CSV, **kwargs):
    return {"raw_response": raw, "error": None, "truncated": False, **kwargs}


class MetricsTests(unittest.TestCase):
    def test_strict_csv_and_decimal_normalization(self):
        result = score(CSV, prediction(HEADER + "  Meal  ,2.0,10,20.000\nTea,1,5,5\n"))
        self.assertTrue(result["document_exact"])
        self.assertEqual(normalize(parse_prediction(HEADER + "Tea,1,0.10,0.10\n")[0])[2],
                         normalize(parse_prediction(HEADER + "Tea,1,0.1,0.1\n")[0])[2])

    def test_row_order_strict_and_order_insensitive(self):
        result = score(CSV, prediction(HEADER + "Tea,1,5,5\nMeal,2,10,20\n"))
        self.assertFalse(result["document_exact"])
        self.assertTrue(result["order_insensitive_exact"])
        self.assertEqual(result["complete_rows"], 2)

    def test_missing_earlier_row_does_not_shift_alignment(self):
        result = score(CSV, prediction(HEADER + "Tea,1,5,5\n"))
        self.assertEqual(result["matched_rows"][0]["reference_index"], 1)
        self.assertEqual(result["missing_reference_indices"], [0])
        self.assertEqual(result["field_correct"]["Rate"], 1)

    def test_duplicate_and_extra_rows_never_receive_double_credit(self):
        result = score(CSV, prediction(CSV + "Tea,1,5,5\nTotal,1,25,25\n"))
        self.assertEqual(result["complete_rows"], 2)
        self.assertEqual(result["extra_prediction_indices"], [2, 3])
        self.assertFalse(result["document_exact"])
        self.assertEqual(summarize([result], 1)["row_precision"]["value"], 0.5)

    def test_fields_and_arithmetic_errors(self):
        result = score(CSV, prediction(HEADER + "Meal,2,11,20\nTeaa,1,5,5\n"))
        self.assertEqual(result["field_correct"], {"Item": 1, "Quantity": 2, "Rate": 1, "Amount": 2})
        self.assertEqual(result["arithmetic_error_indices"], [0])
        self.assertFalse(result["document_exact"])

    def test_empty_markdown_narrative_and_malformed_csv_fail(self):
        for raw in ("", "```csv\n"+CSV+"```", "Here is the CSV:\n"+CSV,
                    HEADER + '"Meal,2,10,20\n', HEADER + "Meal,2,10\n", HEADER + "Meal,2,10,20,extra\n"):
            with self.subTest(raw=raw):
                result = score(CSV, prediction(raw))
                self.assertFalse(result["csv_valid"])
                self.assertFalse(result["document_exact"])

    def test_invalid_or_missing_rates_not_derived(self):
        for rate in ("", "NaN", "INR 10", "-10"):
            result = score(CSV, prediction(HEADER + f"Meal,2,{rate},20\nTea,1,5,5\n"))
            self.assertFalse(result["csv_valid"])
            self.assertEqual(result["field_correct"]["Rate"], 1)
            self.assertEqual(len(result["invalid_numeric_fields"]), 1)

    def test_truncated_output_and_failures_keep_denominator_19(self):
        truncated = score(CSV, prediction(truncated=True))
        self.assertFalse(truncated["eligible_completion"])
        self.assertEqual(truncated["complete_rows"], 0)
        rows = [score(CSV, prediction(error="CUDAError")) for _ in range(19)]
        result = summarize(rows, 19)
        self.assertEqual(result["document_exact"]["denominator"], 19)
        self.assertEqual(result["inference_errors"], 19)
        self.assertEqual(result["fields"]["Item"]["denominator"], 38)
        self.assertEqual(result["row_precision"]["value"], 0)
        with self.assertRaises(ValueError):
            summarize(rows[:-1], 19)


class IdentityTests(unittest.TestCase):
    def test_initialization_and_sequential_switches_freeze_all_parameters(self):
        parameters = {name: Mock(requires_grad=True) for name in ("base", "baseline", "cord_sft")}
        for parameter in parameters.values():
            parameter.requires_grad_.side_effect = lambda flag, parameter=parameter: setattr(parameter, "requires_grad", flag)
        model = Mock()
        model.parameters.side_effect = lambda: list(parameters.values())
        model.peft_config = {name: SimpleNamespace(base_model_name_or_path=MODEL_ID)
                             for name in ("baseline", "cord_sft")}
        model.active_adapters = ["baseline"]
        model.training = True
        model.eval.side_effect = lambda: setattr(model, "training", False)
        def switch(mode):
            model.active_adapters = [mode]
            parameters[mode].requires_grad = True  # PEFT's documented switching behavior.
        model.set_adapter.side_effect = switch
        model.load_adapter.side_effect = lambda *args, **kwargs: switch("baseline")
        processor = Mock(chat_template="fixture-template")
        factories = SimpleNamespace(AutoProcessor=Mock(), BitsAndBytesConfig=Mock(),
                                    Qwen2_5_VLForConditionalGeneration=Mock(), set_seed=Mock())
        factories.AutoProcessor.from_pretrained.return_value = processor
        factories.Qwen2_5_VLForConditionalGeneration.from_pretrained.return_value = Mock(
            config=SimpleNamespace(_commit_hash="a"*40))
        peft = SimpleNamespace(PeftModel=Mock())
        peft.PeftModel.from_pretrained.return_value = model
        torch = SimpleNamespace(cuda=Mock(), bfloat16="bf16", version=SimpleNamespace(cuda="fixture"))
        torch.cuda.is_available.return_value = True
        torch.cuda.device_count.return_value = 1
        torch.cuda.is_bf16_supported.return_value = True
        torch.cuda.mem_get_info.return_value = (23*2**30, 24*2**30)
        torch.cuda.get_device_name.return_value = "Mock L4"
        adapters = {mode: {"path": mode, "adapter_sha256": mode+"-hash"}
                    for mode in ("baseline", "cord_sft")}
        settings = common_settings(parser().parse_args([]), INSTRUCTION, "a"*40)
        with patch.dict("sys.modules", {"torch": torch, "peft": peft, "transformers": factories}), \
             patch("src.trainer.cord.evaluate.verify_loaded_weights") as verify:
            engine = AdapterComparison(adapters, settings)
        self.assertFalse(any(parameter.requires_grad for parameter in parameters.values()))
        self.assertFalse(model.training)
        self.assertFalse(peft.PeftModel.from_pretrained.call_args.kwargs["is_trainable"])
        self.assertFalse(model.load_adapter.call_args.kwargs["is_trainable"])
        self.assertEqual(verify.call_count, 2)
        for mode in ("baseline", "cord_sft", "baseline"):
            engine.select(mode)
            self.assertEqual(model.active_adapters, [mode])
            self.assertFalse(model.training)
            self.assertFalse(any(parameter.requires_grad for parameter in parameters.values()))
        model.set_adapter.side_effect = lambda mode: None
        with self.assertRaisesRegex(RuntimeError, "Wrong/missing active"):
            engine.select("cord_sft")

    def test_inference_freezing_retains_gradient_safety_assertion(self):
        engine = AdapterComparison.__new__(AdapterComparison)
        engine.model = Mock()
        engine.model.parameters.return_value = [Mock(requires_grad=True)]  # Simulate unsuccessful freezing.
        with self.assertRaisesRegex(RuntimeError, "trainable parameters"):
            engine._freeze_for_inference()

    def test_checkpoint_resolution_requires_complete_same_backbone_non_smoke_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [root / mode / "runs/fixture/adapter" for mode in ("baseline", "cord_sft")]
            for mode, path in zip(("baseline", "cord_sft"), paths):
                path.mkdir(parents=True)
                config = {"model_id": MODEL_ID, "expected_optimizer_steps": 82,
                          "instruction": INSTRUCTION, "instruction_sha256": hashlib_prompt(INSTRUCTION),
                          "resolved_model_revision": "a"*40}
                (path.parent / ("training_config.json" if mode == "cord_sft" else "run_config.json")).write_text(json.dumps(config))
                (path.parent / "training_result.json").write_text(json.dumps({"smoke": False, "metrics": {
                    "global_steps_completed": 82, "training_loss": 0.1, "evaluation": {"eval_loss": 0.2}}}))
            with patch("src.trainer.cord.evaluation_data.latest_adapter", side_effect=lambda base: paths[0] if base == root/"baseline" else paths[1]), \
                 patch("src.trainer.cord.evaluation_data.verify_adapter", side_effect=lambda path, model: {
                       "sha256": path.parts[-4], "tensor_count": 288}), \
                 patch("src.trainer.cord.evaluation_data.verify_saved", return_value={"smoke": False}) as canonical:
                metadata, instruction, revision = resolve_checkpoints(root/"baseline", root/"cord_sft")
                self.assertEqual(instruction, INSTRUCTION)
                self.assertEqual(revision, "a"*40)
                canonical.assert_called_once()
                self.assertNotEqual(metadata["baseline"]["adapter_sha256"], metadata["cord_sft"]["adapter_sha256"])
                result = paths[1].parent / "training_result.json"
                failed = json.loads(result.read_text())
                failed["metrics"]["global_steps_completed"] = 81
                result.write_text(json.dumps(failed))
                with self.assertRaisesRegex(ValueError, "complete"):
                    resolve_checkpoints(root/"baseline", root/"cord_sft")

    def test_adapter_modes_and_identical_common_settings(self):
        model = Mock()
        model.active_adapters = ["baseline"]
        model.peft_config = {name: Mock(base_model_name_or_path=MODEL_ID) for name in ("baseline", "cord_sft")}
        assert_adapter_active(model, "baseline")
        with self.assertRaises(RuntimeError):
            assert_adapter_active(model, "cord_sft")
        model.active_adapters = ["cord_sft"]
        assert_adapter_active(model, "cord_sft")
        settings = common_settings(parser().parse_args([]), INSTRUCTION, "a"*40)
        self.assertFalse(settings["do_sample"])
        record = {"image": "held-out.jpg", "image_path": Path("held-out.jpg"), "image_sha256": "hash",
                  "ground_truth_csv": CSV}
        predictions = {mode: [{**prediction(), "mode": mode, "image": record["image"],
                              "image_sha256": "hash", "common_settings_sha256": fingerprint(settings)}]
                       for mode in ("baseline", "cord_sft")}
        self.assertTrue(evaluate_predictions([record], predictions, settings)[0]["baseline"]["metrics"]["document_exact"])
        predictions["cord_sft"][0]["common_settings_sha256"] = "different"
        with self.assertRaises(ValueError):
            evaluate_predictions([record], predictions, settings)

    def test_pointer_checksum_validation_not_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "runs/fixture/adapter"
            adapter.mkdir(parents=True)
            (root / "latest_successful.json").write_text(json.dumps({"adapter_path": "runs/fixture/adapter",
                "model_id": MODEL_ID, "run_id": "fixture", "adapter_sha256": "expected"}))
            (adapter.parent / "training_result.json").write_text(json.dumps({"status": "success", "model_id": MODEL_ID,
                "run_id": "fixture", "metrics": {"global_steps_completed": 82}}))
            for name in ("run_config.json", "training_args.json", "metrics.json", "training_log.json"):
                (adapter.parent / name).write_text("{}")
            with patch("src.trainer.sft.train.checkpoints.verify_adapter", return_value={"sha256": "different"}):
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    latest_adapter(root)


class TestOnlyLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        counts = {"train": 1, "validation": 1, "test": 1}
        context = patch.object(evaluation_data, "EXPECTED_COUNTS", counts)
        context.start()
        self.addCleanup(context.stop)
        for folder in ("images", "labels"):
            (self.root / "test" / folder).mkdir(parents=True)
        annotation = receipt([item()])
        annotation["image_file"] = "cord-test-0001.jpg"
        image_rel, csv_rel = "test/images/cord-test-0001.jpg", "test/labels/cord-test-0001.csv"
        Image.new("RGB", (20, 30), (10, 80, 140)).save(self.root / image_rel)
        text, rows = convert(annotation)
        (self.root / csv_rel).write_text(text)
        image_hash, pixel_hash = inspect_image((self.root / image_rel).read_bytes(), annotation)
        entry = {"source_id": annotation["image_file"], "source_split": "test", "image": image_rel, "csv": csv_rel,
                 "source_line": 2, "annotation_sha256": fingerprint(annotation), "rows": len(rows),
                 "image_sha256": image_hash, "pixel_sha256": pixel_hash, "csv_sha256": sha(self.root / csv_rel)}
        record = {**entry, "source_filename": annotation["image_file"], "target_csv": text,
                  "provenance": {"source_file": "cord-test.jsonl.gz", "source_line": 2,
                                 "annotation_sha256": entry["annotation_sha256"]}, **messages(image_rel, text)}
        hashes = {"train/training.jsonl": "training-fingerprint", "validation/validation.jsonl": "validation-fingerprint"}
        for filename, value in (("test.jsonl", record), ("annotations.jsonl", annotation)):
            (self.root / "test" / filename).write_text(json.dumps(value)+"\n")
            hashes[f"test/{filename}"] = sha(self.root / "test" / filename)
        self.manifest = {"counts": counts, "source": {"repo_id": REPO_ID, "revision": "a"*40,
            "files": {name: {} for name in FILES}}, "file_sha256": hashes,
            "entries": [entry] + [{"source_id": split, "source_split": split,
                                  "image_sha256": split+"-bytes", "pixel_sha256": split+"-pixels"}
                                 for split in ("train", "validation")]}
        self.refresh()

    def refresh(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))
        self.adapters = {"baseline": {"training_config": {}}, "cord_sft": {"training_config": {"dataset": {
            "counts": self.manifest["counts"], "source": self.manifest["source"],
            "dataset_manifest_sha256": sha(self.root / "manifest.json"), "test_artifacts_opened": False,
            "split_sha256": {"train": "training-fingerprint", "validation": "validation-fingerprint"}}}}}

    def test_loads_only_test_without_training_files(self):
        records, metadata = load_test(self.root, self.adapters)
        self.assertEqual(len(records), 1)
        self.assertFalse((self.root / "train").exists())
        self.assertFalse((self.root / "validation").exists())
        self.assertFalse(metadata["train_samples_opened"])

    def test_overlap_and_changed_labels_are_rejected(self):
        self.manifest["entries"][1]["pixel_sha256"] = self.manifest["entries"][0]["pixel_sha256"]
        self.refresh()
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            load_test(self.root, self.adapters)
        self.manifest["entries"][1]["pixel_sha256"] = "train-pixels"
        self.refresh()
        (self.root / "test/labels/cord-test-0001.csv").write_text(CSV)
        with self.assertRaises(ValueError):
            load_test(self.root, self.adapters)


class MockedRunnerTests(unittest.TestCase):
    def scenario(self):
        record = {"image": "held-out.jpg", "image_path": Path("held-out.jpg"),
                  "image_sha256": "image-hash", "pixel_sha256": "pixel-hash", "ground_truth_csv": CSV}
        adapters = {mode: {"path": mode, "adapter_sha256": mode+"-hash", "training_config": {}}
                    for mode in ("baseline", "cord_sft")}
        return record, adapters

    def test_real_generation_wiring_and_report_integrity_using_mocks(self):
        record, adapters = self.scenario()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engine = Mock()
            engine.processor.chat_template = "fixture-template"
            engine.gpu = {"name": "mock"}
            engine.generate.side_effect = [{**prediction(), "generated_tokens": 10},
                                           {**prediction(HEADER+"Tea,1,5,5\n"), "generated_tokens": 8}]
            torch = Mock()
            torch.cuda.max_memory_allocated.return_value = 0
            torch.cuda.max_memory_reserved.return_value = 0
            with patch("src.trainer.cord.evaluate.sys.prefix", "/fixture/.venv"), \
                 patch("src.trainer.cord.evaluate.sys.base_prefix", "/base"), \
                 patch("src.trainer.cord.evaluate.project_path", side_effect=lambda value: root / value), \
                 patch("src.trainer.cord.evaluate.resolve_checkpoints", return_value=(adapters, INSTRUCTION, "a"*40)), \
                 patch("src.trainer.cord.evaluate.load_test", return_value=([record], {"test_split_sha256": "test-hash"})), \
                 patch("src.trainer.cord.evaluate.versions", return_value={name: "fixture" for name in
                       ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "Pillow", "safetensors")}), \
                 patch("src.trainer.cord.evaluate.sha", side_effect=lambda path: sha(path) if Path(path).is_file() else "image-hash"), \
                 patch("src.trainer.cord.evaluate.AdapterComparison", return_value=engine), \
                 patch.dict("sys.modules", {"torch": torch}):
                directory = run(parser().parse_args(["--smoke"]))
            self.assertEqual(engine.generate.call_count, 2)
            for call in engine.generate.call_args_list:
                self.assertEqual(call.args, (record["image_path"], INSTRUCTION))  # No gold answer passed.
            self.assertEqual([call.args[0] for call in engine.select.call_args_list], ["baseline", "cord_sft"])
            engine.close.assert_called_once()
            for filename in ("baseline_predictions.jsonl", "cord_sft_predictions.jsonl", "summary.json",
                             "summary.md", "per_image_comparison.json", "per_image_comparison.csv", "evaluation_manifest.json"):
                self.assertTrue((directory / filename).is_file())
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete_smoke")
            self.assertEqual(summary["models"]["baseline"]["document_exact"]["value"], 1)
            comparisons = json.loads((directory / "per_image_comparison.json").read_text())
            self.assertEqual(comparisons[0]["cord_sft"]["raw_response"], HEADER+"Tea,1,5,5\n")
            self.assertEqual(representatives(comparisons)[0][0], "Baseline wins")
            with patch("builtins.print"):
                inspect_report(directory)
            (directory / "summary.md").write_text("altered report")
            with self.assertRaisesRegex(ValueError, "checksum"):
                inspect_report(directory)

    def test_missing_predictions_cannot_create_fabricated_scores(self):
        record, _ = self.scenario()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            evaluate_predictions([record], {"baseline": [], "cord_sft": []}, {})

    def test_model_loading_failure_never_creates_success_summary(self):
        record, adapters = self.scenario()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("src.trainer.cord.evaluate.sys.prefix", "/fixture/.venv"), \
                 patch("src.trainer.cord.evaluate.sys.base_prefix", "/base"), \
                 patch("src.trainer.cord.evaluate.project_path", side_effect=lambda value: root / value), \
                 patch("src.trainer.cord.evaluate.resolve_checkpoints", return_value=(adapters, INSTRUCTION, "a"*40)), \
                 patch("src.trainer.cord.evaluate.load_test", return_value=([record], {"test_split_sha256": "test-hash"})), \
                 patch("src.trainer.cord.evaluate.versions", return_value={name: "fixture" for name in
                       ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "Pillow", "safetensors")}), \
                 patch("src.trainer.cord.evaluate.AdapterComparison", side_effect=RuntimeError("Cannot load adapter")):
                with self.assertRaises(RuntimeError):
                    run(parser().parse_args(["--smoke"]))
            manifest = next(root.rglob("evaluation_manifest.json"))
            self.assertEqual(json.loads(manifest.read_text())["status"], "failed")
            self.assertFalse(list(root.rglob("summary.json")))
            self.assertFalse(list(root.rglob("baseline_predictions.jsonl")))


if __name__ == "__main__":
    unittest.main()

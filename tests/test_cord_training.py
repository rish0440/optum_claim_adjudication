"""CPU fixtures/mocks only: no model downloads, GPU calls or real CORD test access."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from trainer.sft_qlora_invoice_extraction import data_prep as schema, data_prep as training_data
from trainer.sft_qlora_invoice_extraction.data_prep import convert, fingerprint, messages, sha
from trainer.sft_qlora_invoice_extraction.train import MODEL_ID, OUTPUT, check_arguments, expected_steps, language_targets, parser, publish_success, verify_saved
from trainer.sft_qlora_invoice_extraction.data_prep import BoundedCollator, INSTRUCTION, load_training_data, read_split
from src.utils.paths import project_path
from tests.helpers import item, receipt


class LoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.counts = {"train": 2, "validation": 1, "test": 1}
        for context in (patch.object(schema, "TOTAL_COUNT", 4),
                        patch.object(training_data, "EXPECTED_COUNTS", self.counts)):
            context.start()
            self.addCleanup(context.stop)
        entries, hashes = [], {}
        for offset, split in enumerate(("train", "validation")):
            records, annotations = [], []
            (self.root / split / "images").mkdir(parents=True)
            (self.root / split / "labels").mkdir()
            for index in range(self.counts[split]):
                name = f"cord-{split}-{index}.jpg"
                annotation = receipt([item('Meal, "large"')])
                annotation.update(image_file=name, width=20, height=30+index)
                image_rel, csv_rel = f"{split}/images/{name}", f"{split}/labels/{Path(name).stem}.csv"
                Image.new("RGB", (20, 30+index), (offset*80, index*70, 120)).save(self.root / image_rel)
                text, rows = convert(annotation)
                (self.root / csv_rel).write_text(text, encoding="utf-8")
                image_hash, pixel_hash = training_data.inspect_image((self.root / image_rel).read_bytes(), annotation)
                entry = {"source_id": name, "source_split": split, "image": image_rel, "csv": csv_rel,
                         "source_line": index+1, "rows": len(rows), "annotation_sha256": fingerprint(annotation),
                         "image_sha256": image_hash, "pixel_sha256": pixel_hash, "csv_sha256": sha(self.root / csv_rel)}
                records.append({**entry, "source_filename": name, "target_csv": text,
                    "provenance": {"source_file": f"cord-{split}.jsonl.gz", "source_line": index+1,
                                   "annotation_sha256": entry["annotation_sha256"]}, **messages(image_rel, text)})
                annotations.append(annotation)
                entries.append(entry)
            filename = "training.jsonl" if split == "train" else "validation.jsonl"
            for filename, values in ((filename, records), ("annotations.jsonl", annotations)):
                relative = f"{split}/{filename}"
                (self.root / relative).write_text("".join(json.dumps(row)+"\n" for row in values))
                hashes[relative] = sha(self.root / relative)
        # Manifest inventory only: intentionally NO test directory or test files exist.
        entries.append({"source_id": "held-out.jpg", "source_split": "test",
                        "image_sha256": "held-out-image", "pixel_sha256": "held-out-pixels"})
        self.manifest = {"counts": self.counts, "entries": entries, "file_sha256": hashes,
                         "source": {"repo_id": schema.REPO_ID, "revision": "a"*40,
                                    "files": {name: {} for name in schema.FILES}}}
        self.save_manifest()

    def save_manifest(self):
        (self.root / "manifest.json").write_text(json.dumps(self.manifest))

    def test_counts_paths_pairing_and_no_test_artifacts_opened(self):
        records, metadata = load_training_data(self.root)
        self.assertEqual({key: len(value) for key, value in records.items()}, {"train": 2, "validation": 1})
        self.assertFalse((self.root / "test").exists())
        self.assertFalse(metadata["test_artifacts_opened"])
        self.assertEqual(set(metadata["split_sha256"]), {"train", "validation"})
        self.assertTrue(records["train"][0]["image_path"].is_absolute())
        self.assertIn('"Meal, ""large"""', records["train"][0]["target_csv"])

    def test_test_split_rejected_before_file_access(self):
        with patch.object(Path, "open", side_effect=AssertionError("Unexpected access")):
            with self.assertRaisesRegex(ValueError, "only read"):
                read_split(self.root, "test", self.manifest)

    def test_changed_manifest_and_blank_completion_rejected(self):
        path = self.root / "train/training.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["target_csv"] = ""
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_training_data(self.root)
        self.manifest["file_sha256"]["train/training.jsonl"] = sha(path)
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "CSV"):
            load_training_data(self.root)

    def test_split_leakage_and_duplicate_manifest_rejected(self):
        self.manifest["entries"][0]["source_split"] = "test"
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "split count"):
            load_training_data(self.root)
        self.manifest["entries"][0]["source_split"] = "train"
        self.manifest["entries"][-1]["pixel_sha256"] = self.manifest["entries"][0]["pixel_sha256"]
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "cross-split"):
            load_training_data(self.root)

    def test_hf_builder_reused_and_only_instruction_changes(self):
        records, _ = load_training_data(self.root)
        class Dataset(list):
            def map(self, function):
                return Dataset([{**row, **function(copy.deepcopy(row))} for row in self])
        original = Dataset([{**messages(str(row["image_path"]), row["target_csv"])} for row in records["train"]])
        with patch("trainer.sft_qlora_invoice_extraction.data_prep.make_hf_dataset", return_value=original) as shared:
            dataset = training_data.make_dataset(records["train"], INSTRUCTION)
        shared.assert_called_once_with(records["train"])
        self.assertEqual(dataset[0]["images"], original[0]["images"])
        self.assertEqual(dataset[0]["completion"], original[0]["completion"])
        self.assertEqual(dataset[0]["prompt"][0]["content"][0], {"type": "image"})
        self.assertEqual(dataset[0]["prompt"][0]["content"][1]["text"], INSTRUCTION)


class CollatorTests(unittest.TestCase):
    def fixture(self):
        import torch
        from trainer.sft_qlora_invoice_extraction.runtime import CheckedInvoiceCollator
        expected = "Item,Quantity,Rate,Amount\nMeal,2,10.00,20.00\n"
        example = {"images": [Image.new("RGB", (28, 28))], "prompt": [{"role": "user"}],
                   "completion": [{"role": "assistant", "content": [{"type": "text", "text": expected}]}]}
        processor = Mock()
        processor.tokenizer.get_vocab.return_value = {"<|image_pad|>": 11, "<|vision_start|>": 10, "<|vision_end|>": 12}
        processor.tokenizer.decode.side_effect = lambda ids, **kwargs: "".join(
            expected if token == 30 else "" if token == 99 else "PROMPT" for token in ids)
        batch = {"input_ids": torch.tensor([[10, 11, 12, 20, 30, 99, 0]]),
                 "labels": torch.tensor([[-100, -100, -100, -100, 30, 99, -100]]),
                 "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0]]),
                 "pixel_values": torch.ones((1, 3)), "image_grid_thw": torch.tensor([[1, 1, 1]])}
        with patch("trainer.sft_qlora_invoice_extraction.runtime.DataCollatorForVisionLanguageModeling") as factory:
            factory.return_value.return_value = batch
            checked = CheckedInvoiceCollator(processor)
        return example, batch, checked

    def test_pixels_completion_and_prompt_vision_padding_masks(self):
        example, batch, checked = self.fixture()
        result = BoundedCollator(None, 7, delegate=checked)([example])
        self.assertIs(result, batch)
        self.assertEqual(result["labels"][0].tolist(), [-100, -100, -100, -100, 30, 99, -100])
        self.assertIn("pixel_values", result)

    def test_corrupt_masks_missing_pixels_and_over_budget_fail(self):
        for index in (1, 3, 6):
            example, batch, checked = self.fixture()
            batch["labels"][0, index] = batch["input_ids"][0, index]
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                checked([example])
        example, batch, checked = self.fixture()
        del batch["pixel_values"]
        with self.assertRaisesRegex(RuntimeError, "missing pixels"):
            checked([example])
        example, _, checked = self.fixture()
        with self.assertRaisesRegex(ValueError, "No image/CSV truncation"):
            BoundedCollator(None, 6, delegate=checked)([example])


class ConfigurationCheckpointTests(unittest.TestCase):
    def test_defaults_reproducible_steps_and_smoke(self):
        args = parser().parse_args([])
        check_arguments(args)
        self.assertEqual((args.epochs, args.batch_size, args.grad_accum, args.seed), (2, 1, 4, 42))
        self.assertEqual((args.lora_rank, args.lora_alpha, args.lora_dropout), (8, 16, 0.05))
        self.assertEqual(expected_steps(161, 1, 4, 2), 82)
        self.assertEqual(vars(args), vars(parser().parse_args([])))
        with self.assertRaises(ValueError):
            check_arguments(parser().parse_args(["--smoke", "--max-steps", "2"]))

    def test_other_model_roots_cannot_be_output_destinations(self):
        for output in ("models/other_extraction", "models/qwen_25_finetune_dpo_qlora", "models", "UI"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "protected"):
                check_arguments(parser().parse_args(["--output-dir", output]))
        self.assertEqual(check_arguments(parser().parse_args([])), project_path(OUTPUT).resolve())

    def test_invalid_hyperparameters_fail_before_training(self):
        for flag, value in (("--grad-accum", "0"), ("--learning-rate", "nan"),
                            ("--max-steps", "0"), ("--max-pixels", "1"),
                            ("--lora-dropout", "1"), ("--epochs", "0")):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                check_arguments(parser().parse_args([flag, value]))

    def test_language_lora_targets_exclude_vision(self):
        modules = [(f"model.language_model.layers.0.self_attn.{suffix}", None)
                   for suffix in ("q_proj", "k_proj", "v_proj", "o_proj")]
        model = Mock()
        model.named_modules.return_value = modules
        self.assertEqual(language_targets(model), [name for name, _ in modules])
        model.named_modules.return_value = modules + [("model.visual.q_proj", None)]
        with self.assertRaises(RuntimeError):
            language_targets(model)

    def checkpoint(self, root, smoke):
        run = root / ("smoke/runs" if smoke else "runs") / "fixture"
        (run / "model_weights").mkdir(parents=True)
        config = {"model_id": MODEL_ID, "run_id": run.name, "smoke": smoke, "instruction": INSTRUCTION,
                  "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest()}
        metrics = {"global_steps_completed": 1, "adapter_parameters_changed": True,
                   "training_loss": 1.0, "evaluation": {"eval_loss": 1.1},
                   "initial_adapter_digest": "before", "final_adapter_digest": "after"}
        for name, value in (("training_config.json", config), ("manifest.json", {"counts": {"train": 161}}),
                            ("metrics.json", metrics), ("training_args.json", {"bf16": True})):
            (run / name).write_text(json.dumps(value))
        return run, config, metrics, {"sha256": "verified-weights", "tensor_count": 4}

    def test_smoke_isolated_and_completed_run_pointer_published(self):
        with tempfile.TemporaryDirectory() as tmp, patch("trainer.sft_qlora_invoice_extraction.train.verify_adapter") as verify:
            root = Path(tmp)
            pointer = root / "latest_successful.json"
            pointer.write_text('{"previous":"preserved"}')
            run, config, metrics, artifacts = self.checkpoint(root, True)
            verify.return_value = artifacts
            publish_success(root, run, config, metrics, artifacts)
            self.assertEqual(pointer.read_text(), '{"previous":"preserved"}')
            run, config, metrics, artifacts = self.checkpoint(root, False)
            publish_success(root, run, config, metrics, artifacts)
            self.assertEqual(json.loads(pointer.read_text())["adapter_path"], "runs/fixture/model_weights")
            self.assertFalse(verify_saved(run / "model_weights", root)["smoke"])
            (run / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                verify_saved(run / "model_weights", root)

    def test_failed_zero_step_or_unchanged_run_never_replaces_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pointer = root / "latest_successful.json"
            pointer.write_text('{"previous":"preserved"}')
            run, config, metrics, artifacts = self.checkpoint(root, False)
            for updates in ({"global_steps_completed": 0}, {"adapter_parameters_changed": False},
                            {"training_loss": float("nan")}):
                with self.subTest(updates=updates), self.assertRaises(RuntimeError):
                    publish_success(root, run, config, {**metrics, **updates}, artifacts)
                self.assertEqual(pointer.read_text(), '{"previous":"preserved"}')
            with self.assertRaises(ValueError):
                verify_saved(root / "../other_extraction/model_weights", root)


class ExtractionEvaluationTests(unittest.TestCase):
    def test_default_comparison_needs_only_the_retained_adapter(self):
        from trainer.sft_qlora_invoice_extraction import evaluate
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "runs/fixture/model_weights"
            weights.mkdir(parents=True)
            config = {"model_id": MODEL_ID, "expected_optimizer_steps": 82,
                      "instruction": INSTRUCTION, "instruction_sha256": evaluate.hashlib_prompt(INSTRUCTION),
                      "resolved_model_revision": "a" * 40}
            (weights.parent / "training_config.json").write_text(json.dumps(config))
            (weights.parent / "training_result.json").write_text(json.dumps({
                "smoke": False, "metrics": {"global_steps_completed": 82,
                "training_loss": .1, "evaluation": {"eval_loss": .2}}}))
            with patch.object(evaluate, "latest_adapter", return_value=weights) as latest, \
                 patch.object(evaluate, "verify_adapter", return_value={"sha256": "verified", "tensor_count": 288}), \
                 patch.object(evaluate, "verify_saved", return_value={"smoke": False}):
                adapters, instruction, revision = evaluate.resolve_checkpoints(cord_root=root)
            latest.assert_called_once_with(root.resolve())
            self.assertIsNone(adapters["baseline"]["path"])
            self.assertEqual(adapters["baseline"]["kind"], "pretrained_base")
            self.assertEqual(instruction, INSTRUCTION)
            self.assertEqual(revision, "a" * 40)

    def test_base_comparison_disables_lora_only_during_base_generation(self):
        from contextlib import contextmanager
        from trainer.sft_qlora_invoice_extraction.evaluate import AdapterComparison
        enabled = [True]

        @contextmanager
        def disable_adapter():
            enabled[0] = False
            try:
                yield
            finally:
                enabled[0] = True

        engine = AdapterComparison.__new__(AdapterComparison)
        engine.base_baseline = True
        engine.model = Mock(active_adapters=["cord_sft"])
        engine.model.peft_config = {"cord_sft": Mock(base_model_name_or_path=MODEL_ID)}
        engine.model.parameters.return_value = []
        engine.model.disable_adapter.side_effect = disable_adapter
        engine.select("baseline")
        with engine._adapter_context():
            self.assertFalse(enabled[0])
        self.assertTrue(enabled[0])
        engine.select("cord_sft")
        with engine._adapter_context():
            self.assertTrue(enabled[0])
        engine.model.disable_adapter.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

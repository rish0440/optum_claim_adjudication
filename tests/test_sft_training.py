"""CPU tests for Phase 5 configuration and checkpoint safety; no real training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.trainer.sft.train.train_qlora import check_arguments, parser, train
from src.trainer.sft.train.checkpoints import latest_adapter
from src.trainer.sft.training_data import read_records
from src.utils.paths import ROOT


class SFTPreparationTests(unittest.TestCase):
    def test_conservative_defaults_and_cli_overrides(self):
        args = parser().parse_args([])
        check_arguments(args)
        self.assertEqual((args.max_steps, args.batch_size, args.grad_accum, args.lora_rank), (8, 1, 2, 8))
        self.assertEqual((args.lora_alpha, args.lora_dropout, args.max_pixels), (16, 0.05, 200704))
        args = parser().parse_args(["--max-steps", "10", "--lora-rank", "4", "--lora-alpha", "8", "--dry-run"])
        self.assertEqual((args.max_steps, args.lora_rank, args.lora_alpha, args.dry_run), (10, 4, 8, True))

    def test_reject_invalid_memory_and_training_parameters(self):
        for flag, value in (("--batch-size", "0"), ("--max-steps", "0"),
                            ("--gpu-memory-fraction", "1.1"), ("--learning-rate", "nan"),
                            ("--lora-dropout", "1"), ("--max-pixels", "1")):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                check_arguments(parser().parse_args([flag, value]))

    def test_legacy_train_api_forwards_to_new_configuration(self):
        with patch("src.trainer.sft.train.train_qlora.run") as mocked:
            train(Path("data/synthetic"), Path("models/example"), model_name="example", max_steps=2, grad_accum=3)
        args = mocked.call_args.args[0]
        self.assertEqual((args.model, args.max_steps, args.grad_accum), ("example", 2, 3))

    def test_manifest_override_keeps_split_boundary(self):
        root = ROOT / "data/synthetic"
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
            adapter = root / "runs/test/adapter"
            adapter.mkdir(parents=True)
            (root / "latest_successful.json").write_text(json.dumps({
                "adapter_path": "runs/test/adapter", "model_id": "example", "run_id": "test"}))
            (adapter.parent / "training_result.json").write_text('{"status":"failed"}')
            with self.assertRaisesRegex(RuntimeError, "not successful"):
                latest_adapter(root)


if __name__ == "__main__":
    unittest.main()

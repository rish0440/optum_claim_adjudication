"""CPU tests only: no model loading, CUDA, network or optimization."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from src.utils.paths import ROOT
from src.trainer.preference_optimization.train.config import (
    MODEL_ID, TARGETS, TrainingConfig, dpo_arguments, language_targets, safe_output_root,
)
from src.trainer.preference_optimization.train.dataset import load_preferences, serialize_and_audit
from src.trainer.preference_optimization.train.train_dpo import build_trainer, publish_success, verify_latest


class ToyTokenizer:
    """Small serialization fixture; does not claim to emulate Qwen token IDs."""
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f'<{m["role"]}>{m["content"]}<eos>' for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if i != 0)


def toy_template(row, tokenizer):
    prompt = tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True)
    return {"prompt": prompt, **{
        key: tokenizer.apply_chat_template(row["prompt"]+row[key])[len(prompt):]
        for key in ("chosen", "rejected")}}


def toy_tokenize(row, **kwargs):
    def encode(value):
        parts = value.split("<eos>")
        ids = []
        for i, part in enumerate(parts):
            if i:
                ids.append(0)
            ids.extend(map(ord, part))
        return ids
    return {"prompt_input_ids": encode(row["prompt"]),
            **{key+"_input_ids": encode(row[key])+[0] for key in ("chosen", "rejected")}}


class DPOTrainingTests(unittest.TestCase):
    def test_existing_validated_data_and_test_is_never_opened(self):
        original_text, original_bytes = Path.read_text, Path.read_bytes
        def guarded_text(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.jsonl")
            return original_text(path, *args, **kwargs)
        def guarded_bytes(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.jsonl")
            return original_bytes(path, *args, **kwargs)
        with patch.object(Path, "read_text", guarded_text), patch.object(Path, "read_bytes", guarded_bytes):
            data, audit = load_preferences("data/dpo")
        self.assertEqual([len(data[s]) for s in ("train", "validation")], [60, 10])
        self.assertFalse(audit["test_accessed"])
        self.assertEqual(audit["counts_by_role"]["train"], {"agent_1": 30, "agent_2": 30})
        for records in data.values():
            for row in records:
                self.assertEqual([m["role"] for m in row["prompt"]], ["system", "user"])
                self.assertEqual(row["chosen"][0]["role"], "assistant")
                self.assertNotEqual(row["chosen"], row["rejected"])

    def copy_selected_data(self, directory):
        for name in ("manifest.json", "train.jsonl", "validation.jsonl"):
            (directory/name).write_bytes((ROOT/"data/dpo"/name).read_bytes())

    def test_malformed_duplicate_and_identical_records_rejected(self):
        for mutation in ("malformed", "duplicate", "identical"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                self.copy_selected_data(directory)
                records = [json.loads(line) for line in (directory/"train.jsonl").read_text().splitlines()]
                if mutation == "malformed":
                    records[0]["chosen"][0]["role"] = "user"
                elif mutation == "duplicate":
                    records[1] = records[0]
                else:
                    records[0]["rejected"] = records[0]["chosen"]
                (directory/"train.jsonl").write_text("\n".join(json.dumps(r) for r in records)+"\n")
                with self.assertRaises(ValueError):
                    load_preferences(directory)

    def test_cross_split_case_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            self.copy_selected_data(directory)
            manifest = json.loads((directory/"manifest.json").read_text())
            train = next(a for a in manifest["annotations"] if a["data_split"] == "train")
            validation = next(a for a in manifest["annotations"] if a["data_split"] == "validation")
            validation["synthetic_case_id"] = train["synthetic_case_id"]
            (directory/"manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "leakage"):
                load_preferences(directory)

    def test_completion_boundary_and_no_truncation(self):
        row = {"prompt": [{"role": "system", "content": "Select evidence"},
                          {"role": "user", "content": "Scoped candidates"}],
               "chosen": [{"role": "assistant", "content": '{"selected":0}'}],
               "rejected": [{"role": "assistant", "content": '{"selected":1}'}]}
        rendered, tokens = serialize_and_audit(row, ToyTokenizer(), 2048,
            template_fn=toy_template, tokenize_fn=toy_tokenize)
        self.assertNotIn(row["chosen"][0]["content"], rendered["prompt"])
        self.assertEqual(tokens["chosen_input_ids"][-1], 0)
        with self.assertRaisesRegex(ValueError, "budget"):
            serialize_and_audit(row, ToyTokenizer(), 2, template_fn=toy_template, tokenize_fn=toy_tokenize)

    def test_configuration_and_extraction_protection(self):
        self.assertEqual(TrainingConfig().validate().model, MODEL_ID)
        for values in ({"model": "Qwen/Qwen2.5-VL-3B-Instruct"}, {"rank": 0},
                       {"beta": 0}, {"learning_rate": float("nan")}, {"max_length": 4096}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainingConfig(**values).validate()
        for directory in ("models/qwen25_vl_sft", "models/qwen25_vl_sft/new", "models"):
            with self.assertRaises(ValueError):
                safe_output_root(directory)

    def test_language_only_targets(self):
        names = ["model.layers.0.self_attn."+target for target in TARGETS]
        self.assertEqual(language_targets(names), names)
        with self.assertRaises(ValueError):
            language_targets(names+["visual.attn.q_proj"])

    def test_mocked_trainer_has_genuine_reference_setup(self):
        factory, args_factory, lora_factory = Mock(), Mock(), Mock()
        build_trainer("model", "tokenizer", {"train": "train", "validation": "validation"},
            TrainingConfig(), Path("run"), True, trainer_cls=factory,
            dpo_config_cls=args_factory, lora_config_cls=lora_factory)
        kwargs = factory.call_args.kwargs
        self.assertIsNone(kwargs["ref_model"])
        self.assertFalse(args_factory.call_args.kwargs["reference_free"])
        self.assertIsNone(args_factory.call_args.kwargs["ref_adapter_name"])
        self.assertEqual(lora_factory.call_args.kwargs["task_type"], "CAUSAL_LM")
        self.assertEqual(lora_factory.call_args.kwargs["target_modules"], TARGETS)
        self.assertEqual(kwargs["eval_dataset"], "validation")
        self.assertIsNone(dpo_arguments(TrainingConfig(), Path("run"), True)["max_length"])

    def test_false_success_cannot_replace_previous_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"reasoning"
            run = root/"runs"/"new-run"
            (run/"adapter").mkdir(parents=True)
            pointer = root/"latest_successful.json"
            pointer.write_text('{"previous":"successful"}')
            config = TrainingConfig(max_steps=1, output_dir=str(root))
            valid = dict(global_steps_completed=1, training_loss=.6, eval_loss=.7,
                         policy_calls=1, reference_calls=1, gradient_checks=1, adapter_changed=True)
            for overrides in ({"adapter_changed": False}, {"reference_calls": 0},
                              {"global_steps_completed": 0}, {"eval_loss": float("nan")}):
                with self.assertRaises(RuntimeError):
                    publish_success(root, run, config, {**valid, **overrides}, {})
                self.assertEqual(pointer.read_text(), '{"previous":"successful"}')

    def test_validated_publication_and_pointer_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/"reasoning"
            run = root/"runs"/"new-run"
            (run/"adapter").mkdir(parents=True)
            for name in ("training_config", "training_args", "dataset_manifest", "metrics", "training_log"):
                (run/(name+".json")).write_text("{}")
            info = {"rank": 8, "adapter_sha256": "verified-checksum"}
            metrics = dict(global_steps_completed=1, training_loss=.6, eval_loss=.7,
                           policy_calls=1, reference_calls=1, gradient_checks=1, adapter_changed=True)
            (run/"metrics.json").write_text(json.dumps(metrics))
            (run/"training_config.json").write_text(json.dumps({"model_id": MODEL_ID, "max_steps": 1}))
            (run/"dataset_manifest.json").write_text(json.dumps({
                "train_pairs": 60, "validation_pairs": 10, "test_accessed": False}))
            with patch("src.trainer.preference_optimization.train.train_dpo.verify_adapter", return_value=info):
                publish_success(root, run, TrainingConfig(max_steps=1, output_dir=str(root)), metrics, info)
            pointer_path = root/"latest_successful.json"
            pointer = json.loads(pointer_path.read_text())
            self.assertEqual(pointer["model_id"], MODEL_ID)
            self.assertEqual(pointer["adapter_path"], "runs/new-run/adapter")
            pointer["adapter_path"] = "../../outside/adapter"
            pointer_path.write_text(json.dumps(pointer))
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                verify_latest(root)


if __name__ == "__main__":
    unittest.main()

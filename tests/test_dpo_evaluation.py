"""CPU-only tests with injected model doubles; no weights/network/CUDA loaded."""
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from src.trainer.preference_optimization.eval.evaluate import (
    Backend, completion_batch, evaluate_entries, generation_parameters, load_test, model_mode,
    output_root, preflight,
)
from src.trainer.preference_optimization.eval.metrics import (
    aggregate, assess_generation, preference_score, rate, sanitize_output,
)
from src.trainer.preference_optimization.eval.report import write_reports
from src.trainer.preference_optimization.train.config import MODEL_ID, PINNED
from src.utils.paths import ROOT


def fixture(role="agent_1"):
    if role == "agent_1":
        data = {"date_applicable": True, "items": [{"item_index": 0, "context_required": False,
            "candidates": [{"index": 0, "kind": "coverage", "category": "synthetic", "conditions": "Fixture"}]}]}
        chosen = {"items": [{"item_index": 0, "candidate_index": 0, "suggested_status": "COVERED", "basis": "RULE_MATCH"}]}
        rejected = {"items": [{"item_index": 0, "candidate_index": 0, "suggested_status": "EXCLUDED", "basis": "NO_EVIDENCE"}]}
    else:
        data = {"items": [{"item_index": 0, "facts": [{"index": 0}, {"index": 1}],
                          "concern_facts": [], "required_concerns": []},
                         {"item_index": 1, "facts": [{"index": 2}],
                          "concern_facts": [{"index": 2}], "required_concerns": [2]}]}
        chosen = {"items": [{"item_index": 0, "status": "partial", "fact_indices": [0, 1], "concerns": []},
                            {"item_index": 1, "status": "manual_review", "fact_indices": [2], "concerns": [2]}]}
        rejected = copy.deepcopy(chosen)
        rejected["items"][0]["fact_indices"] = [2]
    record = {"prompt": [{"role": "system", "content": "Test-only scoped selection fixture"},
                          {"role": "user", "content": json.dumps(data)}],
              "chosen": [{"role": "assistant", "content": json.dumps(chosen)}],
              "rejected": [{"role": "assistant", "content": json.dumps(rejected)}]}
    annotation = {"agent_role": role, "pair_id": "fixture-"+role, "scenario_family": "fixture-family",
                  "scenario": "synthetic_unit_fixture", "source_policy_ids": []}
    return record, annotation


class FakeModel:
    def __init__(self):
        self.peft_config = {"dpo": object()}
        self.layer = SimpleNamespace(lora_A={}, merged=False, disable_adapters=False,
                                     active_adapters=["dpo"], scaling={"dpo": 2})
        self.selected = None
    def modules(self):
        return [self.layer]
    def parameters(self):
        return []
    def set_adapter(self, name):
        self.selected = name
    @contextmanager
    def disable_adapter(self):
        self.layer.disable_adapters = True
        try:
            yield
        finally:
            self.layer.disable_adapters = False


class DPOEvaluationTests(unittest.TestCase):
    def test_loads_only_test_and_reuses_canonical_validation(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn(path.name, {"train.jsonl", "validation.jsonl"})
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            entries, audit = load_test()
        self.assertEqual(len(entries), 10)
        self.assertEqual(audit["by_role"], {"agent_1": 5, "agent_2": 5})
        self.assertFalse(audit["train_validation_opened"])

    def test_preflight_uses_existing_latest_verifier_not_training_loader(self):
        args = SimpleNamespace(split="test", max_new_tokens=512, seed=42,
                               dataset_dir="data/dpo", output_dir="reports/dpo/evaluation")
        saved = {"model_id": MODEL_ID, "model": MODEL_ID}
        audit = {"test_accessed": False, "manifest_sha256": "manifest"}
        verify = Mock(return_value={"adapter": str(ROOT/"models/qwen25_3b_reasoning_dpo/runs/test/adapter")})
        def read(path, **kwargs):
            return json.dumps(saved if path.name == "training_config.json" else audit)
        with patch("src.trainer.preference_optimization.eval.evaluate.sys.prefix", str(ROOT.parent/".venv")), \
             patch("src.trainer.preference_optimization.eval.evaluate.sys.base_prefix", "/base"), \
             patch("src.trainer.preference_optimization.eval.evaluate.sys.version_info", (3, 12)), \
             patch("src.trainer.preference_optimization.eval.evaluate.metadata.version", side_effect=lambda name: PINNED.get(name, "preserved-torch")), \
             patch("src.trainer.preference_optimization.eval.evaluate.load_test", return_value=([], {"manifest_sha256": "manifest"})), \
             patch("src.trainer.preference_optimization.eval.evaluate.verify_latest", verify), \
             patch("src.trainer.preference_optimization.eval.evaluate.sha", return_value="hash"), \
             patch.object(Path, "read_text", read):
            _, details = preflight(args)
        verify.assert_called_once_with("models/qwen25_3b_reasoning_dpo")
        self.assertEqual(details["model_id"], MODEL_ID)

    def test_disabled_baseline_enabled_dpo_and_no_merged_weights(self):
        model = FakeModel()
        with model_mode(model, "baseline") as state:
            self.assertTrue(model.layer.disable_adapters)
            self.assertFalse(state["adapter_enabled"])
        with model_mode(model, "dpo") as state:
            self.assertFalse(model.layer.disable_adapters)
            self.assertTrue(state["adapter_enabled"])
            self.assertEqual(model.selected, "dpo")
        model.layer.merged = True
        with self.assertRaises(RuntimeError), model_mode(model, "baseline"):
            pass

    def test_completion_only_mask_and_shift(self):
        batch = completion_batch([11, 12, 13], [21, 22])
        self.assertEqual(batch["input_ids"], [11, 12, 13, 21, 22])
        self.assertEqual(batch["labels"], [-100, -100, -100, 21, 22])
        self.assertEqual(batch["prediction_positions"], [2, 3])
        with self.assertRaises(ValueError):
            completion_batch([], [21])

    def test_logp_forward_alignment_on_cpu_with_mock_model(self):
        import torch
        original_tensor = torch.tensor
        # Production uses CUDA; this test deliberately routes the injected model
        # tensors to CPU and never invokes a CUDA operation.
        def cpu_tensor(*args, **kwargs):
            kwargs["device"] = "cpu"
            return original_tensor(*args, **kwargs)
        model = Mock(return_value=SimpleNamespace(logits=original_tensor(
            [[[0., 0., 0.], [0., 0., 0.], [100., 0., 0.]]])))
        backend = Backend(model, SimpleNamespace(eos_token_id=0, pad_token_id=0), 32)
        with patch.object(torch, "tensor", cpu_tensor):
            value = backend.logp([1, 2], [0, 1])
        self.assertAlmostEqual(value, -2*math.log(3), places=5)
        kwargs = model.call_args.kwargs
        self.assertEqual(kwargs["logits_to_keep"], 3)
        self.assertFalse(kwargs["use_cache"])
        self.assertEqual(kwargs["input_ids"].tolist(), [[1, 2, 0, 1]])

    def test_preference_ranking_ties_and_nonfinite(self):
        self.assertEqual(preference_score(-2, -5)["outcome"], "correct")
        self.assertEqual(preference_score(-5, -2)["outcome"], "incorrect")
        self.assertEqual(preference_score(-2, -2)["outcome"], "tied")
        with self.assertRaises(ValueError):
            preference_score(float("nan"), -2)

    def test_valid_json_and_both_real_wire_schemas(self):
        for role in ("agent_1", "agent_2"):
            record, annotation = fixture(role)
            result = assess_generation(record["chosen"][0]["content"], record, annotation)
            self.assertTrue(result["semantic_correct"])
            self.assertTrue(result["correct_scoped_references"])
            self.assertFalse(assess_generation("not JSON", record, annotation)["valid_json"])
            self.assertFalse(assess_generation('{"items":[]}', record, annotation)["valid_schema"])

    def test_unknown_candidate_unsupported_approval_and_citation(self):
        record, annotation = fixture()
        bad = json.loads(record["chosen"][0]["content"])
        bad["items"][0]["candidate_index"] = 999
        result = assess_generation(json.dumps(bad), record, annotation)
        self.assertEqual(result["unsupported_reference_count"], 1)
        self.assertEqual(result["unsupported_approval_count"], 1)
        bad["items"][0]["policy_ids"] = ["INVENTED"]
        result = assess_generation(json.dumps(bad), record, annotation)
        self.assertFalse(result["valid_schema"])
        self.assertGreater(result["unsupported_reference_count"], 0)

    def test_wrong_item_fact_missing_concern_and_changed_status(self):
        record, annotation = fixture("agent_2")
        result = assess_generation(record["rejected"][0]["content"], record, annotation)
        self.assertEqual(result["wrong_item_reference_count"], 1)
        bad = json.loads(record["chosen"][0]["content"])
        bad["items"][1]["concerns"] = []
        bad["items"][0]["status"] = "approved"
        result = assess_generation(json.dumps(bad), record, annotation)
        self.assertEqual(result["missing_required_concern_count"], 1)
        self.assertFalse(result["semantic_correct"])
        self.assertGreater(result["unsupported_review_count"], 0)
        bad["items"][0]["approved_amount"] = "999999"
        self.assertFalse(assess_generation(json.dumps(bad), record, annotation)["valid_schema"])

    def test_identical_prompt_and_generation_params_no_answer_leak(self):
        record, annotation = fixture()
        model = FakeModel()
        backend = Mock()
        backend.logp.side_effect = [-2, -5, -1, -4]
        backend.generate.return_value = (record["chosen"][0]["content"], False)
        tokens = {"prompt_input_ids": [11, 12], "chosen_input_ids": [21], "rejected_input_ids": [22]}
        predictions, scores = evaluate_entries([(record, annotation)], [tokens], backend, lambda mode: model_mode(model, mode))
        self.assertEqual([call.args for call in backend.generate.call_args_list], [([11, 12],), ([11, 12],)])
        tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=1)
        params = generation_parameters(tokenizer, 128)
        self.assertFalse(params["do_sample"])
        self.assertNotIn("temperature", params)
        self.assertEqual(len(scores), 1)
        metrics = aggregate(predictions, scores)
        self.assertEqual(metrics["baseline"]["overall"]["samples"], 1)
        self.assertEqual(metrics["dpo"]["by_role"]["agent_1"]["preference"]["correct"], 1)
        self.assertIsNone(rate(0, 0)["percent"])
        self.assertIsNone(metrics["baseline"]["overall"]["facts_correct"]["percent"])

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            meta = {"status": "completed", "dataset": {"pairs": 1, "test_sha256": "fixture"},
                    "model_id": MODEL_ID, "adapter": {"adapter_sha256": "fixture", "adapter": "fixture"}}
            write_reports(directory, predictions, scores, metrics, meta)
            self.assertEqual(json.loads((directory/"metadata.json").read_text())["status"], "completed")
            self.assertIn("not independently human verified", (directory/"comparison.md").read_text())
            self.assertTrue((directory/"human_review.jsonl").is_file())

    def test_failure_cannot_publish_fabricated_reports(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(ValueError):
                write_reports(directory, {"baseline": [], "dpo": []}, [], {},
                              {"status": "failed", "dataset": {"pairs": 10}})
            self.assertEqual(list(directory.iterdir()), [])
        with self.assertRaises(ValueError):
            output_root("data/dpo")
        self.assertNotIn("hf_secret", sanitize_output("hf_secret user@example.com"))


if __name__ == "__main__":
    unittest.main()

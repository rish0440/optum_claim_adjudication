"""CPU-only non-test fixtures and mocked HTTP; no GPU/server invocation."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import httpx

from src.inference.remote_client import RemoteVLMError
from src.trainer.preference_optimization.eval.metrics import assess_generation
from src.trainer.preference_optimization.eval.structured_agent2 import (
    BASE_NAME, DPO_NAME, StructuredClient, assess, load_cases, replay_review,
    response_format, selection_diagnostics, server_command,
)
from src.trainer.preference_optimization.train.config import MODEL_ID
from src.trainer.preference_optimization.eval.manual_review_fixture import load_manual_review_fixture
from src.utils.paths import ROOT


def unit_fixture():
    # New structural fixture, never written into preference train/val/test data.
    statuses = ["approved", "partial", "manual_review", "denied"]
    rows, chosen = [], []
    for index, status in enumerate(statuses):
        concerns = [2*index] if status == "manual_review" else []
        rows.append({"item_index": index, "facts": [
            {"index": 2*index, "kind": "decision", "status": status},
            {"index": 2*index+1, "kind": "policy"}],
            "concern_facts": [{"index": slot, "kind": "POLICY_CONTEXT_MISSING"} for slot in concerns],
            "required_concerns": concerns})
        chosen.append({"item_index": index, "status": status,
                       "fact_indices": [2*index, 2*index+1], "concerns": concerns})
    preferred = {"items": chosen}
    rejected = copy.deepcopy(preferred)
    rejected["items"][0]["fact_indices"] = [0]
    record = {"prompt": [{"role": "system", "content": "Return scoped fact selections"},
                          {"role": "user", "content": json.dumps({"items": rows, "totals": {"eligible_total": "10.00"}})}],
              "chosen": [{"role": "assistant", "content": json.dumps(preferred)}],
              "rejected": [{"role": "assistant", "content": json.dumps(rejected)}]}
    return record, {"agent_role": "agent_2"}, preferred


class StructuredAgent2Tests(unittest.TestCase):
    def test_new_manual_review_fixture_is_deterministic_and_outside_datasets(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotIn(path.name, {"train.jsonl", "validation.jsonl", "test.jsonl"})
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            entries, audit = load_manual_review_fixture()
            self.assertEqual((entries, audit), load_manual_review_fixture())
        self.assertEqual(len(entries), 1)
        record, annotation = entries[0]
        manifest = json.loads((ROOT/"data/dpo/manifest.json").read_text())
        self.assertNotIn(annotation["pair_id"], {a["pair_id"] for a in manifest["annotations"]})
        self.assertEqual(audit["canonical_rule_id"], "COV-005")
        expected = json.loads(record["chosen"][0]["content"])
        self.assertEqual(expected["items"], [{"item_index": 0, "status": "manual_review",
            "fact_indices": [0, 1], "concerns": [0]}])
        self.assertEqual([m["role"] for m in record["prompt"]], ["system", "user"])
        self.assertNotIn(record["chosen"][0]["content"], [m["content"] for m in record["prompt"]])
        self.assertIsNone(annotation["deterministic_claim"]["final_reimbursement"])
        self.assertEqual(annotation["deterministic_claim"]["pending_review_total"], "1475.25")
        accepted = assess(record["chosen"][0]["content"], record, annotation, {"finish_reason": "stop"})
        self.assertTrue(accepted["semantic_correct"])
        self.assertTrue(accepted["runtime_validator_accepted"])

    def test_manual_review_missing_concern_and_unsupported_approval_rejected(self):
        entries, _ = load_manual_review_fixture()
        record, annotation = entries[0]
        before = copy.deepcopy(annotation["deterministic_claim"])
        for field, value in (("concerns", []), ("fact_indices", [0]), ("status", "approved"), ("fact_indices", [0, 999])):
            with self.subTest(field=field, value=value):
                bad = json.loads(record["chosen"][0]["content"])
                bad["items"][0][field] = value
                result = assess(json.dumps(bad), record, annotation, {"finish_reason": "stop"})
                self.assertFalse(result["semantic_correct"])
                self.assertFalse(result["runtime_validator_accepted"])
                self.assertTrue(result["failures"])
                self.assertEqual(annotation["deterministic_claim"], before)

    def test_schema_is_actual_contract_without_gold_constraints(self):
        fmt = response_format()
        schema = fmt["json_schema"]["schema"]
        self.assertTrue(fmt["json_schema"]["strict"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), {"items"})
        row = schema["$defs"]["FactSelection"]
        self.assertFalse(row["additionalProperties"])
        self.assertEqual(set(row["required"]), {"item_index", "status", "fact_indices", "concerns"})
        self.assertNotIn("enum", row["properties"]["fact_indices"]["items"])
        self.assertNotIn("enum", row["properties"]["concerns"]["items"])
        self.assertEqual(row["properties"]["fact_indices"]["minItems"], 1)

    def test_approved_cap_review_and_exclusion_selections(self):
        record, annotation, chosen = unit_fixture()
        before = copy.deepcopy(record)
        result = assess_generation(json.dumps(chosen), record, annotation)
        self.assertTrue(result["semantic_correct"])
        self.assertEqual([r["expected_status"] for r in result["items"]], ["approved", "partial", "manual_review", "denied"])
        self.assertEqual(record, before)

    def test_forbidden_totals_missing_facts_wrong_item_and_no_repair(self):
        record, annotation, chosen = unit_fixture()
        for mutation in ("totals", "missing", "wrong_item", "concern", "amount"):
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(chosen)
                if mutation == "totals":
                    value["totals"] = {"eligible_total": "999"}
                elif mutation == "missing":
                    value["items"][0]["fact_indices"] = [0]
                elif mutation == "wrong_item":
                    value["items"][1]["fact_indices"] = [0, 1]
                elif mutation == "concern":
                    value["items"][2]["concerns"] = []
                else:
                    value["items"][0]["approved_amount"] = "999"
                raw = json.dumps(value)
                result = assess_generation(raw, record, annotation)
                self.assertFalse(result["semantic_correct"])
                self.assertEqual(json.loads(raw), value)
        value = copy.deepcopy(chosen)
        value["items"][1]["fact_indices"] = [0, 1]
        self.assertEqual(assess_generation(json.dumps(value), record, annotation)["wrong_item_reference_count"], 2)

    def test_repeated_evidence_and_truncation_stay_invalid(self):
        record, annotation, chosen = unit_fixture()
        for raw in (record["prompt"][1]["content"], '{"items":['):
            result = assess(raw, record, annotation, {"finish_reason": "length"})
            self.assertFalse(result["semantic_correct"])
            self.assertIn("generation_did_not_reach_eos", result["failures"])
        with patch("src.trainer.preference_optimization.eval.structured_agent2.replay_review", return_value={}) as replay:
            result = assess(json.dumps(chosen), record, annotation, {"finish_reason": "length"})
        replay.assert_not_called()
        self.assertFalse(result["semantic_correct"])

    def test_diagnostics_do_not_hide_semantic_errors_behind_totals(self):
        record, _, chosen = unit_fixture()
        chosen["totals"] = {"eligible_total": "999"}
        chosen["items"][0]["fact_indices"] = [2]
        diagnostics = selection_diagnostics(json.dumps(chosen), record)
        self.assertEqual(diagnostics["forbidden_root_fields"], ["totals"])
        self.assertEqual(diagnostics["items"][0]["allowed_fact_indices"], [0, 1])
        self.assertEqual(diagnostics["items"][0]["selected_fact_indices"], [2])

    def test_validation_split_only_and_real_runtime_validator_preserves_amounts(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            self.assertNotEqual(path.name, "test.jsonl")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            entries, audit = load_cases("validation")
        self.assertEqual(len(entries), 5)
        self.assertFalse(audit["independent_generalization_evidence"])
        for record, annotation in entries:
            before = copy.deepcopy(annotation["deterministic_claim"])
            reviewed = replay_review(record["chosen"][0]["content"], annotation)
            for key in ("eligible_total", "copay_amount", "final_reimbursement"):
                self.assertEqual(reviewed["totals"][key], before[key])
            self.assertEqual(annotation["deterministic_claim"], before)
        with self.assertRaisesRegex(ValueError, "diagnostic retest"):
            load_cases("test")

    def test_real_http_invocation_distinct_models_identical_constraints(self):
        record, annotation, chosen = unit_fixture()
        requests = []
        adapter = Path("/synthetic/reasoning-adapter")
        def handler(request):
            payload = json.loads(request.content) if request.content else None
            if request.url.path == "/version":
                return httpx.Response(200, json={"version": "0.11.2"})
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [
                    {"id": BASE_NAME, "root": MODEL_ID, "parent": None},
                    {"id": DPO_NAME, "root": str(adapter), "parent": BASE_NAME}]})
            if request.url.path == "/tokenize":
                return httpx.Response(200, json={"tokens": [1, 2], "count": 2, "max_model_len": 2048})
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.method, "POST")
            requests.append(payload)
            # Deliberately incomplete semantic answer remains uncorrected.
            return httpx.Response(200, json={"model": payload["model"], "usage": {"completion_tokens": 50},
                "choices": [{"finish_reason": "stop", "message": {"content": record["rejected"][0]["content"]}}]})
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            client = StructuredClient("http://127.0.0.1:8002", adapter, http_client=http)
            client.check_models()
            self.assertEqual(client.token_ids(record["prompt"], BASE_NAME), client.token_ids(record["prompt"], DPO_NAME))
            for mode in ("baseline", "dpo"):
                raw, measured = client.generate(record["prompt"], mode)
                self.assertEqual(raw, record["rejected"][0]["content"])
                self.assertTrue(measured["structured_decoding"])
                self.assertEqual(measured["generation_tokens"], 50)
                self.assertFalse(assess_generation(raw, record, annotation)["semantic_correct"])
        self.assertEqual([p["model"] for p in requests], [BASE_NAME, DPO_NAME])
        left, right = copy.deepcopy(requests[0]), copy.deepcopy(requests[1])
        left.pop("model"); right.pop("model")
        self.assertEqual(left, right)
        self.assertEqual(left["messages"], record["prompt"])
        self.assertNotIn(record["chosen"][0]["content"], json.dumps(left))

    def test_wrong_model_mapping_and_no_secret_server_launch(self):
        client = StructuredClient("http://127.0.0.1:8002", "/synthetic/adapter")
        with patch.object(client.transport, "_request", side_effect=[{"version": "0.11.2"}, {"data": []}]):
            with self.assertRaises(RemoteVLMError):
                client.check_models()
        command = server_command({"adapter": "/synthetic/adapter", "rank": 8})
        self.assertIn("Qwen/Qwen2.5-3B-Instruct", command)
        self.assertIn("--port 8002", command)
        self.assertIn("reasoning-dpo=/synthetic/adapter", command)
        self.assertNotIn("invoice-extractor", command)


if __name__ == "__main__":
    unittest.main()

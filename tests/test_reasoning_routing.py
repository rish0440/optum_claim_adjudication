"""Phase 9D CPU tests: real clients/graph, mocked HTTP and Milvus only."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import httpx
from PIL import Image

from tests import test_claim_graph as graph_tests
from src.agents.contracts import AgentFormatError, CandidateSelections
from src.config.reasoning import TEXT_MODEL_ID, reasoning_route, successful_reasoning_adapter
from src.inference.reasoning_client import TextReasoningClient
from src.inference.vlm_client import VLMClient
from src.orchestration.workflow import run_claim_graph


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict("os.environ", {"OPTUM_REASONING_BACKEND": "phase8"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        # Reuse the existing canonical policy / mock-response test fixture.
        self.harness = graph_tests.ClaimGraphTests("test_compilation_and_start_to_end")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        self.adapter = (self.harness.folder / "adapter").resolve()
        self.pointer = patch("src.config.reasoning.successful_reasoning_adapter",
                             return_value=(self.adapter, "a" * 64))
        self.pointer.start()
        self.addCleanup(self.pointer.stop)
        self.requests = []
        self.wrong_identity = False
        self.offline = False
        self.http = httpx.Client(transport=httpx.MockTransport(self.respond))
        self.addCleanup(self.http.close)

    def respond(self, request):
        self.requests.append((request.url.port, request.url.path,
                              json.loads(request.content) if request.content else None))
        if request.url.port == 8002 and self.offline:
            raise httpx.ConnectError("test offline", request=request)
        if request.url.port == 8002 and request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [
                {"id": "reasoning-base", "root": TEXT_MODEL_ID, "parent": None},
                {"id": "reasoning-dpo", "root": str(self.adapter),
                 "parent": "wrong-base" if self.wrong_identity else "reasoning-base"}]})
        body = json.loads(request.content) if request.content else None
        alias = body.get("model") if body else None
        if alias in {"reasoning-base", "reasoning-dpo"}:
            # Unit-test HTTP mock only; production never substitutes model output.
            legacy = {**body, "model": "qwen-base"}
            delegated = httpx.Request(request.method, request.url, json=legacy)
            response = self.harness.respond(delegated)
            value = response.json()
            if "model" in value:
                value["model"] = alias
            return httpx.Response(response.status_code, json=value)
        return self.harness.respond(request)

    def graph(self, backend, **inputs):
        invoice_date = inputs.pop("invoice_date", "2024-06-01")
        if not any(key in inputs for key in ("csv_text", "csv_file", "image")):
            inputs["csv_text"] = graph_tests.CSV
        original_init = TextReasoningClient.__init__
        def text_init(client, endpoint="http://127.0.0.1:8000/v1", **kwargs):
            original_init(client, endpoint, http_client=self.http, **kwargs)
        def image_factory(config):
            real = VLMClient(config, http_client=self.http)
            wrapped = Mock()
            wrapped.extract.side_effect = lambda image: real.extract(image, output_dir=self.harness.folder / "extraction")
            return wrapped
        with patch.dict("os.environ", {"OPTUM_REASONING_BACKEND": backend}), \
             patch.object(TextReasoningClient, "__init__", text_init), \
             patch("src.orchestration.nodes.copilot.VLMClient", side_effect=image_factory):
            return run_claim_graph(invoice_date=invoice_date, policy=self.harness.policy,
                                   retriever=self.harness.retriever, **inputs)

    def test_default_and_opt_in_both_agents_totals_and_schema(self):
        claims = []
        for mode, model, port in (("phase8", "qwen-base", 8000),
                                  ("text_base", "reasoning-base", 8002),
                                  ("text_dpo", "reasoning-dpo", 8002)):
            self.requests.clear()
            report = self.graph(mode, request_id="same-case")
            claims.append(report["claim"])
            self.assertEqual(report["status"], "COMPLETED")
            self.assertTrue(report["reviewed"])
            calls = [(p, b) for p, path, b in self.requests if path == "/v1/chat/completions"]
            self.assertEqual(len(calls), 2)
            self.assertEqual([p for p, _ in calls], [port, port])
            self.assertEqual([b["model"] for _, b in calls], [model, model])
            self.assertEqual([b["response_format"]["json_schema"]["name"] for _, b in calls],
                             ["CandidateSelections", "FactSelections"])
            self.assertTrue(all(b["response_format"]["json_schema"]["strict"] for _, b in calls))
            self.assertEqual(report["metadata"]["model_ids"]["agent_1"], model)
            self.assertEqual(report["metadata"]["model_ids"]["agent_2"], model)
            self.assertEqual(report["metadata"]["agent_1"]["model"], model)
            self.assertEqual(report["metadata"]["agent_2"]["model"], model)
        self.assertEqual(claims[0], claims[1])
        self.assertEqual(claims[1], claims[2])

    def test_dpo_image_stays_on_8000_and_reasoning_on_8002(self):
        image = self.harness.folder / "synthetic.png"
        Image.new("RGB", (32, 32), "white").save(image)
        report = self.graph("text_dpo", image=image)
        self.assertTrue(report["reviewed"])
        calls = [(p, b["model"]) for p, path, b in self.requests if path == "/v1/chat/completions"]
        self.assertEqual(calls, [(8000, "invoice-extractor"), (8002, "reasoning-dpo"), (8002, "reasoning-dpo")])

    def test_reference_manual_review_and_exact_totals_unchanged_across_modes(self):
        path = graph_tests.get_settings().paths.evaluation_ground_truth / "invoice_01_standard.csv"
        claims = []
        for mode in ("phase8", "text_base", "text_dpo"):
            report = self.graph(mode, csv_file=path, invoice_date="2024-01-15")
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertTrue(report["reviewed"])
            claim = report["claim"]
            claims.append(claim)
            for field, value in {"billed_total": "10500.00", "eligible_total": "8300.00",
                                 "disallowed_total": "1000.00", "pending_review_total": "1200.00",
                                 "copay_amount": "830.00"}.items():
                self.assertEqual(claim[field], value)
            self.assertIsNone(claim["final_reimbursement"])
            self.assertEqual(report["agent_2"]["items"][2]["status"], "manual_review")
            self.assertTrue(report["metadata"]["agent_2"]["fact_selections"][2]["concerns"])
        self.assertEqual(claims[0], claims[1])
        self.assertEqual(claims[1], claims[2])

    def test_bad_agent_outputs_never_override_amounts(self):
        for error in ("money", "id", "status", "fact", "wrong_item", "json"):
            self.harness.bad_agent_2 = error
            report = self.graph("text_dpo")
            self.assertFalse(report["reviewed"])
            self.assertEqual(report["status"], "REVIEW_REQUIRED")
            self.assertEqual(report["claim"]["eligible_total"], "1000.00")
            self.assertEqual(report["claim"]["final_reimbursement"], "900.00")
        self.harness.bad_agent_2 = None
        self.harness.bad_agent_1 = "id"
        self.assertEqual(self.graph("text_dpo")["status"], "FAILED")

    def test_wrong_identity_or_unavailable_server_never_falls_back(self):
        for flag in ("wrong_identity", "offline"):
            setattr(self, flag, True)
            self.requests.clear()
            report = self.graph("text_dpo")
            self.assertEqual(report["status"], "FAILED")
            self.assertIsNone(report["claim"])
            self.assertTrue(all(port == 8002 for port, _, _ in self.requests))
            self.assertIn("no alternate model", " ".join(report["warnings"]))
            setattr(self, flag, False)

    def test_excess_context_and_truncation_are_rejected(self):
        self.harness.context_count = 2000
        self.assertEqual(self.graph("text_dpo")["status"], "FAILED")
        self.harness.context_count = 400
        self.harness.truncated = True
        self.assertEqual(self.graph("text_dpo")["status"], "FAILED")

    def test_default_schema_parser_still_rejects_forbidden_fields(self):
        client = TextReasoningClient(http_client=httpx.Client(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json={"data": [{"id": "qwen-base", "parent": None,
                "root": "Qwen/Qwen2.5-VL-3B-Instruct"}]} if request.method == "GET" else
                {"count": 10, "max_model_len": 2048} if request.url.path == "/tokenize" else
                {"model": "qwen-base", "choices": [{"finish_reason": "stop", "message": {
                    "content": '{"items":[],"totals":100}'}}]}))))
        self.addCleanup(client.http_client.close)
        with self.assertRaises(AgentFormatError):
            client.generate("instruction", {}, CandidateSelections, deadline=time.monotonic() + 10)

    def test_invalid_switch_and_extraction_endpoint_rejected(self):
        with patch.dict("os.environ", {"OPTUM_REASONING_BACKEND": "typo"}):
            with self.assertRaises(ValueError):
                reasoning_route()
        with patch.dict("os.environ", {"OPTUM_REASONING_BACKEND": "text_dpo"}):
            with self.assertRaises(ValueError):
                reasoning_route("http://127.0.0.1:8002/v1")


class CheckpointIdentityTests(unittest.TestCase):
    def test_pointer_checksum_and_success_are_required(self):
        with tempfile.TemporaryDirectory() as directory, patch("src.config.reasoning.ROOT", Path(directory)):
            root = Path(directory) / "models/qwen25_3b_reasoning_dpo"
            adapter = root / "runs/test-run/adapter"
            adapter.mkdir(parents=True)
            weights = adapter / "adapter_model.safetensors"
            weights.write_bytes(b"CPU checksum fixture, not real model weights")
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            pointer = {"model_id": TEXT_MODEL_ID, "run_id": "test-run",
                       "adapter_path": "runs/test-run/adapter", "adapter_sha256": digest}
            (root / "latest_successful.json").write_text(json.dumps(pointer))
            result = {"model_id": TEXT_MODEL_ID, "run_id": "test-run", "status": "success",
                      "adapter": {"adapter_sha256": digest}}
            (adapter.parent / "training_result.json").write_text(json.dumps(result))
            (adapter / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": TEXT_MODEL_ID, "peft_type": "LORA", "r": 8}))
            self.assertEqual(successful_reasoning_adapter(), (adapter.resolve(), digest))
            weights.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                successful_reasoning_adapter()
            pointer["adapter_path"] = "../extraction/adapter"
            (root / "latest_successful.json").write_text(json.dumps(pointer))
            with self.assertRaises(ValueError):
                successful_reasoning_adapter()

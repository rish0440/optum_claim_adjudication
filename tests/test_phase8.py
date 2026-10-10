"""Phase 8 CPU tests for real node functions; compiled-graph tests need LangGraph.

No VLM inference, network requests, semantic embedding downloads or Milvus
instance are required for the unit test suite.
"""
import importlib.util
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from src.agentic_workflow.nodes.adjudication.node import calculate_claim
from src.agentic_workflow.nodes.csv_validation.node import validate_csv
from src.agentic_workflow.nodes.invoice_extraction.node import extract_invoice
from src.agentic_workflow.nodes.policy_retrieval.node import retrieve_policy
from src.agentic_workflow.nodes.response_builder.node import build_response
from src.agentic_workflow.nodes.rule_resolution.node import resolve_policy_rules
from src.policy import load_policy
from src.settings import get_settings


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, item, top_k):
        self.calls.append(item)
        return [{"rule_id": "EXC-001", "score": 0.92}]


def run_test_nodes(state, *, retriever=None):
    """Exercise real node logic without claiming to execute a LangGraph runtime."""
    for fn in (extract_invoice, validate_csv):
        state.update(fn(state))
        if state.get("errors"):
            break
    if not state.get("errors"):
        state.update(retrieve_policy(state, retriever=retriever))
    if not state.get("errors"):
        state.update(resolve_policy_rules(state))
    if not state.get("errors"):
        state.update(calculate_claim(state))
    state.update(build_response(state))
    return state["result"]


class Phase8NodeTests(unittest.TestCase):
    def base(self, **kwargs):
        return {"errors": [], "warnings": [], "trace": [], "metadata": {},
                "use_milvus": False, "strict_rag": False, **kwargs}

    def test_five_reference_invoices_through_nodes(self):
        folder = get_settings().paths.evaluation_ground_truth
        expected = ["8550.00", "4545.00", "22410.00", "11754.00", "8055.00"]
        for idx, amount in enumerate(expected, 1):
            path = sorted(folder.glob("invoice_*.json"))[idx - 1]
            with self.subTest(invoice=path.name):
                result = run_test_nodes(self.base(input_mode="invoice_json",
                                                   invoice_json_path=str(path)))
                self.assertEqual(result["status"], "finalized")
                self.assertEqual(Decimal(result["claim"]["final_reimbursement"]), Decimal(amount))
                self.assertEqual(result["extraction_source"], "verified_structured_json")
                self.assertEqual(result["rag_status"], "policy_only_no_vector_search")
                self.assertEqual(result["trace"], ["invoice_extraction", "csv_validation",
                                                  "policy_retrieval", "rule_resolution",
                                                  "adjudication", "response_builder"])
                self.assertIn("not a VLM prediction", " ".join(result["warnings"]))

    def test_csv_input_without_rate_column(self):
        text = "Item,Quantity,Rate,Amount\nBlood Test (CBC),2,,1400\n"
        result = run_test_nodes(self.base(input_mode="csv_text", csv_text=text,
                                          invoice_date="2024-04-18"))
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["claim"]["final_reimbursement"], "1260.00")

    def test_unknown_item_triggers_manual_review(self):
        text = "Item,Quantity,Rate,Amount\nUnlisted therapy,1,100,100\n"
        result = run_test_nodes(self.base(input_mode="csv_text", csv_text=text,
                                          invoice_date="2024-04-18"))
        self.assertEqual(result["status"], "manual_review")
        self.assertIsNone(result["claim"]["final_reimbursement"])
        self.assertEqual(result["claim"]["decisions"][0]["status"], "manual_review")

    def test_invalid_row_cannot_run_financial_nodes(self):
        text = "Item,Quantity,Rate,Amount\nMedicines,2,100,300\n"
        result = run_test_nodes(self.base(input_mode="csv_text", csv_text=text,
                                          invoice_date="2024-04-18"))
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["claim"])
        self.assertEqual(result["error_stage"], "csv_validation")
        self.assertNotIn("adjudication", result["trace"])

    def test_bad_header_is_rejected(self):
        result = run_test_nodes(self.base(input_mode="csv_text",
                                          csv_text="Item,Quantity,Amount\nMedicines,1,30\n",
                                          invoice_date="2024-04-18"))
        self.assertEqual(result["error_stage"], "csv_validation")

    def test_image_requires_actual_gpu_model(self):
        result = run_test_nodes(self.base(input_mode="image", image_path="invoice.jpg",
                                          invoice_date="2024-04-18"))
        self.assertEqual(result["status"], "error")
        self.assertIn("vLLM is on hold", " ".join(result["errors"]))
        self.assertNotIn("adjudication", result["trace"])

    def test_milvus_semantic_candidates_are_used(self):
        retriever = FakeRetriever()
        result = run_test_nodes(self.base(input_mode="csv_text",
                                          csv_text="Item,Quantity,Rate,Amount\nPersonal Care Kit,1,50,50\n",
                                          invoice_date="2024-04-18", use_milvus=True),
                                retriever=retriever)
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["rag_status"], "milvus_live")
        self.assertEqual(retriever.calls, ["Personal Care Kit"])
        self.assertEqual(result["claim"]["decisions"][0]["rule_id"], "EXC-001")

    def test_milvus_error_is_transparently_fallback(self):
        class BrokenRetriever:
            def search(self, item, top_k):
                raise RuntimeError("test-unavailable")
        state = self.base(input_mode="csv_text",
                          csv_text="Item,Quantity,Rate,Amount\nMedicines,1,100,100\n",
                          invoice_date="2024-04-18", use_milvus=True)
        result = run_test_nodes(state, retriever=BrokenRetriever())
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["rag_status"], "policy_fallback_after_milvus_error")
        self.assertIn("test-unavailable", " ".join(result["warnings"]))

    def test_strict_rag_fails_without_claim(self):
        class BrokenRetriever:
            def search(self, item, top_k):
                raise RuntimeError("test-down")
        state = self.base(input_mode="csv_text",
                          csv_text="Item,Quantity,Rate,Amount\nMedicines,1,100,100\n",
                          invoice_date="2024-04-18", use_milvus=True, strict_rag=True)
        result = run_test_nodes(state, retriever=BrokenRetriever())
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_stage"], "policy_retrieval")
        self.assertIsNone(result["claim"])

    def test_forged_rule_cannot_change_cap_or_status(self):
        state = self.base(input_mode="csv_text", invoice_date="2024-04-18",
                          csv_text="Item,Quantity,Rate,Amount\nRoom Charges,1,5000,5000\n")
        for fn in (extract_invoice, validate_csv, retrieve_policy, resolve_policy_rules):
            state.update(fn(state))
        forged = state["resolved_pairs"][0]["rule"].model_copy(update={"maximum_reimbursement": Decimal("999999")})
        state["resolved_pairs"][0]["rule"] = forged
        out = calculate_claim(state)
        self.assertEqual(out["error_stage"], "adjudication")
        self.assertIn("differs from policy", " ".join(out["errors"]))

    def test_missing_json_file_reports_extraction_error(self):
        result = run_test_nodes(self.base(input_mode="invoice_json", invoice_json_path="missing-file.json"))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_stage"], "invoice_extraction")

    def test_remote_vlm_fake_client_extracts_csv(self):
        from src.agentic_workflow.nodes.invoice_extraction.node import extract_invoice
        from types import SimpleNamespace
        class FakeVLM:
            def extract(self, path):
                return SimpleNamespace(csv_text="Item,Quantity,Rate,Amount\nMedicines,1,100,100\n",
                                       model="test-model", latency_ms=50)
        state = self.base(input_mode="image", image_path="dummy.jpg", invoice_date="2024-04-18",
                          use_remote_vlm=True)
        state.update(extract_invoice(state, remote_client=FakeVLM()))
        for fn in (validate_csv, retrieve_policy, resolve_policy_rules, calculate_claim):
            state.update(fn(state))
        result = build_response(state)["result"]
        self.assertEqual(result["status"], "finalized")
        self.assertEqual(result["extraction_source"], "remote_vlm")
        self.assertEqual(result["metadata"]["vlm_latency_ms"], 50)


if importlib.util.find_spec("langgraph") is not None:
    class Phase8CompiledGraphTests(unittest.TestCase):
        def test_compiled_langgraph_executes(self):
            from src.agentic_workflow.workflow import run_claim_workflow
            folder = get_settings().paths.evaluation_ground_truth
            path = folder / "invoice_01_standard.json"
            result = run_claim_workflow(invoice_json=path)
            self.assertEqual(Decimal(result["claim"]["final_reimbursement"]), Decimal("8550"))

        def test_compiled_graph_short_circuits_bad_csv(self):
            from src.agentic_workflow.workflow import run_claim_workflow
            result = run_claim_workflow(csv_text="Broken,csv", invoice_date="2024-04-18")
            self.assertEqual(result["error_stage"], "csv_validation")
            self.assertEqual(result["trace"], ["invoice_extraction", "csv_validation", "response_builder"])


if __name__ == "__main__":
    unittest.main()

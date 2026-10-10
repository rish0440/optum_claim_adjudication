"""CPU-only Phase 8 regression tests; all graph/network work is mocked."""
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from app.main import app
from app import claims
from app.invoice_date import InvoiceDate, extract_invoice_date
from app.claim_report import generate_claim_pdf, money
from src.inference.remote_client import RemoteVLMError


def image_bytes(size=(40, 60)):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, format="PNG")
    return stream.getvalue()


def report_fixture(request_id):
    # Mock service result only: production always invokes the real claim graph.
    decisions = []
    for name, qty, billed, approved, denied, pending, status, rule in (
        ("Room Charges", 2, "7000", "6000", "1000", "0", "partial", "COV-001"),
        ("Consultation", 1, "1500", "1500", "0", "0", "approved", "COV-003"),
        ("Medicines", 1, "1200", "0", "0", "1200", "manual_review", None),
        ("Blood Test", 1, "800", "800", "0", "0", "approved", "COV-010")):
        decisions.append(dict(item=name, quantity=qty, billed_amount=billed, approved_amount=approved,
            disallowed_amount=denied, pending_review_amount=pending, status=status, rule_id=rule,
            explanation="Verified deterministic decision", review_note="Medical relationship required" if pending != "0" else None))
    return dict(request_id=request_id, thread_id=None, status="REVIEW_REQUIRED", reviewed=True,
        claim=dict(invoice_id=request_id, policy_id="TEST-POLICY", currency="INR", claim_status="manual_review",
            decisions=decisions, billed_total="10500", eligible_total="8300", disallowed_total="1000",
            pending_review_total="1200", copay_rate_percent="10", copay_amount="830", final_reimbursement=None, notes=[]),
        extracted_csv="Item,Quantity,Rate,Amount\nBlood Test,1,,800\n", evidence={}, agent_1=None,
        agent_2=None, explanations=[], errors=[], warnings=[], retries={}, trace=[{"node": "invoice_extraction"}], metadata={})


DATE = dict(status="FOUND", invoice_date="2024-01-15", printed_text="2024-01-15",
            source="qwen-base image metadata", verification_status="model_extracted_unverified")


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict("os.environ", {"OPTUM_CLAIM_REPORT_DIR": self.tmp.name,
                          "VLM_BASE_URL": "http://127.0.0.1:8000/v1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(self.tmp.cleanup)

    def upload(self, name="invoice.png", data=None, mime="image/png"):
        return self.client.post("/api/claims/analyze", files={"file": (name, image_bytes() if data is None else data, mime)})

    def run_job(self, request_id, date_result=DATE, graph_result=None):
        directory = self.root / request_id
        directory.mkdir()
        image = directory / "upload.png"
        image.write_bytes(image_bytes())
        claims.jobs[request_id] = dict(request_id=request_id, state="queued", result=None, invoice_date=None, report_available=False)
        claims.slots.acquire()
        with patch("app.claims.extract_invoice_date", return_value=date_result), patch("app.claims._graph", return_value=graph_result or report_fixture(request_id)) as graph, patch("app.claims.generate_claim_pdf") as pdf:
            claims._process(request_id, image, directory)
        self.assertFalse(image.exists())
        self.addCleanup(claims.jobs.pop, request_id, None)
        return claims.jobs[request_id], graph, pdf

    def test_app_and_image_only_ui_legacy_routes_preserved(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Claim Adjudication Copilot", response.text)
        self.assertNotIn('type="date"', response.text)
        self.assertNotIn(".csv,.json", response.text)
        paths = self.client.get("/openapi.json").json()["paths"]
        for route in ("/api/adjudicate", "/api/adjudicate/demo", "/api/claims/analyze"):
            self.assertIn(route, paths)

    def test_nonimages_invalid_mime_corrupt_and_large_uploads(self):
        for name, data, mime, expected in (("invoice.csv", b"a,b", "text/csv", 415),
            ("invoice.png", b"bad", "image/png", 422),
            ("invoice.png", image_bytes(), "image/jpeg", 415),
            ("invoice.png", b"x"*(claims.MAX_BYTES+1), "image/png", 413)):
            self.assertEqual(self.upload(name, data, mime).status_code, expected)
        with patch("app.claims.MAX_PIXELS", 100):
            self.assertEqual(self.upload().status_code, 413)

    def test_upload_queued_controlled_filename(self):
        with patch("app.claims.executor.submit") as submit:
            response = self.upload("../../invoice.png")
        self.assertEqual(response.status_code, 202)
        request_id = response.json()["request_id"]
        self.addCleanup(claims.jobs.pop, request_id, None)
        self.addCleanup(claims.slots.release)
        self.assertEqual(submit.call_args.args[2].name, "upload.png")
        self.assertEqual(submit.call_args.args[2].parent.name, request_id)

    def test_review_graph_wiring_and_exact_null_finances(self):
        request_id = "a"*32
        job, graph, pdf = self.run_job(request_id)
        args = graph.call_args.kwargs
        self.assertEqual(args["invoice_date"], "2024-01-15")
        self.assertEqual(args["request_id"], request_id)
        self.assertEqual(args["image"].name, "upload.png")
        self.assertNotIn("csv_file", args)
        self.assertTrue(job["result"]["reviewed"])
        claim = job["result"]["claim"]
        self.assertEqual(claim["eligible_total"], "8300")
        self.assertEqual(claim["copay_amount"], "830")
        self.assertIsNone(claim["final_reimbursement"])
        self.assertEqual(money(None), "NOT FINALIZED")
        pdf.assert_called_once()
        self.assertEqual(self.client.get(f"/api/claims/{request_id}").status_code, 200)

    def test_graph_facade_calls_existing_phase7_api(self):
        with patch("src.orchestration.workflow.run_claim_graph", return_value={"genuine": True}) as graph:
            result = claims._graph(image=Path("private/upload.png"), invoice_date="2024-01-15", request_id="f"*32)
        self.assertTrue(result["genuine"])
        self.assertEqual(graph.call_args.kwargs["image"], Path("private/upload.png"))

    def test_missing_ambiguous_date_never_invokes_graph(self):
        for index, status in enumerate(("MISSING", "AMBIGUOUS")):
            job, graph, pdf = self.run_job(str(index)*32, dict(status=status, invoice_date=None, printed_text=None))
            self.assertEqual(job["state"], "review_required")
            graph.assert_not_called(); pdf.assert_not_called()
            self.assertIsNone(job["result"])

    def test_gpu_unavailable_fails_without_approval(self):
        request_id = "b"*32
        with patch("app.claims.extract_invoice_date", side_effect=RemoteVLMError("Unavailable")):
            # run_job overrides extraction, so prepare this job directly.
            directory = self.root / request_id; directory.mkdir()
            image = directory / "upload.png"; image.write_bytes(image_bytes())
            claims.jobs[request_id] = dict(request_id=request_id, result=None, report_available=False)
            self.addCleanup(claims.jobs.pop, request_id, None)
            claims.slots.acquire()
            with patch("app.claims._graph") as graph:
                claims._process(request_id, image, directory)
            graph.assert_not_called()
        self.assertEqual(claims.jobs[request_id]["state"], "failed")

    def test_safe_download_and_not_found(self):
        request_id = "c"*32
        directory = self.root / request_id; directory.mkdir()
        (directory / "report.pdf").write_bytes(b"%PDF-1.4\n")
        claims.jobs[request_id] = dict(request_id=request_id, report_available=True)
        self.addCleanup(claims.jobs.pop, request_id, None)
        self.assertEqual(self.client.get(f"/api/claims/{request_id}/report.pdf").status_code, 200)
        self.assertEqual(self.client.get(f"/api/claims/{'d'*32}/report.pdf").status_code, 404)
        with self.assertRaises(HTTPException):
            claims._job("../../etc/passwd")


class DateTests(unittest.TestCase):
    def test_calendar_and_ambiguous_transcription(self):
        InvoiceDate.model_validate({k: DATE[k] for k in ("status", "invoice_date", "printed_text")})
        for iso, printed in (("2024-02-30", "2024-02-30"), ("2024-01-02", "01/02/2024"),
                             ("2024-01-15", "2025-01-15")):
            with self.assertRaises(ValueError):
                InvoiceDate(status="FOUND", invoice_date=iso, printed_text=printed)

    def test_real_image_request_and_model_routing(self):
        client = Mock()
        client._request.side_effect = [{"data": [{"id": "qwen-base", "root": "Qwen/Qwen2.5-VL-3B-Instruct"}]},
            {"model": "qwen-base", "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({k: DATE[k] for k in ("status", "invoice_date", "printed_text")})}}]}]
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "invoice.png"; image.write_bytes(image_bytes())
            result = extract_invoice_date(image, client=client)
        payload = client._request.call_args.args[2]
        self.assertTrue(payload["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(result["verification_status"], "model_extracted_unverified")

    def test_invalid_json_truncation_and_unknown_model_are_rejected(self):
        cards = {"data": [{"id": "qwen-base", "root": "Qwen/Qwen2.5-VL-3B-Instruct"}]}
        for content, finish in (("not JSON", "stop"), ("{}", "length")):
            client = Mock()
            client._request.side_effect = [cards, {"model": "qwen-base", "choices": [{"finish_reason": finish, "message": {"content": content}}]}]
            with tempfile.TemporaryDirectory() as tmp:
                image = Path(tmp) / "invoice.png"; image.write_bytes(image_bytes())
                with self.assertRaises(ValueError):
                    extract_invoice_date(image, client=client)
        client = Mock(); client._request.return_value = {"data": []}
        with self.assertRaises(RemoteVLMError):
            extract_invoice_date("not-opened.png", client=client)


class PDFTests(unittest.TestCase):
    def test_review_and_multipage_item_reports(self):
        # A missing ReportLab installation is an explicit failing dependency,
        # not a silent skip of PDF validation.
        report = report_fixture("e"*32)
        before = copy.deepcopy(report)
        from reportlab.platypus import Paragraph
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "review.pdf"
            with patch("reportlab.platypus.Paragraph", wraps=Paragraph) as paragraphs:
                generate_claim_pdf(report, DATE, path)
            text = " ".join(call.args[0] for call in paragraphs.call_args_list)
            for expected in ("NOT FINALIZED", "Room Charges", "Medicines", "INR 8,300.00", "INR 830.00", "Medical relationship required"):
                self.assertIn(expected, text)
            self.assertEqual(report, before)
            self.assertTrue(path.read_bytes().startswith(b"%PDF-"))
            self.assertGreater(path.stat().st_size, 1000)
            report["claim"]["decisions"] *= 15  # rendering stress only, not an adjudication fixture
            report["claim"]["decisions"][0]["item"] = "Very long hospital billing item "*12
            generate_claim_pdf(report, DATE, Path(tmp) / "many-items.pdf")
        self.assertIsNone(before["claim"]["final_reimbursement"])


if __name__ == "__main__":
    unittest.main()

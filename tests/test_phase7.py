"""Phase 7 local/offline tests, plus five untouched reference invoice evaluations.

No GPU, Milvus service, language model, or embeddings are required.
"""

import json
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import ValidationError

from scripts.run_adjudication import reference_evaluation
from src.adjudication import adjudicate_invoice, parse_invoice_csv
from src.policy import load_policy
from src.rag.resolver import PolicyRuleResolver
from src.schemas import InvoiceGroundTruth, InvoiceLine
from src.settings import get_settings


class Phase7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load_policy()
        cls.resolver = PolicyRuleResolver(cls.policy)

    def invoice(self, *rows, when="2024-06-01", name="edge_case"):
        lines = [InvoiceLine(item=item, quantity=qty, rate=None, amount=amount)
                 for item, qty, amount in rows]
        return InvoiceGroundTruth(
            invoice_id=name, invoice_date=when, image_file="synthetic-case",
            lines=lines,
            billed_total=sum((line.amount for line in lines), Decimal("0")),
            rate_column_visible=False,
        )

    def calculate(self, invoice):
        return adjudicate_invoice(invoice, policy=self.policy, resolver=self.resolver)

    def test_all_five_reference_invoices_match_expected(self):
        result = reference_evaluation(self.policy, self.resolver)
        self.assertEqual(result["invoice_count"], 5)
        self.assertEqual(result["line_count"], 26)
        self.assertEqual(result["metrics"]["invoice_total_exact"], {"correct": 5, "total": 5})
        for key in ("line_rule_id", "line_status", "line_eligible_exact"):
            self.assertEqual(result["metrics"][key], {"correct": 26, "total": 26})

    def test_shared_per_claim_cap_is_not_reapplied(self):
        invoice = self.invoice(("Ambulance", 1, "1000"), ("Ambulance", 1, "1000"))
        answer = self.calculate(invoice)
        self.assertEqual([d.approved_amount for d in answer.decisions],
                         [Decimal("1000"), Decimal("500")])
        self.assertEqual([d.status for d in answer.decisions], ["approved", "partial"])
        self.assertEqual(answer.disallowed_total, Decimal("500"))
        self.assertEqual(answer.eligible_total, Decimal("1500"))
        self.assertEqual(answer.copay_amount, Decimal("150"))
        self.assertEqual(answer.final_reimbursement, Decimal("1350"))

    def test_per_day_and_per_visit_quantity_caps(self):
        invoice = self.invoice(("Room Charges", 3, "12000"),
                               ("Consultation", 2, "4000"),
                               ("ICU Charges", 2, "15000"))
        answer = self.calculate(invoice)
        self.assertEqual([d.approved_amount for d in answer.decisions],
                         [Decimal("9000"), Decimal("3000"), Decimal("12000")])
        self.assertEqual(answer.eligible_total, Decimal("24000"))
        self.assertEqual(answer.final_reimbursement, Decimal("21600"))

    def test_specific_scan_cap_and_exclusion(self):
        invoice = self.invoice(("MRI Scan", 2, "20000"),
                               ("X-Ray", 1, "1800"),
                               ("Personal Care Kit", 1, "500"))
        answer = self.calculate(invoice)
        self.assertEqual([d.rule_id for d in answer.decisions],
                         ["COV-007", "COV-009", "EXC-001"])
        self.assertEqual([d.approved_amount for d in answer.decisions],
                         [Decimal("16000"), Decimal("1000"), Decimal("0")])
        self.assertEqual(answer.final_reimbursement, Decimal("15300"))

    def test_unknown_item_never_finalizes_payout(self):
        invoice = self.invoice(("Medicines", 1, "1000"),
                               ("Unverified novel device", 1, "200"))
        answer = self.calculate(invoice)
        self.assertEqual(answer.claim_status, "manual_review")
        self.assertEqual(answer.eligible_total, Decimal("1000"))
        self.assertEqual(answer.pending_review_total, Decimal("200"))
        self.assertEqual(answer.disallowed_total, Decimal("0"))
        self.assertIsNone(answer.final_reimbursement)
        self.assertEqual(answer.decisions[1].status, "manual_review")
        self.assertEqual(answer.billed_total,
                         answer.eligible_total + answer.disallowed_total + answer.pending_review_total)

    def test_outside_policy_period_requires_review(self):
        invoice = self.invoice(("Room Charges", 1, "1000"), when="2025-01-01")
        answer = self.calculate(invoice)
        self.assertEqual(answer.claim_status, "manual_review")
        self.assertIsNone(answer.final_reimbursement)
        self.assertEqual(answer.pending_review_total, Decimal("1000"))
        self.assertIsNone(answer.decisions[0].rule_id)

    def test_copay_half_up_and_zero_co_pay(self):
        invoice = self.invoice(("Medicines", 1, "10.05"))
        answer = self.calculate(invoice)
        self.assertEqual(answer.copay_amount, Decimal("1.01"))
        self.assertEqual(answer.final_reimbursement, Decimal("9.04"))
        no_copay = self.policy.model_copy(deep=True)
        no_copay.reimbursement_rules.copay.applicable = False
        result = adjudicate_invoice(invoice, policy=no_copay)
        self.assertEqual(result.copay_amount, Decimal("0"))
        self.assertEqual(result.final_reimbursement, Decimal("10.05"))

    def test_missing_rate_is_not_invented(self):
        path = get_settings().paths.evaluation_ground_truth / "invoice_04_diagnostics.csv"
        invoice = parse_invoice_csv(path, invoice_date="2024-04-18")
        self.assertTrue(all(line.rate is None for line in invoice.lines))
        self.assertEqual(self.calculate(invoice).final_reimbursement, Decimal("11754"))

    def test_invalid_csv_and_arithmetic_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "malformed.csv"
            path.write_text("Item,Quantity,Rate,Amount\nMedicines,2,100,300\n", encoding="utf-8")
            with self.assertRaises(ValidationError):
                parse_invoice_csv(path, invoice_date="2024-06-01")
            path.write_text("Item,Quantity,Amount\nMedicines,2,200\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "CSV header"):
                parse_invoice_csv(path, invoice_date="2024-06-01")

    def test_invalid_resolver_rule_is_not_trusted(self):
        class FakeResolver:
            def resolve(self, item, hits):
                return {"rule_id": "MADE-UP-999", "status": "covered",
                        "review_reason": "unverified rule"}
        invoice = self.invoice(("Medicines", 1, "700"))
        result = adjudicate_invoice(invoice, policy=self.policy, resolver=FakeResolver())
        self.assertIsNone(result.decisions[0].rule_id)
        self.assertEqual(result.claim_status, "manual_review")
        self.assertIsNone(result.final_reimbursement)

    def test_retriever_is_optional_and_candidate_search_is_supported(self):
        class FakeRetriever:
            def __init__(self):
                self.requests = []
            def search(self, item, top_k):
                self.requests.append((item, top_k))
                return [{"rule_id": "COV-005", "score": 0.97}]
        retriever = FakeRetriever()
        result = adjudicate_invoice(self.invoice(("Medicines", 1, "100")),
                                    policy=self.policy, resolver=self.resolver,
                                    retriever=retriever)
        self.assertEqual(retriever.requests, [("Medicines", 5)])
        self.assertEqual(result.decisions[0].rule_id, "COV-005")
        self.assertEqual(result.final_reimbursement, Decimal("90"))


if __name__ == "__main__":
    unittest.main()

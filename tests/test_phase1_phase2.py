"""Run: python -m unittest discover -s tests -v  (stdlib only)."""

import csv
import io
import json
import random
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from PIL import Image

from src.adjudication.policy import load_policy, summarize_policy
from src.schemas import InvoiceGroundTruth, InvoiceLine
from src.config.settings import get_settings
from src.trainer.sft.dataset import (
    STYLES, lines_as_csv, make_lines, render_invoice, verify_dataset,
)


class Phase1Tests(unittest.TestCase):
    def test_settings_and_policy(self) -> None:
        conf=get_settings()
        self.assertTrue(conf.paths.policy.is_file())
        summary=summarize_policy(load_policy())
        self.assertEqual(summary["coverage_rules"],25)
        self.assertEqual(summary["exclusion_rules"],20)
        self.assertEqual(summary["copay_percent"],"10")
        self.assertEqual(summary["currency"],"INR")

    def test_five_gold_invoices_and_csv(self) -> None:
        root=get_settings().paths.evaluation_ground_truth
        files=sorted(root.glob("invoice_*.json"))
        self.assertEqual(len(files),5)
        total_rows=0
        for path in files:
            inv=InvoiceGroundTruth.model_validate_json(path.read_text(encoding="utf-8"))
            self.assertTrue((get_settings().paths.optum_invoices / f"{inv.invoice_id}.jpg").is_file())
            with (root / f"{inv.invoice_id}.csv").open(encoding="utf-8",newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),len(inv.lines))
            for actual, expected in zip(rows,inv.lines):
                self.assertEqual(actual["Item"],expected.item)
                self.assertEqual(int(actual["Quantity"]),expected.quantity)
                self.assertEqual(Decimal(actual["Amount"]),expected.amount)
                self.assertEqual(None if actual["Rate"]=="" else Decimal(actual["Rate"]),expected.rate)
            total_rows+=len(rows)
            if inv.invoice_id=="invoice_04_diagnostics":
                self.assertTrue(all(row["Rate"]=="" for row in rows))
        self.assertEqual(total_rows,26)

    def test_expected_claim_totals_reference(self) -> None:
        path=get_settings().paths.evaluation_ground_truth / "expected_adjudication.json"
        data=json.loads(path.read_text(encoding="utf-8"))
        expected={
            "invoice_01_standard":Decimal(8550),
            "invoice_02_exclusions":Decimal(4545),
            "invoice_03_icu_mixed":Decimal(22410),
            "invoice_04_diagnostics":Decimal(11754),
            "invoice_05_nonmedical":Decimal(8055),
        }
        self.assertEqual(len(data),5)
        for claim in data:
            self.assertEqual(Decimal(claim["reimbursement_total"]),expected[claim["invoice_id"]])
            self.assertEqual(
                Decimal(claim["eligible_total"])-Decimal(claim["copay_total"]),
                Decimal(claim["reimbursement_total"]),
            )


class Phase2Tests(unittest.TestCase):
    def test_ten_example_dataset(self) -> None:
        report=verify_dataset()
        self.assertEqual(report["status"],"PASS")
        self.assertEqual(report["total_images"],10)
        self.assertEqual((report["train"],report["validation"]),(8,2))
        self.assertEqual(report["total_rows"],64)
        self.assertEqual(set(report["layout_counts"]),set(STYLES))
        self.assertEqual(report["missing_rate_column_images"],2)

    def test_generator_single_image_each_layout(self) -> None:
        rng=random.Random(99)
        with tempfile.TemporaryDirectory() as tmp:
            for i,style in enumerate(STYLES):
                omitted = style=="borderless"
                lines=make_lines(rng,3,6,omit_rate=omitted)
                self.assertEqual(len(lines_as_csv(lines).splitlines()),len(lines)+1)
                destination=Path(tmp)/f"{style}.jpg"
                render_invoice(destination,lines,rng=rng,style=style,font_choice=i,
                               palette_index=i,invoice_number=i+1)
                with Image.open(destination) as img:
                    self.assertEqual(img.format,"JPEG")
                    self.assertGreater(img.width,800)
                    self.assertGreater(img.height,400)
                parsed=list(csv.DictReader(io.StringIO(lines_as_csv(lines))))
                self.assertTrue(all((x["Rate"]=="")==omitted for x in parsed))

    def test_model_validates_amount_arithmetic(self) -> None:
        with self.assertRaises(ValueError):
            InvoiceLine(item="Room Charges",quantity=2,rate=Decimal(100),amount=Decimal(300))


if __name__ == "__main__":
    unittest.main()

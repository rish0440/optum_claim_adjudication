"""Validate extracted line items, amounts, header and invoice totals."""
import csv
import io
from decimal import Decimal, InvalidOperation

from pydantic import ValidationError

from src.agentic_workflow.state import ClaimState
from src.schemas import InvoiceGroundTruth, InvoiceLine


def validate_csv(state: ClaimState) -> dict:
    """Fail closed on invalid CSV; never repair arithmetic by guessing."""
    try:
        text = state.get("extracted_csv", "")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames != ["Item", "Quantity", "Rate", "Amount"]:
            raise ValueError("CSV header must be exactly Item,Quantity,Rate,Amount")
        lines = []
        for position, row in enumerate(reader, 1):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {position}")
            lines.append(InvoiceLine(item=row["Item"].strip(),
                                     quantity=int(row["Quantity"]),
                                     rate=Decimal(row["Rate"]) if row["Rate"].strip() else None,
                                     amount=Decimal(row["Amount"])))
        if not lines:
            raise ValueError("Invoice CSV has no billable rows")
        if "invoice" in state:
            invoice = state["invoice"]
            if [line.model_dump() for line in invoice.lines] != [line.model_dump() for line in lines]:
                raise ValueError("Source invoice JSON disagrees with validated CSV")
        else:
            if not state.get("invoice_date"):
                raise ValueError("--date YYYY-MM-DD is required for CSV and image input")
            invoice = InvoiceGroundTruth(
                invoice_id=state.get("invoice_id") or "uploaded_invoice",
                image_file=state.get("image_path") or "provided.csv",
                invoice_date=state["invoice_date"],
                lines=lines,
                billed_total=sum((line.amount for line in lines), Decimal("0")),
                rate_column_visible=any(line.rate is not None for line in lines),
            )
        return {"invoice": invoice, "extracted_lines": lines,
                "trace": [*state.get("trace", []), "csv_validation"]}
    except (ValueError, KeyError, ValidationError, InvalidOperation) as exc:
        return {"errors": [*state.get("errors", []), str(exc)],
                "error_stage": "csv_validation",
                "trace": [*state.get("trace", []), "csv_validation"]}

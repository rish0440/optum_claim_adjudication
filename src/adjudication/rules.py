"""Policy rule resolution and safe input loading for deterministic adjudication.

Uses Phase 6's rule resolver; Milvus candidate search is optional. The model or
retriever does NOT set financial limits: amounts always come from policy.json.
"""

import csv
from datetime import date
from decimal import Decimal
from pathlib import Path

from src.policy import load_policy
from src.rag.resolver import PolicyRuleResolver
from src.schemas import InvoiceGroundTruth, InvoiceLine


LIMITED_BY_QUANTITY = {
    "per_day", "per_visit", "per_service", "per_session", "per_procedure",
}


def parse_invoice_csv(path: str | Path, *, invoice_date: str | date,
                      invoice_id: str | None = None) -> InvoiceGroundTruth:
    """Load an extracted CSV. A missing Rate remains null, never guessed."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["Item", "Quantity", "Rate", "Amount"]:
            raise ValueError("CSV header must be exactly Item,Quantity,Rate,Amount")
        lines = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed CSV row: unexpected comma or missing field")
            lines.append(InvoiceLine(
                item=row["Item"].strip(), quantity=int(row["Quantity"]),
                rate=Decimal(row["Rate"]) if row["Rate"].strip() else None,
                amount=Decimal(row["Amount"]),
            ))
    return InvoiceGroundTruth(
        invoice_id=invoice_id or path.stem,
        image_file="not_provided.csv", invoice_date=invoice_date,
        lines=lines, billed_total=sum((r.amount for r in lines), Decimal("0")),
        rate_column_visible=any(line.rate is not None for line in lines),
    )


def resolve_lines(invoice: InvoiceGroundTruth, *, policy=None, resolver=None,
                  retriever=None, hits_by_index: dict[int, list[dict]] | None = None) -> list[dict]:
    """Choose grounded rules, optionally taking semantic candidates from Milvus.

    Unknown or inconsistent choices become manual review, never a new rule.
    """
    policy = policy or load_policy()
    resolver = resolver or PolicyRuleResolver(policy)
    by_id = {rule.rule_id: rule for rule in
             (policy.coverage_rules + policy.exclusion_rules)}
    results = []
    for index, line in enumerate(invoice.lines):
        hits = (hits_by_index.get(index, []) if hits_by_index is not None
                else retriever.search(line.item, top_k=5) if retriever else [])
        decision = resolver.resolve(line.item, hits)
        rule_id = decision.get("rule_id")
        rule = by_id.get(rule_id)
        # Validate both the identity and resolver's status against source rules.
        if rule is None:
            decision = {
                "item": line.item, "status": "manual_review", "rule_id": None,
                "review_reason": decision.get("review_reason") or "No policy rule verified",
                "source": "data/policy.json",
            }
        elif ((rule_id.startswith("EXC-") and decision.get("status") != "excluded")
              or (rule_id.startswith("COV-") and decision.get("status") != "covered")):
            decision = {
                "item": line.item, "status": "manual_review", "rule_id": None,
                "review_reason": "Resolver status conflicts with source policy",
                "source": "data/policy.json",
            }
        results.append({"line": line, "resolution": decision,
                        "rule": by_id.get(decision.get("rule_id"))})
    return results

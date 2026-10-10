"""Deterministic insurance claim calculations; never uses LLM arithmetic.

Exact decimals, quantity-scoped and shared per-claim limits, exclusions,
reconciliation, and non-final manual-review results.
"""

from datetime import date
from decimal import Decimal, ROUND_HALF_UP

from src.adjudication.rules import LIMITED_BY_QUANTITY, resolve_lines
from src.adjudication.policy import load_policy
from src.schemas import ClaimResult, InvoiceGroundTruth, ItemDecision

CENT = Decimal("0.01")
ZERO = Decimal("0.00")


def money(value: Decimal) -> Decimal:
    """Use half-up rounding for INR amounts at the payable boundary."""
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def adjudicate_invoice(invoice: InvoiceGroundTruth, *, policy=None,
                       resolver=None, retriever=None, resolved_pairs=None) -> ClaimResult:
    """Adjudicate verified structured rows under the supplied dated policy.

    A per-claim cap is allocated to qualifying rows in invoice order; unlike
    a per-day cap, it cannot be multiplied by the row's quantity.
    Returns an *unfinalized* result if an item cannot be assigned a rule, or
    when the invoice date is outside policy validity.
    """
    policy = policy or load_policy()
    if not isinstance(invoice, InvoiceGroundTruth):
        invoice = InvoiceGroundTruth.model_validate(invoice)

    applicable = policy.effective_from <= invoice.invoice_date <= policy.effective_to
    used_per_claim: dict[str, Decimal] = {}
    decisions: list[ItemDecision] = []
    notes: list[str] = []
    if not applicable:
        notes.append(
            f"Invoice date {invoice.invoice_date.isoformat()} outside policy validity "
            f"({policy.effective_from} through {policy.effective_to}). No payout finalized."
        )
    else:
        notes.append("Per-claim limits are shared across invoice rows and assigned in row order.")

    # Resolve only if policy applies; otherwise do not make eligibility claims.
    if resolved_pairs is not None:
        # Guard the Phase 8 handoff: no mismatched invoice rows or modified
        # financial policy data may reach the calculator via the graph state.
        if len(resolved_pairs) != len(invoice.lines):
            raise ValueError("Resolved row count does not match the invoice")
        by_id = {r.rule_id: r for r in policy.coverage_rules + policy.exclusion_rules}
        for line, pair in zip(invoice.lines, resolved_pairs):
            if pair.get("line") != line:
                raise ValueError("Resolved rows disagree with invoice rows")
            rule_id = pair.get("resolution", {}).get("rule_id")
            actual_rule = by_id.get(rule_id)
            if rule_id and actual_rule is None:
                raise ValueError("Resolved rule ID not found in policy.json")
            if pair.get("rule") != actual_rule:
                raise ValueError("Resolved policy metadata differs from policy.json")
            expected_status = ("excluded" if rule_id and rule_id.startswith("EXC-")
                               else "covered" if rule_id else "manual_review")
            if pair.get("resolution", {}).get("status") != expected_status:
                raise ValueError("Resolved rule status contradicts policy.json")
    pairs = (resolved_pairs if resolved_pairs is not None else
             resolve_lines(invoice, policy=policy, resolver=resolver,
                           retriever=retriever)) if applicable else [
        {"line": line, "rule": None, "resolution": {
            "review_reason": "Invoice date outside policy effective period"}}
        for line in invoice.lines
    ]

    for pair in pairs:
        line = pair["line"]
        billed = money(line.amount)
        resolved = pair["resolution"]
        rule = pair["rule"]
        rule_id = rule.rule_id if rule else None
        category = rule.category if rule else None
        limit_type = getattr(rule, "limit_type", None)
        limit = getattr(rule, "maximum_reimbursement", None)
        approved = ZERO
        disallowed = ZERO
        pending = ZERO
        review_note = resolved.get("review_reason")
        if not rule:
            status = "manual_review"
            pending = billed
            explanation = f"Review required: {review_note or 'Policy rule not verified'}."
        elif rule_id.startswith("EXC-"):
            status = "denied"
            disallowed = billed
            explanation = f"Excluded under {rule_id} ({category}): {rule.conditions}"
        else:
            if limit_type == "actuals":
                approved = billed
                limit_text = "actual billed amount"
            elif limit_type == "per_claim":
                if limit is None:
                    raise ValueError(f"Missing cap on per-claim rule {rule_id}")
                cap = money(limit)
                remaining = max(ZERO, cap - used_per_claim.get(rule_id, ZERO))
                approved = min(billed, remaining)
                used_per_claim[rule_id] = used_per_claim.get(rule_id, ZERO) + approved
                limit_text = f"INR {cap} shared across claim; available before row INR {remaining}"
            elif limit_type in LIMITED_BY_QUANTITY:
                if limit is None:
                    raise ValueError(f"Missing {limit_type} cap for {rule_id}")
                cap = money(limit)
                approved = min(billed, cap * line.quantity)
                limit_text = f"INR {cap} {limit_type} × {line.quantity}"
            else:
                # Fail closed if the source policy introduces an unsupported scope.
                status = "manual_review"
                pending = billed
                explanation = f"Unsupported policy limit type: {limit_type!r}."
                decisions.append(ItemDecision(
                    item=line.item, quantity=line.quantity,
                    billed_amount=billed, approved_amount=ZERO,
                    disallowed_amount=ZERO, pending_review_amount=pending,
                    status=status, rule_id=rule_id, category=category,
                    limit_type=limit_type, cap_amount=limit,
                    rule_source="data/policy.json", explanation=explanation,
                    review_note=review_note,
                ))
                continue
            disallowed = billed - approved
            status = "approved" if approved == billed else ("denied" if approved == ZERO else "partial")
            explanation = (f"{rule_id} ({category}): {rule.conditions} "
                           f"Applied {limit_text}; eligible INR {approved} of INR {billed}.")
        decisions.append(ItemDecision(
            item=line.item, quantity=line.quantity,
            billed_amount=billed, approved_amount=approved,
            disallowed_amount=disallowed, pending_review_amount=pending,
            status=status, rule_id=rule_id, category=category,
            limit_type=limit_type, cap_amount=limit,
            rule_source="data/policy.json" if rule else None,
            explanation=explanation, review_note=review_note,
        ))
        if review_note and rule:
            notes.append(f"{line.item}: {review_note}")

    billed_total = sum((row.billed_amount for row in decisions), ZERO)
    eligible = sum((row.approved_amount for row in decisions), ZERO)
    disallowed = sum((row.disallowed_amount for row in decisions), ZERO)
    pending = sum((row.pending_review_amount for row in decisions), ZERO)
    copay_rule = policy.reimbursement_rules.copay
    rate = copay_rule.rate_percent if copay_rule.applicable else Decimal("0")
    copay = money(eligible * rate / Decimal("100"))
    has_review = pending != ZERO or any(row.status == "manual_review" for row in decisions)
    final = None if has_review else money(eligible - copay)
    if has_review:
        notes.append("Reimbursement not finalized while item/date review is pending; "
                     "eligible and co-pay totals are provisional.")
    return ClaimResult(
        invoice_id=invoice.invoice_id, policy_id=policy.policy_id,
        currency=policy.currency, claim_status="manual_review" if has_review else "finalized",
        decisions=decisions, billed_total=billed_total, eligible_total=eligible,
        disallowed_total=disallowed, pending_review_total=pending,
        copay_rate_percent=rate, copay_amount=copay,
        final_reimbursement=final, notes=notes,
    )

"""Typed contracts for extraction labels, policy rules, and later adjudication."""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InvoiceLine(BaseModel):
    """One line transcribed from an invoice (not including totals)."""

    item: str = Field(min_length=1)
    quantity: int = Field(ge=1)
    rate: Decimal | None = Field(default=None, ge=0)
    amount: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def check_line_arithmetic(self) -> "InvoiceLine":
        if self.rate is not None and self.rate * self.quantity != self.amount:
            raise ValueError(f"Quantity x rate != amount for {self.item!r}")
        return self


class InvoiceGroundTruth(BaseModel):
    """Hand-transcribed reference label for one held-out Optum invoice."""

    invoice_id: str
    image_file: str
    invoice_date: date
    lines: list[InvoiceLine] = Field(min_length=1)
    billed_total: Decimal = Field(ge=0)
    rate_column_visible: bool = True
    notes: str = ""

    @model_validator(mode="after")
    def check_invoice_total(self) -> "InvoiceGroundTruth":
        if sum((line.amount for line in self.lines), Decimal("0")) != self.billed_total:
            raise ValueError(f"Line amounts do not sum to billed total: {self.invoice_id}")
        if not self.rate_column_visible and any(line.rate is not None for line in self.lines):
            raise ValueError("Unprinted rates must be null, not inferred")
        return self


class CoverageRule(BaseModel):
    rule_id: str
    category: str
    covered: bool
    limit_type: str
    maximum_reimbursement: Decimal | None
    unit: str | None = None
    conditions: str


class ExclusionRule(BaseModel):
    rule_id: str
    category: str
    excluded: bool
    conditions: str


class CopayRule(BaseModel):
    applicable: bool
    rate_percent: Decimal = Field(ge=0, le=100)
    calculation_basis: str


class ReimbursementRules(BaseModel):
    rule_precedence: str
    line_item_calculation: str
    excluded_item_calculation: str
    copay: CopayRule
    final_reimbursement: str


class PolicyDocument(BaseModel):
    """Only fields needed for the Phase 1 policy validation."""

    model_config = ConfigDict(extra="ignore")
    policy_id: str
    policy_name: str
    version: str
    effective_from: date
    effective_to: date
    currency: str
    policy_sections: list[dict]
    coverage_rules: list[CoverageRule]
    exclusion_rules: list[ExclusionRule]
    reimbursement_rules: ReimbursementRules

    @model_validator(mode="after")
    def verify_unique_ids_and_dates(self) -> "PolicyDocument":
        rule_ids = [r.rule_id for r in self.coverage_rules + self.exclusion_rules]
        if len(rule_ids) != len(set(rule_ids)):
            raise ValueError("Policy contains duplicate rule IDs")
        if self.effective_from > self.effective_to:
            raise ValueError("Policy effective dates are reversed")
        return self


class RuleReference(BaseModel):
    rule_id: str
    category: str
    conditions: str


class ItemDecision(BaseModel):
    """Auditable decision for one invoice row; amounts are before co-pay."""

    item: str
    quantity: int = Field(ge=1)
    billed_amount: Decimal = Field(ge=0)
    approved_amount: Decimal = Field(ge=0)
    disallowed_amount: Decimal = Field(ge=0)
    pending_review_amount: Decimal = Field(default=Decimal("0"), ge=0)
    status: Literal["approved", "partial", "denied", "manual_review"]
    rule_id: str | None
    category: str | None = None
    limit_type: str | None = None
    cap_amount: Decimal | None = None
    rule_source: str | None = None
    explanation: str
    review_note: str | None = None

    @model_validator(mode="after")
    def reconcile_amounts(self) -> "ItemDecision":
        if self.approved_amount + self.disallowed_amount + self.pending_review_amount != self.billed_amount:
            raise ValueError("Item decision does not reconcile to billed amount")
        if self.status == "manual_review" and self.pending_review_amount != self.billed_amount:
            raise ValueError("Manual review rows must retain the billed amount as pending")
        return self


class ClaimResult(BaseModel):
    """Invoice adjudication, including safe handling of pending reviews."""

    invoice_id: str
    policy_id: str
    currency: str
    claim_status: Literal["finalized", "manual_review"]
    decisions: list[ItemDecision]
    billed_total: Decimal = Field(ge=0)
    eligible_total: Decimal = Field(ge=0)
    disallowed_total: Decimal = Field(ge=0)
    pending_review_total: Decimal = Field(ge=0)
    copay_rate_percent: Decimal = Field(ge=0, le=100)
    copay_amount: Decimal = Field(ge=0)
    final_reimbursement: Decimal | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def reconcile_claim(self) -> "ClaimResult":
        if self.eligible_total + self.disallowed_total + self.pending_review_total != self.billed_total:
            raise ValueError("Claim totals do not reconcile")
        if self.claim_status == "manual_review" and self.final_reimbursement is not None:
            raise ValueError("Cannot finalize reimbursement while review is pending")
        if self.claim_status == "finalized" and self.final_reimbursement is None:
            raise ValueError("Finalized claim requires a reimbursement amount")
        return self

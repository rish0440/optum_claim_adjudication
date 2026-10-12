"""Strict agent contracts. Model suggestions never replace calculator outputs."""
import re
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt
from src.schema import ClaimResult


class AgentFormatError(ValueError):
    """Recoverable JSON/schema formatting failure; eligible for a bounded retry."""


class AgentSafetyError(ValueError):
    """Unsupported assertion or instruction-like input; never retried."""


def reject_instructions(text: str):
    # A conservative input screen, supplemented by data/instruction separation,
    # citation allowlists and exact fact validation. It cannot detect every injection.
    if re.search(r"ignore\s+(?:all\s+|previous\s+|the\s+)*(?:instructions|rules|policy)|"
                 r"(?:system|assistant|developer)\s*:|<\|(?:im_start|im_end)|"
                 r"(?:reveal|print|read)\s+(?:secrets|credentials|api.key|\.env)|"
                 r"(?:execute|run)\s+(?:shell|python|command)|override\s+(?:policy|instructions)",
                 text, re.IGNORECASE):
        raise AgentSafetyError("Instruction-like content rejected")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicySuggestion(StrictModel):
    item_index: StrictInt = Field(ge=0)
    suggested_status: Literal["COVERED", "EXCLUDED", "REVIEW_REQUIRED"]
    policy_ids: list[str] = Field(max_length=3)
    reason: str = Field(min_length=1, max_length=500)
    needs_review: StrictBool
    supported_conditions: str | None = Field(default=None, max_length=500,
        description="Quoted canonical conditions, not proof that clinical conditions were met")


class Suggestions(StrictModel):
    items: list[PolicySuggestion] = Field(min_length=1)


class CandidateSelection(StrictModel):
    """Small wire contract only; canonical IDs/conditions are resolved in Python."""
    item_index: StrictInt = Field(ge=0)
    candidate_index: StrictInt | None
    suggested_status: Literal["COVERED", "EXCLUDED", "REVIEW_REQUIRED"]
    basis: Literal["RULE_MATCH", "CONDITIONS_UNCERTAIN", "NO_EVIDENCE", "DATE_OUTSIDE", "CONFLICT"]


class CandidateSelections(StrictModel):
    items: list[CandidateSelection] = Field(min_length=1, max_length=50)


class FactSelection(StrictModel):
    item_index: StrictInt = Field(ge=0)
    status: Literal["approved", "partial", "denied", "manual_review"]
    fact_indices: list[StrictInt] = Field(min_length=1, max_length=2)
    concerns: list[StrictInt] = Field(max_length=2,
        description="Select only concern indices supplied for this item; [] means no concern")


class FactSelections(StrictModel):
    items: list[FactSelection] = Field(min_length=1, max_length=50)


class ItemReview(StrictModel):
    item_index: StrictInt = Field(ge=0)
    status: Literal["approved", "partial", "denied", "manual_review"]
    policy_ids: list[str] = Field(max_length=3)
    approved_amount: Decimal = Field(ge=0)
    disallowed_amount: Decimal = Field(ge=0)
    pending_review_amount: Decimal = Field(ge=0)
    explanation: str = Field(min_length=1, max_length=1000)
    needs_review: StrictBool


class ReviewedTotals(StrictModel):
    eligible_total: Decimal = Field(ge=0)
    copay_amount: Decimal = Field(ge=0)
    final_reimbursement: Decimal | None = Field(default=None, ge=0)


class ReviewOutput(StrictModel):
    items: list[ItemReview] = Field(min_length=1)
    totals: ReviewedTotals


class WorkflowReport(StrictModel):
    request_id: str
    thread_id: str | None
    status: Literal["COMPLETED", "REVIEW_REQUIRED", "FAILED"]
    reviewed: bool
    claim: ClaimResult | None
    extracted_csv: str
    evidence: dict[int, list[dict]]
    agent_1: Suggestions | None
    agent_2: ReviewOutput | None
    explanations: list[dict]
    errors: list[str]
    warnings: list[str]
    retries: dict[str, int]
    trace: list[dict]
    metadata: dict

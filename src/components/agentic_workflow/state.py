"""Typed LangGraph state for the Phase 8 claim workflow.

Only one input source is accepted per run. Data stays typed internally so the
Phase 7 validator and calculator remain the source of truth. No fake VLM output.
"""
from typing import Any, Literal, TypedDict

from src.schema import ClaimResult, InvoiceGroundTruth, InvoiceLine
from src.components.agentic_workflow.contracts import Suggestions, ReviewOutput, WorkflowReport


class ClaimState(TypedDict, total=False):
    """Typed values passed between legacy and two-agent LangGraph nodes."""
    # Input modes: verified JSON, previously extracted CSV, raw CSV text, or image.
    input_mode: Literal["invoice_json", "csv_file", "csv_text", "image"]
    invoice_json_path: str
    csv_path: str
    csv_text: str
    image_path: str
    invoice_date: str                  # Mandatory for CSV and image input.
    invoice_id: str
    use_remote_vlm: bool              # Explicit opt-in, never pretend to extract.
    use_milvus: bool                  # Explicit opt-in for semantic retrieval.
    strict_rag: bool                  # Fail if Milvus unavailable (instead of transparent fallback).
    extracted_csv: str
    extraction_source: str
    extracted_lines: list[InvoiceLine]
    invoice: InvoiceGroundTruth
    candidate_rules: dict[int, list[dict[str, Any]]]
    resolved_pairs: list[dict[str, Any]]
    claim_result: ClaimResult
    result: dict[str, Any]
    rag_status: str
    metadata: dict[str, Any]
    trace: list[str]
    warnings: list[str]
    errors: list[str]
    error_stage: str
    # Request-scoped two-agent workflow fields (legacy callers remain compatible).
    request_id: str
    thread_id: str | None
    workflow_status: Literal["RUNNING", "COMPLETED", "REVIEW_REQUIRED", "FAILED"]
    csv_validated: bool
    policy_resolutions: dict[int, dict[str, Any]]
    agent_suggestions: Suggestions
    agent_review: ReviewOutput
    final_report: WorkflowReport
    retry_counters: dict[str, int]
    retry_node: str | None
    max_retries: int
    execution_trace: list[dict[str, Any]]
    model_ids: dict[str, str]
    deadline_monotonic: float
    started_monotonic: float

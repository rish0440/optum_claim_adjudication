"""Produce one JSON-serializable response for successful, review, and failed runs."""
from src.agentic_workflow.state import ClaimState


def build_response(state: ClaimState) -> dict:
    """Report extraction provenance and whether the actual vector search ran."""
    trace = [*state.get("trace", []), "response_builder"]
    errors = state.get("errors", [])
    claim = state.get("claim_result")
    payload = {
        "status": "error" if errors else (claim.claim_status if claim else "error"),
        "input_mode": state.get("input_mode"),
        "extraction_source": state.get("extraction_source", "none"),
        "rag_status": state.get("rag_status", "not_run"),
        "metadata": state.get("metadata", {}),
        "trace": trace,
        "warnings": state.get("warnings", []),
        "errors": errors,
        "error_stage": state.get("error_stage") if errors else None,
        "claim": claim.model_dump(mode="json") if claim and not errors else None,
        "extracted_csv": state.get("extracted_csv", "") if not errors else "",
    }
    if state.get("input_mode") == "invoice_json":
        payload["warnings"] = [*payload["warnings"],
                               "Ground-truth JSON used for component testing; not a VLM prediction"]
    return {"result": payload, "trace": trace}

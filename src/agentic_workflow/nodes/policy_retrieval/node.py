"""Run the Milvus top-K search when requested, or explicitly use policy-only mode.

Milvus failures do not turn into invented candidates. They trigger a clearly
marked fallback (unless --strict-rag requires the graph to fail closed).
"""
from src.agentic_workflow.state import ClaimState
from src.settings import get_settings


def retrieve_policy(state: ClaimState, *, retriever=None) -> dict:
    """Collect semantic candidates by invoice row index with real rule IDs."""
    trace = [*state.get("trace", []), "policy_retrieval"]
    if not state.get("use_milvus", False):
        return {"candidate_rules": {}, "rag_status": "policy_only_no_vector_search",
                "trace": trace}
    try:
        retriever = retriever or __import__(
            "src.rag.retriever", fromlist=["PolicyRetriever"]
        ).PolicyRetriever()
        candidates = {}
        for index, line in enumerate(state["invoice"].lines):
            candidates[index] = retriever.search(line.item, top_k=get_settings().rag.top_k)
        return {"candidate_rules": candidates, "rag_status": "milvus_live",
                "trace": trace}
    except Exception as exc:  # pymilvus has varied exception types by version
        message = f"Milvus policy search unavailable ({type(exc).__name__}): {exc}"
        if state.get("strict_rag", False):
            return {"errors": [*state.get("errors", []), message],
                    "error_stage": "policy_retrieval", "trace": trace}
        return {"candidate_rules": {}, "rag_status": "policy_fallback_after_milvus_error",
                "warnings": [*state.get("warnings", []), message + "; using exact policy resolver only"],
                "trace": trace}
